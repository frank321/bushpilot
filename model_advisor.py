#!/usr/bin/env python3
"""
model_advisor.py — hardware detection + per-model context-size recommendation +
interactive model picker for start_litellm_proxy.sh.

Design notes (see HARDWARE_SPEED_ESTIMATION.md for the full research behind this):
- Context-size recommendation is an EXACT calculation from each model's "kv_config"
  field in models.json (num_kv_heads, head_dim, num_layers, and — critically —
  layer_types/sliding_window to separate context-growing layers from fixed-cost ones).
  kv_config is extracted once from each model's HF config.json ahead of time and baked
  into models.json, rather than read from a local download at runtime — this must work
  identically on a machine that hasn't downloaded any models yet. Not an estimate.
- Token-speed is a genuine ESTIMATE (memory-bandwidth-bound physics + one real
  calibration point per model class), always labeled as such, and left as null when we
  don't have the data to back even a rough number (see MoE handling below).
- Interactive UI goes to stderr. The final machine-readable selection is the ONLY thing
  printed to stdout, as one JSON line, so the calling bash script can capture it cleanly
  with `SELECTION=$(python3 model_advisor.py)` without scraping prompts out of it.

Usage:
  python3 model_advisor.py                  → interactive menu, prints selection JSON
  python3 model_advisor.py --list-json       → non-interactive: print every model's
                                                computed recommendation as JSON, no prompt
  python3 model_advisor.py --tool claude     → also estimate & factor in Claude Code's/
                                                OpenCode's context preload overhead
                                                (system prompt, built-in tools, this
                                                directory's CLAUDE.md/AGENTS.md,
                                                installed skills, configured MCP
                                                servers) — see estimate_preload_overhead
"""
import json
import os
import re
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_JSON = os.path.join(SCRIPT_DIR, "models.json")

MEMORY_CONSTRAINED_KV_BITS = 8   # only used as a FALLBACK now — see recommend_kv_bits_and_context
UNQUANTIZED_KV_BITS = 16        # what MLX uses when --kv-bits is simply not passed (float16 KV)
SAFETY_BUFFER_GB = 3.0       # headroom kept free beyond model+KV, for OS/background apps/spikes
MIN_USABLE_CONTEXT = 12288   # below this, a context window isn't really usable — also used
                              # as the floor for CONVERSATION HEADROOM after tool preload
                              # (see estimate_preload_overhead below): a model that only
                              # "fits" on paper but leaves less than this after Claude
                              # Code's/OpenCode's own preload isn't usable either.

# --- Tool preload-overhead estimation ---------------------------------------------
#
# Claude Code and OpenCode both load a system prompt, built-in tool schemas, any
# CLAUDE.md/AGENTS.md, configured skills, and configured MCP servers' tool schemas
# BEFORE the user types anything. On a small local context window, that preload alone
# can eat most or all of the budget. Below, doc/skill/MCP-COUNT numbers are REAL
# measurements of what's actually on this machine; only the baseline system-prompt +
# built-in-tools figure and the per-MCP-server token cost are approximations, and both
# are labeled and sourced as such — not fabricated, but not exact either.
#
# TOOL_BASELINE_PRELOAD_TOKENS: system prompt + built-in tool schemas with NO MCP
# servers or skills configured yet. Drawn from public reports as of 2026-09, NOT
# measured on this machine: a Sep 2026 HN thread comparing Claude Code (~33k tokens
# before the first user token) vs OpenCode (~7k) in one config; separate token
# breakdowns putting Claude Code's system prompt at ~2.5-3.6k and built-in tool defs at
# ~14-17k (~17-20k combined). Anthropic's own claude-code issue tracker (#45188,
# #46339) shows this figure has swung by tens of thousands of tokens between patch
# releases in the same year — treat these as a rough floor, not a stable spec.
TOOL_BASELINE_PRELOAD_TOKENS = {
    "claude": 18000,
    "opencode": 7000,   # bare-install figure (~0 skills/MCP) — real OpenCode sessions
                         # with skills configured run far higher; see
                         # measure_skills_preload's docstring for why that growth isn't
                         # modeled for OpenCode specifically.
}

# Rough, clearly-approximate per-server token cost — public reports range from ~800
# tokens (a couple of terse tools) to 5,000-8,000+ tokens (verbose internal servers),
# with typical multi-tool setups landing around 2,000-3,500/server. This is the
# unavoidably-fuzzy part; the SERVER COUNT it gets multiplied by is a real count from
# this machine's own config (see measure_mcp_servers).
MCP_SERVER_TOKENS_ESTIMATE = 3000

# Sourced in HARDWARE_SPEED_ESTIMATION.md §4 — cross-checked across two independent
# sources for M5 Pro/Max specifically; M1-M4 figures are long-established/widely
# reproduced. Treat M5-and-later as less certain (documented there, not repeated here).
CHIP_BANDWIDTH_GB_S = {
    "Apple M1": 68.25, "Apple M1 Pro": 200, "Apple M1 Max": 400, "Apple M1 Ultra": 800,
    "Apple M2": 100, "Apple M2 Pro": 200, "Apple M2 Max": 400, "Apple M2 Ultra": 800,
    "Apple M3": 100, "Apple M3 Pro": 150, "Apple M3 Max": 350, "Apple M3 Ultra": 819,
    "Apple M4": 120, "Apple M4 Pro": 273, "Apple M4 Max": 546,
    "Apple M5": 153, "Apple M5 Pro": 307, "Apple M5 Max": 537, "Apple M5 Ultra": 1200,
    "Apple M6": 170,
}
# Derived live in HARDWARE_SPEED_ESTIMATION.md §6 from 3 real measurements (0.86, 0.97,
# 0.88) plus one independent published figure (0.88) — 0.87 sits near the low end of
# that cluster deliberately, so the estimate errs toward underselling speed rather than
# overpromising it.
SPEED_CORRECTION_FACTOR = 0.87


def log(msg):
    print(msg, file=sys.stderr)


def get_current_wired_gb():
    """Returns CURRENTLY-USED wired memory (baseline OS/background-app consumption),
    NOT free memory — the caller must compute free_gb = total_ram_gb - this value.
    Confirmed live methodology for why wired (not free/used from `top`) is the right
    metric to baseline against — see HARDWARE_SPEED_ESTIMATION.md §2 and
    LITELLM_ROUTING_PLAN.md §4.17."""
    out = subprocess.check_output(["vm_stat"], text=True)
    page_size = 4096
    header = out.splitlines()[0]
    if "page size of" in header:
        page_size = int(header.split("page size of")[1].strip().split()[0])
    for line in out.splitlines():
        if "Pages wired down" in line:
            pages = int(line.split(":")[1].strip().rstrip("."))
            return pages * page_size / 1024**3
    raise RuntimeError("Could not find 'Pages wired down' in vm_stat output")


def get_chip_info():
    """Confirmed live: system_profiler SPHardwareDataType SPDisplaysDataType completes
    in ~0.45s on this machine — fast enough to run on every launch, no caching needed."""
    out = subprocess.check_output(
        ["system_profiler", "SPHardwareDataType", "SPDisplaysDataType"], text=True
    )
    chip = None
    total_ram_gb = None
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("Chip:"):
            chip = s.split(":", 1)[1].strip()
        elif s.startswith("Memory:"):
            try:
                total_ram_gb = float(s.split(":", 1)[1].strip().split()[0])
            except (ValueError, IndexError):
                pass
    # "Total Number of Cores" appears twice: once under Hardware (CPU), once under
    # Displays (GPU). Confirmed live via the exact output this produced on this machine.
    core_lines = [line.strip() for line in out.splitlines() if "Total Number of Cores" in line]
    gpu_cores = None
    if len(core_lines) >= 2:
        try:
            gpu_cores = int(core_lines[1].split(":")[1].strip().split()[0])
        except (ValueError, IndexError):
            pass
    return chip, gpu_cores, total_ram_gb


def count_tokens_rough(text):
    """chars/4 heuristic — a standard rough estimate, not a real tokenizer run. Good
    enough to sanity-check "does this fit at all", not exact token accounting."""
    return max(0, len(text) // 4)


def measure_doc_preload(cwd):
    """REAL measurement (not an estimate) of CLAUDE.md/AGENTS.md token cost — these
    files get read verbatim into context, so their actual size is directly countable."""
    total = 0
    found = []
    candidates = [
        os.path.join(cwd, "CLAUDE.md"),
        os.path.join(cwd, "AGENTS.md"),
        os.path.expanduser("~/.claude/CLAUDE.md"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            except OSError:
                continue
            tokens = count_tokens_rough(text)
            total += tokens
            found.append((path, tokens))
    return total, found


_SKILL_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---", re.DOTALL)


def _parse_skill_frontmatter(text):
    m = _SKILL_FRONTMATTER_RE.match(text)
    if not m:
        return None, None
    name = description = None
    for line in m.group(1).splitlines():
        if line.startswith("name:"):
            name = line.split(":", 1)[1].strip()
        elif line.startswith("description:"):
            description = line.split(":", 1)[1].strip()
    return name, description


def measure_skills_preload(cwd):
    """REAL measurement of currently-installed Claude Code skills' name+description
    text — that's what actually gets injected into the skill listing at session start,
    per public per-skill-cost reports of ~75-150 tokens/skill (name + description +
    Claude Code's XML-ish wrapper around each entry). Only covers the two documented
    skill directories below — misses plugin-bundled skills, and doesn't model
    OpenCode's skill mechanism at all (its on-disk convention isn't confirmed here),
    so this under-counts for OpenCode and for plugin-heavy Claude Code setups."""
    total = 0
    count = 0
    for base in (os.path.join(cwd, ".claude", "skills"), os.path.expanduser("~/.claude/skills")):
        if not os.path.isdir(base):
            continue
        for entry in os.listdir(base):
            skill_md = os.path.join(base, entry, "SKILL.md")
            if not os.path.isfile(skill_md):
                continue
            try:
                with open(skill_md, encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            except OSError:
                continue
            name, description = _parse_skill_frontmatter(text)
            if description is None:
                continue
            count += 1
            total += count_tokens_rough((name or "") + description) + 20  # +wrapper allowance
    return total, count


def measure_mcp_servers(cwd):
    """Counts CONFIGURED MCP servers for REAL, from known config file locations — only
    the per-server token COST applied to this count (MCP_SERVER_TOKENS_ESTIMATE) is an
    estimate, not this count itself. May miss non-standard config locations."""
    count = 0
    for path in (
        os.path.join(cwd, ".mcp.json"),
        os.path.expanduser("~/.claude.json"),
        os.path.expanduser("~/.claude/mcp.json"),
    ):
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        servers = data.get("mcpServers")
        if isinstance(servers, dict):
            count += len(servers)
    return count


def estimate_preload_overhead(tool, cwd):
    """Combines the one approximate number (tool baseline) and one approximate
    per-unit cost (MCP_SERVER_TOKENS_ESTIMATE) with everything else measured for real
    on this machine, for the given launch directory."""
    baseline = TOOL_BASELINE_PRELOAD_TOKENS.get(tool, 0)
    doc_tokens, doc_files = measure_doc_preload(cwd)
    skill_tokens, skill_count = measure_skills_preload(cwd)
    mcp_count = measure_mcp_servers(cwd)
    mcp_tokens = mcp_count * MCP_SERVER_TOKENS_ESTIMATE
    return {
        "tool": tool,
        "cwd": cwd,
        "baseline_tokens": baseline,
        "doc_tokens": doc_tokens,
        "doc_files": doc_files,
        "skill_tokens": skill_tokens,
        "skill_count": skill_count,
        "mcp_tokens": mcp_tokens,
        "mcp_count": mcp_count,
        "total": baseline + doc_tokens + skill_tokens + mcp_tokens,
    }


def compute_kv_formula(kv_config, kv_bits=UNQUANTIZED_KV_BITS):
    """Returns (bytes_per_growing_token, fixed_kv_bytes_total, detail_dict) or None if
    kv_config is missing fields we need. See HARDWARE_SPEED_ESTIMATION.md §9 — this
    generalizes both the Qwen3.5 hybrid linear/full-attention case and the Gemma-4
    sliding/full-attention case with one formula: only 'full_attention' layers grow
    with context; 'sliding_attention' layers cost a FIXED amount (sliding_window
    tokens' worth, confirmed via gemma-4's own layer_types + sliding_window fields);
    'linear_attention' layers (Qwen3.5) are approximated as ~0 fixed cost — a real,
    documented approximation, not a solved calculation (their recurrent state is small
    but non-zero and not precisely modeled here).

    kv_config comes from models.json's per-model "kv_config" field, extracted once
    ahead of time from each model's HF config.json (see extraction done when models.json
    was generated) rather than read from disk here — this must work identically on a
    freshly-downloaded model AND on a machine where nothing has been downloaded yet."""
    if kv_config is None:
        return None
    num_kv_heads = kv_config.get("num_key_value_heads")
    head_dim = kv_config.get("head_dim")
    num_layers = kv_config.get("num_hidden_layers")
    if not all([num_kv_heads, head_dim, num_layers]):
        return None

    bytes_per_elem = kv_bits / 8
    layer_types = kv_config.get("layer_types")
    sliding_window = kv_config.get("sliding_window")

    if layer_types:
        growing = sum(1 for t in layer_types if t == "full_attention")
        sliding = sum(1 for t in layer_types if t == "sliding_attention")
        linear = sum(1 for t in layer_types if t == "linear_attention")
        accounted = growing + sliding + linear
        unknown = len(layer_types) - accounted
        # Unrecognized layer-type name: treat as growing (full attention) — the safer
        # overestimate of memory need rather than silently underestimating it.
        growing += unknown
    else:
        growing, sliding, linear = num_layers, 0, 0

    bytes_per_growing_token = 2 * growing * num_kv_heads * head_dim * bytes_per_elem
    fixed_bytes = 0.0
    if sliding and sliding_window:
        fixed_bytes += 2 * sliding * num_kv_heads * head_dim * bytes_per_elem * sliding_window

    detail = {
        "growing_layers": growing, "sliding_layers": sliding, "linear_layers": linear,
        "num_kv_heads": num_kv_heads, "head_dim": head_dim, "sliding_window": sliding_window,
        "kv_bits": kv_bits,
    }
    return bytes_per_growing_token, fixed_bytes, detail


def recommend_max_context(model_entry, free_wired_gb, kv_bits=UNQUANTIZED_KV_BITS,
                           safety_buffer_gb=SAFETY_BUFFER_GB):
    """Returns (recommended_tokens_or_None, reason_string, detail_dict_or_None)."""
    available_gb = free_wired_gb - model_entry["size_gb"] - safety_buffer_gb
    if available_gb <= 0:
        return None, "insufficient_memory_for_model_weights", None

    formula = compute_kv_formula(model_entry.get("kv_config"), kv_bits)
    if formula is None:
        return None, "architecture_fields_missing_from_config", None
    bytes_per_growing_token, fixed_bytes, detail = formula

    available_bytes = available_gb * 1024**3
    usable_bytes = available_bytes - fixed_bytes
    if usable_bytes <= 0:
        return None, "insufficient_memory_for_fixed_kv_cost", detail

    if bytes_per_growing_token <= 0:
        max_ctx = model_entry["max_context_architectural"]
    else:
        max_ctx = int(usable_bytes / bytes_per_growing_token)
        max_ctx = min(max_ctx, model_entry["max_context_architectural"])

    if max_ctx < MIN_USABLE_CONTEXT:
        return None, "computed_context_too_small_to_be_useful", detail

    # Round DOWN to a clean, conservative tier rather than showing a falsely-precise
    # number like "58,392 tokens" — this is an exact calculation but the inputs
    # (free memory especially) fluctuate, so a clean round number is more honest about
    # the real precision available.
    for tier in (262144, 131072, 65536, 32768, 16384, 8192, 4096):
        if max_ctx >= tier:
            return tier, "computed", detail
    return max_ctx, "computed", detail


def estimate_speed_tok_s(model_entry, chip_name, gpu_cores):
    """Returns (estimate_or_None, reason_string). Deliberately returns None rather than
    a guess when we don't have real active-weight data — see gemma-4-26b-a4b-it-4bit's
    entry below, whose MoE routing config is missing num_experts_per_tok."""
    bandwidth = CHIP_BANDWIDTH_GB_S.get(chip_name)
    if bandwidth is None:
        return None, f"no bandwidth data for chip '{chip_name}'"

    active_gb = model_entry.get("active_weight_gb")
    if active_gb is None:
        if model_entry.get("is_moe"):
            return None, "MoE model with unknown active-parameter fraction — no reliable estimate without a live benchmark"
        active_gb = model_entry["size_gb"]  # dense model: full weight size is the active weight

    return round(SPEED_CORRECTION_FACTOR * bandwidth / active_gb, 1), "estimated"


def _apply_preload(ctx, reason, preload_overhead):
    """Shared by both kv_bits branches of recommend_kv_bits_and_context below — applies
    the same tool-preload check build_recommendations always did, so both the
    unquantized and quantized attempts are held to the identical usability bar."""
    fits = ctx is not None
    context_after_preload = None
    if fits and preload_overhead:
        context_after_preload = ctx - preload_overhead
        if context_after_preload < MIN_USABLE_CONTEXT:
            fits = False
            reason = "insufficient_context_after_tool_preload"
    return fits, reason, context_after_preload


def recommend_kv_bits_and_context(model_entry, free_wired_gb, preload_overhead=0,
                                   safety_buffer_gb=SAFETY_BUFFER_GB):
    """Decides BOTH whether to quantize the KV cache AND the resulting context size, in
    one pass — see HARDWARE_SPEED_ESTIMATION.md §14 for the full reasoning and its
    measured basis (§10/§12).

    KV-cache quantization (--kv-bits 8) has a real decode-speed cost (up to 8.7% on
    fast/small models, confirmed live) and confirmed ZERO memory benefit at short
    context (peak_memory was byte-identical with/without it in every short-context
    benchmark run for this project) — its only real purpose is letting a BIGGER
    context fit in the SAME free memory. So: try unquantized first (kv_bits=None in the
    return value means "omit --kv-bits entirely" downstream); only fall back to
    kv_bits=8 if unquantized genuinely doesn't fit. This is a binary fits/doesn't-fit
    decision at whatever context size the model would otherwise get, not a search for
    the context-maximizing bit-width — see §14's own caveat for why this is grounded in
    real data but that data's scope (short context only) doesn't fully cover the regime
    (long context) this decision matters most for.

    Returns dict: kv_bits (None or MEMORY_CONSTRAINED_KV_BITS), recommended_context,
    context_reason, kv_formula_detail, context_after_preload, fits.
    """
    ctx_u, reason_u, detail_u = recommend_max_context(
        model_entry, free_wired_gb, kv_bits=UNQUANTIZED_KV_BITS,
        safety_buffer_gb=safety_buffer_gb,
    )
    fits_u, reason_u, after_u = _apply_preload(ctx_u, reason_u, preload_overhead)
    if fits_u:
        return {
            "kv_bits": None, "recommended_context": ctx_u, "context_reason": reason_u,
            "kv_formula_detail": detail_u, "context_after_preload": after_u, "fits": True,
        }

    ctx_q, reason_q, detail_q = recommend_max_context(
        model_entry, free_wired_gb, kv_bits=MEMORY_CONSTRAINED_KV_BITS,
        safety_buffer_gb=safety_buffer_gb,
    )
    fits_q, reason_q, after_q = _apply_preload(ctx_q, reason_q, preload_overhead)
    if fits_q:
        return {
            "kv_bits": MEMORY_CONSTRAINED_KV_BITS, "recommended_context": ctx_q,
            "context_reason": reason_q, "kv_formula_detail": detail_q,
            "context_after_preload": after_q, "fits": True,
        }

    # Neither fits — report the quantized attempt's reason, since it's the more
    # forgiving of the two (smaller bytes-per-token) and so the more informative "even
    # our best effort doesn't fit" explanation for the user.
    return {
        "kv_bits": None, "recommended_context": ctx_q, "context_reason": reason_q,
        "kv_formula_detail": detail_q, "context_after_preload": after_q, "fits": False,
    }


def build_recommendations(models, free_wired_gb, chip_name, gpu_cores, preload_overhead=0):
    rows = []
    for m in models:
        kv_result = recommend_kv_bits_and_context(m, free_wired_gb, preload_overhead)
        speed, speed_reason = estimate_speed_tok_s(m, chip_name, gpu_cores)
        rows.append({
            "model": m,
            "recommended_context": kv_result["recommended_context"],
            "context_reason": kv_result["context_reason"],
            "kv_formula_detail": kv_result["kv_formula_detail"],
            "kv_bits": kv_result["kv_bits"],
            "estimated_tok_s": speed,
            "speed_reason": speed_reason,
            "context_after_preload": kv_result["context_after_preload"],
            "fits": kv_result["fits"],
        })
    return rows


DIM = "\033[2m"
RESET = "\033[0m"


def print_preload_banner(preload):
    tool_label = {"claude": "Claude Code", "opencode": "OpenCode"}.get(preload["tool"], preload["tool"])
    log(f"  Estimated preload overhead for {tool_label} in {preload['cwd']}:")
    log(f"    baseline (system prompt + built-in tools): ~{preload['baseline_tokens']:,} tokens")
    log("      (public reports, NOT measured on this machine — varies a lot by version)")
    if preload["doc_files"]:
        for path, tokens in preload["doc_files"]:
            log(f"    {os.path.basename(path)}: {tokens:,} tokens (measured: {path})")
    else:
        log("    AGENTS.md/CLAUDE.md: none found")
    if preload["skill_count"]:
        log(f"    skills ({preload['skill_count']} found): ~{preload['skill_tokens']:,} tokens (measured)")
    else:
        log("    skills: none found in .claude/skills")
    if preload["mcp_count"]:
        log(f"    MCP servers ({preload['mcp_count']} found): ~{preload['mcp_tokens']:,} tokens "
            f"(count measured, ~{MCP_SERVER_TOKENS_ESTIMATE:,}/server is a rough estimate)")
    else:
        log("    MCP servers: none found")
    log(f"    {'-' * 60}")
    log(f"    total estimated preload: ~{preload['total']:,} tokens")
    log(f"  A model needs recommended context - preload >= {MIN_USABLE_CONTEXT:,} tokens of real")
    log("  conversation headroom to be usable, i.e. at least "
        f"~{preload['total'] + MIN_USABLE_CONTEXT:,} tokens total.")
    log("=" * 68)
    log("")


def print_menu(rows, free_wired_gb, chip_name, gpu_cores, total_ram_gb, preload=None):
    current_wired_gb = (total_ram_gb - free_wired_gb) if total_ram_gb else None
    log("=" * 68)
    log("  Local Model Advisor")
    log("=" * 68)
    log(f"  Chip: {chip_name or 'unknown'}   Total RAM: {total_ram_gb or '?'} GB")
    log(f"  Currently in use (wired): {current_wired_gb:.1f} GB   Free: {free_wired_gb:.1f} GB")
    log("=" * 68)
    log("  Model availability and recommended context size below are calculated from")
    log("  your system's CURRENT free memory, not a fixed spec. Close background apps")
    log("  or free up memory and re-run this script to refresh what's possible.")
    log("=" * 68)
    log("")
    if preload:
        print_preload_banner(preload)

    fitting = [r for r in rows if r["fits"]]
    not_fitting = [r for r in rows if not r["fits"]]

    if not fitting:
        log("  No model in the list fits your current free memory.")
        log("  Close some background apps and try again.")
        log("")

    i = 0
    for r in rows:
        m = r["model"]
        if r["fits"]:
            i += 1
            speed_str = f"~{r['estimated_tok_s']} tok/s (estimated)" if r["estimated_tok_s"] else "speed unknown"
            if r["kv_bits"]:
                kv_str = f"  [KV cache quantized to {r['kv_bits']}-bit — needed to fit this context in free memory]"
            else:
                kv_str = "  [KV cache unquantized — full decode speed, plenty of free memory for this context]"
            log(f"  {i}) {m['id']}  [{m['display']['tagline']}]")
            log(f"       {m['display']['good_for']}")
            headroom_str = ""
            if r["context_after_preload"] is not None:
                headroom_str = f"  (~{r['context_after_preload']:,} left for conversation after preload)"
            log(f"       size={m['size_gb']}GB  max context={m['max_context_architectural']:,} tokens  "
                f"recommended context={r['recommended_context']:,} tokens  {speed_str}{headroom_str}")
            log(f"      {kv_str}")
            for ex in m["display"]["usage_examples"]:
                log(f"       e.g. {ex}")
            if m["display"].get("known_issues"):
                log(f"       ⚠ {m['display']['known_issues']}")
        else:
            context_reason = r["context_reason"]
            if context_reason == "insufficient_memory_for_model_weights":
                reason = "not enough free memory to load this model"
            elif context_reason == "insufficient_memory_for_fixed_kv_cost":
                reason = "not enough free memory left for this model's context cache"
            elif context_reason == "computed_context_too_small_to_be_useful":
                reason = f"would only fit under {MIN_USABLE_CONTEXT:,} tokens of context — not usable"
            elif context_reason == "architecture_fields_missing_from_config":
                reason = "can't compute (model not downloaded yet / config unavailable)"
            elif context_reason == "insufficient_context_after_tool_preload":
                reason = (
                    f"{r['recommended_context']:,} tokens of context, but tool preload leaves only "
                    f"~{r['recommended_context'] - (preload['total'] if preload else 0):,} for the conversation — below the "
                    f"{MIN_USABLE_CONTEXT:,}-token floor"
                )
            else:
                reason = context_reason
            log(f"{DIM}  —) {m['id']}  [{m['display']['tagline']}]")
            log(f"       {m['display']['good_for']}")
            log(f"       size={m['size_gb']}GB  max context={m['max_context_architectural']:,} tokens  "
                f"✗ unavailable now: {reason}{RESET}")
        log("")

    if not_fitting:
        log(f"  ({len(not_fitting)} model(s) above are greyed out — they don't fit your current free memory)")
        log("")

    return fitting


def main():
    list_json = "--list-json" in sys.argv
    tool = None
    if "--tool" in sys.argv:
        idx = sys.argv.index("--tool")
        if idx + 1 < len(sys.argv):
            tool = sys.argv[idx + 1]

    with open(MODELS_JSON) as f:
        catalog = json.load(f)
    models = catalog["models"]

    current_wired_gb = get_current_wired_gb()
    chip_name, gpu_cores, total_ram_gb = get_chip_info()
    if total_ram_gb is None:
        log("ERROR: could not detect total RAM via system_profiler.")
        sys.exit(2)
    free_wired_gb = total_ram_gb - current_wired_gb

    preload = estimate_preload_overhead(tool, os.getcwd()) if tool else None
    rows = build_recommendations(
        models, free_wired_gb, chip_name, gpu_cores,
        preload_overhead=(preload["total"] if preload else 0),
    )

    if list_json:
        print(json.dumps({
            "current_wired_gb": current_wired_gb, "free_wired_gb": free_wired_gb,
            "chip": chip_name, "gpu_cores": gpu_cores,
            "total_ram_gb": total_ram_gb, "preload_overhead": preload,
            "recommendations": rows,
        }, indent=2))
        return

    fitting = print_menu(rows, free_wired_gb, chip_name, gpu_cores, total_ram_gb, preload)
    if not fitting:
        log("q) Quit")
        # input()'s own prompt argument writes to stdout, which must stay clean for the
        # final JSON line — confirmed live as a real bug during testing (the prompt
        # text ended up concatenated with the JSON, breaking the caller's json.loads).
        # Printing the prompt via log() (stderr) first and calling input() with no
        # argument avoids this.
        print("Select [q]: ", end="", file=sys.stderr, flush=True)
        input()
        sys.exit(1)

    log("q) Quit")
    while True:
        print(f"Select [1-{len(fitting)}] or 'q': ", end="", file=sys.stderr, flush=True)
        choice = input().strip().lower()
        if choice == "q":
            sys.exit(1)
        if choice.isdigit() and 1 <= int(choice) <= len(fitting):
            break
        log("Invalid choice, try again.")

    selected = fitting[int(choice) - 1]
    m = selected["model"]

    # Machine-readable result — the ONLY thing on stdout.
    # kv_bits is None when this model's recommended context fits free memory
    # unquantized (the common case now — see HARDWARE_SPEED_ESTIMATION.md §14); the
    # caller (start_litellm_proxy.sh) must omit --kv-bits entirely in that case rather
    # than treating None as a literal value.
    print(json.dumps({
        "id": m["id"],
        "huggingface": m["huggingface"],
        "size_gb": m["size_gb"],
        "recommended_context": selected["recommended_context"],
        "kv_bits": selected["kv_bits"],
        "estimated_tok_s": selected["estimated_tok_s"],
    }))


if __name__ == "__main__":
    main()
