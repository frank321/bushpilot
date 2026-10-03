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
  don't have the data to back even a rough number (see MoE handling below). A real
  measurement from a PREVIOUS run of this exact model on this exact chip (see
  save_calibration/load_calibration) overrides the formula entirely when one exists —
  "measured on this machine", not an estimate.
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
  python3 model_advisor.py --save-calibration MODEL_ID --decode-tok-s N
                                              [--context-tokens N] [--ttft-s N]
                                              [--peak-memory-gb N]
                                              [--prefill-step-size N]
                                              [--kv-bits-safe-at-long-context 1|0]
                                              → records a real measurement from a
                                                session that already ran (called by
                                                start_mlx_server.sh at teardown, and
                                                by sweep_model.py's richer per-model
                                                sweep), for estimate_speed_tok_s /
                                                recommend_prefill_step_size to prefer
                                                on this model's next launch on this chip
"""
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_JSON = os.path.join(SCRIPT_DIR, "models.json")
# Per-machine learned measurements, one JSON file per (chip, model id) pair -- see
# save_calibration/load_calibration below. Mirrors ~/.bushpilot's other state (the bash
# side's BUSHPILOT_HOME) without model_advisor.py needing to import or shell out to it.
CALIBRATION_DIR = os.path.expanduser("~/.bushpilot/calibration")

MEMORY_CONSTRAINED_KV_BITS = 8   # only used as a FALLBACK now — see recommend_kv_bits_and_context
UNQUANTIZED_KV_BITS = 16        # what MLX uses when --kv-bits is simply not passed (float16 KV)
SAFETY_BUFFER_GB = 3.0       # headroom kept free beyond model+KV, for OS/background apps/spikes
# Mirrors start_mlx_server.sh's own PREFILL_STEP_SIZE default -- two separate copies of the
# same number deliberately, not a shared constant imported across the bash/python boundary:
# bash owns the actual launch flag, this is only the fallback recommend_prefill_step_size
# below returns when no (chip, model) calibration exists yet. Derived from exactly 2 Gemma-4
# models on this one M5 Pro machine (HARDWARE_SPEED_ESTIMATION.md §12) — not assumed to
# generalize, which is the whole reason recommend_prefill_step_size/the sweep harness exist
# (see NEXT_SESSION_HANDOFF_PREFILL_SWEEP.md).
DEFAULT_PREFILL_STEP_SIZE = 2048
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
# servers or skills configured yet. Recalibrated 2026-09-29 (PHASE_PLAN.md Phase 5)
# against REAL measurements via launch_tool.sh, replacing the previous public-report
# -sourced figures (Claude 18,000 / OpenCode 7,000, from a Sep 2026 HN thread — never
# measured on this project's own usage).
#
# Methodology: ran a single-turn, no-tool-call `-p "Say hello."` (claude) /
# `run "Say hello."` (opencode) request through launch_tool.sh from this repo's own
# root (no CLAUDE.md/AGENTS.md/.mcp.json present there, and claude's run used
# --strict-mcp-config), reading the server's own reported total prompt tokens
# (input_tokens, or input+cache.read for a request that hit APC) as ground truth for
# the local model's tokenizer. That total still includes this machine's real ~32
# globally-installed skills (measured separately via measure_skills_preload as
# ~3,518 tokens, since skills are global here, not per-directory) — subtracted back
# out below so estimate_preload_overhead's own skill-token addition (applied for
# whatever skills a real caller's cwd actually has) isn't double-counted on top of
# the baseline, per this phase's own double-counting warning.
#   claude:   22,406 measured total - ~3 (user msg) - 3,518 (skills) ≈ 18,885
#   opencode:  9,906 measured total - ~3 (user msg) - 3,518 (skills) ≈  6,385
# Note: this contradicts an earlier, unreproduced same-day claim in this doc's history
# of ~39,000-44,000 tokens for Claude Code (measured via a Read-tool-call request, not
# a bare "say hello"). A same-methodology Read-tool-call re-test this session measured
# ~24,000 total, consistent with the ~22,406 bare figure here, not the ~39-44K one —
# that earlier figure could not be reproduced and is treated as superseded.
#
# aider/cline/pi added 2026-10-03 (PHASE_PLAN.md's lightweight-tool-CLI research
# session), same ground-truth method (server's own /metrics prompt_tokens for a bare,
# no-edit "say hello"-equivalent request through launch_tool.sh), but with NO skills
# subtraction for any of the three — confirmed live that none of them were loading
# this machine's ~3,518-token Claude-Code-style global skill listing in the first
# place: Aider has no skill/plugin-discovery mechanism at all; Cline has its own
# unrelated one (`cline skill` → `npx skills`) with zero skills installed on this
# machine; Pi has its own too (`~/.pi/agent/skills/`, confirmed in its docs) but it was
# a fresh install with that directory not even existing yet. So each tool's raw
# measured total already IS the clean baseline — nothing to subtract, unlike
# claude/opencode above where this machine's global Claude-Code skills happened to load
# through both of them.
#   aider: --message "Say hello." --yes-always --exit (run against a scratch copy of
#     this repo, not the live checkout — aider's "whole edit format" means a model
#     reply containing anything that looks like code gets auto-applied AND
#     auto-committed with --yes-always; this bit on the first attempt at this exact
#     measurement, which is why the scratch-copy precaution exists here, not for
#     claude/opencode above). Aider always builds a repo-map (confirmed: "Repo-map:
#     using 4096 tokens, auto refresh" in its own startup banner) regardless of the
#     prompt, so this total is inherently repo-shape-dependent, not a pure
#     system-prompt baseline — real for THIS repo's 36 tracked files, not necessarily
#     portable to a differently-sized repo. 6,721 measured total - ~3 (user msg) ≈
#     6,718.
#   cline: a bare `cline "Say hello." --timeout 30` (its default act-mode,
#     auto-approve=true). 5,457 measured total - ~3 (user msg) ≈ 5,454.
#   pi: `pi --provider mlx-direct --model <id> --print "Say hello." --no-session`.
#     1,611 measured total - ~3 (user msg) ≈ 1,608 — by far the smallest of the five
#     tools measured so far, consistent with it being explicitly marketed as a
#     minimal/hackable terminal agent (see PHASE_PLAN.md's research section).
TOOL_BASELINE_PRELOAD_TOKENS = {
    "claude": 18900,
    "opencode": 6400,   # bare-install figure (~0 MCP; global skills subtracted back
                         # out, see above) — real OpenCode sessions with skills
                         # configured run higher; see measure_skills_preload's
                         # docstring for why that growth isn't modeled for OpenCode
                         # specifically.
    "aider": 6700,       # includes this repo's own repo-map cost (see above) — not a
                         # repo-independent figure the way the others are.
    "cline": 5500,
    "pi": 1600,
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

# Decided 2026-09-29 (PHASE_PLAN.md Phase 5): build_recommendations() previously gated
# a model's availability purely on memory fit -- estimated_tok_s was computed and shown
# but never used to exclude anything, so a model estimated at even 1 tok/s would still
# appear fully selectable. Below this, a model is greyed out the same way an
# insufficient-memory model already is.
# Lowered 15 -> 10 (2026-09-29): real-world use of gemma-4-12B-it-8bit at ~17 tok/s
# (estimator's own 0.87 correction factor means the raw estimate for a model actually
# delivering 17 tok/s can land close to 15) felt "pretty fast" in practice, so 15 was
# excluding models that are actually fine to use.
# Lowered 10 -> 7 (2026-09-29): mlx-community/Qwen3.8-27B-4bit running at ~8 tok/s on
# real hardware still works ok in practice. Mainly intended to let 16GB-RAM users reach
# usably-good models that would otherwise be filtered out.
MIN_SPEED_TOK_S = 7


def log(msg):
    print(msg, file=sys.stderr)


def get_current_wired_gb():
    """Returns memory currently UNAVAILABLE to a new model process, so the caller can
    compute free_gb = total_ram_gb - this value. Despite the historical name (kept so
    every call site doesn't change), this is not just "wired" memory.

    REVISED 2026-09-30, after two wrong turns documented here so the next person
    doesn't repeat either:
      1. The original implementation counted ONLY `Pages wired down` as used. Its real
         flaw was ignoring ACTIVE app memory and the compressor -- a machine with 15GB
         of running applications looked identical to an idle one, so a 14GB model
         could be recommended on top of a genuinely busy machine.
      2. A first rewrite then treated macOS's INACTIVE pages (its deliberately-kept
         file cache, which the kernel evicts on demand) as unavailable too. That
         overcorrected badly: an idle 24GB Mac measured only ~12.5GB free and the
         picker greyed out models that demonstrably run (gemma-4-26b ran fine for 14
         minutes on this machine at 65K context) -- a user-visible regression.

    The measure used now is the kernel's own real-time percent-free figure
    (`kern.memorystatus_level`, 0-100) -- the SAME number the launcher's memory
    backstop watches and that `memory_pressure(1)` reports as "System-wide memory free
    percentage". It is the honest middle: evictable cache counts as available (so
    recommendations match what actually runs, per the successful 26B run above), while
    ACTIVE app / compressor / wired consumption genuinely reduces it (so a busy
    machine is correctly seen as busy, fixing the original flaw). Because the backstop
    tears down on this same figure, the fit check and the guard can never disagree
    about what "free" means.

    Fallback (sysctl unavailable, e.g. a future macOS renaming it): vm_stat's
    free+speculative+purgeable+inactive as available -- closer to the kernel view than
    wired-only, still counts cache as reclaimable."""
    total_ram_gb = _get_total_ram_gb()
    if total_ram_gb is None:
        raise RuntimeError("Could not determine total RAM via sysctl hw.memsize")

    try:
        out = subprocess.check_output(["sysctl", "-n", "kern.memorystatus_level"], text=True)
        pct = int(out.strip())
        if 0 <= pct <= 100:
            return total_ram_gb * (100 - pct) / 100
    except (subprocess.CalledProcessError, ValueError):
        pass

    out = subprocess.check_output(["vm_stat"], text=True)
    page_size = 4096
    header = out.splitlines()[0]
    if "page size of" in header:
        page_size = int(header.split("page size of")[1].strip().split()[0])

    pages = {}
    for line in out.splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        try:
            pages[key.strip()] = int(val.strip().rstrip("."))
        except ValueError:
            continue

    reclaimable_keys = ("Pages free", "Pages speculative", "Pages purgeable", "Pages inactive")
    if not any(k in pages for k in reclaimable_keys):
        raise RuntimeError("Could not determine free memory from kern.memorystatus_level or vm_stat")
    reclaimable_gb = sum(pages.get(k, 0) for k in reclaimable_keys) * page_size / 1024**3
    return max(0.0, total_ram_gb - reclaimable_gb)


def _get_total_ram_gb():
    """Total physical RAM in GB via sysctl (fast, exact) -- separate from
    get_chip_info()'s system_profiler parse so this accounting doesn't depend on it."""
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
        return int(out.strip()) / 1024**3
    except (subprocess.CalledProcessError, ValueError):
        return None


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


def effective_size_gb(model_entry):
    """model_entry["size_gb"] is the target weights only (see _schema_notes.size_gb in
    models.json) -- a "serving" block's drafter is a SEPARATE real download that sits in
    memory alongside the target for the whole session, so any memory-fit check has to
    account for both. Kept as a small helper rather than inlined at each call site so the
    two things this affects (recommend_max_context's admission check, compute_bonus_
    concurrency's leftover-memory check) can't drift out of sync with each other. Does
    NOT affect estimate_speed_tok_s's active_gb -- that's about per-token bandwidth cost
    of the TARGET's own forward pass, which the drafter's real-but-comparatively-tiny
    forward pass doesn't change; the drafter's speed impact is already captured in
    serving.measured_speedup instead."""
    return model_entry["size_gb"] + model_entry.get("serving", {}).get("drafter_size_gb", 0)


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
                           safety_buffer_gb=SAFETY_BUFFER_GB, concurrency=1,
                           max_context_tokens=None):
    """Returns (recommended_tokens_or_None, reason_string, detail_dict_or_None).

    concurrency>1 sizes context for N identically-sized concurrent slots instead of
    one -- inflating the per-token/fixed KV costs by N before the fits check is
    mathematically the same as dividing the memory budget by N, and yields the
    per-slot context size such that N copies of it all fit. Used for an explicit
    concurrency request (config.json's "concurrency"); the default auto-detected
    bonus-concurrency path (see compute_bonus_concurrency) instead calls this with
    concurrency=1 (unchanged, context-maximizing behavior) and separately checks
    whether tier-rounding leftover fits extra same-size slots for free.

    max_context_tokens (from config.json's "max_context_tokens" via the
    --max-context-tokens flag) is an OPTIONAL upper bound applied BEFORE tier
    rounding -- deliberately a cap, not an override: the memory-fit math still runs
    unchanged and can only ever shrink the result further, so a cap can never
    produce a context the host can't actually afford (unlike the launcher's
    fixed-model CLI form, whose 3rd positional arg bypasses this check entirely).
    None (the default) means no cap and today's behavior exactly."""
    available_gb = free_wired_gb - effective_size_gb(model_entry) - safety_buffer_gb
    if available_gb <= 0:
        return None, "insufficient_memory_for_model_weights", None

    formula = compute_kv_formula(model_entry.get("kv_config"), kv_bits)
    if formula is None:
        return None, "architecture_fields_missing_from_config", None
    bytes_per_growing_token, fixed_bytes, detail = formula
    bytes_per_growing_token *= concurrency
    fixed_bytes *= concurrency

    available_bytes = available_gb * 1024**3
    usable_bytes = available_bytes - fixed_bytes
    if usable_bytes <= 0:
        return None, "insufficient_memory_for_fixed_kv_cost", detail

    if bytes_per_growing_token <= 0:
        max_ctx = model_entry["max_context_architectural"]
    else:
        max_ctx = int(usable_bytes / bytes_per_growing_token)
        max_ctx = min(max_ctx, model_entry["max_context_architectural"])
    if max_context_tokens is not None:
        max_ctx = min(max_ctx, max_context_tokens)

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


def _calibration_path(chip_name, model_id):
    # Sanitize so "Apple M5 Pro" and arbitrary model ids both become safe filenames --
    # spaces and anything non-alphanumeric collapse to "_" rather than being rejected
    # or silently colliding (e.g. two chip names that only differ in punctuation).
    safe_chip = re.sub(r"[^A-Za-z0-9._-]+", "_", chip_name)
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", model_id)
    return os.path.join(CALIBRATION_DIR, f"{safe_chip}__{safe_model}.json")


def load_calibration(chip_name, model_id):
    """Returns the saved real-world measurement dict for this (chip, model) pair, or
    None if nothing has ever been measured here (the common case for a model's first
    launch on this machine, or any machine that hasn't written to CALIBRATION_DIR yet).

    This is the "has this exact combo actually been run on this machine" check that
    lets estimate_speed_tok_s below prefer a REAL observed decode speed over its
    physics-based formula -- the same provisional-estimate-then-real-measurement
    pattern LM Studio's mlx_engine uses for its prompt-cache disk budget (see
    disk_budget.py's provisional_cache_store_budget_bytes -> final_cache_store_budget_bytes),
    applied here to decode tok/s instead of cache bytes. Missing/corrupt files are
    treated as "nothing measured yet", not an error -- a calibration file is a cache,
    never a required input."""
    path = _calibration_path(chip_name, model_id)
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("decode_tok_s"), (int, float)):
        return None
    return data


def save_calibration(chip_name, model_id, decode_tok_s, context_tokens=None,
                      ttft_s=None, peak_memory_gb=None, prefill_step_size=None,
                      kv_bits_safe_at_long_context=None):
    """Persists one real observation of this model actually decoding on this chip, so
    a FUTURE launch of the identical (chip, model) pair can report a measured tok/s
    instead of SPEED_CORRECTION_FACTOR's physics estimate -- whose own documented
    accuracy range is 0.75x-1.29x across just the 5 models checked so far (see
    HARDWARE_SPEED_ESTIMATION.md §17 item #2). Called by start_mlx_server.sh at
    teardown with whatever that session's own /metrics last reported.

    prefill_step_size/kv_bits_safe_at_long_context (both optional, added for the
    per-(chip, model) sweep harness -- see NEXT_SESSION_HANDOFF_PREFILL_SWEEP.md):
    a condensed summary of the richer `~/.bushpilot/calibration/sweeps/*.sweep.json`
    file a full sweep run produces, written here so recommend_prefill_step_size below
    (and any future reader) never has to parse that separate, larger file just to get
    a launch decision. Deliberately omitted from the payload (not written as an
    explicit null) when not given, exactly like every other optional field here --
    this keeps a plain teardown-time save_calibration_snapshot() call (which never has
    sweep data) byte-identical to before this was added, and keeps load_calibration's
    existing "ignore unknown keys" contract the only thing a reader needs to rely on.

    Deliberately last-write-wins (overwrites any prior file for this exact pair), not
    an average across sessions -- a more recent measurement (different context load,
    different concurrent apps) isn't obviously more or less trustworthy than an older
    one, and there's no principled weighting for averaging stale data with fresh data
    here. The caller decides what's worth saving (e.g. skip a session that generated
    ~0 tokens); this function writes whatever it's given without judging it.

    Written atomically (tmp file + os.replace) so a process killed mid-write never
    leaves a half-written, unparseable calibration file for the next launch to trip
    over."""
    os.makedirs(CALIBRATION_DIR, exist_ok=True)
    path = _calibration_path(chip_name, model_id)
    payload = {
        "decode_tok_s": decode_tok_s,
        "context_tokens": context_tokens,
        "ttft_s": ttft_s,
        "peak_memory_gb": peak_memory_gb,
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if prefill_step_size is not None:
        payload["prefill_step_size"] = prefill_step_size
    if kv_bits_safe_at_long_context is not None:
        payload["kv_bits_safe_at_long_context"] = kv_bits_safe_at_long_context
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f)
    os.replace(tmp_path, path)


def estimate_speed_tok_s(model_entry, chip_name, gpu_cores):
    """Returns (estimate_or_None, reason_string). Deliberately returns None rather than
    a guess when we don't have real active-weight data — see gemma-4-26b-a4b-it-4bit's
    entry if its active_weight_gb were ever removed.

    Checks load_calibration FIRST: a real measurement from a previous run of this
    EXACT model on this EXACT chip always wins over the physics formula below, since
    it's not an estimate at all -- see load_calibration's docstring. Falls through to
    the formula untouched when nothing's been measured yet.

    A "serving" block's measured_speedup (see models.json's _schema_notes.serving) is
    applied AFTER the base bandwidth estimate below, as a multiplier -- not as a flat
    replacement tok/s number. That's deliberate: the multiplier was measured on one
    specific machine (serving.measured_on), but the physics estimate it's layered onto
    already scales per-chip via CHIP_BANDWIDTH_GB_S, so a slower/older Mac than the one
    the drafter was measured on still gets a proportionally scaled-down estimate instead
    of inheriting a number that was never measured on hardware anything like it. The
    underlying accept-rate/speedup ratio could still shift on a very different GPU
    generation than the reference machine -- that residual uncertainty is why the reason
    string keeps calling this an estimate even when a real multiplier backs it, and why
    it's a multiplier on the physics estimate rather than a number treated as ground
    truth on its own.

    model_entry's own optional "speed_correction_factor" field overrides the global
    SPEED_CORRECTION_FACTOR constant when present -- added because that global constant
    is already confirmed NOT to hold across architecture families: Qwen measurements
    cluster at 0.86-0.97, but gemma-4-e4b-it-4bit measured 1.29 and gemma-4-12b-it-4bit
    measured 0.75 (HARDWARE_SPEED_ESTIMATION.md §12/§17 item #2) -- a 4x-wider spread,
    in opposite directions, than one global number can represent. Leave unset in
    models.json until a model has its own live calibration point; the global constant
    remains the right default for anything not yet measured."""
    calibrated = load_calibration(chip_name, model_entry.get("id", ""))
    if calibrated is not None:
        ctx_note = (
            f" at ~{calibrated['context_tokens']:,} tokens of context"
            if isinstance(calibrated.get("context_tokens"), (int, float)) else ""
        )
        return calibrated["decode_tok_s"], (
            f"measured on this machine ({chip_name}){ctx_note}, saved {calibrated.get('saved_at', 'previously')} "
            "— overrides the physics-based estimate below until this model, quantization, "
            "or drafter config changes on this chip (re-run to refresh)"
        )

    bandwidth = CHIP_BANDWIDTH_GB_S.get(chip_name)
    if bandwidth is None:
        return None, f"no bandwidth data for chip '{chip_name}'"

    active_gb = model_entry.get("active_weight_gb")
    if active_gb is None:
        if model_entry.get("type") == "moe":
            return None, "MoE model with unknown active-parameter fraction — no reliable estimate without a live benchmark"
        active_gb = model_entry["size_gb"]  # dense model: full weight size is the active weight

    factor = model_entry.get("speed_correction_factor", SPEED_CORRECTION_FACTOR)
    base = factor * bandwidth / active_gb

    serving = model_entry.get("serving") or {}
    speedup = serving.get("measured_speedup")
    if speedup:
        measured_on = serving.get("measured_on", "reference hardware")
        return round(base * speedup, 1), (
            f"estimated ({speedup}x speculative-decoding multiplier measured on "
            f"{measured_on}, applied to this chip's own bandwidth estimate — "
            "workload- and chip-generation-dependent, treat as directional)"
        )

    return round(base, 1), "estimated"


def _has_sliding_window_layers(kv_config):
    """True when this model's kv_config lists any sliding_attention layer -- used by
    recommend_kv_bits_and_context below to flag a real, upstream-confirmed gap (see
    its docstring), not to redo compute_kv_formula's own byte-budget math."""
    if not kv_config:
        return False
    return any(t == "sliding_attention" for t in kv_config.get("layer_types") or [])


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
                                   safety_buffer_gb=SAFETY_BUFFER_GB, concurrency=1,
                                   max_context_tokens=None):
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

    kv_bits_apc_caveat (in the returned dict) is True when the quantized fallback is
    actually used AND the model has sliding-window (SWA) layers -- flagging a real,
    already-filed upstream gap (mlx-vlm issue #1559, "APC + kv-bits full batch/SWA
    coverage", confirmed/p1, open since Jul 2026, smoke-tested on gemma-4-e2b-it-4bit):
    the server's own APC_ENABLED=1 is unconditional in this project, and upstream has
    confirmed that combining --kv-bits with APC on the batched/concurrent request path
    means SWA layer caches aren't clonable for an exact store, so sliding layers likely
    won't get cached on that path until it's fixed. Degrades gracefully per upstream
    ("prefix reuse still works" per their smoke note) -- NOT corrupted output -- so this
    is surfaced as a caveat for the picker to show (see format_model_body_lines), not a
    reason to refuse the fallback outright; refusing it would make some models that
    could otherwise run look like they don't fit at all, over a bug that degrades
    rather than breaks. Re-check mlx-vlm#1559's status before removing this flag.

    Returns dict: kv_bits (None or MEMORY_CONSTRAINED_KV_BITS), recommended_context,
    context_reason, kv_formula_detail, context_after_preload, fits, kv_bits_apc_caveat.
    """
    ctx_u, reason_u, detail_u = recommend_max_context(
        model_entry, free_wired_gb, kv_bits=UNQUANTIZED_KV_BITS,
        safety_buffer_gb=safety_buffer_gb, concurrency=concurrency,
        max_context_tokens=max_context_tokens,
    )
    fits_u, reason_u, after_u = _apply_preload(ctx_u, reason_u, preload_overhead)
    if fits_u:
        return {
            "kv_bits": None, "recommended_context": ctx_u, "context_reason": reason_u,
            "kv_formula_detail": detail_u, "context_after_preload": after_u, "fits": True,
            "kv_bits_apc_caveat": False,
        }

    ctx_q, reason_q, detail_q = recommend_max_context(
        model_entry, free_wired_gb, kv_bits=MEMORY_CONSTRAINED_KV_BITS,
        safety_buffer_gb=safety_buffer_gb, concurrency=concurrency,
        max_context_tokens=max_context_tokens,
    )
    fits_q, reason_q, after_q = _apply_preload(ctx_q, reason_q, preload_overhead)
    if fits_q:
        return {
            "kv_bits": MEMORY_CONSTRAINED_KV_BITS, "recommended_context": ctx_q,
            "context_reason": reason_q, "kv_formula_detail": detail_q,
            "context_after_preload": after_q, "fits": True,
            "kv_bits_apc_caveat": _has_sliding_window_layers(model_entry.get("kv_config")),
        }

    # Neither fits — report the quantized attempt's reason, since it's the more
    # forgiving of the two (smaller bytes-per-token) and so the more informative "even
    # our best effort doesn't fit" explanation for the user.
    return {
        "kv_bits": None, "recommended_context": ctx_q, "context_reason": reason_q,
        "kv_formula_detail": detail_q, "context_after_preload": after_q, "fits": False,
        "kv_bits_apc_caveat": False,
    }


# A larger --prefill-step-size's measured memory cost (HARDWARE_SPEED_ESTIMATION.md §12) is
# real but comparatively small relative to the model's own footprint, not a multiplier of
# it: going 2048->4096 cost +3.4% of that run's own peak memory for gemma-4-e4b-it-4bit and
# +11.8% for gemma-4-12b-it-4bit, the two data points that exist. Modeled conservatively here
# as a flat 15% margin ABOVE the calibration's own observed peak_memory_gb -- comfortably
# above both observed deltas -- applied ONLY when recommending a step size other than the
# proven-safe default (the default itself needs no extra margin: it's what the calibration's
# own peak_memory_gb was actually measured running at). This is a documented approximation,
# not a re-derivation of §12's relationship for every chip/model -- the sweep harness
# (NEXT_SESSION_HANDOFF_PREFILL_SWEEP.md) is what would eventually replace this with a real
# per-model memory curve if the margin ever proves too tight or too loose in practice.
PREFILL_STEP_SIZE_MEMORY_MARGIN = 1.15


def recommend_prefill_step_size(model_entry, chip_name, free_wired_gb,
                                 safety_buffer_gb=SAFETY_BUFFER_GB):
    """Returns (step_size, reason_string) -- mirrors recommend_kv_bits_and_context's
    try-the-better-option-then-fall-back-to-the-proven-safe-default structure, but for
    --prefill-step-size instead of --kv-bits.

    §12's own finding (HARDWARE_SPEED_ESTIMATION.md) is that `--prefill-step-size`
    affects ONLY time-to-first-token, never decode speed, and that the currently
    hardcoded global default (DEFAULT_PREFILL_STEP_SIZE, 2048) already beat both 512
    and 4096 on both models tested -- but that was 2 Gemma-4 models on 1 machine (this
    one, Apple M5 Pro). A chip with different GPU core count / architecture has no
    reason to share that same optimum (§17 item #8) -- this function is what lets a
    real per-(chip, model) sweep override the global default once one exists, without
    ever hardcoding a second global constant that's just as narrow as the first.

    load_calibration's existing "unknown keys are ignored" contract means this reads
    the SAME lightweight calibration file recommend overridden by load_calibration/
    estimate_speed_tok_s already use -- no new file, no new load path. A calibration
    written before this field existed (no "prefill_step_size" key at all) is treated
    identically to "nothing calibrated yet", not an error.

    Memory safety: a non-default calibrated step size is only ever recommended when
    free_wired_gb comfortably covers the calibration's own observed peak_memory_gb
    plus PREFILL_STEP_SIZE_MEMORY_MARGIN's margin plus the usual safety buffer -- see
    that constant's own comment for where the margin number comes from. A calibration
    with no recorded peak_memory_gb (e.g. hand-written, or from a version of
    save_calibration that predates that field) can't be safety-checked at all, so it
    falls back to the default rather than risk using an unverified value blindly."""
    calibrated = load_calibration(chip_name, model_entry.get("id", ""))
    if calibrated is None or not isinstance(calibrated.get("prefill_step_size"), int):
        return DEFAULT_PREFILL_STEP_SIZE, (
            "default (no calibrated prefill_step_size for this chip+model yet — "
            "see NEXT_SESSION_HANDOFF_PREFILL_SWEEP.md to run the sweep)"
        )

    step_size = calibrated["prefill_step_size"]
    saved_at = calibrated.get("saved_at", "previously")
    if step_size == DEFAULT_PREFILL_STEP_SIZE:
        return step_size, (
            f"calibrated — matches the global default, measured on this machine "
            f"({chip_name}), saved {saved_at}"
        )

    peak_mem = calibrated.get("peak_memory_gb")
    if not isinstance(peak_mem, (int, float)):
        return DEFAULT_PREFILL_STEP_SIZE, (
            f"default (calibrated step size {step_size} has no recorded peak_memory_gb "
            "to safety-check against — falling back rather than risk an unverified value)"
        )

    margin = PREFILL_STEP_SIZE_MEMORY_MARGIN if step_size > DEFAULT_PREFILL_STEP_SIZE else 1.0
    required_gb = peak_mem * margin + safety_buffer_gb
    if free_wired_gb < required_gb:
        return DEFAULT_PREFILL_STEP_SIZE, (
            f"default (calibrated step size {step_size} needs ~{required_gb:.1f}GB free "
            f"with margin; only {free_wired_gb:.1f}GB currently available)"
        )

    return step_size, (
        f"measured on this machine ({chip_name}), saved {saved_at} — overrides the "
        f"global default ({DEFAULT_PREFILL_STEP_SIZE}) until this model or chip changes "
        "(re-run the sweep to refresh)"
    )


# Auto mode never suggests more than this many free bonus slots -- beyond it, more
# concurrent local coding sessions against one model isn't a realistic use case even
# when memory technically allows it (and continuous-batched prefill is CPU/GPU-bound
# too, not just memory-bound).
AUTO_MAX_CONCURRENCY = 4


def compute_bonus_concurrency(model_entry, free_wired_gb, kv_bits_used, recommended_context,
                               safety_buffer_gb=SAFETY_BUFFER_GB):
    """Auto mode (no explicit concurrency requested): recommend_kv_bits_and_context has
    already picked the max context for ONE slot, rounded DOWN to a clean tier (see
    recommend_max_context's own tiering comment) -- which almost always leaves some
    memory unused. This checks whether that rounding leftover fits one or more
    additional, identically-sized slots for free, without shrinking anyone's context,
    and returns the total slot count (>=1)."""
    if recommended_context is None:
        return 1
    formula = compute_kv_formula(model_entry.get("kv_config"),
                                  kv_bits_used if kv_bits_used is not None else UNQUANTIZED_KV_BITS)
    if formula is None:
        return 1
    bytes_per_growing_token, fixed_bytes, _ = formula
    per_slot_bytes = fixed_bytes + bytes_per_growing_token * recommended_context
    if per_slot_bytes <= 0:
        return 1

    available_bytes = (free_wired_gb - effective_size_gb(model_entry) - safety_buffer_gb) * 1024**3
    remaining = available_bytes - per_slot_bytes
    concurrency = 1
    while remaining >= per_slot_bytes and concurrency < AUTO_MAX_CONCURRENCY:
        concurrency += 1
        remaining -= per_slot_bytes
    return concurrency


def build_recommendations(models, free_wired_gb, chip_name, gpu_cores, preload_overhead=0,
                           requested_concurrency=None, max_context_tokens=None):
    """requested_concurrency: None means auto (maximize context for 1 slot, then add
    free bonus slots from tier-rounding leftover -- see compute_bonus_concurrency).
    An explicit int forces that many slots, shrinking context as needed to guarantee
    all of them fit (see recommend_max_context's concurrency param).
    max_context_tokens: optional upper bound on every model's recommended context
    (config.json's "max_context_tokens") -- a cap layered on top of the memory-fit
    math, never a bypass of it (see recommend_max_context's own docstring)."""
    rows = []
    for m in models:
        sizing_concurrency = requested_concurrency if requested_concurrency else 1
        kv_result = recommend_kv_bits_and_context(m, free_wired_gb, preload_overhead,
                                                    concurrency=sizing_concurrency,
                                                    max_context_tokens=max_context_tokens)
        speed, speed_reason = estimate_speed_tok_s(m, chip_name, gpu_cores)

        fits = kv_result["fits"]
        context_reason = kv_result["context_reason"]
        # Speed floor (decided 2026-09-29): estimated_tok_s was computed and shown but
        # never used to exclude anything, so a model estimated at even 1 tok/s still
        # appeared fully selectable. Only applies when there's a real estimate to judge
        # against -- a model with estimated_tok_s is None (the MoE-without-active-
        # weight-data case) keeps today's "shown, speed unknown" behavior rather than
        # being excluded by a floor it can't be measured against.
        if fits and speed is not None and speed < MIN_SPEED_TOK_S:
            fits = False
            context_reason = "estimated_speed_too_slow"

        concurrency = sizing_concurrency
        if fits and not requested_concurrency:
            concurrency = compute_bonus_concurrency(
                m, free_wired_gb, kv_result["kv_bits"], kv_result["recommended_context"],
            )

        prefill_step_size, prefill_step_size_reason = recommend_prefill_step_size(
            m, chip_name, free_wired_gb,
        )

        rows.append({
            "model": m,
            "recommended_context": kv_result["recommended_context"],
            "context_reason": context_reason,
            "kv_formula_detail": kv_result["kv_formula_detail"],
            "kv_bits": kv_result["kv_bits"],
            "kv_bits_apc_caveat": kv_result["kv_bits_apc_caveat"],
            "estimated_tok_s": speed,
            "speed_reason": speed_reason,
            "context_after_preload": kv_result["context_after_preload"],
            "fits": fits,
            "concurrency": concurrency,
            "prefill_step_size": prefill_step_size,
            "prefill_step_size_reason": prefill_step_size_reason,
        })
    return rows


DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"
GREEN = "\033[32m"
YELLOW = "\033[33m"


def use_color(force=False):
    """Color only on a real terminal and when NO_COLOR isn't set. stdout is a pipe into
    fzf here, so the TTY check is on stderr (where the banner is painted).

    `force=True` (--color) skips the stderr-isatty check entirely -- for the
    --list-compact picker path specifically, stderr isn't a reliable signal: every
    fzf reload/bind command in start_mlx_server.sh redirects this process's stderr to
    /dev/null (to swallow system_profiler's unrelated hw.cpufamily noise), which makes
    sys.stderr.isatty() False on every single reload after the picker's first instant
    -- not just on some other machine, on EVERY machine, every launch. The caller
    already knows this output is headed into fzf's --ansi-enabled screen, so it passes
    --color to assert that rather than relying on a TTY check that --color's own
    plumbing defeats. NO_COLOR is still honored even when forced -- that's an explicit
    user accessibility opt-out, not a detection heuristic, so it should never be
    overridden."""
    if "NO_COLOR" in os.environ:
        return False
    return True if force else sys.stderr.isatty()


def print_preload_banner(preload):
    tool_label = {"claude": "Claude Code", "opencode": "OpenCode"}.get(preload["tool"], preload["tool"])
    log(f"  Estimated preload overhead for {tool_label} in {preload['cwd']}:")
    log(f"    baseline (system prompt + built-in tools): ~{preload['baseline_tokens']:,} tokens")
    log("      (measured live on this machine 2026-09-29, see TOOL_BASELINE_PRELOAD_TOKENS — still varies by tool version)")
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


def describe_unavailable_reason(r, preload):
    """Human-readable reason a model is greyed out -- shared by the plain numbered
    menu (print_menu) and --list-compact's hidden-model summary line."""
    context_reason = r["context_reason"]
    if context_reason == "insufficient_memory_for_model_weights":
        return "not enough free memory to load this model"
    elif context_reason == "insufficient_memory_for_fixed_kv_cost":
        return "not enough free memory left for this model's context cache"
    elif context_reason == "computed_context_too_small_to_be_useful":
        return f"would only fit under {MIN_USABLE_CONTEXT:,} tokens of context — not usable"
    elif context_reason == "architecture_fields_missing_from_config":
        return "can't compute (model not downloaded yet / config unavailable)"
    elif context_reason == "insufficient_context_after_tool_preload":
        return (
            f"{r['recommended_context']:,} tokens of context, but tool preload leaves only "
            f"~{r['recommended_context'] - (preload['total'] if preload else 0):,} for the conversation — below the "
            f"{MIN_USABLE_CONTEXT:,}-token floor"
        )
    elif context_reason == "estimated_speed_too_slow":
        return f"estimated ~{r['estimated_tok_s']} tok/s — below the {MIN_SPEED_TOK_S} tok/s usability floor"
    else:
        return context_reason


def format_model_body_lines(m, r=None):
    """The descriptive block shared between the plain numbered menu (print_menu) and
    --detail's fzf preview pane (Phase 6): good_for, usage examples, known_issues, and
    the KV-cache note. The text itself doesn't change between the two renderers, only
    where/how it's displayed -- indentation and surrounding chrome are each caller's
    own job."""
    lines = [m["display"]["good_for"], ""]
    for ex in m["display"]["usage_examples"]:
        lines.append(f"e.g. {ex}")
    if m["display"].get("known_issues"):
        lines.append(f"⚠ {m['display']['known_issues']}")
    if r is not None:
        lines.append("")
        if r["kv_bits"]:
            lines.append(f"[KV cache quantized to {r['kv_bits']}-bit — needed to fit this context in free memory]")
            if r.get("kv_bits_apc_caveat"):
                lines.append(
                    "[⚠ this model's sliding-window layers + quantized KV cache + prefix caching is a "
                    "known upstream gap (mlx-vlm #1559) — prefix reuse still works, but sliding layers "
                    "likely won't cache on concurrent requests until it's fixed]"
                )
        else:
            lines.append("[KV cache unquantized — full decode speed, plenty of free memory for this context]")
        if r.get("concurrency", 1) > 1:
            lines.append(f"[{r['concurrency']} concurrent requests supported at this context size, without exceeding safe memory]")
    return lines


def format_context_k(tokens):
    """Decimal-K notation matching Phase 6's locked mockup (131072 -> '131K'), not
    binary-K -- a colloquial round number, not a precise one."""
    return f"{round(tokens / 1000)}K"


DRAFTER_LABELS = {"dflash": "DFlash", "mtp": "MTP", "eagle3": "EAGLE3"}
DRAFTER_ID_SUFFIXES = ("-dflash-vlm", "-dflash", "-mtp")


def model_engine(m):
    """Inference engine that serves this entry. Entries without a serving block run on
    plain mlx_vlm.server (see models.json's _schema_notes.serving)."""
    return (m.get("serving") or {}).get("backend", "mlx_vlm")


def model_drafter_label(m):
    """Speculative-decoding method of the attached drafter ('DFlash', 'MTP', ...), or '—'
    when the entry has no drafter. mlx_dspark entries carry it in `mode`, mlx_vlm entries
    in `draft_kind`."""
    sv = m.get("serving") or {}
    if not (sv.get("drafter") or sv.get("draft_model")):
        return "—"
    kind = sv.get("draft_kind") or sv.get("mode") or "?"
    return DRAFTER_LABELS.get(kind, kind)


def model_display_name(m):
    """The id minus its engine/drafter suffix, since those have their own columns. Display
    only -- the full id stays the unique key for selection, config.json and the CLI."""
    if model_drafter_label(m) == "—":
        return m["id"]
    for suffix in DRAFTER_ID_SUFFIXES:
        if m["id"].endswith(suffix):
            return m["id"][: -len(suffix)]
    return m["id"]


def model_input_label(m):
    """Input modalities this entry actually serves, e.g. 'text+img+vid'. The model's own
    `modalities` are narrowed to text when the serving variant drops the vision tower
    (serving.multimodal_capable false -- every mlx_dspark entry)."""
    if (m.get("serving") or {}).get("multimodal_capable") is False:
        mods = {"text"}
    else:
        mods = set(m.get("modalities") or ["text"])
    return "+".join(["text"] + [lbl for key, lbl in (("image", "img"), ("video", "vid")) if key in mods])


def model_tools_label(m):
    """Whether the server returns structured tool_calls for this model, per its recorded
    tool_calling test ('tools' / 'no tools'); 'tools?' when it has never been tested."""
    verdict = (m.get("tool_calling") or {}).get("verdict")
    return {"SUPPORTED": "tools", "NOT_SUPPORTED": "no tools"}.get(verdict, "tools?")


def short_unavailable_reason(r, free_wired_gb, preload=None):
    """One-line version of describe_unavailable_reason for the greyed-out picker rows,
    with the actual numbers where we have them."""
    if r["context_reason"] == "insufficient_memory_for_model_weights":
        need = effective_size_gb(r["model"]) + SAFETY_BUFFER_GB
        return f"needs ~{math.ceil(need)} GB free, you have {free_wired_gb:.0f} GB"
    return describe_unavailable_reason(r, preload)


def compact_row_cells(r, reason=None):
    """Plain-text cells of one --list-compact row, in display order: name (with a leading
    tested/untested marker), type, engine, drafter, input, tools, released, size, context,
    KV-cache precision, speed, and a last column (concurrency, or why the model doesn't fit when `reason` is
    given)."""
    m = r["model"]
    if r["fits"]:
        ctx_str = f"{format_context_k(r['recommended_context'])} tok"
        kv_str = f"{r['kv_bits']}-bit" if r["kv_bits"] else f"{UNQUANTIZED_KV_BITS}-bit"
    else:
        ctx_str = kv_str = "—"
    mark = "○" if m.get("status") == "untested" else "✓"
    speed = f"~{r['estimated_tok_s']} tok/s" if r["estimated_tok_s"] is not None else (
        "speed unknown" if r["fits"] else "—")
    return [
        f"{mark} {model_display_name(m)}",
        "MoE" if m.get("type") == "moe" else "Dense",
        model_engine(m),
        model_drafter_label(m),
        model_input_label(m),
        model_tools_label(m),
        m.get("released") or "—",
        f"{m['size_gb']:g}GB",
        ctx_str,
        kv_str,
        speed,
        reason or (f"×{r['concurrency']} concurrent" if r["concurrency"] > 1 else "1 concurrent"),
    ]


# Column indexes in compact_row_cells. Numeric columns right-align, text columns
# left-align; the last column is never padded.
COL_NAME, COL_TYPE, COL_ENGINE, COL_DRAFTER, COL_INPUT, COL_TOOLS, COL_RELEASED, \
    COL_SIZE, COL_CTX, COL_KV, COL_SPEED, COL_LAST = range(12)
COMPACT_RIGHT_ALIGNED = {COL_SIZE, COL_SPEED}

# Dropped first, in this order, on terminals too narrow to show every column (see
# visible_columns below). Name/Size/Context/KV/Speed are what the fit-and-speed decision
# is actually made from and are never dropped; everything in this list is still shown in
# full in the --detail preview pane, so narrowing the list loses nothing, just layout.
OPTIONAL_COLUMN_DROP_ORDER = [COL_DRAFTER, COL_INPUT, COL_TOOLS, COL_RELEASED, COL_ENGINE, COL_TYPE]


# Header words double as --sort values (fzf's click-header passes the clicked word), so
# each is a single lowercase-able word that SORT_KEYS below accepts. The first cell is
# indented past the "✓ " tested marker so "Name" sits over the model names.
COMPACT_HEADERS = ["  Name", "Type", "Engine", "Drafter", "Input", "Tools", "Released",
                   "Size", "Context", "KV", "Speed", "Notes"]


def compact_row_widths(rows):
    """Max width of each padded column across `rows`, never narrower than its header plus
    room for the " ▼" sort arrow, so the columns don't shift when the sort changes."""
    cells = [compact_row_cells(r) for r in rows]
    return tuple(max([len(COMPACT_HEADERS[k]) + 2] + [len(c[k]) for c in cells]) for k in range(COL_LAST))


def visible_columns(widths, term_width):
    """Which of the COL_NAME..COL_SPEED columns fit side by side in term_width columns,
    as an ordered list of column indexes (Notes is handled separately -- see
    format_model_compact_row -- since it's unpadded and gets truncated instead of
    dropped). `term_width` of None means "no limit" (today's behavior: every column).

    Columns are never reordered, only removed, one at a time from the back of
    OPTIONAL_COLUMN_DROP_ORDER, so the picker degrades gracefully as the window shrinks
    rather than jumping straight to a bare Name/Size/Context/KV/Speed table."""
    cols = list(range(COL_LAST))
    if term_width is None:
        return cols

    def total_width(cs):
        return sum(widths[c] for c in cs) + 2 * max(0, len(cs) - 1)

    for victim in OPTIONAL_COLUMN_DROP_ORDER:
        if total_width(cols) <= term_width:
            break
        if victim in cols:
            cols.remove(victim)
    return cols


def cap_name_width(widths, cols, term_width, min_name_width=12):
    """Even with every optional column dropped, Name/Size/Context/KV/Speed can still
    overflow a narrow terminal -- Name is sized to the single longest model id in the
    whole catalog, not to what's actually on screen. Name is the last remaining column
    with any real slack (Size/Context/KV/Speed are short, fixed-shape strings), so once
    the other visible columns' widths are pinned down, whatever's left over becomes
    Name's budget; format_model_compact_row truncates individual cells to it with an
    ellipsis. Never shrinks below min_name_width -- past that point the name is useless
    and the row is left to overflow rather than mangled further (the full id is always
    one keystroke away in the --detail preview)."""
    if term_width is None or COL_NAME not in cols:
        return widths
    # The gap count matches format_model_compact_row's `2 * len(padded)` exactly: one
    # "  " between every pair of visible columns, plus one more before Notes (even when
    # Notes ends up empty) -- len(cols) gaps in total, not len(cols) - 1. The header row
    # always shows the literal "Notes" label (never truncated, never empty like a data
    # row's can be), so its width is reserved here too -- otherwise the header alone
    # would run len("Notes") past term_width even when every data row fits.
    others = sum(widths[c] for c in cols if c != COL_NAME) + 2 * len(cols) + len(COMPACT_HEADERS[COL_LAST])
    budget = term_width - others
    if budget >= widths[COL_NAME] or budget < min_name_width:
        return widths
    widths = list(widths)
    widths[COL_NAME] = budget
    return tuple(widths)


def _truncate_cell(text, width):
    """Shorten `text` to `width` characters with a trailing ellipsis if it's longer --
    used where a column's width was capped to the terminal rather than to content (see
    cap_name_width), since plain ljust/rjust only pad, they never truncate."""
    if width is None or len(text) <= width:
        return text
    if width <= 1:
        return text[:width]
    return text[: width - 1] + "…"


def format_compact_header_row(widths, cols, sort_key=None, descending=False, force_color=False):
    """First line of --list-compact, consumed by fzf's --header-lines=1: same
    `id<TAB>display<TAB>reason` shape as a data row (so --with-nth applies to it
    identically and the columns line up), id left as a placeholder. The column the list
    is currently sorted by gets a ▼ (high to low / newest first) or ▲ (low to high / A-Z).
    `cols` is the ordered subset of columns visible_columns() decided fits the terminal;
    a column dropped for width still keeps its sort-arrow label logic correct for the
    columns that remain. `force_color` is --list-compact's --color flag, forwarded to
    use_color() -- see its docstring for why stderr's own isatty status can't be trusted
    here."""
    labels = list(COMPACT_HEADERS)
    if sort_key in COLUMN_SORT_KEYS:
        k = COLUMN_SORT_KEYS.index(sort_key)
        labels[k] += " ▼" if descending else " ▲"
    padded = [labels[k].rjust(widths[k]) if k in COMPACT_RIGHT_ALIGNED else labels[k].ljust(widths[k])
              for k in cols]
    line = "  ".join(padded + [labels[COL_LAST]])
    return f"-\t{BOLD + line + RESET if use_color(force_color) else line}\t"


def format_model_compact_row(r, widths=None, cols=None, reason=None, term_width=None, force_color=False):
    """One row for --list-compact: `id<TAB>display<TAB>reason`. Field 1 (id) is used
    internally (fzf's {1} for the preview command, `cut -f1` for the final --resolve
    call); field 2 is what fzf shows and matches against (the caller hides the others
    with --with-nth=2); field 3 is empty for models that fit and the why-not text for
    the ones that don't, so the caller can refuse them. Columns are padded to `widths`
    (computed across all rows by print_list_compact) so they line up -- tab stops alone
    don't. `cols` restricts which padded columns actually get joined into the line (see
    visible_columns); columns left out here are still shown in full in --detail.

    Notes (the last cell) is never padded, but on a narrow terminal a long why-not reason
    can still run the line past term_width on its own -- so unlike the other columns it's
    truncated with an ellipsis to whatever room is left, rather than dropped outright."""
    # No known_issues marker here -- every model in this catalog carries some caveat
    # note (severity varies widely), so it wouldn't discriminate anything in this
    # compact view. Full known_issues text is always in the --detail preview pane.
    cells = compact_row_cells(r, reason)
    w = widths or (0,) * COL_LAST
    cols = cols if cols is not None else list(range(COL_LAST))
    cells[COL_NAME] = _truncate_cell(cells[COL_NAME], w[COL_NAME])
    padded = [cells[k].rjust(w[k]) if k in COMPACT_RIGHT_ALIGNED else cells[k].ljust(w[k])
              for k in cols]
    notes = cells[COL_LAST]
    if term_width is not None:
        fixed_width = sum(len(p) for p in padded) + 2 * len(padded)  # the "  " before Notes too
        room = term_width - fixed_width
        if room < 1:
            notes = ""
        elif len(notes) > room:
            notes = (notes[: room - 1] + "…") if room > 1 else "…"
    line = "  ".join(padded + [notes])
    if use_color(force_color):
        if not r["fits"]:
            line = f"{DIM}{line}{RESET}"
        else:
            # Named 16-color ANSI only (never 256/RGB), so the user's terminal decides
            # the actual hues. Green = fits unquantized; yellow = needs KV quantization
            # to fit. Meaning is also carried by the KV column text. Keyed by original
            # column index (not position in `cols`) since COL_CTX/KV/ENGINE may have
            # been dropped by visible_columns on a narrow terminal.
            hue = YELLOW if r["kv_bits"] else GREEN
            colored = dict(zip(cols, padded))
            if COL_CTX in colored:
                colored[COL_CTX] = f"{hue}{colored[COL_CTX]}{RESET}"
            if COL_KV in colored:
                colored[COL_KV] = f"{hue}{colored[COL_KV]}{RESET}"
            if COL_ENGINE in colored:
                colored[COL_ENGINE] = f"{DIM}{colored[COL_ENGINE]}{RESET}"
            line = "  ".join([colored[k] for k in cols] + [f"{DIM}{notes}{RESET}"])
    return f"{r['model']['id']}\t{line}\t{reason or ''}"


SORT_KEYS = ("released", "speed", "size", "name", "type", "engine", "drafter", "input",
             "tools", "context", "kv")

# Sort key for each compact_row_cells column (None = not sortable), for header clicks.
COLUMN_SORT_KEYS = ("name", "type", "engine", "drafter", "input", "tools", "released",
                    "size", "context", "kv", "speed", None)


def sort_key_for_click(char_index, widths, cols):
    """Map fzf's FZF_CLICK_HEADER_COLUMN (1-based character index into the displayed
    header line, calibrated against fzf 0.74: index 1 is the first character of the
    header text, indent excluded) to the sort key of the column under it. Clicks in the
    2-space gap between columns count toward the column on their left. `cols` is the
    same visible-column subset the header was actually rendered with (see
    visible_columns) -- columns dropped for width were never on screen to click."""
    start = 1
    chosen = None
    for k in list(cols) + [COL_LAST]:
        if char_index >= start:
            chosen = COLUMN_SORT_KEYS[k]
        width = widths[k] if k < COL_LAST else len(COMPACT_HEADERS[COL_LAST])
        start += width + 2
    return chosen


# Keys whose natural order is high-to-low / newest-first; the rest are A-Z.
DESCENDING_SORT_KEYS = {"released", "speed", "size", "context", "kv", "input", "tools"}


def sort_rows(rows, sort_key, reverse=False):
    """Tested-and-fitting first, then untested-and-fitting, then the models that don't
    fit -- under every sort, so a fast untested model never outranks a tested one --
    and `sort_key` orders within each group. Numeric/date keys (released, speed, size,
    context, kv) put the biggest or newest first with missing values last; the rest sort
    A-Z; `reverse` flips that within each group (the groups themselves never move). Ties
    keep a model's variants together by name. Stable sorts, innermost first. Returns
    (rows, descending) where `descending` is the direction actually applied."""
    def display(r):
        m = r["model"]
        return (model_display_name(m), model_engine(m), model_drafter_label(m))

    keys = {
        "released": lambda r: r["model"].get("released") or "",
        "speed": lambda r: r["estimated_tok_s"] if r["estimated_tok_s"] is not None else -1,
        "size": lambda r: r["model"]["size_gb"],
        "context": lambda r: r["recommended_context"] or 0,
        "kv": lambda r: (r["kv_bits"] or UNQUANTIZED_KV_BITS) if r["fits"] else 0,
        # more modalities first; "tools" first, then untested, then no tools
        "input": lambda r: len(compact_row_cells(r)[4]),
        "tools": lambda r: {"tools": 2, "tools?": 1, "no tools": 0}[model_tools_label(r["model"])],
        "type": lambda r: r["model"].get("type", "dense"),
        "engine": lambda r: model_engine(r["model"]),
        "drafter": lambda r: model_drafter_label(r["model"]),
        "name": display,
    }
    descending = (sort_key in DESCENDING_SORT_KEYS) != reverse
    rows = sorted(rows, key=display)
    rows.sort(key=keys[sort_key], reverse=descending)
    rows.sort(key=lambda r: (not r["fits"], r["model"].get("status") == "untested"))
    return rows, descending


def next_sort(requested_key, state_path):
    """Pick the sort for this render and remember it in `state_path` ("key reversed").
    Asking for the key the list is already sorted by flips its direction -- clicking the
    same header twice reverses it -- while a different key starts in its natural order.
    requested_key None (a click on a non-sortable spot) keeps the current sort as is."""
    prev_key, prev_rev = None, False
    if state_path:
        try:
            with open(state_path) as f:
                prev_key, flag = f.read().split()
                prev_rev = flag == "1"
        except (OSError, ValueError):
            pass
    if requested_key is None:
        key, rev = (prev_key, prev_rev) if prev_key in SORT_KEYS else ("released", False)
    elif requested_key == prev_key:
        key, rev = requested_key, not prev_rev
    else:
        key, rev = requested_key, False
    if state_path:
        try:
            with open(state_path, "w") as f:
                f.write(f"{key} {int(rev)}")
        except OSError:
            pass
    return key, rev


def print_list_compact(rows, chip_name, gpu_cores, total_ram_gb, free_wired_gb, preload=None,
                       sort_key=None, banner=True, sort_col=None, sort_state=None,
                       term_width=None, force_color=False):
    """--list-compact: the fzf-facing presentation mode. Every catalog model is listed;
    the ones that don't fit your free memory are dimmed and carry the reason, so you can
    see what a bigger machine would unlock. Rows go to stdout (fzf's list input); the
    header banner goes to stderr, so it's visible in the terminal right above where fzf
    paints (suppressed with banner=False when fzf re-runs this to re-sort). `sort_key`
    of None means "keep whatever's active" (see next_sort) rather than resetting to the
    released-first default -- callers that actually want a specific sort (ctrl-r/s/t/o)
    pass one explicitly. `sort_col` is a header click (see sort_key_for_click);
    `sort_state` is the file remembering the current sort so a second request for the
    same key reverses it. `term_width` (the caller's actual terminal width --
    start_mlx_server.sh passes fzf's own $FZF_COLUMNS) is what keeps the line from
    running off the right edge of a narrower window or terminal; None means render every
    column at full width, as before. `force_color` is --color -- see use_color()'s
    docstring: start_mlx_server.sh's fzf reload/bind commands all redirect this
    process's stderr to /dev/null to swallow system_profiler's noise, which defeats the
    stderr-isatty auto-detection on every single reload, on every machine -- not just
    the picker's first frame. The caller passes --color because it already knows this
    output is going into fzf's --ansi-enabled screen."""
    fitting_count = sum(1 for r in rows if r["fits"])
    if banner:
        log(f"  Local Model Advisor — {chip_name or 'unknown chip'}   "
            f"RAM: {total_ram_gb or '?'}GB   Free: {free_wired_gb:.1f}GB   "
            f"{fitting_count}/{len(rows)} models fit")
        if preload:
            print_preload_banner(preload)

    widths = compact_row_widths(rows)
    cols = visible_columns(widths, term_width)
    widths = cap_name_width(widths, cols, term_width)
    requested = sort_key_for_click(sort_col, widths, cols) if sort_col is not None else sort_key
    sort_key, reverse = next_sort(requested, sort_state)
    rows, descending = sort_rows(rows, sort_key, reverse)
    print(format_compact_header_row(widths, cols, sort_key, descending, force_color=force_color))
    for r in rows:
        reason = None if r["fits"] else short_unavailable_reason(r, free_wired_gb, preload)
        print(format_model_compact_row(r, widths, cols, reason, term_width=term_width,
                                        force_color=force_color))
    return [r for r in rows if r["fits"]]


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
            headroom_str = ""
            if r["context_after_preload"] is not None:
                headroom_str = f"  (~{r['context_after_preload']:,} left for conversation after preload)"
            conc_str = f"  concurrency={r['concurrency']}" if r["concurrency"] > 1 else ""
            log(f"  {i}) {m['id']}  [{m['display']['tagline']}]")
            log(f"       size={m['size_gb']}GB  max context={m['max_context_architectural']:,} tokens  "
                f"recommended context={r['recommended_context']:,} tokens  {speed_str}{headroom_str}{conc_str}")
            for line in format_model_body_lines(m, r):
                log(f"       {line}" if line else "")
        else:
            reason = describe_unavailable_reason(r, preload)
            log(f"{DIM}  —) {m['id']}  [{m['display']['tagline']}]")
            log(f"       {m['display']['good_for']}")
            log(f"       size={m['size_gb']}GB  max context={m['max_context_architectural']:,} tokens  "
                f"✗ unavailable now: {reason}{RESET}")
        log("")

    if not_fitting:
        log(f"  ({len(not_fitting)} model(s) above are greyed out — they don't fit your current free memory)")
        log("")

    return fitting


def _flag_value(argv, flag):
    if flag in argv:
        idx = argv.index(flag)
        if idx + 1 < len(argv):
            return argv[idx + 1]
    return None


def emit_selection_json(m, selected):
    # Machine-readable result — the ONLY thing on stdout.
    # kv_bits is None when this model's recommended context fits free memory
    # unquantized (the common case now — see HARDWARE_SPEED_ESTIMATION.md §14); the
    # caller (start_mlx_server.sh) must omit --kv-bits entirely in that case rather
    # than treating None as a literal value.
    print(json.dumps({
        "id": m["id"],
        "huggingface": m["huggingface"],
        "size_gb": m["size_gb"],
        "recommended_context": selected["recommended_context"],
        "kv_bits": selected["kv_bits"],
        "kv_bits_apc_caveat": selected["kv_bits_apc_caveat"],
        "estimated_tok_s": selected["estimated_tok_s"],
        "concurrency": selected["concurrency"],
        "prefill_step_size": selected["prefill_step_size"],
    }))


def main():
    argv = sys.argv[1:]
    list_json = "--list-json" in argv
    list_compact = "--list-compact" in argv
    detail_id = _flag_value(argv, "--detail")
    resolve_id = _flag_value(argv, "--resolve")
    tool = _flag_value(argv, "--tool")
    concurrency_flag = _flag_value(argv, "--concurrency")
    # Explicit request wins; anything absent/invalid falls back to auto mode (None)
    # rather than erroring, since this is a convenience override, not a required arg.
    requested_concurrency = None
    if concurrency_flag:
        try:
            requested_concurrency = max(1, int(concurrency_flag))
        except ValueError:
            requested_concurrency = None

    # Same convenience-override philosophy as --concurrency: an optional upper bound
    # on every model's recommended context (config.json's "max_context_tokens",
    # forwarded by start_mlx_server.sh's ADVISOR_FLAGS). A cap, never a bypass --
    # the memory-fit math still runs underneath it. Invalid values fall back to no
    # cap rather than erroring.
    max_context_flag = _flag_value(argv, "--max-context-tokens")
    max_context_tokens = None
    if max_context_flag:
        try:
            max_context_tokens = int(max_context_flag)
            if max_context_tokens < 1:
                max_context_tokens = None
        except ValueError:
            max_context_tokens = None

    with open(MODELS_JSON) as f:
        catalog = json.load(f)
    models = catalog["models"]

    current_wired_gb = get_current_wired_gb()
    chip_name, gpu_cores, total_ram_gb = get_chip_info()

    # Standalone, catalog-fit-independent write path: start_mlx_server.sh's
    # full_teardown calls this with whatever that session's own /metrics last
    # reported, so a FUTURE launch of the identical (chip, model) pair can report a
    # measured decode speed instead of estimate_speed_tok_s's physics formula — see
    # save_calibration's docstring. Doesn't need total_ram_gb/free_wired_gb or any of
    # the fit-check machinery below; a session that already ran doesn't need its own
    # launch-time fit decision repeated just to record what actually happened.
    save_calibration_id = _flag_value(argv, "--save-calibration")
    if save_calibration_id is not None:
        decode_flag = _flag_value(argv, "--decode-tok-s")
        if decode_flag is None:
            log("ERROR: --save-calibration requires --decode-tok-s")
            sys.exit(2)
        try:
            decode_tok_s = float(decode_flag)
        except ValueError:
            log(f"ERROR: --decode-tok-s value '{decode_flag}' is not a number")
            sys.exit(2)
        if decode_tok_s <= 0:
            log(f"Skipping calibration save: non-positive decode_tok_s ({decode_tok_s}) — nothing real to record.")
            return
        context_flag = _flag_value(argv, "--context-tokens")
        ttft_flag = _flag_value(argv, "--ttft-s")
        peak_mem_flag = _flag_value(argv, "--peak-memory-gb")
        try:
            context_tokens = int(context_flag) if context_flag else None
            ttft_s = float(ttft_flag) if ttft_flag else None
            peak_memory_gb = float(peak_mem_flag) if peak_mem_flag else None
        except ValueError:
            context_tokens = ttft_s = peak_memory_gb = None
        # Both optional, additive (see save_calibration's own docstring) -- written by
        # the sweep harness (sweep_model.py), never by the plain teardown-time
        # save_calibration_snapshot() path, which has no sweep data to offer.
        prefill_step_size_flag = _flag_value(argv, "--prefill-step-size")
        kv_bits_safe_flag = _flag_value(argv, "--kv-bits-safe-at-long-context")
        try:
            prefill_step_size = int(prefill_step_size_flag) if prefill_step_size_flag else None
        except ValueError:
            prefill_step_size = None
        kv_bits_safe_at_long_context = None
        if kv_bits_safe_flag is not None:
            kv_bits_safe_at_long_context = kv_bits_safe_flag.strip().lower() in ("1", "true", "yes")
        save_calibration(chip_name, save_calibration_id, decode_tok_s,
                          context_tokens=context_tokens, ttft_s=ttft_s,
                          peak_memory_gb=peak_memory_gb,
                          prefill_step_size=prefill_step_size,
                          kv_bits_safe_at_long_context=kv_bits_safe_at_long_context)
        log(f"Saved calibration for '{save_calibration_id}' on '{chip_name}': {decode_tok_s} tok/s")
        return

    if total_ram_gb is None:
        log("ERROR: could not detect total RAM via system_profiler.")
        sys.exit(2)
    free_wired_gb = total_ram_gb - current_wired_gb

    preload = estimate_preload_overhead(tool, os.getcwd()) if tool else None
    rows = build_recommendations(
        models, free_wired_gb, chip_name, gpu_cores,
        preload_overhead=(preload["total"] if preload else 0),
        requested_concurrency=requested_concurrency,
        max_context_tokens=max_context_tokens,
    )

    # Phase 6: three non-interactive presentation modes for the fzf-based picker.
    # None of these touch recommend_kv_bits_and_context/build_recommendations --
    # purely how the same computed rows get rendered or looked up by id.
    if detail_id is not None:
        row = next((r for r in rows if r["model"]["id"] == detail_id), None)
        if row is None:
            log(f"ERROR: unknown model id '{detail_id}'")
            sys.exit(2)
        print(row["model"]["id"])
        print(row["model"]["display"]["tagline"])
        print("")
        for line in format_model_body_lines(row["model"], row):
            print(line)
        return

    if resolve_id is not None:
        row = next((r for r in rows if r["model"]["id"] == resolve_id), None)
        if row is None:
            log(f"ERROR: unknown model id '{resolve_id}'")
            sys.exit(2)
        if not row["fits"]:
            log(f"ERROR: '{resolve_id}' does not currently fit free memory.")
            sys.exit(3)
        emit_selection_json(row["model"], row)
        return

    if list_compact:
        # fzf's click-header passes whatever word was clicked ("Notes", "✓", ...), so an
        # unrecognized value falls back to the default instead of failing the reload.
        # Omitting --sort entirely (as the resize/start reloads below do) is NOT the same
        # as asking for "released" -- it means "whatever's already active", so a window
        # resize or the initial start-up render never resets the user's chosen sort (see
        # next_sort: None preserves the prior state, falling back to "released" only when
        # there isn't one yet).
        sort_arg = _flag_value(argv, "--sort")
        sort_key = sort_arg.lower() if sort_arg else None
        if sort_key is not None and sort_key not in SORT_KEYS:
            sort_key = "released"
        sort_col = _flag_value(argv, "--sort-col")
        # Caller's real terminal width (start_mlx_server.sh passes `tput cols`), used to
        # drop/truncate columns so the row never runs off the right edge of the window --
        # see visible_columns. Absent or garbage falls back to unlimited width (today's
        # pre-existing behavior) rather than guessing.
        term_width_flag = _flag_value(argv, "--term-width")
        term_width = None
        if term_width_flag:
            try:
                term_width = int(term_width_flag)
                if term_width < 20:
                    term_width = None
            except ValueError:
                term_width = None
        print_list_compact(rows, chip_name, gpu_cores, total_ram_gb, free_wired_gb, preload,
                           sort_key=sort_key, banner="--no-banner" not in argv,
                           sort_col=int(sort_col) if sort_col and sort_col.isdigit() else None,
                           sort_state=_flag_value(argv, "--sort-state"),
                           term_width=term_width, force_color="--color" in argv)
        return

    if list_json:
        print(json.dumps({
            "current_wired_gb": current_wired_gb, "free_wired_gb": free_wired_gb,
            "chip": chip_name, "gpu_cores": gpu_cores,
            "total_ram_gb": total_ram_gb, "preload_overhead": preload,
            # Exposed so callers (launch_tool.sh's attach-time preload-fit check) don't
            # have to hardcode a second copy of this constant.
            "min_usable_context": MIN_USABLE_CONTEXT,
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
    emit_selection_json(selected["model"], selected)


if __name__ == "__main__":
    main()
