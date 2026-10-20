#!/usr/bin/env python3
"""
sweep_model.py — comprehensive per-(chip, model) calibration sweep: prefill-step-size
TTFT optimum, decode-speed-vs-context-depth curve, KV-bits decode-speed cost AT LONG
CONTEXT, Automatic Prefix Caching (APC) hit-rate/correctness, tool-calling/multimodal/
drafter re-verification, concurrency throughput, and a thermal-throttling check.

See docs/NEXT_SESSION_HANDOFF_PREFILL_SWEEP.md for the full spec/rationale this implements
("the sweep spec" section) and docs/HARDWARE_SPEED_ESTIMATION.md §12/§13/§17/§19 for the
prior art each tier extends. Short version of WHY this exists: `start_mlx_server.sh`'s
hardcoded `PREFILL_STEP_SIZE=2048` was derived from sweeping exactly 2 models on 1
machine -- this script makes that a per-(chip, model) SELF-CALIBRATING measurement
instead of a bigger version of the same hardcoded global constant, reusing the
calibration-file infrastructure in model_advisor.py (save_calibration/load_calibration/
recommend_prefill_step_size).

Design: standalone, NOT a mode of start_mlx_server.sh. It manages its OWN short-lived
`mlx_vlm.server` subprocess(es) directly (same binary/venv/env-var shape as that
script's non-dspark launch branch, deliberately NOT reusing that script's own
interactive picker/dashboard/memory-backstop machinery) -- this is unavoidable for the
prefill-step-size axis specifically, since `--prefill-step-size` is a server LAUNCH
flag, not a per-request parameter, so sweeping it requires relaunching the server with
each candidate value. Every OTHER tier below reuses whichever one of those launches is
still live rather than relaunching again, maximizing what gets measured per model-load
per docs/NEXT_SESSION_HANDOFF_PREFILL_SWEEP.md's own stated priority.

Scope: covers both the `mlx_vlm` backend (including `--draft-model` speculative decoding)
and the `mlx_dspark` backend (`mlx-dspark serve --drafter ...`). Because `mlx-dspark`
self-calibrates its own prefill path and has no `--prefill-step-size` launch flag, the
`prefill_step_size` tier is automatically skipped when `serving.backend == "mlx_dspark"`
while all other tiers (`decode_vs_context`, `apc_cache_test`, `tool_calling_reverified`,
`speculative_decoding`, `concurrency_throughput`, `thermal_check`) run against
`mlx-dspark serve`.

Usage:
  python3 sweep_model.py --model-id gemma-4-e4b-it-4bit
  python3 sweep_model.py --model-id gemma-4-e4b-it-4bit --step-sizes 2048,4096 \\
      --context-depths 100,4096 --concurrency-levels 1,2 --thermal-requests 8
      # a smaller/faster pass for validating the harness itself
  python3 sweep_model.py --untested-cached --sync-verified
      # batch-sweep all locally cached untested models and sync into models.json
  python3 sweep_model.py --all-cached --force
      # re-sweep every locally cached catalog model on this chip

Writes:
  ~/.bushpilot/calibration/sweeps/<chip>__<model_id>.sweep.json  (rich diagnostics,
    written incrementally after each tier completes, so a crash mid-sweep doesn't
    lose everything already measured)
  ~/.bushpilot/calibration/<chip>__<model_id>.json  (the existing lightweight
    calibration file model_advisor.py's load_calibration/estimate_speed_tok_s/
    recommend_prefill_step_size already read -- gets 2 new optional keys derived from
    the sweep above: prefill_step_size, kv_bits_safe_at_long_context. Written via
    model_advisor.save_calibration(), never hand-rolled, so that function's own
    atomic-write/last-write-wins contract is the only place this logic lives.)
"""
import argparse
import glob
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
import model_advisor as ma  # noqa: E402  (path insert above must run first)

BUSHPILOT_HOME = os.path.expanduser("~/.bushpilot")
BUSHPILOT_VENV = os.path.join(BUSHPILOT_HOME, "venv")
MLX_PYTHON = os.path.join(BUSHPILOT_VENV, "bin", "python")
MLX_DSPARK = os.path.join(BUSHPILOT_VENV, "bin", "mlx-dspark")
HF_HOME = os.path.join(BUSHPILOT_HOME, "models")
APC_DISK_PATH = os.path.join(BUSHPILOT_HOME, "apc-cache")
SWEEP_DIR = os.path.join(ma.CALIBRATION_DIR, "sweeps")

# Deliberately a different port from start_mlx_server.sh's production default
# (38472 via config.json/CONFIG_PORT) -- this script's own server instances must
# never collide with (or be mistaken for) a real tool-attached session.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 38973

# §12's exact 3 candidate values -- see docs/HARDWARE_SPEED_ESTIMATION.md §12/§17 item #8
# for why these 3 and why the result (2048 lowest-TTFT, non-monotonic) is a measured
# finding on 2 models/1 machine, not an understood-and-tuned optimum.
DEFAULT_STEP_SIZES = [512, 2048, 4096]
# Short baseline, a "real coding session" medium depth, and a genuinely long depth --
# mirrors the handoff doc's own illustrative schema (100 / 7200ish / 16384). 4096
# replaces the doc's 7200 only so a single depth list can serve both the
# decode-speed-vs-context curve AND double as the step-size sweep's test depth
# without needing two separate constants that drift apart.
DEFAULT_CONTEXT_DEPTHS = [100, 4096, 16384]
DEFAULT_CONCURRENCY_LEVELS = [1, 2, 4]
DEFAULT_THERMAL_REQUESTS = 40
# §13's own measured prefix length (~4,450 tokens) -- kept close to that, not copied
# exactly, since this script's rough token-counting (chars/4, see build_filler_prompt)
# can't hit an exact figure anyway.
DEFAULT_APC_PREFIX_TOKENS = 4096
# §12/§17 item #6's own data: every model tested so far shows a ≤1% KV-bits decode
# slowdown at SHORT context; the long-context regime is this project's sharpest named
# gap (§17 #6/#13) precisely because nobody has measured whether that floor holds once
# quantization's memory benefit actually kicks in. 10% is a deliberately generous
# threshold (not "noise-level" like the ~1% short-context floor) chosen so a real but
# modest long-context cost doesn't get waved through as "safe" -- documented judgment
# call, not a re-derivation of §12's relationship for every model.
KV_BITS_LONG_CONTEXT_SAFE_SLOWDOWN_PCT = 10.0
WARMUP_RUNS = 1
MEASURED_RUNS = 3
SHORT_PROMPT = "Write a 200-word short story about a robot exploring an old library."
FILLER_SENTENCE = " The weather today is mild and pleasant with light clouds. "


def log(msg):
    print(f"[sweep] {msg}", file=sys.stderr)


# --- Pure logic (unit-testable without a real server) ------------------------------

def count_tokens_rough(text):
    """Same chars/4 heuristic model_advisor.py already uses for preload estimation --
    good enough to aim a prompt at an approximate depth; the REAL token count used for
    every recorded metric below always comes back from the server's own response
    (`timings.prompt_n` + `timings.cache_n`), never from this estimate."""
    return max(0, len(text) // 4)


def build_filler_prompt(target_tokens, instruction="Reply with just the word done.", nonce=None):
    """§12's own filler-prompt methodology (repeated short sentences) -- kept as one
    of two prompt shapes (see build_code_prompt for the other) so the step-size sweep
    isn't making the same single-prompt-shape assumption §17 item #8 flags as
    unconfirmed ("genuinely different prompt content... could shift where the sweet
    spot sits").

    nonce: see build_code_prompt's docstring for why this exists -- a real,
    confirmed-live bug this session, not a hypothetical one."""
    prefix = f"[sweep-nonce:{nonce}]\n" if nonce is not None else ""
    repeats = max(1, target_tokens // count_tokens_rough(FILLER_SENTENCE))
    return prefix + FILLER_SENTENCE * repeats + f"\n\n{instruction}"


def build_short_decode_prompt(target_tokens=None, nonce=None):
    """§12's EXACT original short-context prompt ("Write a 200-word short story...")
    -- deliberately NOT build_filler_prompt's "Reply with just the word done."
    default, which invites a trivially short completion. Confirmed live this session:
    using the filler default for the shallow point of decode_speed_vs_context produced
    an unreliable decode_tok_s (early-EOS completions a few tokens long, the exact
    small-sample problem §12's own methodology note already flags for its long-prompt
    runs: "every long-prompt run's predicted_n is 17, not the requested max_tokens=50
    ... a smaller sample per run than intended"). This prompt is what §12 originally
    used BECAUSE it reliably produces a long enough completion (predicted_n=150,
    hitting max_tokens, not EOS) to average cleanly -- reused here verbatim so this
    tier's own short-context decode_tok_s point is trustworthy enough to eventually
    feed a models.json speed_correction_factor update (remaining-work item #6), not
    just directionally similar.

    target_tokens is accepted-but-ignored so this has the same call signature as
    build_filler_prompt/build_code_prompt for run_measured_requests' prompt_fn slot --
    this prompt's length isn't meant to scale with a target depth, it's always §12's
    one short baseline prompt."""
    prefix = f"[sweep-nonce:{nonce}]\n" if nonce is not None else ""
    return prefix + SHORT_PROMPT


def build_code_prompt(target_tokens, repo_file=None,
                       instruction="In one sentence, summarize what this code does.",
                       nonce=None):
    """A REAL source file from this repo (default: model_advisor.py itself) repeated/
    truncated to roughly target_tokens, instead of only filler prose -- closes part of
    §17 item #8's caveat ("genuinely different prompt content... e.g. code") by
    actually trying one, rather than repeating the caveat unaddressed.

    nonce: when given, prepended as the prompt's very FIRST characters -- NOT the
    suffix -- so it lands in APC's first 16-token hash block and invalidates reuse of
    every block after it too (prefix caching only reuses a BLOCK-SEQUENCE match
    starting from block 0; a mismatched first block means nothing downstream can
    match either, confirmed against this project's own APC description in
    docs/HARDWARE_SPEED_ESTIMATION.md §13). Exists because of a real bug confirmed live
    this session: APC_ENABLED=1 is unconditional in production (and in every
    SweepServer launch here, deliberately, to stay production-representative) and its
    DISK tier survives process restarts (§13's own "surviving the model-switch
    restarts this project's own tooling causes" finding) -- repeating this exact
    function's output with no nonce across (a) a warmup-then-measured loop's own
    iterations and (b) separate step-size/kv-bits relaunches both self-contaminated
    "cold" TTFT into a near-instant cache hit (observed: 0.03s TTFT at 16,384 tokens
    of context, versus the multi-second figure §12 measured at less than half that
    depth with no APC in the picture at all). Every caller that wants a genuinely cold
    prefill measurement MUST pass a nonce that's unique per request, not just per
    tier/launch -- see run_measured_requests, which does exactly that."""
    prefix = f"[sweep-nonce:{nonce}]\n" if nonce is not None else ""
    repo_file = repo_file or os.path.join(SCRIPT_DIR, "model_advisor.py")
    try:
        with open(repo_file, encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except OSError:
        return build_filler_prompt(target_tokens, instruction, nonce=nonce)
    if not text:
        return build_filler_prompt(target_tokens, instruction, nonce=nonce)
    target_chars = max(1, target_tokens * 4)
    body = text
    while count_tokens_rough(body) < target_tokens:
        body += "\n" + text
    body = body[:target_chars]
    return f"{prefix}{instruction}\n\n```python\n{body}\n```"


def average_runs(runs, key):
    """Plain arithmetic mean of one numeric field across a list of measured-run
    dicts, skipping any run that's missing or non-numeric for that field (a single
    flaky/erroring run shouldn't poison the whole average) -- None if nothing usable
    was found at all, rather than a misleading 0.0."""
    vals = [r[key] for r in runs if isinstance(r.get(key), (int, float))]
    if not vals:
        return None
    return sum(vals) / len(vals)


def pick_best_step_size(prefill_step_size_sweep):
    """Lowest-TTFT entry wins -- §12's own reframing ("the metric that actually
    matters here is TTFT, not a throughput number", see that section's own writeup) is
    carried through here rather than re-litigated. Entries with no usable ttft_s
    (e.g. every run in that step size errored) are never picked; returns
    DEFAULT_STEP_SIZES's middle value (today's hardcoded global default) if nothing
    usable exists at all, rather than crashing the whole sweep over one bad step."""
    usable = [e for e in prefill_step_size_sweep if isinstance(e.get("ttft_s"), (int, float))]
    if not usable:
        return ma.DEFAULT_PREFILL_STEP_SIZE
    return min(usable, key=lambda e: e["ttft_s"])["step_size"]


def kv_bits_safe_at_long_context(decode_speed_vs_context, deepest_depth):
    """True when the quantized-KV decode slowdown AT THE DEEPEST TESTED DEPTH stays
    within KV_BITS_LONG_CONTEXT_SAFE_SLOWDOWN_PCT -- closes §17 item #6/#13's sharpest
    named gap (short-context-only data backing the existing dynamic --kv-bits
    decision) with a real long-context measurement. Returns None (not True/False) when
    either side of the comparison is missing -- "untested", not a guessed verdict."""
    unquant = next((e for e in decode_speed_vs_context
                     if e.get("context_tokens") == deepest_depth and e.get("kv_bits") is None), None)
    quant = next((e for e in decode_speed_vs_context
                   if e.get("context_tokens") == deepest_depth and e.get("kv_bits") is not None), None)
    if not unquant or not quant:
        return None
    base = unquant.get("decode_tok_s")
    quantized = quant.get("decode_tok_s")
    if not isinstance(base, (int, float)) or not isinstance(quantized, (int, float)) or base <= 0:
        return None
    slowdown_pct = (base - quantized) / base * 100.0
    return slowdown_pct <= KV_BITS_LONG_CONTEXT_SAFE_SLOWDOWN_PCT


def derive_lightweight_calibration_fields(sweep_data):
    """Condenses the rich sweep JSON into exactly the fields save_calibration() needs
    for the lightweight file (decode_tok_s/context_tokens/ttft_s/peak_memory_gb from
    the WINNING step size's own measured run, plus the optional sweep keys) -- see
    docs/NEXT_SESSION_HANDOFF_PREFILL_SWEEP.md's storage-design section for why these two
    files are kept deliberately separate. Returns a plain kwargs dict ready to
    **-expand into model_advisor.save_calibration(), or None if the sweep didn't even
    get far enough to produce a usable step-size measurement (e.g. every launch in
    this sweep failed).

    When `prefill_step_size_sweep` contains measurements at one or more `context_tokens`
    depths, also populates `prefill_step_size_by_context` (mapping `str(depth)` ->
    `{"step_size": ..., "peak_memory_gb": ..., "ttft_s": ...}`) so
    `recommend_prefill_step_size(..., context_tokens=...)` can select the optimum for the
    recommended context depth (closing §17 item #8). When multiple depths are present in
    one sweep, the flat top-level `prefill_step_size` / `decode_tok_s` fields are taken
    from the deepest swept depth."""
    step_sweep = sweep_data.get("prefill_step_size_sweep") or []
    decode_curve = sweep_data.get("decode_speed_vs_context") or []

    by_depth_entries = {}
    for entry in step_sweep:
        ctx = entry.get("context_tokens")
        if isinstance(ctx, int) and not isinstance(ctx, bool) and ctx > 0:
            by_depth_entries.setdefault(ctx, []).append(entry)

    by_context = {}
    for ctx in sorted(by_depth_entries):
        d_entries = by_depth_entries[ctx]
        usable_d = [e for e in d_entries if isinstance(e.get("ttft_s"), (int, float))]
        if not usable_d:
            continue
        best_d = pick_best_step_size(usable_d)
        winner_d = next((e for e in usable_d if e.get("step_size") == best_d), None)
        if winner_d is not None:
            rec = {"step_size": best_d}
            if isinstance(winner_d.get("peak_memory_gb"), (int, float)):
                rec["peak_memory_gb"] = winner_d["peak_memory_gb"]
            if isinstance(winner_d.get("ttft_s"), (int, float)):
                rec["ttft_s"] = winner_d["ttft_s"]
            by_context[str(ctx)] = rec

    winner = None
    best_step_size = ma.DEFAULT_PREFILL_STEP_SIZE
    if step_sweep:
        primary_pool = by_depth_entries[max(by_depth_entries)] if by_depth_entries else step_sweep
        best_step_size = pick_best_step_size(primary_pool)
        winner = next((e for e in primary_pool if e.get("step_size") == best_step_size), None)
    if winner is None or not isinstance(winner.get("decode_tok_s"), (int, float)):
        # Fallback when prefill_step_size_sweep was skipped (e.g. mlx_dspark backend or
        # --skip-tiers prefill_step_size): use the deepest unquantized decode_speed_vs_context
        # point so decode_tok_s / decode_tok_s_by_context / kv_bits_safe_at_long_context still
        # land in the lightweight calibration file.
        usable_curve = [
            d for d in decode_curve
            if d.get("kv_bits") is None
            and isinstance(d.get("decode_tok_s"), (int, float))
            and d["decode_tok_s"] > 0
            and isinstance(d.get("context_tokens"), (int, float))
        ]
        if not usable_curve:
            return None
        winner = max(usable_curve, key=lambda d: d["context_tokens"])
        best_step_size = ma.DEFAULT_PREFILL_STEP_SIZE

    deepest = max((d["context_tokens"] for d in sweep_data.get("decode_speed_vs_context", [])
                   if isinstance(d.get("context_tokens"), (int, float))), default=None)
    kv_safe = None
    if deepest is not None:
        kv_safe = kv_bits_safe_at_long_context(sweep_data.get("decode_speed_vs_context", []), deepest)

    decode_by_context = {}
    for entry in sweep_data.get("decode_speed_vs_context") or []:
        ctx = entry.get("context_tokens")
        tok_s = entry.get("decode_tok_s")
        if (
            isinstance(ctx, (int, float))
            and not isinstance(ctx, bool)
            and int(ctx) > 0
            and isinstance(tok_s, (int, float))
            and not isinstance(tok_s, bool)
            and tok_s > 0
        ):
            depth_key = str(int(ctx))
            # Prefer the unquantized (kv_bits is None) baseline at each depth when both
            # unquantized and quantized runs exist in decode_speed_vs_context.
            if entry.get("kv_bits") is None or depth_key not in decode_by_context:
                decode_by_context[depth_key] = round(float(tok_s), 2)

    out = {
        "decode_tok_s": winner["decode_tok_s"],
        "context_tokens": winner.get("context_tokens"),
        "ttft_s": winner.get("ttft_s"),
        "peak_memory_gb": winner.get("peak_memory_gb"),
        "prefill_step_size": best_step_size,
        "kv_bits_safe_at_long_context": kv_safe,
    }
    if by_context:
        out["prefill_step_size_by_context"] = by_context
    if decode_by_context:
        out["decode_tok_s_by_context"] = decode_by_context
    return out


def derive_speed_correction_factor(sweep_data, model_entry, chip_name, gpu_cores=None):
    """Derives the empirical `speed_correction_factor` for `model_entry` on `chip_name`
    by inverting `estimate_speed_tok_s`'s bandwidth formula against the sweep's
    shallowest unquantized `decode_speed_vs_context` measurement (falling back to the
    winning step-size run's `decode_tok_s` if `decode_speed_vs_context` is absent):
        speed_correction_factor = round(decode_tok_s * active_gb / bandwidth_gb_s, 3)
    Returns None for drafter-backed entries (`serving.drafter` / `serving.draft_model`),
    because `speed_correction_factor` is a base-architecture property whereas speculative
    decoding speedup is represented via `serving.measured_speedup`.
    Closes docs/HARDWARE_SPEED_ESTIMATION.md §17 item #2 / NEXT_SESSION_HANDOFF_PREFILL_SWEEP.md
    remaining-work item #6."""
    serving = model_entry.get("serving") or {}
    if serving.get("drafter") or serving.get("draft_model"):
        return None
    bw = ma.CHIP_BANDWIDTH_GB_S.get(chip_name)
    if not bw or bw <= 0:
        return None
    active_gb = model_entry.get("active_weight_gb")
    if active_gb is None:
        if model_entry.get("type") == "moe":
            return None
        active_gb = model_entry.get("size_gb")
    if not isinstance(active_gb, (int, float)) or active_gb <= 0:
        return None

    curve = [
        d for d in (sweep_data.get("decode_speed_vs_context") or [])
        if d.get("kv_bits") is None and isinstance(d.get("decode_tok_s"), (int, float))
        and d["decode_tok_s"] > 0 and isinstance(d.get("context_tokens"), (int, float))
    ]
    measured_tok_s = None
    if curve:
        shallowest_entry = min(curve, key=lambda d: d["context_tokens"])
        measured_tok_s = shallowest_entry["decode_tok_s"]
    else:
        derived = derive_lightweight_calibration_fields(sweep_data)
        if derived and isinstance(derived.get("decode_tok_s"), (int, float)) and derived["decode_tok_s"] > 0:
            measured_tok_s = derived["decode_tok_s"]
    if measured_tok_s is None:
        return None
    return round(measured_tok_s * active_gb / bw, 3)


def is_multimodal_entry(model_entry):
    """Architectural capability (models.json's own modalities list) narrowed by
    serving.multimodal_capable when a drafter is attached -- same rule
    model_advisor.py's picker column already applies (see models.json's
    _schema_notes.modalities)."""
    if "image" not in (model_entry.get("modalities") or []):
        return False
    serving = model_entry.get("serving") or {}
    return serving.get("multimodal_capable", True)


# A tiny 8x8 solid-color RGB PNG, base64-encoded -- just enough for a real multimodal
# request to exercise the vision tower's forward pass for a SPEED measurement; this is
# not a correctness test (no claim is made about what the model says about the
# image), matching the Tier 3 table's own framing ("currently undocumented —
# correctness was verified, speed never was"). Deliberately NOT 1x1 -- confirmed live
# on this machine: a 1x1 PNG decodes (via PIL) to a (1, 1, 1)-shaped array that
# mlx_vlm's own image preprocessor can't handle ("TypeError: Cannot handle this data
# type: (1, 1, 1), |u1", a real 500 from a live sweep run), not something anticipated
# ahead of time. 8x8 is comfortably past whatever minimum size trips that edge case
# while still being effectively free to transmit/decode.
TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAIAAABLbSncAAAAEklEQVR4nGP4z8CAFWEXHbQSACj/"
    "P8Fu7N9hAAAAAElFTkSuQmCC"
)



def normalize_dspark_timings(x_dspark, metrics=None):
    """Synthesizes an mlx_vlm-compatible `timings` dict from mlx-dspark's per-response
    `x_mlx_dspark` block (plus an optional `/metrics` snapshot for `peak_memory` and
    `draft_acceptance`). Lets every measurement tier below treat `mlx_vlm` and
    `mlx-dspark` identically."""
    if not isinstance(x_dspark, dict):
        return None
    prompt_tokens = x_dspark.get("prompt_tokens") or 0
    cached_tokens = x_dspark.get("cached_tokens") or 0
    prompt_n = max(0, prompt_tokens - cached_tokens)
    ttft_s = x_dspark.get("ttft_seconds")
    if ttft_s is None:
        ttft_s = x_dspark.get("prefill_seconds")
    prompt_ms = (ttft_s * 1000.0) if isinstance(ttft_s, (int, float)) else None
    prefill_tok_s = x_dspark.get("prefill_tokens_per_sec")
    if prefill_tok_s is None:
        prefill_s = x_dspark.get("prefill_seconds")
        if isinstance(prefill_s, (int, float)) and prefill_s > 0:
            prefill_tok_s = prompt_n / prefill_s
    peak_mem_gb = None
    draft_acceptance = None
    if isinstance(metrics, dict):
        peak_bytes = (metrics.get("memory") or {}).get("peak_bytes")
        if isinstance(peak_bytes, (int, float)) and peak_bytes > 0:
            peak_mem_gb = peak_bytes / 1e9
        acc = (metrics.get("rounds") or {}).get("draft_acceptance")
        if isinstance(acc, (int, float)):
            draft_acceptance = acc
    return {
        "prompt_n": prompt_n,
        "cache_n": cached_tokens,
        "predicted_n": x_dspark.get("completion_tokens"),
        "prompt_ms": prompt_ms,
        "prompt_per_second": prefill_tok_s,
        "predicted_per_second": (
            x_dspark.get("decode_tokens_per_sec") or x_dspark.get("tokens_per_sec")
        ),
        "peak_memory": peak_mem_gb,
        "draft_kind": x_dspark.get("mode"),
        "accept_len": x_dspark.get("accept_len"),
        "target_forwards": x_dspark.get("target_forwards"),
        "draft_acceptance_rate": draft_acceptance,
    }


def extract_draft_stats_from_timings(t):
    """Extracts `(draft_acceptance_rate, mean_accept_len)` from either an `mlx_vlm`
    `GenerationTimings` dict (`draft_n`, `draft_n_accepted`, `draft_rounds`) or a
    normalized `mlx-dspark` timings dict (`draft_acceptance_rate`, `accept_len`)."""
    if not isinstance(t, dict):
        return None, None
    acc_rate = t.get("draft_acceptance_rate")
    mean_accept = t.get("accept_len")
    draft_n = t.get("draft_n")
    draft_n_accepted = t.get("draft_n_accepted")
    draft_rounds = t.get("draft_rounds")
    if acc_rate is None and isinstance(draft_n, int) and draft_n > 0:
        acc_rate = (draft_n_accepted or 0) / draft_n
    if mean_accept is None and isinstance(draft_rounds, int) and draft_rounds > 0:
        mean_accept = 1.0 + (draft_n_accepted or 0) / draft_rounds
    return acc_rate, mean_accept


# --- HTTP helpers --------------------------------------------------------------------

def _http_json(method, url, body=None, timeout=180):
    """Returns (ok, status_or_None, parsed_json_or_None, error_str_or_None). Never
    raises -- every caller gets a uniform result shape to branch on, same philosophy as
    verify_max_kv_size.sh's send_recall_probe (http_code/error capture, no exception
    propagation out of the probe itself)."""
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return True, resp.status, (json.loads(raw) if raw else {}), None
    except urllib.error.HTTPError as e:
        try:
            parsed = json.loads(e.read())
        except Exception:
            parsed = None
        return False, e.code, parsed, str(e)
    except Exception as e:
        return False, None, None, str(e)


def get_metrics(host, port, timeout=5):
    ok, _status, parsed, _err = _http_json("GET", f"http://{host}:{port}/metrics", timeout=timeout)
    return parsed if ok and isinstance(parsed, dict) else None


def chat_completion(host, port, model_hf_id, content, max_tokens=50, temperature=0.0,
                     tools=None, timeout=180):
    """Sends one /v1/chat/completions request and returns a flat result dict:
    ok, http_code, error, text, finish_reason, usage, timings (mlx_vlm's own
    per-request GenerationTimings block -- prompt_n/cache_n/predicted_n/prompt_ms/
    prompt_per_second/predicted_ms/predicted_per_second/peak_memory/draft_* -- see
    mlx_vlm/server/schemas.py's GenerationTimings, or a normalized timings dict
    synthesized from mlx-dspark's `x_mlx_dspark` block when talking to an
    `mlx-dspark serve` backend)."""
    body = {
        "model": model_hf_id,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if tools is not None:
        body["tools"] = tools
    ok, status, parsed, err = _http_json(
        "POST", f"http://{host}:{port}/v1/chat/completions", body, timeout=timeout,
    )
    out = {"ok": ok, "http_code": status, "error": err, "text": None,
           "finish_reason": None, "usage": None, "timings": None}
    if isinstance(parsed, dict):
        choices = parsed.get("choices") or []
        if choices:
            out["finish_reason"] = choices[0].get("finish_reason")
            out["text"] = (choices[0].get("message") or {}).get("content")
        out["usage"] = parsed.get("usage")
        out["timings"] = parsed.get("timings")
        if out["timings"] is None and isinstance(parsed.get("x_mlx_dspark"), dict):
            metrics = get_metrics(host, port)
            out["timings"] = normalize_dspark_timings(parsed["x_mlx_dspark"], metrics=metrics)
    return out


def multimodal_chat_completion(host, port, model_hf_id, prompt, max_tokens=50, timeout=180):
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{TINY_PNG_B64}"}},
    ]
    return chat_completion(host, port, model_hf_id, content, max_tokens=max_tokens, timeout=timeout)


def get_health(host, port, timeout=5):
    ok, status, parsed, err = _http_json("GET", f"http://{host}:{port}/health", timeout=timeout)
    return ok, parsed, err


def get_cache_stats(host, port, timeout=5):
    ok, status, parsed, err = _http_json("GET", f"http://{host}:{port}/v1/cache/stats", timeout=timeout)
    return parsed if ok else None


def reset_cache(host, port, timeout=5):
    _http_json("POST", f"http://{host}:{port}/v1/cache/reset", timeout=timeout)


def wait_for_health(host, port, pid, timeout_s=600):
    """Mirrors start_mlx_server.sh's own wait_for_url -- polls /health once per
    second, bailing out early (instead of burning the full timeout) if the watched
    PID has already exited, since a crashed launch has nothing left to come up."""
    start = time.monotonic()
    for _ in range(timeout_s):
        ok, _, _ = get_health(host, port, timeout=5)
        if ok:
            return True, time.monotonic() - start
        if pid is not None:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return False, time.monotonic() - start
        time.sleep(1)
    return False, time.monotonic() - start


# --- Server process management (mlx_vlm and mlx-dspark backends) ---------------------

# Every live SweepServer registers itself here on start() and deregisters on stop() --
# lets _handle_termination_signal below tear down whatever's currently running
# instead of leaking an orphaned mlx_vlm.server or mlx-dspark process. Confirmed as a
# REAL (not hypothetical) risk this session: an external kill of this script mid-sweep
# (a shell timeout, in that instance) left the question of whether the in-flight server
# subprocess would be reaped genuinely unclear -- Python's subprocess.Popen does NOT
# kill children automatically when its own parent is killed unless the parent
# explicitly handles the signal, matching exactly why start_mlx_server.sh's own
# full_teardown traps INT/HUP/TERM rather than assuming the OS handles it.
_ACTIVE_SERVERS = []


def _handle_termination_signal(signum, _frame):
    log(f"Received signal {signum} — tearing down any live sweep server before exiting...")
    for server in list(_ACTIVE_SERVERS):
        server.stop()
    sys.exit(1)


signal.signal(signal.SIGINT, _handle_termination_signal)
signal.signal(signal.SIGTERM, _handle_termination_signal)


class SweepServer:
    """Launches/tears down one `mlx_vlm.server` (with optional `--draft-model` /
    `--draft-kind` / `--draft-block-size`) or `mlx-dspark serve` subprocess with a
    specific `(prefill_step_size, kv_bits, max_num_seqs)` combination -- the exact same
    flag shape as `start_mlx_server.sh`'s launch branches, deliberately reimplemented
    here rather than imported/shelled-out-to from that script, since this needs direct
    control over relaunching with different flags repeatedly in a tight loop."""

    def __init__(self, mlx_model_hf, host, port, context_window, log_path,
                 prefill_step_size=ma.DEFAULT_PREFILL_STEP_SIZE, kv_bits=None, max_num_seqs=4,
                 apc_num_blocks=ma.MLX_VLM_APC_NUM_BLOCKS, serving=None):
        self.mlx_model_hf = mlx_model_hf
        self.host = host
        self.port = port
        self.context_window = context_window
        self.log_path = log_path
        self.prefill_step_size = prefill_step_size
        self.kv_bits = kv_bits
        self.max_num_seqs = max_num_seqs
        self.apc_num_blocks = apc_num_blocks if apc_num_blocks is not None else ma.MLX_VLM_APC_NUM_BLOCKS
        self.serving = dict(serving) if isinstance(serving, dict) else {}
        self.proc = None
        self.load_time_s = None

    def build_command_args(self):
        backend = self.serving.get("backend")
        if backend == "mlx_dspark":
            mode = self.serving.get("mode") or "dflash"
            args = [
                MLX_DSPARK, "serve",
                "--model", self.mlx_model_hf,
                "--host", self.host,
                "--port", str(self.port),
                "--context-window", str(self.context_window),
                "--mode", str(mode),
            ]
            drafter = self.serving.get("drafter")
            if drafter:
                args += ["--drafter", str(drafter)]
            if self.kv_bits:
                args += ["--kv-bits", str(self.kv_bits)]
            else:
                args += ["--kv-bits", "0"]
                if self.max_num_seqs and self.max_num_seqs > 1:
                    args += ["--max-batch", str(self.max_num_seqs)]
            return args

        args = [
            MLX_PYTHON, "-m", "mlx_vlm.server",
            "--model", self.mlx_model_hf,
            "--host", self.host,
            "--port", str(self.port),
            "--max-kv-size", str(self.context_window),
            "--prefill-step-size", str(self.prefill_step_size),
            "--max-num-seqs", str(self.max_num_seqs),
        ]
        if self.kv_bits:
            args += [
                "--kv-bits", str(self.kv_bits),
                "--kv-group-size", "64",
                "--quantized-kv-start", "0",
                "--kv-quant-scheme", "uniform",
            ]
        draft_model = self.serving.get("draft_model")
        if draft_model:
            args += ["--draft-model", str(draft_model)]
            draft_kind = self.serving.get("draft_kind")
            if draft_kind:
                args += ["--draft-kind", str(draft_kind)]
            draft_block_size = self.serving.get("draft_block_size")
            if draft_block_size is not None and draft_block_size != "":
                args += ["--draft-block-size", str(draft_block_size)]
        return args

    def start(self, timeout_s=600):
        args = self.build_command_args()
        os.makedirs(APC_DISK_PATH, exist_ok=True)
        env = dict(os.environ)
        env["HF_HOME"] = HF_HOME
        env["APC_ENABLED"] = "1"
        env["APC_NUM_BLOCKS"] = str(self.apc_num_blocks)
        env["APC_DISK_PATH"] = APC_DISK_PATH
        env["APC_DISK_MAX_GB"] = "5"
        log(f"Launching: {' '.join(args)}")
        logf = open(self.log_path, "a")
        self.proc = subprocess.Popen(args, stdout=logf, stderr=subprocess.STDOUT, env=env)
        logf.close()
        _ACTIVE_SERVERS.append(self)
        healthy, elapsed = wait_for_health(self.host, self.port, self.proc.pid, timeout_s=timeout_s)
        self.load_time_s = elapsed
        if not healthy:
            log(f"Server failed to become healthy within {timeout_s}s — check {self.log_path}")
            self.stop()
            return False
        log(f"Healthy after {elapsed:.1f}s (prefill_step_size={self.prefill_step_size}, "
            f"kv_bits={self.kv_bits})")
        return True

    def stop(self, timeout_s=20):
        if self in _ACTIVE_SERVERS:
            _ACTIVE_SERVERS.remove(self)
        if self.proc is None:
            return
        try:
            self.proc.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            self.proc = None
            return
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                self.proc = None
                return
            time.sleep(0.5)
        try:
            self.proc.kill()
        except ProcessLookupError:
            pass
        self.proc = None


# --- Memory safety -------------------------------------------------------------------

def memory_guard_ok(model_entry, safety_buffer_gb=ma.SAFETY_BUFFER_GB):
    """Same philosophy as start_mlx_server.sh's own pre-launch admission check --
    refuses to even attempt a launch that's already known not to fit, rather than
    discovering that the hard way mid-sweep. Returns (ok, free_wired_gb, required_gb)."""
    chip_name, _gpu_cores, total_ram_gb = ma.get_chip_info()
    current_wired_gb = ma.get_current_wired_gb()
    free_wired_gb = total_ram_gb - current_wired_gb
    required_gb = ma.effective_size_gb(model_entry) + safety_buffer_gb
    return free_wired_gb >= required_gb, free_wired_gb, required_gb


def concurrency_memory_guard_ok(model_entry, context_tokens, concurrency, kv_bits,
                                 safety_buffer_gb=ma.SAFETY_BUFFER_GB, apc_num_blocks=None):
    """Tier 4's own safety rule (docs/NEXT_SESSION_HANDOFF_PREFILL_SWEEP.md remaining-work
    item #5): reuses compute_kv_formula + mlx_vlm_apc_pool_bytes directly (the exact same
    math compute_bonus_concurrency/recommend_max_context already trust) to check whether
    `concurrency` identically-sized slots at `context_tokens` each plus the shared APC
    pool would ACTUALLY fit free memory before firing them all at once -- the sweep must
    never trigger the emergency memory backstop it exists to help avoid."""
    chip_name, _gpu_cores, total_ram_gb = ma.get_chip_info()
    current_wired_gb = ma.get_current_wired_gb()
    free_wired_gb = total_ram_gb - current_wired_gb
    effective_kv_bits = kv_bits if kv_bits else ma.UNQUANTIZED_KV_BITS
    kv_group_size = (
        ma.DEFAULT_KV_GROUP_SIZE if effective_kv_bits < ma.UNQUANTIZED_KV_BITS else None
    )
    formula = ma.compute_kv_formula(
        model_entry.get("kv_config"), effective_kv_bits, kv_group_size=kv_group_size,
    )
    if formula is None:
        return True  # No kv_config to check against -- can't refuse on data we don't have.
    bytes_per_growing_token, fixed_bytes, _ = formula
    drafter_growing, drafter_fixed = ma.compute_drafter_kv_formula(
        model_entry, kv_bits=effective_kv_bits, kv_group_size=kv_group_size,
    )
    raw_growing = bytes_per_growing_token + drafter_growing
    per_slot_bytes = (fixed_bytes + drafter_fixed) + raw_growing * context_tokens
    apc_pool_bytes = ma.mlx_vlm_apc_pool_bytes(
        model_entry, raw_growing, apc_num_blocks=apc_num_blocks,
    )
    total_kv_bytes = per_slot_bytes * concurrency + apc_pool_bytes
    required_gb = ma.effective_size_gb(model_entry) + safety_buffer_gb + total_kv_bytes / 1024**3
    return free_wired_gb >= required_gb


# --- Sweep writer (incremental, crash-safe) -------------------------------------------

def sweep_path(chip_name, model_id):
    safe_chip = re.sub(r"[^A-Za-z0-9._-]+", "_", chip_name)
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", model_id)
    os.makedirs(SWEEP_DIR, exist_ok=True)
    return os.path.join(SWEEP_DIR, f"{safe_chip}__{safe_model}.sweep.json")


def write_sweep_json(path, data):
    """Atomic write (tmp + os.replace), same pattern as model_advisor.save_calibration
    -- called after every tier completes so a crash mid-sweep never loses everything
    already measured (docs/NEXT_SESSION_HANDOFF_PREFILL_SWEEP.md remaining-work item #3)."""
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, path)


# --- Catalog health (no running server needed) ---------------------------------------

def check_hf_repo_resolves(huggingface_id, timeout=15):
    """Side effect of touching every model: does this entry's HF repo still 404?
    Closes §17 item #14's NVFP4 case (a repo with a manifest but zero resolving weight
    files) early rather than discovering it mid-launch. A plain HEAD-ish GET against
    the HF API's model-info endpoint, no huggingface_hub dependency required."""
    url = f"https://huggingface.co/api/models/{huggingface_id}"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


# --- Tier runners ----------------------------------------------------------------------

def run_measured_requests(host, port, model_hf_id, prompt_fn, context_tokens, max_tokens,
                           warmup=WARMUP_RUNS, measured=MEASURED_RUNS, kv_bits=None,
                           tag=""):
    """Shared by every tier that needs §12's warmup-then-average discipline: sends
    `warmup` discarded requests, then `measured` requests whose timings get averaged.
    Returns (avg_dict, raw_runs) -- avg_dict is None if every measured run errored.

    Each of the warmup+measured iterations gets its own UNIQUE prompt (via
    prompt_fn's nonce param -- see build_code_prompt's docstring for the full story)
    rather than reusing one prompt object across every iteration. Confirmed live this
    session: reusing identical content let the warmup request's own APC cache write
    silently warm every "measured" request that followed it (self-contamination
    within one batch), AND let a PRIOR launch's disk-persisted cache warm this batch's
    very first request too (cross-launch contamination) -- APC_ENABLED=1 is
    unconditional in every SweepServer launch here, deliberately, to stay
    production-representative, so the fix has to be "never repeat exact content", not
    "turn caching off". Averaging now happens across N independently-cold prefills
    instead of N repeats of one cache-warmed prefill -- arguably more honest than the
    original repeat-same-prompt discipline anyway, since real production traffic is
    essentially never byte-identical request to request either."""
    runs = []
    for i in range(warmup + measured):
        prompt = prompt_fn(context_tokens, nonce=uuid.uuid4().hex)
        result = chat_completion(host, port, model_hf_id, prompt, max_tokens=max_tokens)
        if i < warmup:
            continue
        if not result["ok"] or not result.get("timings"):
            log(f"{tag}: run {i} failed/no timings (http={result['http_code']} err={result['error']})")
            continue
        t = result["timings"]
        acc_rate, mean_accept = extract_draft_stats_from_timings(t)
        run_rec = {
            "ttft_s": (t.get("prompt_ms") or 0) / 1000.0,
            "prompt_tok_s": t.get("prompt_per_second"),
            "decode_tok_s": t.get("predicted_per_second"),
            "peak_memory_gb": t.get("peak_memory"),
            "prompt_n": t.get("prompt_n"),
            "cache_n": t.get("cache_n"),
        }
        if isinstance(acc_rate, (int, float)):
            run_rec["draft_acceptance_rate"] = acc_rate
        if isinstance(mean_accept, (int, float)):
            run_rec["mean_accept_len"] = mean_accept
        runs.append(run_rec)
    if not runs:
        return None, runs
    avg = {
        "context_tokens": context_tokens,
        "ttft_s": average_runs(runs, "ttft_s"),
        "prompt_tok_s": average_runs(runs, "prompt_tok_s"),
        "decode_tok_s": average_runs(runs, "decode_tok_s"),
        "peak_memory_gb": average_runs(runs, "peak_memory_gb"),
        "kv_bits": kv_bits,
    }
    avg_acc = average_runs(runs, "draft_acceptance_rate")
    if avg_acc is not None:
        avg["draft_acceptance_rate"] = round(avg_acc, 4)
    avg_accept_len = average_runs(runs, "mean_accept_len")
    if avg_accept_len is not None:
        avg["mean_accept_len"] = round(avg_accept_len, 3)
    return avg, runs


def run_prefill_step_size_tier(model_entry, host, port, step_sizes, sweep_context_tokens,
                                max_num_seqs, log_dir, apc_num_blocks=ma.MLX_VLM_APC_NUM_BLOCKS):
    """Tier 1 headline metric: launches the server once per candidate step size (with
    --kv-bits 8 held constant, matching §12's exact methodology so step size is the
    only variable), sends the warmup+measured discipline at one or more context depths
    (`sweep_context_tokens` may be a single int or an iterable of ints, tested in
    ascending order within the same server launch so multi-depth calibration requires
    zero extra model loads), and records TTFT/prompt tok/s/decode tok/s/peak memory for
    each. Returns (list_of_entries, winning_step_size_at_deepest_depth)."""
    if isinstance(sweep_context_tokens, (list, tuple)):
        depths = sorted(int(d) for d in sweep_context_tokens if int(d) > 0)
    else:
        depths = [int(sweep_context_tokens)]
    max_depth = max(depths)
    entries = []
    for step_size in step_sizes:
        server = SweepServer(
            model_entry["huggingface"], host, port, context_window=max_depth + 4096,
            log_path=os.path.join(log_dir, f"step{step_size}.log"),
            prefill_step_size=step_size, kv_bits=ma.MEMORY_CONSTRAINED_KV_BITS,
            max_num_seqs=max_num_seqs, apc_num_blocks=apc_num_blocks,
            serving=model_entry.get("serving"),
        )
        if not server.start():
            for d in depths:
                entries.append({
                    "step_size": step_size, "context_tokens": d,
                    "ttft_s": None, "error": "launch_failed",
                })
            continue
        try:
            for d in depths:
                avg, _runs = run_measured_requests(
                    host, port, model_entry["huggingface"], build_code_prompt,
                    d, max_tokens=50, kv_bits=ma.MEMORY_CONSTRAINED_KV_BITS,
                    tag=f"step_size={step_size} depth={d}",
                )
                entry = {"step_size": step_size, "context_tokens": d}
                if avg:
                    entry.update({k: avg[k] for k in ("ttft_s", "prompt_tok_s", "decode_tok_s", "peak_memory_gb")})
                else:
                    entry["error"] = "no_successful_measured_runs"
                entries.append(entry)
        finally:
            server.stop()
    deepest_entries = [e for e in entries if e.get("context_tokens") == max_depth] or entries
    return entries, pick_best_step_size(deepest_entries)


def run_decode_vs_context_and_kv_bits_tier(model_entry, host, port, context_depths,
                                            best_step_size, max_num_seqs, log_dir,
                                            apc_num_blocks=ma.MLX_VLM_APC_NUM_BLOCKS):
    """Tier 1's remaining metrics (decode speed vs. context depth curve, real peak
    memory at each depth) PLUS the KV-bits-at-long-context comparison -- two server
    launches at best_step_size (one unquantized, one with --kv-bits 8), each sent
    through every depth in context_depths. Reusing the winning step size here (not
    relaunching per depth) is the "maximize what's measured per model-load" principle
    in action -- decode speed is independently confirmed unaffected by
    --prefill-step-size (§12), so this is a safe reuse, not a confound.

    Also runs the APC cache test (Tier 2) and the tool-calling/multimodal/drafter
    re-verification (Tier 3) against the UNQUANTIZED launch, since those are
    orthogonal to the kv-bits question and this is the natural single long-lived
    instance for them -- returns a dict with every tier's results bundled together.

    IMPORTANT ordering note, confirmed live from mlx_vlm's own source
    (server/generation.py: `peak_memory=mx.get_peak_memory() / 1e9`, with no matching
    `mx.reset_peak_memory()` anywhere in that file): `peak_memory` is MLX's own
    process-lifetime high-water mark, NOT a true per-request figure, reset only by a
    fresh process (a new SweepServer launch). Confirmed as a real artifact this
    session: an earlier ordering ran decode_speed_vs_context's depth loop BEFORE
    multimodal_speed, and multimodal_speed's own peak_memory_gb came back bit-for-bit
    identical to the deepest depth just tested (a tiny image+short-text request has no
    business needing 8GB) -- a stale carried-over reading, not that request's own
    peak. Tool-calling/multimodal now run FIRST in this function, before the depth
    loop inflates the process-wide high-water mark, so their own peak_memory_gb
    readings (where collected) are at least closer to their own true cost. The depth
    loop's OWN peak_memory_gb readings are still only reliable in ASCENDING depth
    order within one process (true here since context_depths is tested in whatever
    order the caller passed, and every default/documented usage sorts ascending) --
    not fixed structurally, just documented."""
    results = {"decode_speed_vs_context": [], "apc_cache_test": None,
               "tool_calling_reverified": None, "multimodal_speed": None,
               "speculative_decoding": None}

    deepest = max(context_depths)
    for kv_bits, label in ((None, "unquantized"), (ma.MEMORY_CONSTRAINED_KV_BITS, "kv8")):
        server = SweepServer(
            model_entry["huggingface"], host, port, context_window=deepest + 4096,
            log_path=os.path.join(log_dir, f"depth_{label}.log"),
            prefill_step_size=best_step_size, kv_bits=kv_bits, max_num_seqs=max_num_seqs,
            apc_num_blocks=apc_num_blocks, serving=model_entry.get("serving"),
        )
        if not server.start():
            for depth in context_depths:
                results["decode_speed_vs_context"].append(
                    {"context_tokens": depth, "kv_bits": kv_bits, "error": "launch_failed"})
            continue
        try:
            # The unquantized launch is this sweep's one long-lived "primary" instance
            # -- run every tier that doesn't care about kv_bits here, so a kv_bits=8
            # relaunch below isn't needed a second time for the same checks. Run
            # BEFORE the depth loop below -- see this function's own docstring for why
            # (process-wide peak_memory high-water mark).
            if kv_bits is None:
                results["tool_calling_reverified"] = run_tool_calling_tier(model_entry, host, port)
                results["multimodal_speed"] = run_multimodal_tier(model_entry, host, port)

            shallowest = min(context_depths)
            for depth in context_depths:
                if depth == shallowest:
                    # §12's own exact short-context methodology (max_tokens=150,
                    # the real story prompt) -- see build_short_decode_prompt's
                    # docstring for why the filler-prompt default would corrupt this
                    # specific point's decode_tok_s via early EOS.
                    prompt_fn, request_max_tokens = build_short_decode_prompt, 150
                else:
                    prompt_fn, request_max_tokens = build_code_prompt, 50
                avg, _runs = run_measured_requests(
                    host, port, model_entry["huggingface"], prompt_fn, depth,
                    max_tokens=request_max_tokens, kv_bits=kv_bits,
                    tag=f"depth={depth} kv_bits={kv_bits}",
                )
                if avg:
                    results["decode_speed_vs_context"].append(avg)
                else:
                    results["decode_speed_vs_context"].append(
                        {"context_tokens": depth, "kv_bits": kv_bits, "error": "no_successful_measured_runs"})

            if kv_bits is None:
                results["speculative_decoding"] = run_drafter_tier(
                    model_entry, results["decode_speed_vs_context"],
                )
                results["apc_cache_test"] = run_apc_tier(
                    model_entry, host, port, prefix_tokens=DEFAULT_APC_PREFIX_TOKENS,
                    kv_bits_quantized=False,
                )
            else:
                # kv_bits_apc_caveat (model_advisor.py's recommend_kv_bits_and_context)
                # flags that --kv-bits + APC + sliding-window layers is a real,
                # upstream-confirmed gap (mlx-vlm#1559) ON THIS EXACT COMBINATION --
                # this is the one place that combination actually gets exercised live.
                if ma._has_sliding_window_layers(model_entry.get("kv_config")):
                    results["apc_cache_test_kv_bits_quantized"] = run_apc_tier(
                        model_entry, host, port, prefix_tokens=DEFAULT_APC_PREFIX_TOKENS,
                        kv_bits_quantized=True,
                    )
        finally:
            server.stop()
    return results


def run_apc_tier(model_entry, host, port, prefix_tokens, kv_bits_quantized):
    """Tier 2: resends the same long prefix twice (§13's own "simulate a coding
    agent's repeated prior-conversation resend" methodology) and records hit rate +
    cold/warm TTFT + exact-output-match -- also the natural place to finally check the
    kv_bits_apc_caveat gap live (mlx-vlm#1559) when kv_bits_quantized is True, instead
    of only flagging it as a caveat string in the picker.

    The prefix gets ONE nonce generated here, reused for BOTH the cold and warm
    requests (not a fresh nonce per request -- that would defeat the entire test,
    since "warm" specifically means "the SAME prefix as the cold request"). What the
    nonce guards against is a DIFFERENT contamination than build_code_prompt's own
    docstring describes: without it, re-running this exact function in a LATER sweep
    (confirmed live this session -- re-running the validation pass twice in one
    afternoon) would silently inherit the FIRST run's disk-persisted cache entry for
    this same unnonced prefix, making even the "cold" request of the second run come
    back already warm (observed: cold_ttft_s dropped from 0.76s on the first run to
    0.026s on an immediate second run against the literal same prefix text).
    reset_cache() alone doesn't prevent this -- confirmed from mlx_vlm's own source
    (apc.py's clear() only touches the in-memory pool, never the disk tier)."""
    reset_cache(host, port)
    prefix = build_code_prompt(prefix_tokens, nonce=uuid.uuid4().hex)
    suffix = "\n\nIn one short sentence, what does the last function above do?"
    cold = chat_completion(host, port, model_entry["huggingface"], prefix + suffix, max_tokens=40)
    if not cold["ok"] or not cold.get("timings"):
        return {"tested": False, "reason": f"cold request failed: {cold['error']}"}
    warm = chat_completion(host, port, model_entry["huggingface"], prefix + suffix, max_tokens=40)
    if not warm["ok"] or not warm.get("timings"):
        return {"tested": False, "reason": f"warm request failed: {warm['error']}"}

    ct, wt = cold["timings"], warm["timings"]
    cache_n = wt.get("cache_n") or 0
    total_prompt = (wt.get("prompt_n") or 0) + cache_n
    hit_rate = (cache_n / total_prompt) if total_prompt else None
    return {
        "tested": True,
        "prefix_tokens": total_prompt,
        "cold_ttft_s": (ct.get("prompt_ms") or 0) / 1000.0,
        "cold_decode_tok_s": ct.get("predicted_per_second"),
        "warm_ttft_s": (wt.get("prompt_ms") or 0) / 1000.0,
        "warm_decode_tok_s": wt.get("predicted_per_second"),
        "hit_rate": hit_rate,
        "output_matches_cold_exactly": cold.get("text") == warm.get("text"),
        "tested_with_kv_bits_quantized": kv_bits_quantized,
    }


# Mirrors the live methodology docs/PROJECT_STATUS.md's "Live-tested tool-calling support"
# section documents (a web_search tool schema, temperature=0, finish_reason check) --
# the exact schema text was never committed verbatim anywhere in this repo, so this is
# a faithful reconstruction of that description, not a copy of a committed fixture.
WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current information.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The search query."}},
            "required": ["query"],
        },
    },
}


def run_tool_calling_tier(model_entry, host, port):
    """Tier 3: re-runs the live tool-call check so models.json's tool_calling.verdict
    (a one-time, point-in-time measurement) gets a fresh, repeatable confirmation
    every sweep instead of going stale silently."""
    result = chat_completion(
        host, port, model_entry["huggingface"],
        "What is the current weather in Tokyo? Use the web_search tool to find out.",
        max_tokens=100, tools=[WEB_SEARCH_TOOL],
    )
    if not result["ok"]:
        return {"verdict": "ERROR", "matches_catalog": False, "error": result["error"]}
    verdict = "SUPPORTED" if result.get("finish_reason") == "tool_calls" else "NOT_SUPPORTED"
    catalog_verdict = (model_entry.get("tool_calling") or {}).get("verdict")
    return {"verdict": verdict, "matches_catalog": verdict == catalog_verdict}


def run_multimodal_tier(model_entry, host, port):
    """Tier 3: if this serving variant retains image input, measure TTFT/decode speed
    WITH an image attached -- correctness was already verified elsewhere (models.json's
    modalities field is itself a correctness claim); speed with an image was, per the
    handoff doc, never measured before this. Nonced for the same cross-sweep-run
    cache-contamination reason as everywhere else in this file (see build_code_prompt's
    docstring) -- this is a TTFT/speed measurement, not a repeat-prefix test."""
    if not is_multimodal_entry(model_entry):
        return {"tested": False, "reason": "model is text-only (or this serving variant drops vision)"}
    result = multimodal_chat_completion(
        host, port, model_entry["huggingface"],
        f"[sweep-nonce:{uuid.uuid4().hex}]\nDescribe this image in one word.", max_tokens=20,
    )
    if not result["ok"] or not result.get("timings"):
        return {"tested": True, "ok": False, "error": result["error"]}
    t = result["timings"]
    return {
        "tested": True, "ok": True,
        "ttft_s": (t.get("prompt_ms") or 0) / 1000.0,
        "decode_tok_s": t.get("predicted_per_second"),
        "peak_memory_gb": t.get("peak_memory"),
    }


def run_drafter_tier(model_entry, decode_speed_vs_context=None):
    """Tier 3: records speculative-decoding metrics across context depths when a drafter
    (`serving.draft_model` on `mlx_vlm` or `serving.drafter` on `mlx_dspark`) is
    attached to `model_entry`. Summarizes the unquantized (`kv_bits is None`)
    `decode_speed_vs_context` runs (`decode_tok_s`, `mean_accept_len`,
    `draft_acceptance_rate`) by context depth."""
    serving = model_entry.get("serving") or {}
    draft_model = serving.get("draft_model") or serving.get("drafter")
    if not draft_model:
        return {"drafter_attached": False}
    by_context = []
    for d in (decode_speed_vs_context or []):
        if d.get("kv_bits") is not None or not isinstance(d.get("decode_tok_s"), (int, float)):
            continue
        point = {
            "context_tokens": d.get("context_tokens"),
            "decode_tok_s": d.get("decode_tok_s"),
        }
        if isinstance(d.get("mean_accept_len"), (int, float)):
            point["mean_accept_len"] = d["mean_accept_len"]
        if isinstance(d.get("draft_acceptance_rate"), (int, float)):
            point["draft_acceptance_rate"] = d["draft_acceptance_rate"]
        by_context.append(point)
    if by_context:
        return {
            "drafter_attached": True,
            "drafter": draft_model,
            "backend": serving.get("backend") or "mlx_vlm",
            "mode": serving.get("mode") or serving.get("draft_kind") or "default",
            "tested_this_sweep": True,
            "by_context": by_context,
        }
    return {
        "drafter_attached": True, "drafter": draft_model,
        "tested_this_sweep": False,
        "reason": "no usable unquantized decode_speed_vs_context runs recorded in this sweep",
    }


def run_concurrency_tier(model_entry, host, port, levels, context_tokens, kv_bits,
                         apc_num_blocks=ma.MLX_VLM_APC_NUM_BLOCKS):
    """Tier 4: fires `concurrency` simultaneous requests at a SHALLOWER context depth
    than the step-size/decode-curve tiers' deepest point (remaining-work item #5),
    guarded by concurrency_memory_guard_ok before ever firing. Aggregate tok/s = total
    completion tokens / wall-clock span; per-request tok/s = the mean of each
    request's own predicted_per_second.

    Each concurrent slot gets its own uniquely-nonced prompt (see
    build_code_prompt's docstring) -- N real users hitting a shared server
    concurrently essentially never send byte-identical content, and letting them
    collide into a single shared APC cache entry here would understate real
    concurrent-load prefill cost, not just measure a cache-adjacent corner case.

    Uses an instruction that invites an actual multi-sentence reply, NOT
    build_filler_prompt's own default ("Reply with just the word done.") --
    confirmed live this session: with that default, the model obliges literally
    (completion_tokens in the single digits), so aggregate_tok_s (completion tokens /
    wall-clock span) ends up dominated by fixed per-request overhead (connection +
    TTFT) rather than reflecting anything about sustained concurrent decode
    throughput -- observed concurrency=1 aggregate_tok_s of ~5 tok/s next to a
    per_request_tok_s (the server's own measured decode rate) of ~57 tok/s on the
    same request, an order of magnitude apart purely from the completion being too
    short to amortize fixed costs over."""
    import concurrent.futures
    results = []
    for concurrency in levels:
        if not concurrency_memory_guard_ok(
            model_entry, context_tokens, concurrency, kv_bits, apc_num_blocks=apc_num_blocks
        ):
            results.append({"concurrency": concurrency, "error": "memory_guard_refused"})
            continue
        prompts = [build_filler_prompt(
            context_tokens, instruction="Write a detailed 150-word paragraph in response.",
            nonce=uuid.uuid4().hex,
        ) for _ in range(concurrency)]
        start = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(chat_completion, host, port, model_entry["huggingface"],
                                    prompts[i], 50) for i in range(concurrency)]
            outs = [f.result() for f in futures]
        elapsed = time.monotonic() - start
        ok_outs = [o for o in outs if o["ok"] and o.get("timings")]
        total_completion = sum((o["usage"] or {}).get("completion_tokens", 0) for o in ok_outs)
        per_request = average_runs(
            [{"r": o["timings"].get("predicted_per_second")} for o in ok_outs], "r")
        results.append({
            "concurrency": concurrency,
            "aggregate_tok_s": (total_completion / elapsed) if elapsed > 0 else None,
            "per_request_tok_s": per_request,
            "successful_requests": len(ok_outs),
        })
    return results


def run_thermal_tier(model_entry, host, port, num_requests):
    """Tier 4: fires num_requests short SEQUENTIAL requests (not concurrent -- thermal
    throttling is about sustained total compute over time, not instantaneous load) and
    tracks decode tok/s across the run. Apple Silicon fanless/low-fan configs can
    throttle under sustained load; this was previously never logged anywhere.

    Deliberately reuses one fixed SHORT_PROMPT across every iteration, unlike the
    other tiers above -- this measures decode_tok_s (predicted_per_second), which is
    the per-TOKEN generation rate AFTER prefill and is unaffected by APC cache
    hits/misses (confirmed in this project's own §12: "decode speed is unaffected by
    --prefill-step-size", the same prefill/decode independence that makes caching the
    prefill phase irrelevant to a decode-rate measurement). Reusing identical content
    is actually DESIRABLE here: it keeps the (already-trivial) prefill cost constant
    across all num_requests iterations, so whatever decode-rate drift shows up is
    attributable to sustained compute/thermal effects, not request-to-request prompt
    variance."""
    rates = []
    for _ in range(num_requests):
        result = chat_completion(host, port, model_entry["huggingface"], SHORT_PROMPT, max_tokens=30)
        if result["ok"] and result.get("timings"):
            rate = result["timings"].get("predicted_per_second")
            if isinstance(rate, (int, float)):
                rates.append(rate)
    if not rates:
        return {"requests_run": 0, "throttling_suspected": None}
    first, last = rates[0], rates[-1]
    max_rate, min_rate = max(rates), min(rates)
    # A real, sustained drop from the run's OWN peak -- not just first-vs-last noise --
    # is the signal; >15% off peak by the end is a documented, generous threshold
    # (judgment call, not a derived physical limit).
    throttling_suspected = (max_rate - last) / max_rate > 0.15 if max_rate > 0 else None
    return {
        "requests_run": len(rates), "first_decode_tok_s": first, "last_decode_tok_s": last,
        "max_decode_tok_s": max_rate, "min_decode_tok_s": min_rate,
        "throttling_suspected": throttling_suspected,
    }


# --- Orchestration & Batch discovery -------------------------------------------------

def load_catalog_models(models_path=None):
    with open(models_path or ma.MODELS_JSON) as f:
        catalog = json.load(f)
    return catalog.get("models") or []


def load_model_entry(model_id):
    models = load_catalog_models()
    return next((m for m in models if m["id"] == model_id), None)


def is_repo_cached(repo_id, hf_home=None):
    """Returns True when `repo_id` has a downloaded snapshot with `config.json`
    under `<hf_home>/hub/models--<org>--<name>/snapshots/*/config.json` (same
    local-cache check as `start_mlx_server.sh`)."""
    if not repo_id or not isinstance(repo_id, str):
        return False
    base = hf_home or HF_HOME
    repo_dir = f"models--{repo_id.replace('/', '--')}"
    pattern = os.path.join(base, "hub", repo_dir, "snapshots", "*", "config.json")
    return bool(glob.glob(pattern))


def is_model_cached(model_entry, hf_home=None):
    """Returns True when the model's primary HuggingFace repo (and its drafter repo,
    if configured in `serving.drafter` or `serving.draft_model`) is cached locally."""
    if not isinstance(model_entry, dict):
        return False
    hf_id = model_entry.get("huggingface")
    if not is_repo_cached(hf_id, hf_home=hf_home):
        return False
    serving = model_entry.get("serving") or {}
    drafter = serving.get("drafter") or serving.get("draft_model")
    if drafter and not is_repo_cached(drafter, hf_home=hf_home):
        return False
    return True


def select_cached_models(catalog_models, chip_name, hf_home=None,
                         untested_only=False, force=False, sweeps_dir=None):
    """Filters `catalog_models` to locally cached entries for batch sweeping.
    - When `untested_only=True`, only includes models with `status != "tested"`.
    - When `force=False`, skips models that already have a `.sweep.json` file for
      `chip_name` in `sweeps_dir` (defaults to `SWEEP_DIR`)."""
    selected = []
    target_sweeps_dir = sweeps_dir or SWEEP_DIR
    safe_chip = re.sub(r"[^A-Za-z0-9._-]+", "_", chip_name)
    for m in catalog_models or []:
        if not isinstance(m, dict) or not m.get("id"):
            continue
        if untested_only and m.get("status") == "tested":
            continue
        if not is_model_cached(m, hf_home=hf_home):
            continue
        if not force:
            safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", m["id"])
            existing = os.path.join(target_sweeps_dir, f"{safe_chip}__{safe_model}.sweep.json")
            if os.path.isfile(existing):
                continue
        selected.append(m)
    return selected


def maybe_sync_verified():
    """Calls `add_model.sync_verified_results(write=True)` to ingest newly produced
    sweep and verification artifacts into `models.json`."""
    import add_model
    updated = add_model.sync_verified_results(write=True)
    if updated:
        log(f"--sync-verified updated {len(updated)} catalog entry/entries in {ma.MODELS_JSON}:")
        for mid, fields, promoted in updated:
            status_tag = " [PROMOTED -> tested]" if promoted else ""
            log(f"  - {mid}: {', '.join(fields)}{status_tag}")
    else:
        log(f"--sync-verified: {ma.MODELS_JSON} is already up to date with local verification/sweep files.")
    return updated


def read_load_time_hint():
    """Reads start_mlx_server.sh's own best-effort load-time file (see that script's
    MLX_LOAD_TIME_FILE) if one exists -- lets the sweep's "load" field reflect a REAL
    production launch's measured time when this sweep is run right after one, rather
    than always reporting null just because this script manages its own separate
    short-lived instances for the actual measurement tiers above."""
    path = os.path.join(BUSHPILOT_HOME, "state", "mlx-load-time.txt")
    try:
        with open(path) as f:
            return float(f.read().strip())
    except (OSError, ValueError):
        return None


def run_sweep(model_id, host=DEFAULT_HOST, port=DEFAULT_PORT,
              step_sizes=None, context_depths=None, concurrency_levels=None,
              thermal_requests=DEFAULT_THERMAL_REQUESTS,
              apc_prefix_tokens=DEFAULT_APC_PREFIX_TOKENS, max_num_seqs=4,
              skip_tiers=None, output_path=None, step_size_depths=None):
    step_sizes = step_sizes or DEFAULT_STEP_SIZES
    context_depths = context_depths or DEFAULT_CONTEXT_DEPTHS
    concurrency_levels = concurrency_levels or DEFAULT_CONCURRENCY_LEVELS
    skip_tiers = set(skip_tiers or [])

    model_entry = load_model_entry(model_id)
    if model_entry is None:
        log(f"ERROR: unknown model id '{model_id}' in {ma.MODELS_JSON}")
        return 2
    serving = model_entry.get("serving") or {}
    backend = serving.get("backend")
    if backend == "mlx_dspark" and "prefill_step_size" not in skip_tiers:
        log(f"Note: '{model_id}' uses backend=mlx_dspark (no --prefill-step-size flag) — "
            "automatically skipping prefill_step_size tier.")
        skip_tiers.add("prefill_step_size")

    chip_name, gpu_cores, total_ram_gb = ma.get_chip_info()
    if chip_name is None or total_ram_gb is None:
        log("ERROR: could not detect chip/RAM via system_profiler.")
        return 2

    ok, free_gb, required_gb = memory_guard_ok(model_entry)
    if not ok:
        log(f"ERROR: refusing to start — need ~{required_gb:.1f}GB free, only "
            f"{free_gb:.1f}GB available. Close other apps/models and retry.")
        return 1

    rec = ma.recommend_kv_bits_and_context(model_entry, free_gb)
    apc_num_blocks = rec.get("apc_num_blocks") or ma.MLX_VLM_APC_NUM_BLOCKS

    out_path = output_path or sweep_path(chip_name, model_id)
    # Model-specific subdirectory -- confirmed live this session: a shared
    # ~/.bushpilot/logs/sweeps/ directory meant two different models' step512.log
    # etc. silently collided/appended into each other's files across separate runs,
    # purely a debugging-log clarity issue (never affected the actual sweep JSON
    # data, which is always written to the correct per-model path above), but
    # confusing enough to fix outright.
    log_dir = os.path.join(BUSHPILOT_HOME, "logs", "sweeps",
                            re.sub(r"[^A-Za-z0-9._-]+", "_", model_id))
    os.makedirs(log_dir, exist_ok=True)

    catalog_health = {"hf_repo_resolves": check_hf_repo_resolves(model_entry["huggingface"])}
    drafter_repo = serving.get("drafter") or serving.get("draft_model")
    if drafter_repo:
        catalog_health["drafter_repo_resolves"] = check_hf_repo_resolves(drafter_repo)

    data = {
        "schema_version": 1,
        "model_id": model_id,
        "chip": chip_name,
        "total_ram_gb": total_ram_gb,
        "swept_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "load": {"load_time_s": read_load_time_hint()},
        "catalog_health": catalog_health,
    }
    write_sweep_json(out_path, data)
    log(f"Sweep file: {out_path}")

    template_patch_res = ma.check_and_patch_chat_template(HF_HOME, model_entry["huggingface"])
    if template_patch_res == "patched":
        log(f"Applied chat-template patch for '{model_entry['huggingface']}' before sweeping.")

    if "prefill_step_size" not in skip_tiers:
        # Must be genuinely deep -- a prompt that doesn't exceed even the SMALLEST
        # candidate step size gets processed in a single chunk regardless of which
        # step size is configured, showing zero effect no matter what (§12's own
        # methodology note: a ~27-31 token prompt never exceeds 512, so that section
        # deliberately reused a ~7,200-token filler prompt instead). Confirmed live
        # this session: an earlier version of this line picked context_depths' own
        # SHALLOWEST point instead of its deepest, silently producing a meaningless
        # "winning step size" result (every candidate single-chunked the ~100-token
        # test prompt identically) -- never caught by any unit test since it only
        # shows up against a real multi-chunk-length prompt. Guarded with a floor of
        # 4x the largest step size being tested, not just max(context_depths) alone,
        # so a user-supplied --context-depths that's all shallow still gets a
        # meaningful multi-chunk test rather than silently repeating that bug.
        if step_size_depths:
            sweep_context_tokens = sorted(
                max(int(d), min(step_sizes) * 2) for d in step_size_depths if int(d) > 0
            )
        else:
            sweep_context_tokens = max(max(context_depths), max(step_sizes) * 4)
        log(f"=== Tier 1a: prefill-step-size sweep {step_sizes} @ {sweep_context_tokens} tokens ===")
        entries, best_step_size = run_prefill_step_size_tier(
            model_entry, host, port, step_sizes, sweep_context_tokens=sweep_context_tokens,
            max_num_seqs=max_num_seqs, log_dir=log_dir, apc_num_blocks=apc_num_blocks,
        )
        data["prefill_step_size_sweep"] = entries
        log(f"Winning step size: {best_step_size}")
        write_sweep_json(out_path, data)
    else:
        best_step_size = ma.DEFAULT_PREFILL_STEP_SIZE

    if "decode_vs_context" not in skip_tiers:
        log(f"=== Tier 1b-3: decode-vs-context + kv-bits-long-context + APC + "
            f"tool-calling + multimodal (depths={context_depths}) ===")
        bundle = run_decode_vs_context_and_kv_bits_tier(
            model_entry, host, port, context_depths, best_step_size, max_num_seqs, log_dir,
            apc_num_blocks=apc_num_blocks,
        )
        data.update(bundle)
        write_sweep_json(out_path, data)

    if "concurrency" not in skip_tiers or "thermal" not in skip_tiers:
        concurrency_context = min(context_depths) if context_depths else 512
        server = SweepServer(
            model_entry["huggingface"], host, port, context_window=concurrency_context + 2048,
            log_path=os.path.join(log_dir, "tier4.log"),
            prefill_step_size=best_step_size, kv_bits=None, max_num_seqs=max(concurrency_levels),
            apc_num_blocks=apc_num_blocks, serving=serving,
        )
        if server.start():
            try:
                if "concurrency" not in skip_tiers:
                    log(f"=== Tier 4a: concurrency throughput {concurrency_levels} "
                        f"@ {concurrency_context} tokens ===")
                    data["concurrency_throughput"] = run_concurrency_tier(
                        model_entry, host, port, concurrency_levels, concurrency_context, kv_bits=None,
                        apc_num_blocks=apc_num_blocks,
                    )
                    write_sweep_json(out_path, data)
                if "thermal" not in skip_tiers:
                    log(f"=== Tier 4b: thermal check ({thermal_requests} sequential requests) ===")
                    data["thermal_check"] = run_thermal_tier(model_entry, host, port, thermal_requests)
                    write_sweep_json(out_path, data)
            finally:
                server.stop()
        else:
            data["concurrency_throughput"] = [{"error": "launch_failed"}]
            data["thermal_check"] = {"error": "launch_failed"}
            write_sweep_json(out_path, data)

    derived_scf = derive_speed_correction_factor(data, model_entry, chip_name, gpu_cores)
    if derived_scf is not None:
        data["derived_speed_correction_factor"] = derived_scf
        write_sweep_json(out_path, data)
        log(f"Derived speed_correction_factor for '{model_id}' on '{chip_name}': {derived_scf}")

    derived = derive_lightweight_calibration_fields(data)
    if derived:
        ma.save_calibration(chip_name, model_id, **derived)
        log(f"Derived lightweight calibration: prefill_step_size={derived['prefill_step_size']} "
            f"prefill_step_size_by_context={derived.get('prefill_step_size_by_context')} "
            f"kv_bits_safe_at_long_context={derived['kv_bits_safe_at_long_context']}")
    else:
        log("Could not derive lightweight calibration fields — neither step-size tier "
            "nor decode_speed_vs_context produced a usable measurement.")

    log(f"Done. Full results: {out_path}")
    return 0


def run_batch_sweep(untested_only=False, force=False, sync_verified=False, **sweep_kwargs):
    chip_name, _gpu_cores, total_ram_gb = ma.get_chip_info()
    if chip_name is None or total_ram_gb is None:
        log("ERROR: could not detect chip/RAM via system_profiler.")
        return 2
    catalog_models = load_catalog_models()
    candidates = select_cached_models(
        catalog_models, chip_name, hf_home=HF_HOME,
        untested_only=untested_only, force=force, sweeps_dir=SWEEP_DIR,
    )
    mode_label = "untested cached" if untested_only else "all cached"
    if not candidates:
        log(f"No {mode_label} models to sweep on '{chip_name}' "
            f"(use --force to re-sweep models that already have a .sweep.json).")
        if sync_verified:
            maybe_sync_verified()
        return 0
    log(f"Batch sweep ({mode_label}): {len(candidates)} model(s) queued on '{chip_name}': "
        f"{', '.join(m['id'] for m in candidates)}")
    failures = 0
    for idx, m in enumerate(candidates, 1):
        mid = m["id"]
        log(f"=== [{idx}/{len(candidates)}] Sweeping '{mid}' ===")
        rc = run_sweep(mid, **sweep_kwargs)
        if rc != 0:
            log(f"WARNING: sweep for '{mid}' exited with status {rc}; continuing batch.")
            failures += 1
    if sync_verified:
        maybe_sync_verified()
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-id", default=None, help="models.json catalog id, not the HF repo")
    parser.add_argument("--all-cached", action="store_true",
                        help="sweep all locally cached catalog models not yet swept on this chip")
    parser.add_argument("--untested-cached", action="store_true",
                        help="sweep locally cached catalog models with status != 'tested' "
                             "not yet swept on this chip")
    parser.add_argument("--force", action="store_true",
                        help="in batch mode, re-sweep models even if a .sweep.json already exists")
    parser.add_argument("--sync-verified", action="store_true",
                        help="run add_model.py --sync-verified after sweeping to update models.json")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--step-sizes", default=",".join(map(str, DEFAULT_STEP_SIZES)))
    parser.add_argument("--step-size-depths", default="",
                        help="optional comma list of context depths to sweep in Tier 1a "
                             "(within one server launch per step size)")
    parser.add_argument("--context-depths", default=",".join(map(str, DEFAULT_CONTEXT_DEPTHS)))
    parser.add_argument("--concurrency-levels", default=",".join(map(str, DEFAULT_CONCURRENCY_LEVELS)))
    parser.add_argument("--thermal-requests", type=int, default=DEFAULT_THERMAL_REQUESTS)
    parser.add_argument("--apc-prefix-tokens", type=int, default=DEFAULT_APC_PREFIX_TOKENS)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--skip-tiers", default="",
                         help="comma list from: prefill_step_size,decode_vs_context,concurrency,thermal")
    parser.add_argument("--output", default=None, help="override the sweep JSON output path")
    args = parser.parse_args()

    if not args.model_id and not args.all_cached and not args.untested_cached:
        parser.error("one of --model-id, --all-cached, or --untested-cached is required")

    sweep_kwargs = {
        "host": args.host,
        "port": args.port,
        "step_sizes": [int(x) for x in args.step_sizes.split(",") if x],
        "context_depths": [int(x) for x in args.context_depths.split(",") if x],
        "concurrency_levels": [int(x) for x in args.concurrency_levels.split(",") if x],
        "thermal_requests": args.thermal_requests,
        "apc_prefix_tokens": args.apc_prefix_tokens,
        "max_num_seqs": args.max_num_seqs,
        "skip_tiers": [t for t in args.skip_tiers.split(",") if t],
        "output_path": args.output,
        "step_size_depths": [int(x) for x in args.step_size_depths.split(",") if x] or None,
    }

    if args.all_cached or args.untested_cached:
        return run_batch_sweep(
            untested_only=bool(args.untested_cached and not args.all_cached),
            force=args.force,
            sync_verified=args.sync_verified,
            **sweep_kwargs,
        )

    rc = run_sweep(args.model_id, **sweep_kwargs)
    if rc == 0 and args.sync_verified:
        maybe_sync_verified()
    return rc


if __name__ == "__main__":
    sys.exit(main())
