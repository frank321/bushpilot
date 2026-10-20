"""--prefill-step-size recommendation -- time-to-first-token tuning, independent of
the --kv-bits/context decision in kv.py. See recommend_prefill_step_size's own
docstring for the memory-safety check this applies to a calibrated value."""
from .calibration import load_calibration
from .kv import SAFETY_BUFFER_GB

# Mirrors start_mlx_server.sh's own PREFILL_STEP_SIZE default -- two separate copies of the
# same number deliberately, not a shared constant imported across the bash/python boundary:
# bash owns the actual launch flag, this is only the fallback recommend_prefill_step_size
# below returns when no (chip, model) calibration exists yet. Derived from exactly 2 Gemma-4
# models on this one M5 Pro machine (docs/HARDWARE_SPEED_ESTIMATION.md §12) — not assumed to
# generalize, which is the whole reason recommend_prefill_step_size/the sweep harness exist
# (see docs/NEXT_SESSION_HANDOFF_PREFILL_SWEEP.md).
DEFAULT_PREFILL_STEP_SIZE = 2048


# A larger --prefill-step-size's measured memory cost (docs/HARDWARE_SPEED_ESTIMATION.md §12) is
# real but comparatively small relative to the model's own footprint, not a multiplier of
# it: going 2048->4096 cost +3.4% of that run's own peak memory for gemma-4-e4b-it-4bit and
# +11.8% for gemma-4-12b-it-4bit, the two data points that exist. Modeled conservatively here
# as a flat 15% margin ABOVE the calibration's own observed peak_memory_gb -- comfortably
# above both observed deltas -- applied ONLY when recommending a step size other than the
# proven-safe default (the default itself needs no extra margin: it's what the calibration's
# own peak_memory_gb was actually measured running at). This is a documented approximation,
# not a re-derivation of §12's relationship for every chip/model -- the sweep harness
# (docs/NEXT_SESSION_HANDOFF_PREFILL_SWEEP.md) is what would eventually replace this with a real
# per-model memory curve if the margin ever proves too tight or too loose in practice.
PREFILL_STEP_SIZE_MEMORY_MARGIN = 1.15


def _select_depth_calibrated_step_size(calibrated, context_tokens):
    """If calibrated contains a valid prefill_step_size_by_context dict and
    context_tokens is a positive number, returns (step_size, peak_mem, matched_depth)
    for the swept context depth closest to context_tokens (ties broken toward the
    deeper depth so long-context memory/TTFT behavior is preferred at the midpoint).
    Returns None if no depth-keyed entry is usable."""
    if not isinstance(context_tokens, (int, float)) or context_tokens <= 0:
        return None
    by_ctx = calibrated.get("prefill_step_size_by_context")
    if not isinstance(by_ctx, dict) or not by_ctx:
        return None
    candidates = {}
    for raw_depth, entry in by_ctx.items():
        try:
            depth = int(raw_depth)
        except (TypeError, ValueError):
            continue
        if depth <= 0:
            continue
        if isinstance(entry, dict):
            step = entry.get("step_size")
            peak = entry.get("peak_memory_gb", calibrated.get("peak_memory_gb"))
        elif isinstance(entry, int) and not isinstance(entry, bool):
            step = entry
            peak = calibrated.get("peak_memory_gb")
        else:
            continue
        if isinstance(step, int) and not isinstance(step, bool) and step > 0:
            candidates[depth] = (step, peak)
    if not candidates:
        return None
    matched_depth = min(candidates.keys(), key=lambda d: (abs(d - context_tokens), -d))
    step_size, peak_mem = candidates[matched_depth]
    return step_size, peak_mem, matched_depth


def recommend_prefill_step_size(model_entry, chip_name, free_wired_gb,
                                 safety_buffer_gb=SAFETY_BUFFER_GB,
                                 context_tokens=None):
    """Returns (step_size, reason_string) -- mirrors recommend_kv_bits_and_context's
    try-the-better-option-then-fall-back-to-the-proven-safe-default structure, but for
    --prefill-step-size instead of --kv-bits.

    §12's own finding (docs/HARDWARE_SPEED_ESTIMATION.md) is that `--prefill-step-size`
    affects ONLY time-to-first-token, never decode speed, and that the currently
    hardcoded global default (DEFAULT_PREFILL_STEP_SIZE, 2048) already beat both 512
    and 4096 on both models tested at ~7,200 tokens -- while §20 showed that on
    gemma-4-12b-it-4bit at 16,384 tokens, 512 beat 2048 by ~16% (20.13s vs 23.97s).
    When `context_tokens` (e.g. recommended_context) is provided and the calibration
    file carries `prefill_step_size_by_context`, this selects the calibrated step size
    for the closest swept context depth, falling back to the flat `prefill_step_size`
    field when no depth map is present.

    Memory safety: a non-default calibrated step size is only ever recommended when
    free_wired_gb comfortably covers the calibration's own observed peak_memory_gb
    plus PREFILL_STEP_SIZE_MEMORY_MARGIN's margin plus the usual safety buffer -- see
    that constant's own comment for where the margin number comes from. A calibration
    with no recorded peak_memory_gb (e.g. hand-written, or from a version of
    save_calibration that predates that field) can't be safety-checked at all, so it
    falls back to the default rather than risk using an unverified value blindly."""
    calibrated = load_calibration(chip_name, model_entry.get("id", ""))
    if calibrated is None:
        return DEFAULT_PREFILL_STEP_SIZE, (
            "default (no calibrated prefill_step_size for this chip+model yet — "
            "see docs/NEXT_SESSION_HANDOFF_PREFILL_SWEEP.md to run the sweep)"
        )

    depth_match = _select_depth_calibrated_step_size(calibrated, context_tokens)
    if depth_match is not None:
        step_size, peak_mem, matched_depth = depth_match
        depth_note = f" at {matched_depth:,} tokens"
    else:
        raw_step = calibrated.get("prefill_step_size")
        if not isinstance(raw_step, int) or isinstance(raw_step, bool) or raw_step <= 0:
            return DEFAULT_PREFILL_STEP_SIZE, (
                "default (no calibrated prefill_step_size for this chip+model yet — "
                "see docs/NEXT_SESSION_HANDOFF_PREFILL_SWEEP.md to run the sweep)"
            )
        step_size = raw_step
        peak_mem = calibrated.get("peak_memory_gb")
        depth_note = ""

    saved_at = calibrated.get("saved_at", "previously")
    if step_size == DEFAULT_PREFILL_STEP_SIZE:
        return step_size, (
            f"calibrated — matches the global default, measured on this machine "
            f"({chip_name}){depth_note}, saved {saved_at}"
        )

    if not isinstance(peak_mem, (int, float)) or peak_mem <= 0:
        return DEFAULT_PREFILL_STEP_SIZE, (
            f"default (calibrated step size {step_size}{depth_note} has no recorded peak_memory_gb "
            "to safety-check against — falling back rather than risk an unverified value)"
        )

    margin = PREFILL_STEP_SIZE_MEMORY_MARGIN if step_size > DEFAULT_PREFILL_STEP_SIZE else 1.0
    required_gb = peak_mem * margin + safety_buffer_gb
    if free_wired_gb < required_gb:
        return DEFAULT_PREFILL_STEP_SIZE, (
            f"default (calibrated step size {step_size}{depth_note} needs ~{required_gb:.1f}GB free "
            f"with margin; only {free_wired_gb:.1f}GB currently available)"
        )

    return step_size, (
        f"measured on this machine ({chip_name}){depth_note}, saved {saved_at} — overrides the "
        f"global default ({DEFAULT_PREFILL_STEP_SIZE}) until this model or chip changes "
        "(re-run the sweep to refresh)"
    )
