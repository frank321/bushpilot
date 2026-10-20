"""Token-generation speed estimation: a memory-bandwidth-bound physics formula,
overridden by a real per-(chip, model) calibration measurement when one exists
(see load_calibration)."""
from .calibration import load_calibration

# Sourced in docs/HARDWARE_SPEED_ESTIMATION.md §4 — cross-checked across two independent
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
# Derived live in docs/HARDWARE_SPEED_ESTIMATION.md §6 from 3 real measurements (0.86, 0.97,
# 0.88) plus one independent published figure (0.88) — 0.87 sits near the low end of
# that cluster deliberately, so the estimate errs toward underselling speed rather than
# overpromising it.
SPEED_CORRECTION_FACTOR = 0.87

# Architecture-family correction factors derived from live calibration sweeps on Apple
# Silicon (docs/HARDWARE_SPEED_ESTIMATION.md §6, §12, §17 item #2, §20). Used when a
# catalog entry does not specify its own per-model "speed_correction_factor" (for
# example, drafter variants or sibling quantizations of a measured architecture family),
# before falling back to the global SPEED_CORRECTION_FACTOR (0.87) for unmeasured
# architecture families.
ARCHITECTURE_SPEED_CORRECTION_FACTORS = {
    'gemma4 (elastic/MatFormer "E4B" variant)': 1.281,
    "gemma4_unified": 0.756,
    "qwen3_5": 0.87,
}


def resolve_speed_correction_factor(model_entry):
    """Returns the speed correction factor for `model_entry` in priority order:
    1. Explicit per-model `model_entry["speed_correction_factor"]`
    2. Measured architecture-family factor from `ARCHITECTURE_SPEED_CORRECTION_FACTORS`
    3. Global `SPEED_CORRECTION_FACTOR` (0.87) fallback."""
    explicit = model_entry.get("speed_correction_factor")
    if isinstance(explicit, (int, float)) and not isinstance(explicit, bool) and explicit > 0:
        return float(explicit)
    arch = model_entry.get("architecture")
    if isinstance(arch, str) and arch in ARCHITECTURE_SPEED_CORRECTION_FACTORS:
        return ARCHITECTURE_SPEED_CORRECTION_FACTORS[arch]
    return SPEED_CORRECTION_FACTOR


def resolve_calibrated_decode_speed(calibrated, context_tokens=None):
    """Selects `(decode_tok_s, matched_context_tokens)` from a calibration dict.
    When `context_tokens` is given and `calibrated["decode_tok_s_by_context"]` has one
    or more valid depth entries, picks the depth closest to `context_tokens` (ties
    broken toward the deeper depth, matching `recommend_prefill_step_size`). Otherwise
    falls back to the flat top-level `(calibrated["decode_tok_s"], calibrated.get("context_tokens"))`."""
    if not isinstance(calibrated, dict):
        return None, None
    by_ctx = calibrated.get("decode_tok_s_by_context")
    if (
        isinstance(by_ctx, dict)
        and by_ctx
        and isinstance(context_tokens, (int, float))
        and not isinstance(context_tokens, bool)
        and context_tokens > 0
    ):
        parsed = {}
        for k, v in by_ctx.items():
            try:
                depth = int(k)
            except (TypeError, ValueError):
                continue
            if depth > 0 and isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
                parsed[depth] = float(v)
        if parsed:
            target = int(context_tokens)
            chosen_depth = min(parsed.keys(), key=lambda d: (abs(d - target), -d))
            return parsed[chosen_depth], chosen_depth
    return calibrated.get("decode_tok_s"), calibrated.get("context_tokens")


def estimate_speed_tok_s(model_entry, chip_name, gpu_cores, context_tokens=None):
    """Returns (estimate_or_None, reason_string). Deliberately returns None rather than
    a guess when we don't have real active-weight data — see gemma-4-26b-a4b-it-4bit's
    entry if its active_weight_gb were ever removed.

    Checks load_calibration FIRST: a real measurement from a previous run of this
    EXACT model on this EXACT chip always wins over the physics formula below, since
    it's not an estimate at all -- see load_calibration's docstring. When `context_tokens`
    is provided and the calibration file carries `decode_tok_s_by_context`, selects the
    measured decode rate at the nearest swept context depth. Falls through to the
    formula untouched when nothing's been measured yet.

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
    measured 0.75 (docs/HARDWARE_SPEED_ESTIMATION.md §12/§17 item #2) -- a 4x-wider spread,
    in opposite directions, than one global number can represent. Leave unset in
    models.json until a model has its own live calibration point; the global constant
    remains the right default for anything not yet measured."""
    serving = model_entry.get("serving") or {}
    expected_drafter = serving.get("drafter") or serving.get("draft_model") or ""
    calibrated = load_calibration(chip_name, model_entry.get("id", ""))
    if calibrated is not None:
        recorded_drafter = calibrated.get("drafter")
        if recorded_drafter is None or recorded_drafter == expected_drafter:
            cal_tok_s, cal_ctx = resolve_calibrated_decode_speed(calibrated, context_tokens=context_tokens)
            if isinstance(cal_tok_s, (int, float)) and cal_tok_s > 0:
                ctx_note = (
                    f" at ~{int(cal_ctx):,} tokens of context"
                    if isinstance(cal_ctx, (int, float)) else ""
                )
                return round(cal_tok_s, 1), (
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

    factor = resolve_speed_correction_factor(model_entry)
    base = factor * bandwidth / active_gb

    speedup = serving.get("measured_speedup")
    if speedup:
        measured_on = serving.get("measured_on", "reference hardware")
        return round(base * speedup, 1), (
            f"estimated ({speedup}x speculative-decoding multiplier measured on "
            f"{measured_on}, applied to this chip's own bandwidth estimate — "
            "workload- and chip-generation-dependent, treat as directional)"
        )

    return round(base, 1), "estimated"
