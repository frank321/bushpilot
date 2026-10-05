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
    measured 0.75 (docs/HARDWARE_SPEED_ESTIMATION.md §12/§17 item #2) -- a 4x-wider spread,
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
