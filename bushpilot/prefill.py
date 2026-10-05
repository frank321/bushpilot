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


def recommend_prefill_step_size(model_entry, chip_name, free_wired_gb,
                                 safety_buffer_gb=SAFETY_BUFFER_GB):
    """Returns (step_size, reason_string) -- mirrors recommend_kv_bits_and_context's
    try-the-better-option-then-fall-back-to-the-proven-safe-default structure, but for
    --prefill-step-size instead of --kv-bits.

    §12's own finding (docs/HARDWARE_SPEED_ESTIMATION.md) is that `--prefill-step-size`
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
            "see docs/NEXT_SESSION_HANDOFF_PREFILL_SWEEP.md to run the sweep)"
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
