"""Per-(chip, model) real-world measurement cache under ~/.bushpilot/calibration --
a cache that's always safe to ignore, never a required input (see
load_calibration's docstring)."""
import json
import os
import re
from datetime import datetime, timezone

# Per-machine learned measurements, one JSON file per (chip, model id) pair -- see
# save_calibration/load_calibration below. Mirrors ~/.bushpilot's other state (the bash
# side's BUSHPILOT_HOME) without model_advisor.py needing to import or shell out to it.
CALIBRATION_DIR = os.path.expanduser("~/.bushpilot/calibration")


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
    docs/HARDWARE_SPEED_ESTIMATION.md §17 item #2). Called by start_mlx_server.sh at
    teardown with whatever that session's own /metrics last reported.

    prefill_step_size/kv_bits_safe_at_long_context (both optional, added for the
    per-(chip, model) sweep harness -- see docs/NEXT_SESSION_HANDOFF_PREFILL_SWEEP.md):
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
