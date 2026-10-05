"""Tests for sweep_model.py's pure-logic pieces -- prompt builders, averaging,
step-size/kv-bits derivation, and the memory-safety guards (chip/RAM detection
monkeypatched, exactly like test_recommendation_math.py does for model_advisor.py).

Deliberately NOT tested here: SweepServer's real subprocess launch/teardown, the
chat_completion/_http_json HTTP layer, or run_sweep's live orchestration -- those need
a real mlx_vlm.server instance and are exercised by actually running the harness (see
docs/NEXT_SESSION_HANDOFF_PREFILL_SWEEP.md), not by this pure-logic unit suite.
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pytest

import model_advisor as ma
import sweep_model as sm
from bushpilot import calibration as _calibration


# --- prompt builders -----------------------------------------------------------------

def test_build_filler_prompt_scales_with_target_tokens():
    short = sm.build_filler_prompt(50)
    long = sm.build_filler_prompt(500)
    assert len(long) > len(short)
    assert "Reply with just the word done." in short


def test_build_code_prompt_uses_a_real_source_file_by_default():
    prompt = sm.build_code_prompt(200)
    assert "```python" in prompt
    assert "model_advisor.py" in prompt or "def " in prompt  # real code content, not filler


def test_build_code_prompt_falls_back_to_filler_when_file_missing():
    prompt = sm.build_code_prompt(50, repo_file="/no/such/file.py")
    # Falls back to the filler-prompt shape (repeated sentences), but keeps the
    # code-prompt's own instruction text, not the filler builder's default one.
    assert "summarize what this code does" in prompt
    assert "weather today is mild" in prompt


def test_build_short_decode_prompt_is_the_s12_story_prompt():
    prompt = sm.build_short_decode_prompt()
    assert prompt == sm.SHORT_PROMPT


def test_build_short_decode_prompt_ignores_target_tokens():
    # Accepts-but-ignores target_tokens -- same call signature as the other two
    # builders for run_measured_requests' prompt_fn slot, but this prompt is always
    # §12's one short baseline, never scaled.
    assert sm.build_short_decode_prompt(100) == sm.build_short_decode_prompt(99999)


def test_build_short_decode_prompt_nonce_goes_first():
    prompt = sm.build_short_decode_prompt(nonce="abc123")
    assert prompt.startswith("[sweep-nonce:abc123]")
    assert sm.SHORT_PROMPT in prompt


def test_build_code_prompt_roughly_hits_target_length():
    prompt = sm.build_code_prompt(1000)
    # Rough chars/4 heuristic -- not exact, just in the right ballpark (allow 2x slack
    # since the instruction/code-fence wrapper adds a bit of overhead on top).
    assert sm.count_tokens_rough(prompt) >= 1000


# --- average_runs ----------------------------------------------------------------------

def test_average_runs_computes_plain_mean():
    runs = [{"x": 10.0}, {"x": 20.0}, {"x": 30.0}]
    assert sm.average_runs(runs, "x") == 20.0


def test_average_runs_skips_non_numeric_entries():
    runs = [{"x": 10.0}, {"x": None}, {"x": "error"}, {"x": 30.0}]
    assert sm.average_runs(runs, "x") == 20.0


def test_average_runs_returns_none_when_nothing_usable():
    assert sm.average_runs([{"x": None}, {"x": "error"}], "x") is None
    assert sm.average_runs([], "x") is None


# --- pick_best_step_size ---------------------------------------------------------------

def test_pick_best_step_size_picks_lowest_ttft():
    entries = [
        {"step_size": 512, "ttft_s": 1.716},
        {"step_size": 2048, "ttft_s": 1.647},
        {"step_size": 4096, "ttft_s": 1.823},
    ]
    assert sm.pick_best_step_size(entries) == 2048


def test_pick_best_step_size_ignores_entries_with_no_ttft():
    entries = [
        {"step_size": 512, "error": "launch_failed"},
        {"step_size": 2048, "ttft_s": 1.647},
    ]
    assert sm.pick_best_step_size(entries) == 2048


def test_pick_best_step_size_falls_back_to_default_when_nothing_usable():
    entries = [{"step_size": 512, "error": "launch_failed"}]
    assert sm.pick_best_step_size(entries) == ma.DEFAULT_PREFILL_STEP_SIZE


# --- kv_bits_safe_at_long_context -------------------------------------------------------

def test_kv_bits_safe_at_long_context_true_when_slowdown_within_threshold():
    curve = [
        {"context_tokens": 16384, "kv_bits": None, "decode_tok_s": 68.0},
        {"context_tokens": 16384, "kv_bits": 8, "decode_tok_s": 65.0},  # ~4.4% slowdown
    ]
    assert sm.kv_bits_safe_at_long_context(curve, 16384) is True


def test_kv_bits_safe_at_long_context_false_when_slowdown_exceeds_threshold():
    curve = [
        {"context_tokens": 16384, "kv_bits": None, "decode_tok_s": 68.0},
        {"context_tokens": 16384, "kv_bits": 8, "decode_tok_s": 50.0},  # ~26% slowdown
    ]
    assert sm.kv_bits_safe_at_long_context(curve, 16384) is False


def test_kv_bits_safe_at_long_context_none_when_comparison_missing():
    curve = [{"context_tokens": 16384, "kv_bits": None, "decode_tok_s": 68.0}]
    assert sm.kv_bits_safe_at_long_context(curve, 16384) is None
    assert sm.kv_bits_safe_at_long_context([], 16384) is None


# --- derive_lightweight_calibration_fields ----------------------------------------------

def test_derive_lightweight_calibration_fields_happy_path():
    sweep_data = {
        "prefill_step_size_sweep": [
            {"step_size": 512, "ttft_s": 1.716, "decode_tok_s": 72.29, "context_tokens": 7200, "peak_memory_gb": 6.25},
            {"step_size": 2048, "ttft_s": 1.647, "decode_tok_s": 71.60, "context_tokens": 7200, "peak_memory_gb": 6.49},
            {"step_size": 4096, "ttft_s": 1.823, "decode_tok_s": 72.08, "context_tokens": 7200, "peak_memory_gb": 6.71},
        ],
        "decode_speed_vs_context": [
            {"context_tokens": 16384, "kv_bits": None, "decode_tok_s": 68.0},
            {"context_tokens": 16384, "kv_bits": 8, "decode_tok_s": 65.0},
        ],
    }
    derived = sm.derive_lightweight_calibration_fields(sweep_data)
    assert derived["prefill_step_size"] == 2048  # lowest TTFT
    assert derived["decode_tok_s"] == 71.60
    assert derived["context_tokens"] == 7200
    assert derived["peak_memory_gb"] == 6.49
    assert derived["kv_bits_safe_at_long_context"] is True


def test_derive_lightweight_calibration_fields_none_when_step_size_tier_empty():
    assert sm.derive_lightweight_calibration_fields({}) is None
    assert sm.derive_lightweight_calibration_fields({"prefill_step_size_sweep": []}) is None


def test_derive_lightweight_calibration_fields_none_when_winner_has_no_decode_speed():
    sweep_data = {"prefill_step_size_sweep": [{"step_size": 2048, "error": "launch_failed"}]}
    assert sm.derive_lightweight_calibration_fields(sweep_data) is None


def test_derive_lightweight_calibration_fields_kv_bits_safe_none_without_long_context_data():
    sweep_data = {
        "prefill_step_size_sweep": [
            {"step_size": 2048, "ttft_s": 1.6, "decode_tok_s": 71.6, "context_tokens": 4096, "peak_memory_gb": 6.49},
        ],
    }
    derived = sm.derive_lightweight_calibration_fields(sweep_data)
    assert derived["kv_bits_safe_at_long_context"] is None


# --- is_multimodal_entry ---------------------------------------------------------------

def test_is_multimodal_entry_true_for_vision_capable_model(sliding_window_model):
    assert sm.is_multimodal_entry(sliding_window_model) is True  # has "image" modality


def test_is_multimodal_entry_false_for_text_only_model(dense_model):
    dense_model["modalities"] = ["text"]
    assert sm.is_multimodal_entry(dense_model) is False


def test_is_multimodal_entry_false_when_serving_narrows_it(sliding_window_model):
    sliding_window_model["serving"] = {"backend": "mlx_dspark", "multimodal_capable": False}
    assert sm.is_multimodal_entry(sliding_window_model) is False


# --- memory guards (chip/RAM detection monkeypatched) -----------------------------------

def test_memory_guard_ok_true_when_plenty_of_room(dense_model, monkeypatch):
    monkeypatch.setattr(ma, "get_chip_info", lambda: ("Apple M5 Pro", 20, 48.0))
    monkeypatch.setattr(ma, "get_current_wired_gb", lambda: 10.0)  # 38GB free
    ok, free_gb, required_gb = sm.memory_guard_ok(dense_model)
    assert ok is True
    assert free_gb == 38.0


def test_memory_guard_ok_false_when_insufficient_room(dense_model, monkeypatch):
    monkeypatch.setattr(ma, "get_chip_info", lambda: ("Apple M5 Pro", 20, 8.0))
    monkeypatch.setattr(ma, "get_current_wired_gb", lambda: 6.0)  # 2GB free, model needs ~9.3
    ok, free_gb, required_gb = sm.memory_guard_ok(dense_model)
    assert ok is False
    assert required_gb > free_gb


def test_concurrency_memory_guard_ok_true_with_plenty_of_room(dense_model, monkeypatch):
    monkeypatch.setattr(ma, "get_chip_info", lambda: ("Apple M5 Pro", 20, 48.0))
    monkeypatch.setattr(ma, "get_current_wired_gb", lambda: 10.0)
    assert sm.concurrency_memory_guard_ok(dense_model, context_tokens=2048, concurrency=4,
                                           kv_bits=None) is True


def test_concurrency_memory_guard_ok_false_when_tight(dense_model, monkeypatch):
    monkeypatch.setattr(ma, "get_chip_info", lambda: ("Apple M5 Pro", 20, 10.0))
    monkeypatch.setattr(ma, "get_current_wired_gb", lambda: 2.0)  # 8GB free
    assert sm.concurrency_memory_guard_ok(dense_model, context_tokens=200000, concurrency=4,
                                           kv_bits=None) is False


def test_concurrency_memory_guard_ok_true_when_no_kv_config():
    model_entry = {"id": "no-kv-config", "size_gb": 1.0}
    assert sm.concurrency_memory_guard_ok(model_entry, context_tokens=2048, concurrency=4,
                                           kv_bits=None) is True


# --- sweep_path / write_sweep_json ------------------------------------------------------

def test_sweep_path_sanitizes_and_lands_in_sweeps_subdir(tmp_path, monkeypatch):
    monkeypatch.setattr(_calibration, "CALIBRATION_DIR", str(tmp_path))
    monkeypatch.setattr(sm, "SWEEP_DIR", os.path.join(str(tmp_path), "sweeps"))
    path = sm.sweep_path("Apple M5 Pro", "gemma-4-e4b-it-4bit")
    assert " " not in os.path.basename(path)
    assert path.endswith(".sweep.json")
    assert os.path.dirname(path) == os.path.join(str(tmp_path), "sweeps")


def test_write_sweep_json_round_trips(tmp_path):
    path = os.path.join(str(tmp_path), "test.sweep.json")
    data = {"model_id": "test", "prefill_step_size_sweep": [{"step_size": 2048}]}
    sm.write_sweep_json(path, data)
    import json
    with open(path) as f:
        loaded = json.load(f)
    assert loaded == data


def test_write_sweep_json_is_atomic_no_leftover_tmp_file(tmp_path):
    path = os.path.join(str(tmp_path), "test.sweep.json")
    sm.write_sweep_json(path, {"a": 1})
    assert not os.path.exists(f"{path}.tmp")
    assert os.path.exists(path)
