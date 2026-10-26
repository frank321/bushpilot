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

import model_advisor as ma  # noqa: E402 -- must follow the sys.path.insert above
import sweep_model as sm  # noqa: E402
from bushpilot import calibration as _calibration  # noqa: E402

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
    assert derived["decode_tok_s_by_context"] == {"16384": 68.0}


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


def test_concurrency_memory_guard_ok_true_when_no_kv_config(monkeypatch):
    # chip/RAM detection must be monkeypatched here like every sibling test in this
    # section: the real get_chip_info() shells out to `system_profiler`, which is
    # macOS-only and raises FileNotFoundError on the Linux CI runner. The assertion
    # below is unaffected either way -- the guard returns True at the "no kv_config"
    # early exit before free memory is consulted -- but the call must not reach the
    # host binary for this suite to stay pure-logic (see AGENTS.md).
    monkeypatch.setattr(ma, "get_chip_info", lambda: ("Apple M5 Pro", 20, 48.0))
    monkeypatch.setattr(ma, "get_current_wired_gb", lambda: 10.0)
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


def test_concurrency_memory_guard_ok_accounts_for_apc_pool(dense_model, monkeypatch):
    monkeypatch.setattr(ma, "get_chip_info", lambda: ("Apple M5 Pro", 20, 16.0))
    # 6.3 GB weights + 3.0 GB safety buffer = 9.3 GB base.
    # With 6.45 GB wired -> 9.55 GB free (0.25 GB headroom above base).
    # Without APC pool (apc_num_blocks=0), 1 slot @ 2048 tokens (~0.008 GB) fits easily.
    # With full 4096-block APC pool (~0.25 GB) + 1 slot @ 2048 tokens (~0.008 GB) = ~0.258 GB > 0.25 GB -> refused!
    monkeypatch.setattr(ma, "get_current_wired_gb", lambda: 6.45)
    assert sm.concurrency_memory_guard_ok(
        dense_model, context_tokens=2048, concurrency=1, kv_bits=None, apc_num_blocks=0
    ) is True
    assert sm.concurrency_memory_guard_ok(
        dense_model, context_tokens=2048, concurrency=1, kv_bits=None,
        apc_num_blocks=ma.MLX_VLM_APC_NUM_BLOCKS,
    ) is False


def test_sweep_server_stores_custom_apc_num_blocks():
    srv = sm.SweepServer(
        "mlx-community/test", "127.0.0.1", 38472, 16384, "/tmp/test.log",
        prefill_step_size=2048, kv_bits=8, max_num_seqs=1, apc_num_blocks=1024,
    )
    assert srv.apc_num_blocks == 1024


def test_derive_lightweight_calibration_fields_multi_depth_populates_by_context():
    sweep_data = {
        "prefill_step_size_sweep": [
            {"step_size": 512, "ttft_s": 0.95, "decode_tok_s": 70.0, "context_tokens": 4096, "peak_memory_gb": 6.8},
            {"step_size": 2048, "ttft_s": 0.80, "decode_tok_s": 71.0, "context_tokens": 4096, "peak_memory_gb": 7.0},
            {"step_size": 512, "ttft_s": 3.55, "decode_tok_s": 69.2, "context_tokens": 16384, "peak_memory_gb": 7.84},
            {"step_size": 2048, "ttft_s": 3.85, "decode_tok_s": 68.5, "context_tokens": 16384, "peak_memory_gb": 8.43},
        ],
    }
    derived = sm.derive_lightweight_calibration_fields(sweep_data)
    # Deepest depth (16384) wins for the flat top-level prefill_step_size
    assert derived["prefill_step_size"] == 512
    assert derived["context_tokens"] == 16384
    assert derived["decode_tok_s"] == 69.2
    # Both depths are recorded in prefill_step_size_by_context with their respective winners
    by_ctx = derived["prefill_step_size_by_context"]
    assert by_ctx["4096"]["step_size"] == 2048
    assert by_ctx["4096"]["peak_memory_gb"] == 7.0
    assert by_ctx["16384"]["step_size"] == 512
    assert by_ctx["16384"]["peak_memory_gb"] == 7.84


def test_derive_speed_correction_factor_inverts_bandwidth_formula(dense_model):
    # Apple M5 Pro bandwidth = 307 GB/s; dense_model active_weight_gb = 6.3 GB.
    # At 62.42 tok/s: round(62.42 * 6.3 / 307, 3) = 1.281
    sweep_data = {
        "decode_speed_vs_context": [
            {"context_tokens": 100, "kv_bits": None, "decode_tok_s": 62.42},
            {"context_tokens": 16384, "kv_bits": None, "decode_tok_s": 58.10},
        ],
    }
    scf = sm.derive_speed_correction_factor(sweep_data, dense_model, "Apple M5 Pro", 20)
    assert scf == 1.281
    # Unknown chip returns None cleanly
    assert sm.derive_speed_correction_factor(sweep_data, dense_model, "Apple Z9000", 20) is None


def test_derive_speed_correction_factor_skips_drafter_models(drafter_model, dspark_drafter_model):
    sweep_data = {
        "decode_speed_vs_context": [
            {"context_tokens": 100, "kv_bits": None, "decode_tok_s": 110.0},
        ],
    }
    assert sm.derive_speed_correction_factor(sweep_data, drafter_model, "Apple M5 Pro", 20) is None
    assert sm.derive_speed_correction_factor(sweep_data, dspark_drafter_model, "Apple M5 Pro", 20) is None


# --- SweepServer.build_command_args (#4: draft-model + mlx-dspark support) -------------

def test_sweep_server_build_command_args_standard_mlx_vlm():
    srv = sm.SweepServer(
        "mlx-community/test-vlm", "127.0.0.1", 38472, 16384, "/tmp/test.log",
        prefill_step_size=2048, kv_bits=8, max_num_seqs=2,
    )
    cmd = srv.build_command_args()
    assert cmd[:4] == [sm.MLX_PYTHON, "-m", "mlx_vlm.server", "--model"]
    assert "--prefill-step-size" in cmd
    assert cmd[cmd.index("--prefill-step-size") + 1] == "2048"
    assert "--kv-bits" in cmd
    assert cmd[cmd.index("--kv-bits") + 1] == "8"
    assert "--draft-model" not in cmd


def test_sweep_server_build_command_args_mlx_vlm_with_draft_model():
    srv = sm.SweepServer(
        "mlx-community/test-vlm", "127.0.0.1", 38472, 16384, "/tmp/test.log",
        prefill_step_size=2048,
        serving={
            "backend": "mlx_vlm",
            "draft_model": "mlx-community/test-dflash-bf16",
            "draft_kind": "dflash",
            "draft_block_size": 16,
        },
    )
    cmd = srv.build_command_args()
    assert cmd[cmd.index("--draft-model") + 1] == "mlx-community/test-dflash-bf16"
    assert cmd[cmd.index("--draft-kind") + 1] == "dflash"
    assert cmd[cmd.index("--draft-block-size") + 1] == "16"


def test_sweep_server_build_command_args_mlx_dspark():
    srv = sm.SweepServer(
        "mlx-community/test-dspark", "127.0.0.1", 38472, 16384, "/tmp/test.log",
        max_num_seqs=2,
        serving={
            "backend": "mlx_dspark",
            "mode": "dflash",
            "drafter": "z-lab/test-dflash",
        },
    )
    cmd = srv.build_command_args()
    assert cmd[:3] == [sm.MLX_DSPARK, "serve", "--model"]
    assert cmd[cmd.index("--mode") + 1] == "dflash"
    assert cmd[cmd.index("--drafter") + 1] == "z-lab/test-dflash"
    assert cmd[cmd.index("--context-window") + 1] == "16384"
    assert cmd[cmd.index("--kv-bits") + 1] == "0"
    assert cmd[cmd.index("--max-batch") + 1] == "2"

    # With quantized KV, mlx_dspark passes --kv-bits 8 and suppresses --max-batch
    srv_q = sm.SweepServer(
        "mlx-community/test-dspark", "127.0.0.1", 38472, 16384, "/tmp/test.log",
        kv_bits=8, max_num_seqs=2,
        serving={"backend": "mlx_dspark", "drafter": "z-lab/test-dflash"},
    )
    cmd_q = srv_q.build_command_args()
    assert cmd_q[cmd_q.index("--kv-bits") + 1] == "8"
    assert "--max-batch" not in cmd_q


# --- normalize_dspark_timings & extract_draft_stats_from_timings -----------------------

def test_normalize_dspark_timings_maps_dspark_fields():
    x_dspark = {
        "prompt_tokens": 360,
        "cached_tokens": 60,
        "completion_tokens": 90,
        "ttft_seconds": 0.25,
        "prefill_tokens_per_sec": 1200.0,
        "decode_tokens_per_sec": 92.5,
        "mode": "dflash",
        "accept_len": 4.1,
        "target_forwards": 22,
    }
    metrics = {
        "memory": {"peak_bytes": int(8.5e9)},
        "rounds": {"draft_acceptance": 0.72},
    }
    norm = sm.normalize_dspark_timings(x_dspark, metrics=metrics)
    assert norm["prompt_n"] == 300
    assert norm["cache_n"] == 60
    assert norm["predicted_n"] == 90
    assert norm["prompt_ms"] == 250.0
    assert norm["prompt_per_second"] == 1200.0
    assert norm["predicted_per_second"] == 92.5
    assert norm["peak_memory"] == 8.5
    assert norm["draft_kind"] == "dflash"
    assert norm["accept_len"] == 4.1
    assert norm["target_forwards"] == 22
    assert norm["draft_acceptance_rate"] == 0.72


def test_extract_draft_stats_from_timings_supports_vlm_and_dspark_shapes():
    # mlx_vlm shape: draft_n_accepted / draft_n, and 1 + draft_n_accepted / draft_rounds
    vlm_t = {
        "draft_n": 40,
        "draft_n_accepted": 30,
        "draft_rounds": 10,
    }
    acc_rate, mean_accept = sm.extract_draft_stats_from_timings(vlm_t)
    assert acc_rate == 0.75
    assert mean_accept == 4.0

    # dspark normalized shape
    dspark_t = {"draft_acceptance_rate": 0.68, "accept_len": 3.4}
    acc_rate2, mean_accept2 = sm.extract_draft_stats_from_timings(dspark_t)
    assert acc_rate2 == 0.68
    assert mean_accept2 == 3.4


def test_run_drafter_tier_summarizes_by_context(drafter_model, dense_model):
    assert sm.run_drafter_tier(dense_model) == {"drafter_attached": False}
    curve = [
        {
            "context_tokens": 100,
            "kv_bits": None,
            "decode_tok_s": 95.2,
            "mean_accept_len": 4.2,
            "draft_acceptance_rate": 0.74,
        },
        {
            "context_tokens": 16384,
            "kv_bits": None,
            "decode_tok_s": 81.0,
            "mean_accept_len": 3.6,
            "draft_acceptance_rate": 0.65,
        },
        {
            "context_tokens": 16384,
            "kv_bits": 8,
            "decode_tok_s": 78.0,
            "mean_accept_len": 3.5,
            "draft_acceptance_rate": 0.63,
        },
    ]
    res = sm.run_drafter_tier(drafter_model, decode_speed_vs_context=curve)
    assert res["drafter_attached"] is True
    assert res["tested_this_sweep"] is True
    assert res["drafter"] == "mlx-community/Test-Drafter-assistant-bf16"
    assert res["mode"] == "mtp"
    assert len(res["by_context"]) == 2
    assert res["by_context"][0]["context_tokens"] == 100
    assert res["by_context"][0]["mean_accept_len"] == 4.2
    assert res["by_context"][1]["context_tokens"] == 16384
    assert res["by_context"][1]["draft_acceptance_rate"] == 0.65


def test_derive_lightweight_calibration_fields_fallback_without_prefill_tier():
    # mlx-dspark sweeps skip prefill_step_size_sweep; calibration still derives
    # decode_tok_s and decode_tok_s_by_context from decode_speed_vs_context.
    sweep_data = {
        "prefill_step_size_sweep": [],
        "decode_speed_vs_context": [
            {"context_tokens": 100, "kv_bits": None, "decode_tok_s": 105.4, "peak_memory_gb": 7.1},
            {"context_tokens": 16384, "kv_bits": None, "decode_tok_s": 88.2, "peak_memory_gb": 8.4},
            {"context_tokens": 16384, "kv_bits": 8, "decode_tok_s": 85.0, "peak_memory_gb": 7.6},
        ],
    }
    derived = sm.derive_lightweight_calibration_fields(sweep_data)
    assert derived is not None
    assert derived["prefill_step_size"] == ma.DEFAULT_PREFILL_STEP_SIZE
    assert derived["decode_tok_s"] == 88.2
    assert derived["context_tokens"] == 16384
    assert derived["peak_memory_gb"] == 8.4
    assert derived["kv_bits_safe_at_long_context"] is True
    assert derived["decode_tok_s_by_context"] == {"100": 105.4, "16384": 88.2}


# --- Batch cached-model discovery (#3: --all-cached / --untested-cached) ---------------

def _make_fake_hf_cache_snap(hf_home, repo_id, with_config=True):
    snap = os.path.join(
        str(hf_home), "hub", "models--" + repo_id.replace("/", "--"), "snapshots", "abc123"
    )
    os.makedirs(snap, exist_ok=True)
    if with_config:
        with open(os.path.join(snap, "config.json"), "w") as f:
            f.write("{}")
    return snap


def test_is_repo_cached_and_is_model_cached(tmp_path, dense_model, drafter_model):
    hf_home = tmp_path / "hf"
    assert sm.is_repo_cached(dense_model["huggingface"], hf_home=str(hf_home)) is False

    _make_fake_hf_cache_snap(hf_home, dense_model["huggingface"], with_config=False)
    assert sm.is_repo_cached(dense_model["huggingface"], hf_home=str(hf_home)) is False

    _make_fake_hf_cache_snap(hf_home, dense_model["huggingface"], with_config=True)
    assert sm.is_repo_cached(dense_model["huggingface"], hf_home=str(hf_home)) is True
    assert sm.is_model_cached(dense_model, hf_home=str(hf_home)) is True

    # Drafter model requires both target and drafter repos to be cached
    _make_fake_hf_cache_snap(hf_home, drafter_model["huggingface"], with_config=True)
    assert sm.is_model_cached(drafter_model, hf_home=str(hf_home)) is False
    _make_fake_hf_cache_snap(hf_home, drafter_model["serving"]["draft_model"], with_config=True)
    assert sm.is_model_cached(drafter_model, hf_home=str(hf_home)) is True


def test_select_cached_models_filters_by_status_and_existing_sweep(tmp_path, dense_model):
    hf_home = tmp_path / "hf"
    sweeps_dir = tmp_path / "sweeps"
    sweeps_dir.mkdir()

    m_tested = dict(dense_model, id="tested-m", huggingface="org/tested-m", status="tested")
    m_untested = dict(dense_model, id="untested-m", huggingface="org/untested-m", status="untested")
    m_uncached = dict(dense_model, id="uncached-m", huggingface="org/uncached-m", status="untested")

    _make_fake_hf_cache_snap(hf_home, "org/tested-m", with_config=True)
    _make_fake_hf_cache_snap(hf_home, "org/untested-m", with_config=True)
    models = [m_tested, m_untested, m_uncached]

    # All cached (force=False, no existing sweep files)
    selected = sm.select_cached_models(
        models, chip_name="Apple M5 Pro", untested_only=False, force=False,
        hf_home=str(hf_home), sweeps_dir=str(sweeps_dir),
    )
    assert [m["id"] for m in selected] == ["tested-m", "untested-m"]

    # Untested only
    selected_u = sm.select_cached_models(
        models, chip_name="Apple M5 Pro", untested_only=True, force=False,
        hf_home=str(hf_home), sweeps_dir=str(sweeps_dir),
    )
    assert [m["id"] for m in selected_u] == ["untested-m"]

    # Existing sweep file skips unless force=True
    existing_sweep = sweeps_dir / "Apple_M5_Pro__untested-m.sweep.json"
    existing_sweep.write_text("{}")
    selected_skip = sm.select_cached_models(
        models, chip_name="Apple M5 Pro", untested_only=True, force=False,
        hf_home=str(hf_home), sweeps_dir=str(sweeps_dir),
    )
    assert selected_skip == []

    selected_force = sm.select_cached_models(
        models, chip_name="Apple M5 Pro", untested_only=True, force=True,
        hf_home=str(hf_home), sweeps_dir=str(sweeps_dir),
    )
    assert [m["id"] for m in selected_force] == ["untested-m"]


def test_dspark_supports_kv_bits_gemma_vs_qwen():
    gemma_dspark = sm.load_model_entry("gemma-4-12b-it-4bit-dflash")
    qwen_dspark = sm.load_model_entry("qwen3.5-9b-mlx-4bit-dflash")
    gemma_vlm_drafter = sm.load_model_entry("gemma-4-12b-it-4bit-dflash-vlm")
    gpt_oss = sm.load_model_entry("gpt-oss-20b-mxfp4-q8")
    assert ma._dspark_supports_kv_bits(gemma_dspark) is False
    assert ma._dspark_supports_kv_bits(qwen_dspark) is True
    assert ma._dspark_supports_kv_bits(gemma_vlm_drafter) is True
    assert ma._model_supports_kv_bits(gemma_dspark) is False
    assert ma._model_supports_kv_bits(qwen_dspark) is True
    assert ma._model_supports_kv_bits(gemma_vlm_drafter) is True
    assert ma._model_supports_kv_bits(gpt_oss) is False
    # Even when memory is too tight for unquantized, recommend_kv_bits_and_context
    # must never fall back to kv_bits=8 on gpt-oss (which crashes SDPA on attention sinks).
    rec_tight = ma.recommend_kv_bits_and_context(gpt_oss, free_wired_gb=13.5)
    assert rec_tight["kv_bits"] is None




