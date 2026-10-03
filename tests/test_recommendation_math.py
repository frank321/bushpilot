"""Tests for the memory/context/speed recommendation math: compute_kv_formula,
recommend_max_context, estimate_speed_tok_s, recommend_kv_bits_and_context,
compute_bonus_concurrency, build_recommendations.

This is the project's actual core value proposition -- "will this model fit, at what
context size, how fast" -- and until now it was only ever checked by hand on whatever
Mac happened to be at hand (see IMPLEMENTATION_NOTES.md's "Testing Strategy"). These
tests use synthetic chip/memory inputs instead of real hardware, so the ARITHMETIC can
be verified on any machine, in CI, offline -- they deliberately don't replace
verify_max_kv_size.sh / verify_long_context_memory.sh, which check things no unit test
can (whether mlx_vlm.server itself actually enforces --max-kv-size on real hardware).

Every expected number below was derived by hand from the formulas in model_advisor.py
and cross-checked by running the real functions once to confirm (see the session that
added this file) -- not copied from a prior implementation that could itself be wrong.
"""
import model_advisor as ma


# --- effective_size_gb --------------------------------------------------------------

def test_effective_size_gb_without_drafter(dense_model):
    assert ma.effective_size_gb(dense_model) == 6.3


def test_effective_size_gb_adds_drafter_weight(drafter_model):
    # 4.5 (target) + 0.16 (drafter) -- the drafter is a separate real download that
    # sits in memory for the whole session alongside the target.
    assert ma.effective_size_gb(drafter_model) == 4.5 + 0.16


# --- compute_kv_formula --------------------------------------------------------------

def test_compute_kv_formula_returns_none_without_kv_config():
    assert ma.compute_kv_formula(None) is None


def test_compute_kv_formula_returns_none_when_required_fields_missing():
    assert ma.compute_kv_formula({"num_key_value_heads": 2}) is None


def test_compute_kv_formula_hybrid_linear_and_full_attention(dense_model):
    # 6 linear_attention (~0 fixed cost) + 2 full_attention (growing) layers.
    bytes_per_token, fixed_bytes, detail = ma.compute_kv_formula(
        dense_model["kv_config"], kv_bits=16)
    assert detail["growing_layers"] == 2
    assert detail["linear_layers"] == 6
    assert detail["sliding_layers"] == 0
    # 2 (bytes_per_elem=16/8) * 2 growing layers * 2 kv_heads * 256 head_dim = 4096
    assert bytes_per_token == 4096.0
    assert fixed_bytes == 0.0


def test_compute_kv_formula_scales_with_kv_bits():
    kv_config = {
        "num_key_value_heads": 2, "head_dim": 256, "num_hidden_layers": 8,
        "layer_types": ["linear_attention"] * 6 + ["full_attention"] * 2,
    }
    bytes_16, _, _ = ma.compute_kv_formula(kv_config, kv_bits=16)
    bytes_8, _, _ = ma.compute_kv_formula(kv_config, kv_bits=8)
    assert bytes_16 == 4096.0
    assert bytes_8 == 2048.0
    assert bytes_16 == bytes_8 * 2


def test_compute_kv_formula_sliding_window_adds_fixed_cost(sliding_window_model):
    bytes_per_token, fixed_bytes, detail = ma.compute_kv_formula(
        sliding_window_model["kv_config"], kv_bits=16)
    assert detail["growing_layers"] == 2
    assert detail["sliding_layers"] == 6
    # 2 * 2 growing * 4 kv_heads * 128 head_dim = 4096 (same per-token cost shape)
    assert bytes_per_token == 4096.0
    # 2 * 6 sliding * 4 kv_heads * 128 head_dim * 2 bytes/elem * 512 window = 6,291,456
    assert fixed_bytes == 6291456.0


def test_compute_kv_formula_no_layer_types_treats_every_layer_as_growing():
    kv_config = {"num_key_value_heads": 2, "head_dim": 128, "num_hidden_layers": 4}
    bytes_per_token, fixed_bytes, detail = ma.compute_kv_formula(kv_config, kv_bits=16)
    assert detail["growing_layers"] == 4
    assert fixed_bytes == 0.0


def test_compute_kv_formula_unrecognized_layer_type_counts_as_growing():
    # Deliberate safety choice (see the function's docstring): an unrecognized layer
    # type name is the SAFER overestimate of memory need, not silently dropped.
    kv_config = {
        "num_key_value_heads": 2, "head_dim": 128, "num_hidden_layers": 3,
        "layer_types": ["full_attention", "some_future_layer_type", "linear_attention"],
    }
    _, _, detail = ma.compute_kv_formula(kv_config, kv_bits=16)
    assert detail["growing_layers"] == 2  # full_attention + the unrecognized one
    assert detail["linear_layers"] == 1


# --- recommend_max_context ------------------------------------------------------------

def test_recommend_max_context_insufficient_for_model_weights(dense_model):
    # 6.3 (weights) + 3.0 (safety buffer) = 9.3 GB floor; 9.0 free doesn't clear it.
    ctx, reason, detail = ma.recommend_max_context(dense_model, free_wired_gb=9.0)
    assert ctx is None
    assert reason == "insufficient_memory_for_model_weights"
    assert detail is None


def test_recommend_max_context_too_small_to_be_useful(dense_model):
    # Just over the weights+buffer floor, but not enough left for MIN_USABLE_CONTEXT.
    ctx, reason, _ = ma.recommend_max_context(dense_model, free_wired_gb=9.3)
    assert ctx is None
    assert reason == "computed_context_too_small_to_be_useful"


def test_recommend_max_context_rounds_down_to_a_clean_tier(dense_model):
    ctx, reason, _ = ma.recommend_max_context(dense_model, free_wired_gb=10)
    assert reason == "computed"
    assert ctx == 131072


def test_recommend_max_context_capped_at_architectural_max(dense_model):
    ctx, reason, _ = ma.recommend_max_context(dense_model, free_wired_gb=1000)
    assert reason == "computed"
    assert ctx == dense_model["max_context_architectural"] == 262144


def test_recommend_max_context_sliding_window_model(sliding_window_model):
    ctx, reason, _ = ma.recommend_max_context(sliding_window_model, free_wired_gb=16)
    assert reason == "computed"
    assert ctx == 262144


def test_recommend_max_context_max_context_tokens_is_a_cap_not_an_override(dense_model):
    # The cap is applied BEFORE tier rounding, so it can only shrink the result --
    # 20000 rounds down to the 16384 tier, never up past the cap.
    ctx, reason, _ = ma.recommend_max_context(
        dense_model, free_wired_gb=1000, max_context_tokens=20000)
    assert reason == "computed"
    assert ctx == 16384
    assert ctx <= 20000


def test_recommend_max_context_concurrency_shrinks_per_slot_context(dense_model):
    ctx_1, _, _ = ma.recommend_max_context(dense_model, free_wired_gb=12, concurrency=1)
    ctx_4, _, _ = ma.recommend_max_context(dense_model, free_wired_gb=12, concurrency=4)
    assert ctx_1 == 262144  # one slot: plenty of room, hits the architectural cap
    assert ctx_4 < ctx_1  # four slots dividing the same memory: each gets less


# --- estimate_speed_tok_s -------------------------------------------------------------

def test_estimate_speed_tok_s_dense_model(dense_model):
    speed, reason = ma.estimate_speed_tok_s(dense_model, "Apple M3 Max", 40)
    # 0.87 (SPEED_CORRECTION_FACTOR) * 350 (M3 Max bandwidth) / 6.3 (active_gb) = 48.33...
    assert speed == 48.3
    assert reason == "estimated"


def test_estimate_speed_tok_s_unknown_chip_returns_none(dense_model):
    speed, reason = ma.estimate_speed_tok_s(dense_model, "Apple Z9000", 40)
    assert speed is None
    assert "no bandwidth data" in reason


def test_estimate_speed_tok_s_moe_without_active_weight_is_unknown(moe_model_no_active_weight):
    speed, reason = ma.estimate_speed_tok_s(moe_model_no_active_weight, "Apple M3 Max", 40)
    assert speed is None
    assert "MoE model with unknown active-parameter fraction" in reason


def test_estimate_speed_tok_s_dense_falls_back_to_size_gb_when_active_weight_missing():
    m = {"size_gb": 10.0, "type": "dense"}
    speed, reason = ma.estimate_speed_tok_s(m, "Apple M3 Max", 40)
    # 0.87 * 350 / 10.0 = 30.45 -> 30.4 (banker's/round-half-even at the .05 boundary)
    assert speed == 30.4
    assert reason == "estimated"


def test_estimate_speed_tok_s_applies_measured_speedup_multiplier(drafter_model):
    speed, reason = ma.estimate_speed_tok_s(drafter_model, "Apple M3 Max", 40)
    # base = 0.87 * 350 / 4.5 = 67.6667; speed = round(base * 1.5, 1) = 101.5
    assert speed == 101.5
    assert "1.5x speculative-decoding multiplier" in reason
    assert "estimated" in reason


def test_estimate_speed_tok_s_uses_per_model_speed_correction_factor_override(dense_model):
    # Same shape as test_estimate_speed_tok_s_dense_model, but with a model-specific
    # override (gap: one global SPEED_CORRECTION_FACTOR is confirmed not to hold across
    # architecture families -- see HARDWARE_SPEED_ESTIMATION.md \u00a717 item #2) in place
    # of the 0.87 global constant.
    dense_model["speed_correction_factor"] = 1.29
    speed, reason = ma.estimate_speed_tok_s(dense_model, "Apple M3 Max", 40)
    # 1.29 * 350 / 6.3 = 71.666... -> 71.7, not the 0.87-based 48.3 from the other test.
    assert speed == 71.7
    assert reason == "estimated"


def test_estimate_speed_tok_s_prefers_saved_calibration_over_formula(dense_model, tmp_path, monkeypatch):
    # A real measurement from a previous run on this exact (chip, model) pair always
    # wins over the physics formula -- see load_calibration's docstring.
    monkeypatch.setattr(ma, "CALIBRATION_DIR", str(tmp_path))
    ma.save_calibration("Apple M3 Max", dense_model["id"], 55.5, context_tokens=4096)
    speed, reason = ma.estimate_speed_tok_s(dense_model, "Apple M3 Max", 40)
    assert speed == 55.5
    assert "measured on this machine" in reason
    assert "4,096" in reason


def test_estimate_speed_tok_s_falls_back_to_formula_when_no_calibration_saved(dense_model, tmp_path, monkeypatch):
    monkeypatch.setattr(ma, "CALIBRATION_DIR", str(tmp_path))
    speed, reason = ma.estimate_speed_tok_s(dense_model, "Apple M3 Max", 40)
    assert speed == 48.3
    assert reason == "estimated"


# --- save_calibration / load_calibration -----------------------------------------------

def test_load_calibration_returns_none_when_nothing_saved(tmp_path, monkeypatch):
    monkeypatch.setattr(ma, "CALIBRATION_DIR", str(tmp_path))
    assert ma.load_calibration("Apple M5 Pro", "some-model") is None


def test_save_then_load_calibration_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr(ma, "CALIBRATION_DIR", str(tmp_path))
    ma.save_calibration("Apple M5 Pro", "gemma-4-e4b-it-4bit", 71.6,
                         context_tokens=7200, ttft_s=1.65, peak_memory_gb=6.49)
    loaded = ma.load_calibration("Apple M5 Pro", "gemma-4-e4b-it-4bit")
    assert loaded["decode_tok_s"] == 71.6
    assert loaded["context_tokens"] == 7200
    assert loaded["ttft_s"] == 1.65
    assert loaded["peak_memory_gb"] == 6.49
    assert "saved_at" in loaded


def test_save_calibration_overwrites_prior_measurement_for_same_pair(tmp_path, monkeypatch):
    monkeypatch.setattr(ma, "CALIBRATION_DIR", str(tmp_path))
    ma.save_calibration("Apple M5 Pro", "some-model", 10.0)
    ma.save_calibration("Apple M5 Pro", "some-model", 20.0)
    assert ma.load_calibration("Apple M5 Pro", "some-model")["decode_tok_s"] == 20.0


def test_load_calibration_ignores_corrupt_file(tmp_path, monkeypatch):
    monkeypatch.setattr(ma, "CALIBRATION_DIR", str(tmp_path))
    path = ma._calibration_path("Apple M5 Pro", "some-model")
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("not valid json{{{")
    assert ma.load_calibration("Apple M5 Pro", "some-model") is None


def test_calibration_path_sanitizes_chip_name_spaces(tmp_path, monkeypatch):
    monkeypatch.setattr(ma, "CALIBRATION_DIR", str(tmp_path))
    ma.save_calibration("Apple M5 Pro", "some-model", 10.0)
    files = list(tmp_path.iterdir())
    assert len(files) == 1
    assert " " not in files[0].name


def test_save_calibration_omits_sweep_fields_when_not_given(tmp_path, monkeypatch):
    # Additive-only contract (see save_calibration's own docstring) -- a plain
    # teardown-time save (no sweep data) must look byte-identical to before
    # prefill_step_size/kv_bits_safe_at_long_context existed, not write explicit nulls.
    monkeypatch.setattr(ma, "CALIBRATION_DIR", str(tmp_path))
    ma.save_calibration("Apple M5 Pro", "some-model", 10.0)
    loaded = ma.load_calibration("Apple M5 Pro", "some-model")
    assert "prefill_step_size" not in loaded
    assert "kv_bits_safe_at_long_context" not in loaded


def test_save_calibration_round_trips_sweep_fields_when_given(tmp_path, monkeypatch):
    monkeypatch.setattr(ma, "CALIBRATION_DIR", str(tmp_path))
    ma.save_calibration("Apple M5 Pro", "gemma-4-e4b-it-4bit", 71.6,
                         context_tokens=7200, ttft_s=1.65, peak_memory_gb=6.49,
                         prefill_step_size=2048, kv_bits_safe_at_long_context=True)
    loaded = ma.load_calibration("Apple M5 Pro", "gemma-4-e4b-it-4bit")
    assert loaded["prefill_step_size"] == 2048
    assert loaded["kv_bits_safe_at_long_context"] is True


# --- recommend_kv_bits_and_context (the kv-quantization fallback decision) ----------

def test_recommend_kv_bits_and_context_prefers_unquantized_when_it_fits(dense_model):
    result = ma.recommend_kv_bits_and_context(dense_model, free_wired_gb=24)
    assert result["fits"] is True
    assert result["kv_bits"] is None  # None means "omit --kv-bits entirely"
    assert result["recommended_context"] == 262144


def test_recommend_kv_bits_and_context_falls_back_to_8bit_when_needed(dense_model):
    # At this free memory, unquantized's context rounds to below MIN_USABLE_CONTEXT,
    # but halving KV bytes/token with 8-bit KV lets it clear the floor.
    result = ma.recommend_kv_bits_and_context(dense_model, free_wired_gb=9.34)
    assert result["fits"] is True
    assert result["kv_bits"] == ma.MEMORY_CONSTRAINED_KV_BITS
    assert result["recommended_context"] == 16384
    # dense_model has no sliding-window layers -- the mlx-vlm #1559 gap doesn't apply.
    assert result["kv_bits_apc_caveat"] is False


def test_recommend_kv_bits_and_context_flags_apc_caveat_for_sliding_window_fallback(sliding_window_model):
    # Real, upstream-confirmed gap (mlx-vlm issue #1559): --kv-bits + APC + sliding-
    # window (SWA) layers don't compose correctly on the batched request path. Only
    # set when the quantized fallback is ACTUALLY used -- a model with sliding-window
    # layers that fits unquantized never hits this combo at all (see the next test).
    result = ma.recommend_kv_bits_and_context(sliding_window_model, free_wired_gb=7.55)
    assert result["fits"] is True
    assert result["kv_bits"] == ma.MEMORY_CONSTRAINED_KV_BITS
    assert result["kv_bits_apc_caveat"] is True


def test_recommend_kv_bits_and_context_no_apc_caveat_when_sliding_window_model_fits_unquantized(sliding_window_model):
    result = ma.recommend_kv_bits_and_context(sliding_window_model, free_wired_gb=24)
    assert result["fits"] is True
    assert result["kv_bits"] is None
    assert result["kv_bits_apc_caveat"] is False


def test_recommend_kv_bits_and_context_neither_fits(dense_model):
    result = ma.recommend_kv_bits_and_context(dense_model, free_wired_gb=9.0)
    assert result["fits"] is False
    assert result["kv_bits_apc_caveat"] is False
    assert result["context_reason"] == "insufficient_memory_for_model_weights"


def test_recommend_kv_bits_and_context_preload_overhead_can_fail_an_otherwise_fitting_model(dense_model):
    # Fits on memory alone, but a tool's preload (system prompt + docs/skills/MCP) eats
    # so much of the context that what's left for conversation drops below the floor.
    result = ma.recommend_kv_bits_and_context(dense_model, free_wired_gb=24, preload_overhead=260000)
    assert result["fits"] is False
    assert result["context_reason"] == "insufficient_context_after_tool_preload"
    assert result["context_after_preload"] == 262144 - 260000


# --- recommend_prefill_step_size (per-(chip, model) sweep override of the global default)

def test_recommend_prefill_step_size_defaults_when_nothing_calibrated(dense_model, tmp_path, monkeypatch):
    monkeypatch.setattr(ma, "CALIBRATION_DIR", str(tmp_path))
    step_size, reason = ma.recommend_prefill_step_size(dense_model, "Apple M3 Max", 24)
    assert step_size == ma.DEFAULT_PREFILL_STEP_SIZE == 2048
    assert "default" in reason
    assert "no calibrated prefill_step_size" in reason


def test_recommend_prefill_step_size_defaults_when_calibration_predates_the_field(dense_model, tmp_path, monkeypatch):
    # A calibration file saved before prefill_step_size existed at all (e.g. by an
    # older save_calibration_snapshot()) has no such key -- treated identically to
    # "nothing calibrated yet", not an error.
    monkeypatch.setattr(ma, "CALIBRATION_DIR", str(tmp_path))
    ma.save_calibration("Apple M3 Max", dense_model["id"], 55.5, context_tokens=4096)
    step_size, reason = ma.recommend_prefill_step_size(dense_model, "Apple M3 Max", 24)
    assert step_size == ma.DEFAULT_PREFILL_STEP_SIZE
    assert "default" in reason


def test_recommend_prefill_step_size_reports_calibrated_default_as_calibrated(dense_model, tmp_path, monkeypatch):
    # Calibrated value happens to equal the global default -- still a real
    # measurement, worth saying so, not silently indistinguishable from "unmeasured".
    monkeypatch.setattr(ma, "CALIBRATION_DIR", str(tmp_path))
    ma.save_calibration("Apple M3 Max", dense_model["id"], 71.6, peak_memory_gb=6.49,
                         prefill_step_size=2048)
    step_size, reason = ma.recommend_prefill_step_size(dense_model, "Apple M3 Max", 24)
    assert step_size == 2048
    assert "calibrated" in reason
    assert "matches the global default" in reason


def test_recommend_prefill_step_size_uses_calibrated_non_default_when_memory_allows(dense_model, tmp_path, monkeypatch):
    monkeypatch.setattr(ma, "CALIBRATION_DIR", str(tmp_path))
    ma.save_calibration("Apple M3 Max", dense_model["id"], 68.0, peak_memory_gb=6.71,
                         prefill_step_size=4096)
    # 6.71 * 1.15 margin + 3.0 safety buffer ≈ 10.7GB required -- 24GB clears it easily.
    step_size, reason = ma.recommend_prefill_step_size(dense_model, "Apple M3 Max", 24)
    assert step_size == 4096
    assert "measured on this machine" in reason


def test_recommend_prefill_step_size_falls_back_when_memory_too_tight_for_non_default(dense_model, tmp_path, monkeypatch):
    monkeypatch.setattr(ma, "CALIBRATION_DIR", str(tmp_path))
    ma.save_calibration("Apple M3 Max", dense_model["id"], 68.0, peak_memory_gb=6.71,
                         prefill_step_size=4096)
    # Same calibration as above, but far less free memory this time -- 6.71*1.15+3.0
    # ≈ 10.7GB required, well above the 8GB on offer here, so falls back to default.
    step_size, reason = ma.recommend_prefill_step_size(dense_model, "Apple M3 Max", 8)
    assert step_size == ma.DEFAULT_PREFILL_STEP_SIZE
    assert "default" in reason
    assert "needs ~" in reason


def test_recommend_prefill_step_size_falls_back_when_calibration_has_no_peak_memory(dense_model, tmp_path, monkeypatch):
    # A non-default step size with no recorded peak_memory_gb can't be safety-checked
    # at all -- falls back rather than risk using an unverified value blindly.
    monkeypatch.setattr(ma, "CALIBRATION_DIR", str(tmp_path))
    ma.save_calibration("Apple M3 Max", dense_model["id"], 68.0, prefill_step_size=512)
    step_size, reason = ma.recommend_prefill_step_size(dense_model, "Apple M3 Max", 24)
    assert step_size == ma.DEFAULT_PREFILL_STEP_SIZE
    assert "no recorded peak_memory_gb" in reason


def test_recommend_prefill_step_size_lower_non_default_needs_no_extra_margin(dense_model, tmp_path, monkeypatch):
    # A step size BELOW the default (e.g. 512) was already observed using LESS memory
    # than the default in §12's own data -- no 15% margin applied in that direction.
    monkeypatch.setattr(ma, "CALIBRATION_DIR", str(tmp_path))
    ma.save_calibration("Apple M3 Max", dense_model["id"], 72.0, peak_memory_gb=6.25,
                         prefill_step_size=512)
    step_size, reason = ma.recommend_prefill_step_size(dense_model, "Apple M3 Max", 9.3)
    assert step_size == 512
    assert "measured on this machine" in reason


# --- compute_bonus_concurrency --------------------------------------------------------

def test_compute_bonus_concurrency_caps_at_auto_max(dense_model):
    # Plenty of memory left over after one max-context slot -- caps at
    # AUTO_MAX_CONCURRENCY rather than packing in as many slots as technically fit.
    concurrency = ma.compute_bonus_concurrency(dense_model, 24, None, 262144)
    assert concurrency == ma.AUTO_MAX_CONCURRENCY == 4


def test_compute_bonus_concurrency_one_when_no_room_for_a_second_slot(dense_model):
    concurrency = ma.compute_bonus_concurrency(dense_model, 9.4, None, 16384)
    assert concurrency == 1


def test_compute_bonus_concurrency_one_when_context_is_none(dense_model):
    assert ma.compute_bonus_concurrency(dense_model, 24, None, None) == 1


# --- build_recommendations (end-to-end) ----------------------------------------------

def test_build_recommendations_marks_fitting_and_non_fitting_models(dense_model, moe_model_no_active_weight):
    # moe_model_no_active_weight is a 50GB model; 24GB free can't fit it.
    rows = ma.build_recommendations([dense_model, moe_model_no_active_weight], free_wired_gb=24,
                                     chip_name="Apple M3 Max", gpu_cores=40)
    by_id = {r["model"]["id"]: r for r in rows}

    dense_row = by_id[dense_model["id"]]
    assert dense_row["fits"] is True
    assert dense_row["recommended_context"] == 262144
    assert dense_row["estimated_tok_s"] == 48.3
    assert dense_row["concurrency"] == 4
    # No calibration saved for this (chip, model) pair -- falls back to the global
    # default, same as recommend_prefill_step_size's own unit tests.
    assert dense_row["prefill_step_size"] == ma.DEFAULT_PREFILL_STEP_SIZE
    assert "default" in dense_row["prefill_step_size_reason"]

    moe_row = by_id[moe_model_no_active_weight["id"]]
    assert moe_row["fits"] is False
    assert moe_row["context_reason"] == "insufficient_memory_for_model_weights"
    assert moe_row["estimated_tok_s"] is None


def test_build_recommendations_excludes_models_estimated_too_slow():
    # Fits in memory easily, but a slow chip + large active-weight combination puts
    # the speed estimate below MIN_SPEED_TOK_S -- excluded even though it "fits".
    slow_model = {
        "id": "test-slow", "size_gb": 6.3, "active_weight_gb": 20.0, "type": "dense",
        "max_context_architectural": 262144,
        "kv_config": {
            "num_key_value_heads": 2, "head_dim": 256, "num_hidden_layers": 8,
            "layer_types": ["linear_attention"] * 6 + ["full_attention"] * 2,
        },
    }
    rows = ma.build_recommendations([slow_model], free_wired_gb=24, chip_name="Apple M1", gpu_cores=8)
    row = rows[0]
    assert row["estimated_tok_s"] < ma.MIN_SPEED_TOK_S
    assert row["fits"] is False
    assert row["context_reason"] == "estimated_speed_too_slow"


def test_build_recommendations_explicit_concurrency_shrinks_context_instead_of_auto_bonus(dense_model):
    auto_rows = ma.build_recommendations([dense_model], free_wired_gb=24,
                                          chip_name="Apple M3 Max", gpu_cores=40)
    forced_rows = ma.build_recommendations([dense_model], free_wired_gb=24,
                                            chip_name="Apple M3 Max", gpu_cores=40,
                                            requested_concurrency=2)
    assert auto_rows[0]["concurrency"] == 4  # auto bonus picked its own count
    assert forced_rows[0]["concurrency"] == 2  # explicit request always honored exactly
