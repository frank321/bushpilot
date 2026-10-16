"""Combines kv/speed/prefill into one row per model -- build_recommendations() is
the advisor's main entry point, called once per launch with the full catalog."""
from .kv import (
    DEFAULT_KV_GROUP_SIZE,
    SAFETY_BUFFER_GB,
    UNQUANTIZED_KV_BITS,
    compute_drafter_kv_formula,
    compute_kv_formula,
    effective_size_gb,
    mlx_vlm_apc_pool_bytes,
    prefix_cache_cost_multiplier,
    recommend_kv_bits_and_context,
)
from .prefill import recommend_prefill_step_size
from .speed import estimate_speed_tok_s

# Decided 2026-09-29 (docs/PHASE_PLAN.md Phase 5): build_recommendations() previously gated
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


# Auto mode never suggests more than this many free bonus slots -- beyond it, more
# concurrent local coding sessions against one model isn't a realistic use case even
# when memory technically allows it (and continuous-batched prefill is CPU/GPU-bound
# too, not just memory-bound).
AUTO_MAX_CONCURRENCY = 4


def compute_bonus_concurrency(model_entry, free_wired_gb, kv_bits_used, recommended_context,
                               safety_buffer_gb=SAFETY_BUFFER_GB, prefix_cache_slots=None,
                               apc_num_blocks=None):
    """Auto mode (no explicit concurrency requested): recommend_kv_bits_and_context has
    already picked the max context for ONE slot, rounded DOWN to a clean tier (see
    recommend_max_context's own tiering comment) -- which almost always leaves some
    memory unused. This checks whether that rounding leftover fits one or more
    additional, identically-sized slots for free, without shrinking anyone's context,
    and returns the total slot count (>=1).

    prefix_cache_slots: same multiplier recommend_max_context applied to pick
    recommended_context in the first place (see prefix_cache_cost_multiplier) -- must be
    re-applied here too, or a bonus slot's true cost (live KV + mlx-dspark's own
    always-on prefix-cache copies) would be underestimated and this could offer more
    bonus concurrency than free memory actually allows. mlx_vlm's own APC pool (see
    mlx_vlm_apc_pool_bytes) gets the same treatment but subtracted ONCE up front instead
    of per-slot -- it's one pool shared by the whole server process, not duplicated per
    bonus slot."""
    if recommended_context is None:
        return 1
    kv_bits = kv_bits_used if kv_bits_used is not None else UNQUANTIZED_KV_BITS
    kv_group_size = DEFAULT_KV_GROUP_SIZE if kv_bits < UNQUANTIZED_KV_BITS else None
    formula = compute_kv_formula(model_entry.get("kv_config"), kv_bits, kv_group_size=kv_group_size)
    if formula is None:
        return 1
    target_bytes_per_growing_token, fixed_bytes, _ = formula
    drafter_growing, drafter_fixed = compute_drafter_kv_formula(
        model_entry, kv_bits=kv_bits, kv_group_size=kv_group_size,
    )
    raw_bytes_per_growing_token = target_bytes_per_growing_token + drafter_growing
    fixed_bytes += drafter_fixed

    cache_multiplier = prefix_cache_cost_multiplier(model_entry, prefix_cache_slots)
    bytes_per_growing_token = raw_bytes_per_growing_token * cache_multiplier
    fixed_bytes *= cache_multiplier
    per_slot_bytes = fixed_bytes + bytes_per_growing_token * recommended_context
    if per_slot_bytes <= 0:
        return 1

    apc_pool_bytes = mlx_vlm_apc_pool_bytes(
        model_entry, raw_bytes_per_growing_token, apc_num_blocks=apc_num_blocks,
    )
    available_bytes = (free_wired_gb - effective_size_gb(model_entry) - safety_buffer_gb) * 1024**3
    available_bytes -= apc_pool_bytes
    remaining = available_bytes - per_slot_bytes
    concurrency = 1
    while remaining >= per_slot_bytes and concurrency < AUTO_MAX_CONCURRENCY:
        concurrency += 1
        remaining -= per_slot_bytes
    return concurrency


def build_recommendations(models, free_wired_gb, chip_name, gpu_cores, preload_overhead=0,
                           requested_concurrency=None, max_context_tokens=None,
                           dspark_prefix_cache_slots=None):
    """requested_concurrency: None means auto (maximize context for 1 slot, then add
    free bonus slots from tier-rounding leftover -- see compute_bonus_concurrency).
    An explicit int forces that many slots, shrinking context as needed to guarantee
    all of them fit (see recommend_max_context's concurrency param).
    max_context_tokens: optional upper bound on every model's recommended context
    (config.json's "max_context_tokens") -- a cap layered on top of the memory-fit
    math, never a bypass of it (see recommend_max_context's own docstring).
    dspark_prefix_cache_slots: forwarded to recommend_kv_bits_and_context/
    compute_bonus_concurrency for every model -- a no-op for anything not served
    through mlx_dspark (see prefix_cache_cost_multiplier). None means mlx-dspark's own
    default slot count, matching what start_mlx_server.sh actually launches with when
    config.json doesn't override it."""
    rows = []
    for m in models:
        sizing_concurrency = requested_concurrency if requested_concurrency else 1
        kv_result = recommend_kv_bits_and_context(m, free_wired_gb, preload_overhead,
                                                    concurrency=sizing_concurrency,
                                                    max_context_tokens=max_context_tokens,
                                                    prefix_cache_slots=dspark_prefix_cache_slots)
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
                prefix_cache_slots=dspark_prefix_cache_slots,
                apc_num_blocks=kv_result.get("apc_num_blocks"),
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
            "apc_num_blocks": kv_result.get("apc_num_blocks"),
            "estimated_tok_s": speed,
            "speed_reason": speed_reason,
            "context_after_preload": kv_result["context_after_preload"],
            "fits": fits,
            "concurrency": concurrency,
            "prefill_step_size": prefill_step_size,
            "prefill_step_size_reason": prefill_step_size_reason,
        })
    return rows
