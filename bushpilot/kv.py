"""Context-size recommendation: KV-cache byte math and the policy for how much
context (and whether to quantize the KV cache) a model gets, given free memory.
Pure arithmetic -- no imports, no I/O."""


MEMORY_CONSTRAINED_KV_BITS = 8   # used when unquantized fails or unlocks a bigger tier — see recommend_kv_bits_and_context
UNQUANTIZED_KV_BITS = 16        # what MLX uses when --kv-bits is simply not passed (float16 KV)
DEFAULT_KV_GROUP_SIZE = 64      # MLX's default --kv-group-size (KVCache.to_quantized adds 4 bytes FP16 scale+bias per group)
SAFETY_BUFFER_GB = 3.0       # headroom kept free beyond model+KV, for OS/background apps/spikes


MIN_USABLE_CONTEXT = 12288   # below this, a context window isn't really usable — also used
                              # as the floor for CONVERSATION HEADROOM after tool preload
                              # (see estimate_preload_overhead below): a model that only
                              # "fits" on paper but leaves less than this after Claude
                              # Code's/OpenCode's own preload isn't usable either.

# Unquantized KV that only reaches the 16k tier (< 32,768) when the model/config
# supports more is upgraded to 8-bit KV if 8-bit unlocks a strictly larger tier (32k+)
# without triggering mlx-vlm #1559's SWA+APC caveat.
COMFORTABLE_UNQUANTIZED_CONTEXT = 32768


# mlx-dspark (serving.backend == "mlx_dspark") always runs its own prefix-cache, a small
# LRU of up to this many full conversation checkpoints kept resident in RAM at once (see
# mlx_dspark/prefix_cache.py's own docstring: "a small LRU of conversations (default 2
# slots) -- so an agent process and a chat window hitting the same server don't evict
# each other every turn"). start_mlx_server.sh only passes --prefix-cache-slots when
# config.json's dspark_prefix_cache_slots is set; unset means mlx-dspark's own default,
# which this constant mirrors. Each resident slot can be nearly as large as the live
# generation's own KV cache (a slot snapshots "the target KV cache (and, for DSpark, the
# drafter's context cache)" per that same docstring), and there's no RAM ceiling of its
# own by default either (--prefix-cache-max-ram-mb also unset unless config.json's
# dspark_prefix_cache_max_ram_mb is set) -- so without accounting for it here, a model
# sized to just fit ONE live KV cache can push well past free memory in real multi-turn
# use once the cache fills, surfacing only reactively as repeated memory-guard sheds
# (confirmed live: a qwen3.8-27b-4bit-dflash session recommended the full 262144-token
# architectural max at launch, free_wired_gb=41.28GB, then measured peak_bytes=45.15GB
# mid-session with 16 memory-guard sheds and 2 full cache-empties -- the ~6-9GB gap
# between the formula's budget and the real peak matches an extra, unbudgeted KV-sized
# structure almost exactly).
DSPARK_DEFAULT_PREFIX_CACHE_SLOTS = 2


def effective_size_gb(model_entry):
    """model_entry["size_gb"] is the target weights only (see _schema_notes.size_gb in
    models.json) -- a "serving" block's drafter is a SEPARATE real download that sits in
    memory alongside the target for the whole session, so any memory-fit check has to
    account for both. Kept as a small helper rather than inlined at each call site so the
    two things this affects (recommend_max_context's admission check, compute_bonus_
    concurrency's leftover-memory check) can't drift out of sync with each other. Does
    NOT affect estimate_speed_tok_s's active_gb -- that's about per-token bandwidth cost
    of the TARGET's own forward pass, which the drafter's real-but-comparatively-tiny
    forward pass doesn't change; the drafter's speed impact is already captured in
    serving.measured_speedup instead."""
    return model_entry["size_gb"] + model_entry.get("serving", {}).get("drafter_size_gb", 0)


def prefix_cache_cost_multiplier(model_entry, prefix_cache_slots=None):
    """How many KV-cache-sized copies to budget for, beyond the one live generation
    already costs. 1 (no-op) for every backend except mlx_dspark: that backend always
    keeps its own prefix-cache LRU resident (see DSPARK_DEFAULT_PREFIX_CACHE_SLOTS'
    comment), so a model served through it needs memory for the live KV cache PLUS up
    to `slots` more full-size cached snapshots, worst case all resident at once.

    prefix_cache_slots=None means "whatever start_mlx_server.sh will actually launch
    with" -- mirrors --prefix-cache-slots being omitted, which is mlx-dspark's own
    default (DSPARK_DEFAULT_PREFIX_CACHE_SLOTS). An explicit value (from config.json's
    dspark_prefix_cache_slots, forwarded by model_advisor.py's --prefix-cache-slots)
    overrides it so the sizing math never disagrees with what actually gets launched."""
    if model_entry.get("serving", {}).get("backend") != "mlx_dspark":
        return 1
    slots = prefix_cache_slots if prefix_cache_slots is not None else DSPARK_DEFAULT_PREFIX_CACHE_SLOTS
    return 1 + max(0, slots)


# mlx_vlm (every model NOT served through mlx_dspark -- with or without a drafter, see
# start_mlx_server.sh's launch branches) always runs with its own native Automatic
# Prefix Caching (APC) too, unconditionally (see docs/HARDWARE_SPEED_ESTIMATION.md §13):
# start_mlx_server.sh exports APC_ENABLED=1, APC_NUM_BLOCKS=4096 by default (or the
# advisor's scaled-down MLX_VLM_APC_CONSTRAINED_NUM_BLOCKS when memory is tight) for
# every mlx_vlm launch. These constants mirror that real memory consumer -- same
# "deliberately duplicated constants" pairing as DEFAULT_PREFILL_STEP_SIZE/
# PREFILL_STEP_SIZE (see AGENTS.md): if APC_NUM_BLOCKS changes in start_mlx_server.sh,
# change MLX_VLM_APC_NUM_BLOCKS here too. APC_BLOCK_SIZE itself is mlx_vlm's own
# hard-coded default (not an env var this project sets, so there is nothing on the bash
# side to mirror beyond APC_NUM_BLOCKS) -- confirmed from mlx_vlm/apc.py's own source
# per §13's research.
MLX_VLM_APC_NUM_BLOCKS = 4096
MLX_VLM_APC_CONSTRAINED_NUM_BLOCKS = 1024  # 16,384 cached tokens when full 65k APC pool won't fit
MLX_VLM_APC_BLOCK_SIZE = 16


def mlx_vlm_apc_pool_bytes(model_entry, bytes_per_growing_token, apc_num_blocks=None):
    """mlx_vlm's own APC pool's documented worst case: apc_num_blocks * APC_BLOCK_SIZE
    tokens' worth of KV, at whatever per-token rate the CURRENT recommend_max_context
    attempt is using for bytes_per_growing_token (so the 8-bit kv_bits fallback prices
    in ~half as much as the unquantized attempt tried first, matching compute_kv_
    formula's own kv_bits scaling -- pass the RAW, pre-concurrency/pre-prefix_cache_
    cost_multiplier value here, not the already-scaled one).

    A no-op (returns 0) for mlx_dspark-served models, which never run mlx_vlm's APC at
    all (see prefix_cache_cost_multiplier for that backend's own, different cache cost
    instead -- the two are mutually exclusive per model, never both nonzero).

    This is ONE pool SHARED across every request/conversation the server sees, not one
    per concurrent slot and not one per cached-conversation slot -- unlike bytes_per_
    growing_token/fixed_bytes, callers must NOT scale this by concurrency.

    Confirmed lazy allocation (real memory grows only as the cache actually fills, per
    §13's own research into mlx_vlm/apc.py) means this is a worst-case ceiling, not a
    guaranteed cost -- and that same §13 research flags this ceiling as "never
    approached, let alone tested past" in this project's own real use. Priced in as a
    safety margin against that documented worst case, not a reproduction of an observed
    crash the way DSPARK_DEFAULT_PREFIX_CACHE_SLOTS's own live-measured gap was."""
    if model_entry.get("serving", {}).get("backend") == "mlx_dspark":
        return 0
    blocks = MLX_VLM_APC_NUM_BLOCKS if apc_num_blocks is None else max(0, apc_num_blocks)
    return blocks * MLX_VLM_APC_BLOCK_SIZE * bytes_per_growing_token


def compute_kv_formula(kv_config, kv_bits=UNQUANTIZED_KV_BITS, kv_group_size=None):
    """Returns (bytes_per_growing_token, fixed_kv_bytes_total, detail_dict) or None if
    kv_config is missing fields we need. See docs/HARDWARE_SPEED_ESTIMATION.md §9 — this
    generalizes both the Qwen3.5 hybrid linear/full-attention case and the Gemma-4
    sliding/full-attention case with one formula: only 'full_attention' layers grow
    with context; 'sliding_attention' layers cost a FIXED amount (sliding_window
    tokens' worth, confirmed via gemma-4's own layer_types + sliding_window fields);
    'linear_attention' layers (Qwen3.5) are approximated as ~0 fixed cost — a real,
    documented approximation, not a solved calculation (their recurrent state is small
    but non-zero and not precisely modeled here).

    Also supports Gemma-4's per-layer KV geometry (confirmed in mlx_vlm/models/gemma4/
    language.py):
    - num_kv_shared_layers: trailing layers that reuse earlier layers' KV caches and
      allocate 0 new cache entries (make_cache slices layer_types[:num_layers -
      num_kv_shared_layers]).
    - global_head_dim / num_global_key_value_heads: full_attention layers can use a
      distinct head_dim (e.g. 512 vs 256) and KV head count (e.g. 1-4 vs 8-16) from
      sliding_attention layers.
    - kv_group_size: when quantized (kv_bits < 16) and kv_group_size is provided (e.g.
      DEFAULT_KV_GROUP_SIZE=64), accounts for the 4 bytes of FP16 scale+bias per group
      in MLX's affine group quantization (bytes_per_elem = kv_bits/8 + 4/kv_group_size).

    kv_config comes from models.json's per-model "kv_config" field, extracted once
    ahead of time from each model's HF config.json (see extraction done when models.json
    was generated) rather than read from disk here — this must work identically on a
    freshly-downloaded model AND on a machine where nothing has been downloaded yet."""
    if kv_config is None:
        return None
    num_kv_heads = kv_config.get("num_key_value_heads")
    head_dim = kv_config.get("head_dim")
    num_layers = kv_config.get("num_hidden_layers")
    if not all([num_kv_heads, head_dim, num_layers]):
        return None

    bytes_per_elem = kv_bits / 8
    if kv_group_size and kv_bits < UNQUANTIZED_KV_BITS:
        bytes_per_elem += 4.0 / kv_group_size

    global_head_dim = kv_config.get("global_head_dim") or head_dim
    global_kv_heads = kv_config.get("num_global_key_value_heads") or num_kv_heads
    num_kv_shared_layers = kv_config.get("num_kv_shared_layers") or 0

    layer_types = kv_config.get("layer_types")
    sliding_window = kv_config.get("sliding_window")

    if layer_types:
        active_layer_types = layer_types
        if 0 < num_kv_shared_layers <= len(layer_types):
            active_layer_types = layer_types[:len(layer_types) - num_kv_shared_layers]
        growing = sum(1 for t in active_layer_types if t == "full_attention")
        sliding = sum(1 for t in active_layer_types if t == "sliding_attention")
        linear = sum(1 for t in active_layer_types if t == "linear_attention")
        accounted = growing + sliding + linear
        unknown = len(active_layer_types) - accounted
        # Unrecognized layer-type name: treat as growing (full attention) — the safer
        # overestimate of memory need rather than silently underestimating it.
        growing += unknown
    else:
        growing = max(0, num_layers - num_kv_shared_layers)
        sliding, linear = 0, 0

    bytes_per_growing_token = 2 * growing * global_kv_heads * global_head_dim * bytes_per_elem
    fixed_bytes = 0.0
    if sliding and sliding_window:
        fixed_bytes += 2 * sliding * num_kv_heads * head_dim * bytes_per_elem * sliding_window

    detail = {
        "growing_layers": growing, "sliding_layers": sliding, "linear_layers": linear,
        "num_kv_heads": num_kv_heads, "head_dim": head_dim,
        "global_kv_heads": global_kv_heads, "global_head_dim": global_head_dim,
        "num_kv_shared_layers": num_kv_shared_layers,
        "sliding_window": sliding_window,
        "kv_bits": kv_bits, "kv_group_size": kv_group_size,
    }
    return bytes_per_growing_token, fixed_bytes, detail


def compute_drafter_kv_formula(model_entry, kv_bits=UNQUANTIZED_KV_BITS, kv_group_size=None):
    """Returns (drafter_bytes_per_growing_token, drafter_fixed_kv_bytes) for a model with
    a speculative-decoding drafter whose KV config is recorded in
    serving.drafter_kv_config, or (0.0, 0.0) when absent.

    While effective_size_gb() accounts for the drafter's static weights
    (serving.drafter_size_gb), a standalone drafter model (DFlash or MTP with unshared
    layers) also allocates its own KV cache across the conversation context alongside
    the target model."""
    drafter_kv_config = model_entry.get("serving", {}).get("drafter_kv_config")
    if not drafter_kv_config:
        return 0.0, 0.0
    formula = compute_kv_formula(drafter_kv_config, kv_bits=kv_bits, kv_group_size=kv_group_size)
    if formula is None:
        return 0.0, 0.0
    return formula[0], formula[1]


def recommend_max_context(model_entry, free_wired_gb, kv_bits=UNQUANTIZED_KV_BITS,
                           safety_buffer_gb=SAFETY_BUFFER_GB, concurrency=1,
                           max_context_tokens=None, prefix_cache_slots=None,
                           apc_num_blocks=None):
    """Returns (recommended_tokens_or_None, reason_string, detail_dict_or_None).

    concurrency>1 sizes context for N identically-sized concurrent slots instead of
    one -- inflating the per-token/fixed KV costs by N before the fits check is
    mathematically the same as dividing the memory budget by N, and yields the
    per-slot context size such that N copies of it all fit. Used for an explicit
    concurrency request (config.json's "concurrency"); the default auto-detected
    bonus-concurrency path (see compute_bonus_concurrency) instead calls this with
    concurrency=1 (unchanged, context-maximizing behavior) and separately checks
    whether tier-rounding leftover fits extra same-size slots for free.

    max_context_tokens (from config.json's "max_context_tokens" via the
    --max-context-tokens flag) is an OPTIONAL upper bound applied BEFORE tier
    rounding -- deliberately a cap, not an override: the memory-fit math still runs
    unchanged and can only ever shrink the result further, so a cap can never
    produce a context the host can't actually afford (unlike the launcher's
    fixed-model CLI form, whose 3rd positional arg bypasses this check entirely).
    None (the default) means no cap and today's behavior exactly.

    prefix_cache_slots: see prefix_cache_cost_multiplier -- a no-op for every backend
    except mlx_dspark, where it prices in that backend's always-on, by-default-
    unbounded prefix-cache LRU alongside the live generation's own KV cost BEFORE the
    fits check, instead of discovering the gap live via memory-guard sheds/swap.

    apc_num_blocks: optional override for MLX_VLM_APC_NUM_BLOCKS (e.g.
    MLX_VLM_APC_CONSTRAINED_NUM_BLOCKS when scaling down the APC pool on a
    memory-constrained machine)."""
    available_gb = free_wired_gb - effective_size_gb(model_entry) - safety_buffer_gb
    if available_gb <= 0:
        return None, "insufficient_memory_for_model_weights", None

    kv_group_size = DEFAULT_KV_GROUP_SIZE if kv_bits < UNQUANTIZED_KV_BITS else None
    formula = compute_kv_formula(model_entry.get("kv_config"), kv_bits, kv_group_size=kv_group_size)
    if formula is None:
        return None, "architecture_fields_missing_from_config", None
    target_bytes_per_growing_token, fixed_bytes, detail = formula

    drafter_growing, drafter_fixed = compute_drafter_kv_formula(
        model_entry, kv_bits=kv_bits, kv_group_size=kv_group_size,
    )
    raw_bytes_per_growing_token = target_bytes_per_growing_token + drafter_growing
    fixed_bytes += drafter_fixed
    detail["drafter_bytes_per_growing_token"] = drafter_growing
    detail["drafter_fixed_bytes"] = drafter_fixed

    bytes_per_growing_token = raw_bytes_per_growing_token * concurrency
    fixed_bytes *= concurrency

    cache_multiplier = prefix_cache_cost_multiplier(model_entry, prefix_cache_slots)
    bytes_per_growing_token *= cache_multiplier
    fixed_bytes *= cache_multiplier
    detail["prefix_cache_cost_multiplier"] = cache_multiplier

    # mlx_vlm's own APC pool (see mlx_vlm_apc_pool_bytes) is priced in off the RAW,
    # pre-concurrency/pre-cache_multiplier rate -- it's one pool shared by the whole
    # server process, not one per concurrent slot or per mlx-dspark-style cached slot
    # (and mutually exclusive with cache_multiplier above: never both nonzero for the
    # same model, since each is gated to the opposite backend).
    apc_pool_bytes = mlx_vlm_apc_pool_bytes(
        model_entry, raw_bytes_per_growing_token, apc_num_blocks=apc_num_blocks,
    )
    fixed_bytes += apc_pool_bytes
    detail["mlx_vlm_apc_pool_bytes"] = apc_pool_bytes
    detail["apc_num_blocks"] = (
        0 if model_entry.get("serving", {}).get("backend") == "mlx_dspark"
        else (MLX_VLM_APC_NUM_BLOCKS if apc_num_blocks is None else max(0, apc_num_blocks))
    )

    available_bytes = available_gb * 1024**3
    usable_bytes = available_bytes - fixed_bytes
    if usable_bytes <= 0:
        return None, "insufficient_memory_for_fixed_kv_cost", detail

    if bytes_per_growing_token <= 0:
        max_ctx = model_entry["max_context_architectural"]
    else:
        max_ctx = int(usable_bytes / bytes_per_growing_token)
        max_ctx = min(max_ctx, model_entry["max_context_architectural"])
    if max_context_tokens is not None:
        max_ctx = min(max_ctx, max_context_tokens)

    if max_ctx < MIN_USABLE_CONTEXT:
        return None, "computed_context_too_small_to_be_useful", detail

    # Round DOWN to a clean, conservative tier rather than showing a falsely-precise
    # number like "58,392 tokens" — this is an exact calculation but the inputs
    # (free memory especially) fluctuate, so a clean round number is more honest about
    # the real precision available.
    for tier in (262144, 131072, 65536, 32768, 16384, 8192, 4096):
        if max_ctx >= tier:
            return tier, "computed", detail
    return max_ctx, "computed", detail


def _has_sliding_window_layers(kv_config):
    """True when this model's kv_config lists any sliding_attention layer -- used by
    recommend_kv_bits_and_context below to flag a real, upstream-confirmed gap (see
    its docstring), not to redo compute_kv_formula's own byte-budget math."""
    if not kv_config:
        return False
    return any(t == "sliding_attention" for t in kv_config.get("layer_types") or [])


def _dspark_supports_kv_bits(model_entry):
    """True unless `model_entry` is served through `mlx_dspark` on a target that
    `mlx_dspark` routes to `mlx_vlm` (Gemma-4 / sliding-window-attention targets), where
    `mlx_dspark.target.Target.__init__` raises `ValueError: --kv-bits is supported for
    mlx-lm text targets only` whenever `--kv-bits` is non-zero (confirmed live during
    `gemma-4-12b-it-4bit-dflash` sweep)."""
    if not isinstance(model_entry, dict):
        return True
    serving = model_entry.get("serving") or {}
    if serving.get("backend") != "mlx_dspark":
        return True
    if _has_sliding_window_layers(model_entry.get("kv_config")):
        return False
    hf = (model_entry.get("huggingface") or "").lower()
    if "gemma" in hf:
        return False
    return True


def _apply_preload(ctx, reason, preload_overhead):
    """Shared by both kv_bits branches of recommend_kv_bits_and_context below — applies
    the same tool-preload check build_recommendations always did, so both the
    unquantized and quantized attempts are held to the identical usability bar."""
    fits = ctx is not None
    context_after_preload = None
    if fits and preload_overhead:
        context_after_preload = ctx - preload_overhead
        if context_after_preload < MIN_USABLE_CONTEXT:
            fits = False
            reason = "insufficient_context_after_tool_preload"
    return fits, reason, context_after_preload


def recommend_kv_bits_and_context(model_entry, free_wired_gb, preload_overhead=0,
                                   safety_buffer_gb=SAFETY_BUFFER_GB, concurrency=1,
                                   max_context_tokens=None, prefix_cache_slots=None):
    """Decides whether to quantize the KV cache, the APC pool block ceiling, and the
    resulting context size in one pass — see docs/HARDWARE_SPEED_ESTIMATION.md §14 for
    the full reasoning and its measured basis (§10/§12).

    1. Try unquantized FP16 KV first (kv_bits=None downstream means "omit --kv-bits").
       If unquantized reaches at least COMFORTABLE_UNQUANTIZED_CONTEXT (32,768) or the
       model's/config's own max context ceiling, return it immediately for full decode
       speed.
    2. If unquantized is cramped at the 16k tier (< COMFORTABLE_UNQUANTIZED_CONTEXT)
       despite the model/config supporting more, OR if unquantized doesn't fit at all,
       evaluate 8-bit KV (MEMORY_CONSTRAINED_KV_BITS):
       - When unquantized fit at 16k, upgrade to 8-bit KV only if 8-bit unlocks a
         strictly larger context tier AND the model has no sliding-window layers (so we
         never voluntarily trade a working unquantized cache for the mlx-vlm #1559
         SWA+APC caveat).
       - When unquantized didn't fit at all, accept 8-bit KV if it fits (surfacing
         kv_bits_apc_caveat when SWA layers are present).
    3. If even 8-bit KV fails to fit with the full 4,096-block (65,536-token) mlx_vlm
       APC pool, retry 8-bit KV with MLX_VLM_APC_CONSTRAINED_NUM_BLOCKS (1,024 blocks =
       16,384 cached tokens) so memory-constrained Macs can still run the model with a
       smaller prefix cache instead of being blocked by the 65k APC reservation.

    Returns dict: kv_bits (None or MEMORY_CONSTRAINED_KV_BITS), recommended_context,
    context_reason, kv_formula_detail, context_after_preload, fits, kv_bits_apc_caveat,
    apc_num_blocks.
    """
    default_apc_blocks = (
        None if model_entry.get("serving", {}).get("backend") == "mlx_dspark"
        else MLX_VLM_APC_NUM_BLOCKS
    )
    has_swa = _has_sliding_window_layers(model_entry.get("kv_config"))
    effective_cap = model_entry["max_context_architectural"]
    if max_context_tokens is not None:
        effective_cap = min(effective_cap, max_context_tokens)

    ctx_u, reason_u, detail_u = recommend_max_context(
        model_entry, free_wired_gb, kv_bits=UNQUANTIZED_KV_BITS,
        safety_buffer_gb=safety_buffer_gb, concurrency=concurrency,
        max_context_tokens=max_context_tokens, prefix_cache_slots=prefix_cache_slots,
    )
    fits_u, reason_u, after_u = _apply_preload(ctx_u, reason_u, preload_overhead)
    if not _dspark_supports_kv_bits(model_entry):
        return {
            "kv_bits": None, "recommended_context": ctx_u, "context_reason": reason_u,
            "kv_formula_detail": detail_u, "context_after_preload": after_u, "fits": fits_u,
            "kv_bits_apc_caveat": False, "apc_num_blocks": default_apc_blocks,
        }
    if fits_u and (
        ctx_u >= COMFORTABLE_UNQUANTIZED_CONTEXT
        or ctx_u >= effective_cap
        or has_swa
    ):
        return {
            "kv_bits": None, "recommended_context": ctx_u, "context_reason": reason_u,
            "kv_formula_detail": detail_u, "context_after_preload": after_u, "fits": True,
            "kv_bits_apc_caveat": False, "apc_num_blocks": default_apc_blocks,
        }

    ctx_q, reason_q, detail_q = recommend_max_context(
        model_entry, free_wired_gb, kv_bits=MEMORY_CONSTRAINED_KV_BITS,
        safety_buffer_gb=safety_buffer_gb, concurrency=concurrency,
        max_context_tokens=max_context_tokens, prefix_cache_slots=prefix_cache_slots,
    )
    fits_q, reason_q, after_q = _apply_preload(ctx_q, reason_q, preload_overhead)

    if fits_u:
        # Unquantized fit at < 32k (16k tier) on a non-SWA model whose cap allows more:
        # upgrade to 8-bit KV if it unlocks a strictly larger context tier.
        if fits_q and ctx_q > ctx_u:
            return {
                "kv_bits": MEMORY_CONSTRAINED_KV_BITS, "recommended_context": ctx_q,
                "context_reason": reason_q, "kv_formula_detail": detail_q,
                "context_after_preload": after_q, "fits": True,
                "kv_bits_apc_caveat": False, "apc_num_blocks": default_apc_blocks,
            }
        return {
            "kv_bits": None, "recommended_context": ctx_u, "context_reason": reason_u,
            "kv_formula_detail": detail_u, "context_after_preload": after_u, "fits": True,
            "kv_bits_apc_caveat": False, "apc_num_blocks": default_apc_blocks,
        }

    if fits_q:
        return {
            "kv_bits": MEMORY_CONSTRAINED_KV_BITS, "recommended_context": ctx_q,
            "context_reason": reason_q, "kv_formula_detail": detail_q,
            "context_after_preload": after_q, "fits": True,
            "kv_bits_apc_caveat": has_swa,
            "apc_num_blocks": default_apc_blocks,
        }

    # Dynamic APC pool scaling for mlx_vlm models: if 8-bit KV failed because the
    # default 4,096-block (65,536-token) APC pool reservation ate the remaining memory
    # budget, retry with MLX_VLM_APC_CONSTRAINED_NUM_BLOCKS (1,024 blocks = 16,384
    # tokens).
    if (
        default_apc_blocks is not None
        and reason_q in (
            "insufficient_memory_for_fixed_kv_cost",
            "computed_context_too_small_to_be_useful",
            "insufficient_context_after_tool_preload",
        )
    ):
        ctx_c, reason_c, detail_c = recommend_max_context(
            model_entry, free_wired_gb, kv_bits=MEMORY_CONSTRAINED_KV_BITS,
            safety_buffer_gb=safety_buffer_gb, concurrency=concurrency,
            max_context_tokens=max_context_tokens, prefix_cache_slots=prefix_cache_slots,
            apc_num_blocks=MLX_VLM_APC_CONSTRAINED_NUM_BLOCKS,
        )
        fits_c, reason_c, after_c = _apply_preload(ctx_c, reason_c, preload_overhead)
        if fits_c:
            return {
                "kv_bits": MEMORY_CONSTRAINED_KV_BITS, "recommended_context": ctx_c,
                "context_reason": reason_c, "kv_formula_detail": detail_c,
                "context_after_preload": after_c, "fits": True,
                "kv_bits_apc_caveat": has_swa,
                "apc_num_blocks": MLX_VLM_APC_CONSTRAINED_NUM_BLOCKS,
            }

    # Neither fits — report the quantized attempt's reason, since it's the more
    # forgiving of the two (smaller bytes-per-token) and so the more informative "even
    # our best effort doesn't fit" explanation for the user.
    return {
        "kv_bits": None, "recommended_context": ctx_q, "context_reason": reason_q,
        "kv_formula_detail": detail_q, "context_after_preload": after_q, "fits": False,
        "kv_bits_apc_caveat": False, "apc_num_blocks": default_apc_blocks,
    }
