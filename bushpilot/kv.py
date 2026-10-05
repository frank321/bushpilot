"""Context-size recommendation: KV-cache byte math and the policy for how much
context (and whether to quantize the KV cache) a model gets, given free memory.
Pure arithmetic -- no imports, no I/O."""


MEMORY_CONSTRAINED_KV_BITS = 8   # only used as a FALLBACK now — see recommend_kv_bits_and_context
UNQUANTIZED_KV_BITS = 16        # what MLX uses when --kv-bits is simply not passed (float16 KV)
SAFETY_BUFFER_GB = 3.0       # headroom kept free beyond model+KV, for OS/background apps/spikes


MIN_USABLE_CONTEXT = 12288   # below this, a context window isn't really usable — also used
                              # as the floor for CONVERSATION HEADROOM after tool preload
                              # (see estimate_preload_overhead below): a model that only
                              # "fits" on paper but leaves less than this after Claude
                              # Code's/OpenCode's own preload isn't usable either.


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


def compute_kv_formula(kv_config, kv_bits=UNQUANTIZED_KV_BITS):
    """Returns (bytes_per_growing_token, fixed_kv_bytes_total, detail_dict) or None if
    kv_config is missing fields we need. See docs/HARDWARE_SPEED_ESTIMATION.md §9 — this
    generalizes both the Qwen3.5 hybrid linear/full-attention case and the Gemma-4
    sliding/full-attention case with one formula: only 'full_attention' layers grow
    with context; 'sliding_attention' layers cost a FIXED amount (sliding_window
    tokens' worth, confirmed via gemma-4's own layer_types + sliding_window fields);
    'linear_attention' layers (Qwen3.5) are approximated as ~0 fixed cost — a real,
    documented approximation, not a solved calculation (their recurrent state is small
    but non-zero and not precisely modeled here).

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
    layer_types = kv_config.get("layer_types")
    sliding_window = kv_config.get("sliding_window")

    if layer_types:
        growing = sum(1 for t in layer_types if t == "full_attention")
        sliding = sum(1 for t in layer_types if t == "sliding_attention")
        linear = sum(1 for t in layer_types if t == "linear_attention")
        accounted = growing + sliding + linear
        unknown = len(layer_types) - accounted
        # Unrecognized layer-type name: treat as growing (full attention) — the safer
        # overestimate of memory need rather than silently underestimating it.
        growing += unknown
    else:
        growing, sliding, linear = num_layers, 0, 0

    bytes_per_growing_token = 2 * growing * num_kv_heads * head_dim * bytes_per_elem
    fixed_bytes = 0.0
    if sliding and sliding_window:
        fixed_bytes += 2 * sliding * num_kv_heads * head_dim * bytes_per_elem * sliding_window

    detail = {
        "growing_layers": growing, "sliding_layers": sliding, "linear_layers": linear,
        "num_kv_heads": num_kv_heads, "head_dim": head_dim, "sliding_window": sliding_window,
        "kv_bits": kv_bits,
    }
    return bytes_per_growing_token, fixed_bytes, detail


def recommend_max_context(model_entry, free_wired_gb, kv_bits=UNQUANTIZED_KV_BITS,
                           safety_buffer_gb=SAFETY_BUFFER_GB, concurrency=1,
                           max_context_tokens=None):
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
    None (the default) means no cap and today's behavior exactly."""
    available_gb = free_wired_gb - effective_size_gb(model_entry) - safety_buffer_gb
    if available_gb <= 0:
        return None, "insufficient_memory_for_model_weights", None

    formula = compute_kv_formula(model_entry.get("kv_config"), kv_bits)
    if formula is None:
        return None, "architecture_fields_missing_from_config", None
    bytes_per_growing_token, fixed_bytes, detail = formula
    bytes_per_growing_token *= concurrency
    fixed_bytes *= concurrency

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
                                   max_context_tokens=None):
    """Decides BOTH whether to quantize the KV cache AND the resulting context size, in
    one pass — see docs/HARDWARE_SPEED_ESTIMATION.md §14 for the full reasoning and its
    measured basis (§10/§12).

    KV-cache quantization (--kv-bits 8) has a real decode-speed cost (up to 8.7% on
    fast/small models, confirmed live) and confirmed ZERO memory benefit at short
    context (peak_memory was byte-identical with/without it in every short-context
    benchmark run for this project) — its only real purpose is letting a BIGGER
    context fit in the SAME free memory. So: try unquantized first (kv_bits=None in the
    return value means "omit --kv-bits entirely" downstream); only fall back to
    kv_bits=8 if unquantized genuinely doesn't fit. This is a binary fits/doesn't-fit
    decision at whatever context size the model would otherwise get, not a search for
    the context-maximizing bit-width — see §14's own caveat for why this is grounded in
    real data but that data's scope (short context only) doesn't fully cover the regime
    (long context) this decision matters most for.

    kv_bits_apc_caveat (in the returned dict) is True when the quantized fallback is
    actually used AND the model has sliding-window (SWA) layers -- flagging a real,
    already-filed upstream gap (mlx-vlm issue #1559, "APC + kv-bits full batch/SWA
    coverage", confirmed/p1, open since Jul 2026, smoke-tested on gemma-4-e2b-it-4bit):
    the server's own APC_ENABLED=1 is unconditional in this project, and upstream has
    confirmed that combining --kv-bits with APC on the batched/concurrent request path
    means SWA layer caches aren't clonable for an exact store, so sliding layers likely
    won't get cached on that path until it's fixed. Degrades gracefully per upstream
    ("prefix reuse still works" per their smoke note) -- NOT corrupted output -- so this
    is surfaced as a caveat for the picker to show (see format_model_body_lines), not a
    reason to refuse the fallback outright; refusing it would make some models that
    could otherwise run look like they don't fit at all, over a bug that degrades
    rather than breaks. Re-check mlx-vlm#1559's status before removing this flag.

    Returns dict: kv_bits (None or MEMORY_CONSTRAINED_KV_BITS), recommended_context,
    context_reason, kv_formula_detail, context_after_preload, fits, kv_bits_apc_caveat.
    """
    ctx_u, reason_u, detail_u = recommend_max_context(
        model_entry, free_wired_gb, kv_bits=UNQUANTIZED_KV_BITS,
        safety_buffer_gb=safety_buffer_gb, concurrency=concurrency,
        max_context_tokens=max_context_tokens,
    )
    fits_u, reason_u, after_u = _apply_preload(ctx_u, reason_u, preload_overhead)
    if fits_u:
        return {
            "kv_bits": None, "recommended_context": ctx_u, "context_reason": reason_u,
            "kv_formula_detail": detail_u, "context_after_preload": after_u, "fits": True,
            "kv_bits_apc_caveat": False,
        }

    ctx_q, reason_q, detail_q = recommend_max_context(
        model_entry, free_wired_gb, kv_bits=MEMORY_CONSTRAINED_KV_BITS,
        safety_buffer_gb=safety_buffer_gb, concurrency=concurrency,
        max_context_tokens=max_context_tokens,
    )
    fits_q, reason_q, after_q = _apply_preload(ctx_q, reason_q, preload_overhead)
    if fits_q:
        return {
            "kv_bits": MEMORY_CONSTRAINED_KV_BITS, "recommended_context": ctx_q,
            "context_reason": reason_q, "kv_formula_detail": detail_q,
            "context_after_preload": after_q, "fits": True,
            "kv_bits_apc_caveat": _has_sliding_window_layers(model_entry.get("kv_config")),
        }

    # Neither fits — report the quantized attempt's reason, since it's the more
    # forgiving of the two (smaller bytes-per-token) and so the more informative "even
    # our best effort doesn't fit" explanation for the user.
    return {
        "kv_bits": None, "recommended_context": ctx_q, "context_reason": reason_q,
        "kv_formula_detail": detail_q, "context_after_preload": after_q, "fits": False,
        "kv_bits_apc_caveat": False,
    }
