"""bushpilot — hardware-aware local model advisor.

Flat compatibility namespace. This package used to be one 1,743-line
model_advisor.py script; the root-level model_advisor.py is now a thin shim
over this package (kept so every start_mlx_server.sh/launch_tool.sh call site,
the release bundle, and `import model_advisor` in the test suite all keep
working unchanged). Each submodule below owns one concern:

  hardware    chip/GPU/memory detection
  kv          context-size + KV-cache-quantization math (pure, no imports)
  speed       token/s estimation (physics formula + real calibration override)
  prefill     --prefill-step-size recommendation
  calibration per-(chip, model) real-measurement cache under ~/.bushpilot/
  preload     Claude Code/OpenCode/etc. context preload-overhead estimation
  recommend   combines the above into one row per model (build_recommendations)
  picker      row model (cells), sort state machine (sort), rendering (display)
  cli         flag parsing + main()

This __init__ re-exports the public surface so `import bushpilot` (or the
model_advisor.py shim's `from bushpilot import *`) sees every name at the top
level, matching the pre-split flat shape the test suite and bash scripts
expect. Add new names to BOTH their owning submodule and the import/__all__
lists below, or they will silently not reach the shim.
"""
from .calibration import CALIBRATION_DIR, _calibration_path, load_calibration, save_calibration
from .cli import MODELS_JSON, _flag_value, emit_selection_json, main
from .hardware import (
    DEFAULT_METAL_WIRED_LIMIT_FRACTION,
    _cap_free_by_metal_limit,
    get_chip_info,
    get_current_wired_gb,
    get_metal_wired_limit_gb,
    log,
)
from .kv import (
    COMFORTABLE_UNQUANTIZED_CONTEXT,
    DEFAULT_KV_GROUP_SIZE,
    DSPARK_DEFAULT_PREFIX_CACHE_SLOTS,
    MEMORY_CONSTRAINED_KV_BITS,
    MIN_USABLE_CONTEXT,
    MLX_VLM_APC_BLOCK_SIZE,
    MLX_VLM_APC_CONSTRAINED_NUM_BLOCKS,
    MLX_VLM_APC_NUM_BLOCKS,
    SAFETY_BUFFER_GB,
    UNQUANTIZED_KV_BITS,
    compute_drafter_kv_formula,
    compute_kv_formula,
    effective_size_gb,
    mlx_vlm_apc_pool_bytes,
    prefix_cache_cost_multiplier,
    recommend_kv_bits_and_context,
    recommend_max_context,
)
from .picker.cells import (
    COL_CTX,
    COL_DRAFTER,
    COL_ENGINE,
    COL_INPUT,
    COL_KV,
    COL_LAST,
    COL_NAME,
    COL_RELEASED,
    COL_SIZE,
    COL_SPEED,
    COL_TOOLS,
    COL_TYPE,
    COMPACT_HEADERS,
    COMPACT_RIGHT_ALIGNED,
    DRAFTER_ID_SUFFIXES,
    DRAFTER_LABELS,
    OPTIONAL_COLUMN_DROP_ORDER,
    _truncate_cell,
    cap_name_width,
    compact_row_cells,
    compact_row_widths,
    format_context_k,
    model_display_name,
    model_drafter_label,
    model_engine,
    model_input_label,
    model_tools_label,
    visible_columns,
)
from .picker.display import (
    BOLD,
    DIM,
    GREEN,
    RESET,
    YELLOW,
    describe_unavailable_reason,
    format_compact_header_row,
    format_model_body_lines,
    format_model_compact_row,
    print_list_compact,
    print_menu,
    print_preload_banner,
    short_unavailable_reason,
    use_color,
)
from .picker.sort import (
    COLUMN_SORT_KEYS,
    DESCENDING_SORT_KEYS,
    SORT_KEYS,
    next_sort,
    sort_key_for_click,
    sort_rows,
)
from .prefill import (
    DEFAULT_PREFILL_STEP_SIZE,
    PREFILL_STEP_SIZE_MEMORY_MARGIN,
    recommend_prefill_step_size,
)
from .preload import (
    MCP_SERVER_TOKENS_ESTIMATE,
    TOOL_BASELINE_PRELOAD_TOKENS,
    count_tokens_rough,
    estimate_preload_overhead,
    measure_doc_preload,
    measure_mcp_servers,
    measure_skills_preload,
)
from .recommend import (
    AUTO_MAX_CONCURRENCY,
    MIN_SPEED_TOK_S,
    build_recommendations,
    compute_bonus_concurrency,
)
from .speed import CHIP_BANDWIDTH_GB_S, SPEED_CORRECTION_FACTOR, estimate_speed_tok_s
from .template_patches import check_and_patch_chat_template, find_chat_template_path

__all__ = [
    "AUTO_MAX_CONCURRENCY", "BOLD", "CALIBRATION_DIR", "CHIP_BANDWIDTH_GB_S",
    "COL_CTX", "COL_DRAFTER", "COL_ENGINE", "COL_INPUT", "COL_KV", "COL_LAST",
    "COL_NAME", "COL_RELEASED", "COL_SIZE", "COL_SPEED", "COL_TOOLS", "COL_TYPE",
    "COLUMN_SORT_KEYS", "COMFORTABLE_UNQUANTIZED_CONTEXT", "COMPACT_HEADERS",
    "COMPACT_RIGHT_ALIGNED", "DEFAULT_KV_GROUP_SIZE",
    "DEFAULT_METAL_WIRED_LIMIT_FRACTION", "DEFAULT_PREFILL_STEP_SIZE",
    "DESCENDING_SORT_KEYS", "DIM", "DRAFTER_ID_SUFFIXES", "DRAFTER_LABELS",
    "DSPARK_DEFAULT_PREFIX_CACHE_SLOTS", "GREEN", "MCP_SERVER_TOKENS_ESTIMATE",
    "MEMORY_CONSTRAINED_KV_BITS", "MIN_SPEED_TOK_S", "MIN_USABLE_CONTEXT",
    "MLX_VLM_APC_BLOCK_SIZE", "MLX_VLM_APC_CONSTRAINED_NUM_BLOCKS",
    "MLX_VLM_APC_NUM_BLOCKS", "MODELS_JSON", "OPTIONAL_COLUMN_DROP_ORDER",
    "PREFILL_STEP_SIZE_MEMORY_MARGIN", "RESET", "SAFETY_BUFFER_GB", "SORT_KEYS",
    "SPEED_CORRECTION_FACTOR", "TOOL_BASELINE_PRELOAD_TOKENS", "UNQUANTIZED_KV_BITS",
    "YELLOW", "_calibration_path", "_cap_free_by_metal_limit", "_flag_value",
    "_truncate_cell", "cap_name_width", "check_and_patch_chat_template",
    "compact_row_cells", "compact_row_widths", "compute_bonus_concurrency",
    "compute_drafter_kv_formula", "compute_kv_formula", "count_tokens_rough",
    "describe_unavailable_reason", "effective_size_gb", "emit_selection_json",
    "estimate_preload_overhead", "estimate_speed_tok_s", "find_chat_template_path",
    "format_compact_header_row", "format_context_k", "format_model_body_lines",
    "format_model_compact_row", "get_chip_info", "get_current_wired_gb",
    "get_metal_wired_limit_gb", "load_calibration", "log", "main",
    "measure_doc_preload", "measure_mcp_servers", "measure_skills_preload",
    "mlx_vlm_apc_pool_bytes", "model_display_name", "model_drafter_label",
    "model_engine", "model_input_label", "model_tools_label", "next_sort",
    "prefix_cache_cost_multiplier", "print_list_compact", "print_menu",
    "print_preload_banner", "build_recommendations", "recommend_kv_bits_and_context",
    "recommend_max_context", "recommend_prefill_step_size", "save_calibration",
    "short_unavailable_reason", "sort_key_for_click", "sort_rows", "use_color",
    "visible_columns",
]
