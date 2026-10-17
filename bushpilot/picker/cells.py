"""One picker row's cell contents and column layout -- compact_row_cells, column
widths, which columns fit a given terminal width, and name truncation. The
--list-compact table's row model."""
import glob
import os

from ..kv import UNQUANTIZED_KV_BITS


def is_model_cached(m):
    """True when the model's HuggingFace repo already has a downloaded snapshot in
    HF_HOME (defaults to ~/.bushpilot/models). Pure local filesystem check."""
    hf_repo = m.get("huggingface")
    if not hf_repo or "/" not in hf_repo:
        return False
    hf_home = os.path.expanduser(os.environ.get("HF_HOME", "~/.bushpilot/models"))
    repo_dir = f"models--{hf_repo.replace('/', '--')}"
    pattern = os.path.join(hf_home, "hub", repo_dir, "snapshots", "*", "config.json")
    return bool(glob.glob(pattern))


def format_context_k(tokens):
    """Decimal-K notation matching Phase 6's locked mockup (131072 -> '131K'), not
    binary-K -- a colloquial round number, not a precise one."""
    return f"{round(tokens / 1000)}K"


DRAFTER_LABELS = {"dflash": "DFlash", "mtp": "MTP", "eagle3": "EAGLE3"}
DRAFTER_ID_SUFFIXES = ("-dflash-vlm", "-dflash", "-mtp")


def model_engine(m):
    """Inference engine that serves this entry. Entries without a serving block run on
    plain mlx_vlm.server (see models.json's _schema_notes.serving)."""
    return (m.get("serving") or {}).get("backend", "mlx_vlm")


def model_drafter_label(m):
    """Speculative-decoding method of the attached drafter ('DFlash', 'MTP', ...), or '—'
    when the entry has no drafter. mlx_dspark entries carry it in `mode`, mlx_vlm entries
    in `draft_kind`."""
    sv = m.get("serving") or {}
    if not (sv.get("drafter") or sv.get("draft_model")):
        return "—"
    kind = sv.get("draft_kind") or sv.get("mode") or "?"
    return DRAFTER_LABELS.get(kind, kind)


def model_display_name(m):
    """The id minus its engine/drafter suffix, since those have their own columns. Display
    only -- the full id stays the unique key for selection, config.json and the CLI."""
    if model_drafter_label(m) == "—":
        return m["id"]
    for suffix in DRAFTER_ID_SUFFIXES:
        if m["id"].endswith(suffix):
            return m["id"][: -len(suffix)]
    return m["id"]


def model_input_label(m):
    """Input modalities this entry actually serves, e.g. 'text+img+vid'. The model's own
    `modalities` are narrowed to text when the serving variant drops the vision tower
    (serving.multimodal_capable false -- every mlx_dspark entry)."""
    if (m.get("serving") or {}).get("multimodal_capable") is False:
        mods = {"text"}
    else:
        mods = set(m.get("modalities") or ["text"])
    return "+".join(["text"] + [lbl for key, lbl in (("image", "img"), ("video", "vid")) if key in mods])


def model_tools_label(m):
    """Whether the server returns structured tool_calls for this model, per its recorded
    tool_calling test ('tools' / 'no tools'); 'tools?' when it has never been tested."""
    verdict = (m.get("tool_calling") or {}).get("verdict")
    return {"SUPPORTED": "tools", "NOT_SUPPORTED": "no tools"}.get(verdict, "tools?")


def compact_row_cells(r, reason=None):
    """Plain-text cells of one --list-compact row, in display order: name (with a leading
    tested/untested marker), type, engine, drafter, input, tools, released, size, context,
    KV-cache precision, speed, and a last column (concurrency, or why the model doesn't fit when `reason` is
    given)."""
    m = r["model"]
    if r["fits"]:
        ctx_str = f"{format_context_k(r['recommended_context'])} tok"
        kv_str = f"{r['kv_bits']}-bit" if r["kv_bits"] else f"{UNQUANTIZED_KV_BITS}-bit"
    else:
        ctx_str = kv_str = "—"
    mark = "○" if m.get("status") == "untested" else "✓"
    if r["estimated_tok_s"] is not None:
        prefix = "" if r.get("speed_source") == "calibrated" else "~"
        speed = f"{prefix}{round(r['estimated_tok_s'], 1)} tok/s"
    else:
        speed = "speed unknown" if r["fits"] else "—"
    if reason:
        notes = reason
    else:
        notes = f"×{r['concurrency']} concurrent" if r["concurrency"] > 1 else "1 concurrent"
        if is_model_cached(m):
            notes += " · cached"
    return [
        f"{mark} {model_display_name(m)}",
        "MoE" if m.get("type") == "moe" else "Dense",
        model_engine(m),
        model_drafter_label(m),
        model_input_label(m),
        model_tools_label(m),
        m.get("released") or "—",
        f"{m['size_gb']:g}GB",
        ctx_str,
        kv_str,
        speed,
        notes,
    ]


# Column indexes in compact_row_cells. Numeric columns right-align, text columns
# left-align; the last column is never padded.
COL_NAME, COL_TYPE, COL_ENGINE, COL_DRAFTER, COL_INPUT, COL_TOOLS, COL_RELEASED, \
    COL_SIZE, COL_CTX, COL_KV, COL_SPEED, COL_LAST = range(12)
COMPACT_RIGHT_ALIGNED = {COL_SIZE, COL_SPEED}

# Dropped first, in this order, on terminals too narrow to show every column (see
# visible_columns below). Name/Size/Context/KV/Speed are what the fit-and-speed decision
# is actually made from and are never dropped; everything in this list is still shown in
# full in the --detail preview pane, so narrowing the list loses nothing, just layout.
OPTIONAL_COLUMN_DROP_ORDER = [COL_DRAFTER, COL_INPUT, COL_TOOLS, COL_RELEASED, COL_ENGINE, COL_TYPE]


# Header words double as --sort values (fzf's click-header passes the clicked word), so
# each is a single lowercase-able word that SORT_KEYS below accepts. The first cell is
# indented past the "✓ " tested marker so "Name" sits over the model names.
COMPACT_HEADERS = ["  Name", "Type", "Engine", "Drafter", "Input", "Tools", "Released",
                   "Size", "Context", "KV", "Speed", "Notes"]


def compact_row_widths(rows):
    """Max width of each padded column across `rows`, never narrower than its header plus
    room for the " ▼" sort arrow, so the columns don't shift when the sort changes."""
    cells = [compact_row_cells(r) for r in rows]
    return tuple(max([len(COMPACT_HEADERS[k]) + 2] + [len(c[k]) for c in cells]) for k in range(COL_LAST))


def visible_columns(widths, term_width):
    """Which of the COL_NAME..COL_SPEED columns fit side by side in term_width columns,
    as an ordered list of column indexes (Notes is handled separately -- see
    format_model_compact_row -- since it's unpadded and gets truncated instead of
    dropped). `term_width` of None means "no limit" (today's behavior: every column).

    Columns are never reordered, only removed, one at a time from the back of
    OPTIONAL_COLUMN_DROP_ORDER, so the picker degrades gracefully as the window shrinks
    rather than jumping straight to a bare Name/Size/Context/KV/Speed table."""
    cols = list(range(COL_LAST))
    if term_width is None:
        return cols

    def total_width(cs):
        return sum(widths[c] for c in cs) + 2 * max(0, len(cs) - 1)

    for victim in OPTIONAL_COLUMN_DROP_ORDER:
        if total_width(cols) <= term_width:
            break
        if victim in cols:
            cols.remove(victim)
    return cols


def cap_name_width(widths, cols, term_width, min_name_width=12):
    """Even with every optional column dropped, Name/Size/Context/KV/Speed can still
    overflow a narrow terminal -- Name is sized to the single longest model id in the
    whole catalog, not to what's actually on screen. Name is the last remaining column
    with any real slack (Size/Context/KV/Speed are short, fixed-shape strings), so once
    the other visible columns' widths are pinned down, whatever's left over becomes
    Name's budget; format_model_compact_row truncates individual cells to it with an
    ellipsis. Never shrinks below min_name_width -- past that point the name is useless
    and the row is left to overflow rather than mangled further (the full id is always
    one keystroke away in the --detail preview)."""
    if term_width is None or COL_NAME not in cols:
        return widths
    # The gap count matches format_model_compact_row's `2 * len(padded)` exactly: one
    # "  " between every pair of visible columns, plus one more before Notes (even when
    # Notes ends up empty) -- len(cols) gaps in total, not len(cols) - 1. The header row
    # always shows the literal "Notes" label (never truncated, never empty like a data
    # row's can be), so its width is reserved here too -- otherwise the header alone
    # would run len("Notes") past term_width even when every data row fits.
    others = sum(widths[c] for c in cols if c != COL_NAME) + 2 * len(cols) + len(COMPACT_HEADERS[COL_LAST])
    budget = term_width - others
    if budget >= widths[COL_NAME] or budget < min_name_width:
        return widths
    widths = list(widths)
    widths[COL_NAME] = budget
    return tuple(widths)


def _truncate_cell(text, width):
    """Shorten `text` to `width` characters with a trailing ellipsis if it's longer --
    used where a column's width was capped to the terminal rather than to content (see
    cap_name_width), since plain ljust/rjust only pad, they never truncate."""
    if width is None or len(text) <= width:
        return text
    if width <= 1:
        return text[:width]
    return text[: width - 1] + "…"
