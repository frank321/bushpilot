"""Sort-key state machine for --list-compact: fzf header clicks map to a sort key
(sort_key_for_click), applying it orders the rows (sort_rows), and the active sort
persists across reloads (next_sort)."""
import os

from ..kv import UNQUANTIZED_KV_BITS
from .cells import (
    COL_LAST, COMPACT_HEADERS, compact_row_cells, model_display_name, model_drafter_label,
    model_engine, model_tools_label,
)


SORT_KEYS = ("released", "speed", "size", "name", "type", "engine", "drafter", "input",
             "tools", "context", "kv")

# Sort key for each compact_row_cells column (None = not sortable), for header clicks.
COLUMN_SORT_KEYS = ("name", "type", "engine", "drafter", "input", "tools", "released",
                    "size", "context", "kv", "speed", None)


def sort_key_for_click(char_index, widths, cols):
    """Map fzf's FZF_CLICK_HEADER_COLUMN (1-based character index into the displayed
    header line, calibrated against fzf 0.74: index 1 is the first character of the
    header text, indent excluded) to the sort key of the column under it. Clicks in the
    2-space gap between columns count toward the column on their left. `cols` is the
    same visible-column subset the header was actually rendered with (see
    visible_columns) -- columns dropped for width were never on screen to click."""
    start = 1
    chosen = None
    for k in list(cols) + [COL_LAST]:
        if char_index >= start:
            chosen = COLUMN_SORT_KEYS[k]
        width = widths[k] if k < COL_LAST else len(COMPACT_HEADERS[COL_LAST])
        start += width + 2
    return chosen


# Keys whose natural order is high-to-low / newest-first; the rest are A-Z.
DESCENDING_SORT_KEYS = {"released", "speed", "size", "context", "kv", "input", "tools"}


def sort_rows(rows, sort_key, reverse=False):
    """Tested-and-fitting first, then untested-and-fitting, then the models that don't
    fit -- under every sort, so a fast untested model never outranks a tested one --
    and `sort_key` orders within each group. Numeric/date keys (released, speed, size,
    context, kv) put the biggest or newest first with missing values last; the rest sort
    A-Z; `reverse` flips that within each group (the groups themselves never move). Ties
    keep a model's variants together by name. Stable sorts, innermost first. Returns
    (rows, descending) where `descending` is the direction actually applied."""
    def display(r):
        m = r["model"]
        return (model_display_name(m), model_engine(m), model_drafter_label(m))

    keys = {
        "released": lambda r: r["model"].get("released") or "",
        "speed": lambda r: r["estimated_tok_s"] if r["estimated_tok_s"] is not None else -1,
        "size": lambda r: r["model"]["size_gb"],
        "context": lambda r: r["recommended_context"] or 0,
        "kv": lambda r: (r["kv_bits"] or UNQUANTIZED_KV_BITS) if r["fits"] else 0,
        # more modalities first; "tools" first, then untested, then no tools
        "input": lambda r: len(compact_row_cells(r)[4]),
        "tools": lambda r: {"tools": 2, "tools?": 1, "no tools": 0}[model_tools_label(r["model"])],
        "type": lambda r: r["model"].get("type", "dense"),
        "engine": lambda r: model_engine(r["model"]),
        "drafter": lambda r: model_drafter_label(r["model"]),
        "name": display,
    }
    descending = (sort_key in DESCENDING_SORT_KEYS) != reverse
    rows = sorted(rows, key=display)
    rows.sort(key=keys[sort_key], reverse=descending)
    rows.sort(key=lambda r: (not r["fits"], r["model"].get("status") == "untested"))
    return rows, descending


def next_sort(requested_key, state_path):
    """Pick the sort for this render and remember it in `state_path` ("key reversed").
    Asking for the key the list is already sorted by flips its direction -- clicking the
    same header twice reverses it -- while a different key starts in its natural order.
    requested_key None (a click on a non-sortable spot) keeps the current sort as is."""
    prev_key, prev_rev = None, False
    if state_path:
        try:
            with open(state_path) as f:
                prev_key, flag = f.read().split()
                prev_rev = flag == "1"
        except (OSError, ValueError):
            pass
    if requested_key is None:
        key, rev = (prev_key, prev_rev) if prev_key in SORT_KEYS else ("released", False)
    elif requested_key == prev_key:
        key, rev = requested_key, not prev_rev
    else:
        key, rev = requested_key, False
    if state_path:
        try:
            with open(state_path, "w") as f:
                f.write(f"{key} {int(rev)}")
        except OSError:
            pass
    return key, rev
