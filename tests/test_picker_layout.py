"""Tests for the --list-compact picker's terminal-width-aware column layout:
compact_row_widths, visible_columns, cap_name_width, _truncate_cell,
format_compact_header_row, format_model_compact_row, sort_key_for_click.

This is the exact code that was debugged live across several iterations in the
session that added this file (an off-by-two in the gap-width math, a color-highlight
index bug after switching to variable-width columns, rows silently overflowing a
narrow terminal) -- every one of those would have been caught immediately by the
"no rendered line exceeds term_width" property test below. That's the single most
important invariant in this file; the rest are supporting unit tests for the pieces
that invariant is built from.
"""
import pytest

import model_advisor as ma

# use_color is read at call time by format_compact_header_row/format_model_compact_row,
# both defined in bushpilot/picker/display.py -- same reasoning as calibration.py's
# CALIBRATION_DIR in test_recommendation_math.py: patch it where it's actually looked up.
from bushpilot.picker import display as _display

# --- compact_row_widths ---------------------------------------------------------------

def test_compact_row_widths_matches_hand_verified_values(sample_rows):
    # Verified once against the live function output when this test was written (see
    # this file's module docstring) -- not re-derived by eyeballing compact_row_cells.
    widths = ma.compact_row_widths(sample_rows)
    assert widths == (25, 6, 8, 9, 12, 8, 10, 6, 9, 6, 11)


def test_compact_row_widths_never_narrower_than_header_plus_arrow_room(sample_rows):
    widths = ma.compact_row_widths(sample_rows)
    for k, width in enumerate(widths):
        assert width >= len(ma.COMPACT_HEADERS[k]) + 2


# --- visible_columns: the optional-column drop order --------------------------------

def test_visible_columns_keeps_everything_when_term_width_is_none(sample_rows):
    widths = ma.compact_row_widths(sample_rows)
    assert ma.visible_columns(widths, None) == list(range(ma.COL_LAST))


def test_visible_columns_keeps_everything_when_it_already_fits(sample_rows):
    widths = ma.compact_row_widths(sample_rows)
    assert ma.visible_columns(widths, 500) == list(range(ma.COL_LAST))


def test_visible_columns_drops_optional_columns_in_documented_order(sample_rows):
    widths = ma.compact_row_widths(sample_rows)
    # Breakpoints below were read off the real function output for this fixture's
    # widths (25, 6, 8, 9, 12, 8, 10, 6, 9, 6, 11) -- each one point narrower than
    # where the previous optional column got dropped.
    assert ma.visible_columns(widths, 129) == [0, 1, 2, 4, 5, 6, 7, 8, 9, 10]       # no Drafter
    assert ma.visible_columns(widths, 118) == [0, 1, 2, 5, 6, 7, 8, 9, 10]          # no Drafter/Input
    assert ma.visible_columns(widths, 97) == [0, 1, 2, 6, 7, 8, 9, 10]             # no .../Tools
    assert ma.visible_columns(widths, 85) == [0, 1, 2, 7, 8, 9, 10]                # no .../Released
    assert ma.visible_columns(widths, 73) == [0, 1, 7, 8, 9, 10]                   # no .../Engine
    assert ma.visible_columns(widths, 72) == [0, 7, 8, 9, 10]                      # no .../Type


def test_visible_columns_never_drops_the_essential_columns(sample_rows):
    widths = ma.compact_row_widths(sample_rows)
    essential = {ma.COL_NAME, ma.COL_SIZE, ma.COL_CTX, ma.COL_KV, ma.COL_SPEED}
    # Even at an absurdly narrow width, Name/Size/Context/KV/Speed are never dropped --
    # only truncated/overflowed (see cap_name_width), because they're what the
    # fit-and-speed decision is actually made from.
    cols = ma.visible_columns(widths, 1)
    assert set(cols) == essential


def test_visible_columns_drop_order_matches_the_documented_constant(sample_rows):
    # The drop order itself is a module-level constant precisely so it's visible and
    # testable independent of any one fixture's specific widths.
    assert ma.OPTIONAL_COLUMN_DROP_ORDER == [
        ma.COL_DRAFTER, ma.COL_INPUT, ma.COL_TOOLS, ma.COL_RELEASED, ma.COL_ENGINE, ma.COL_TYPE,
    ]


# --- cap_name_width: the last-resort lever once every optional column is gone -------

def test_cap_name_width_noop_when_term_width_is_none(sample_rows):
    widths = ma.compact_row_widths(sample_rows)
    cols = list(range(ma.COL_LAST))
    assert ma.cap_name_width(widths, cols, None) == widths


def test_cap_name_width_shrinks_name_even_without_dropping_any_column(sample_rows):
    # At term_width=130 every column still technically fits (total width is exactly
    # 130), but the header's literal "Notes" text wasn't accounted for yet -- Name
    # still gets capped down from 25 to make room for it.
    widths = ma.compact_row_widths(sample_rows)
    cols = ma.visible_columns(widths, 130)
    assert cols == list(range(ma.COL_LAST))
    capped = ma.cap_name_width(widths, cols, 130)
    assert capped[ma.COL_NAME] == 18
    assert all(capped[k] == widths[k] for k in cols if k != ma.COL_NAME)


def test_cap_name_width_respects_the_floor(sample_rows):
    # Once every optional column is already gone, squeezing further would require
    # shrinking Name below min_name_width -- the function refuses and leaves the
    # original (overflowing) width instead of producing an unreadable name.
    widths = ma.compact_row_widths(sample_rows)
    cols = ma.visible_columns(widths, 20)
    assert set(cols) == {ma.COL_NAME, ma.COL_SIZE, ma.COL_CTX, ma.COL_KV, ma.COL_SPEED}
    capped = ma.cap_name_width(widths, cols, 20)
    assert capped[ma.COL_NAME] == widths[ma.COL_NAME]  # unchanged, not forced below the floor


def test_cap_name_width_noop_if_name_column_already_dropped():
    # Defensive case: cap_name_width should be a no-op if COL_NAME somehow isn't in
    # cols at all (never happens via visible_columns, but the function shouldn't
    # assume its caller never passes a cols list without it).
    widths = (25, 6, 8, 9, 12, 8, 10, 6, 9, 6, 11)
    cols = [ma.COL_SIZE, ma.COL_CTX, ma.COL_KV, ma.COL_SPEED]
    assert ma.cap_name_width(widths, cols, 40) == widths


# --- _truncate_cell --------------------------------------------------------------------

def test_truncate_cell_leaves_short_text_untouched():
    assert ma._truncate_cell("short", 20) == "short"


def test_truncate_cell_passthrough_when_width_is_none():
    assert ma._truncate_cell("a very long piece of text indeed", None) == "a very long piece of text indeed"


def test_truncate_cell_adds_ellipsis_when_too_long():
    result = ma._truncate_cell("a-very-long-model-name-indeed", 10)
    assert result == "a-very-lo…"
    assert len(result) == 10


def test_truncate_cell_width_one_has_no_room_for_ellipsis():
    assert ma._truncate_cell("abcdef", 1) == "a"


# --- end-to-end rendering: the invariant that matters most --------------------------


@pytest.mark.parametrize("term_width", [20, 40, 54, 60, 72, 73, 85, 97, 118, 129, 130, 160, 250])
def test_rendered_rows_never_exceed_term_width_except_at_the_documented_floor(sample_rows, term_width):
    """The property that was actually broken (three separate times) in the session
    that added this test: compute widths/cols/caps the same way print_list_compact
    does, render every row and the header, and assert nothing runs past term_width --
    UNLESS cap_name_width's own documented floor kicked in (min_name_width=12), in
    which case a bounded, known amount of overflow (exactly the uncapped Name width)
    is the deliberately-accepted trade-off, not a silent regression."""
    widths = ma.compact_row_widths(sample_rows)
    cols = ma.visible_columns(widths, term_width)
    capped = ma.cap_name_width(widths, cols, term_width)
    at_the_floor = capped[ma.COL_NAME] == widths[ma.COL_NAME] and capped[ma.COL_NAME] > 12

    header_line = ma.format_compact_header_row(capped, cols, sort_key="released").split("\t")[1]
    assert len(header_line) <= term_width or at_the_floor

    for r in sample_rows:
        reason = None if r["fits"] else ma.short_unavailable_reason(r, 24, None)
        row_line = ma.format_model_compact_row(r, capped, cols, reason, term_width=term_width).split("\t")[1]
        assert len(row_line) <= term_width or at_the_floor


def test_notes_column_truncated_with_ellipsis_when_room_is_tight(sample_rows):
    moe_row = next(r for r in sample_rows if r["model"]["id"] == "test-moe-unknown-active")
    reason = ma.short_unavailable_reason(moe_row, 24, None)
    assert reason == "needs ~53 GB free, you have 24 GB"  # the long reason this test needs

    widths = ma.compact_row_widths(sample_rows)
    cols = ma.visible_columns(widths, 60)
    capped = ma.cap_name_width(widths, cols, 60)
    line = ma.format_model_compact_row(moe_row, capped, cols, reason, term_width=60).split("\t")[1]
    assert len(line) <= 60
    assert line.endswith("…") or reason not in line  # truncated, not silently dropped-then-overflowed


def test_notes_column_empty_when_no_room_left_at_all(sample_rows):
    dense_row = next(r for r in sample_rows if r["model"]["id"] == "test-dense-4bit")
    widths = ma.compact_row_widths(sample_rows)
    cols = ma.visible_columns(widths, 60)
    capped = ma.cap_name_width(widths, cols, 60)
    # dense_row fits, so its Notes text is "1 concurrent"/"×N concurrent". The padded
    # columns + the mandatory "  " gap before Notes stay the same length regardless of
    # whether Notes itself is present -- so the width that leaves exactly zero room for
    # Notes is the full (untruncated-Notes) line's length minus just the Notes text.
    full_line = ma.format_model_compact_row(dense_row, capped, cols, None, term_width=None).split("\t")[1]
    notes_text = "×4 concurrent"
    assert full_line.endswith(notes_text)
    fixed_only_width = len(full_line) - len(notes_text)
    line = ma.format_model_compact_row(dense_row, capped, cols, None, term_width=fixed_only_width).split("\t")[1]
    assert len(line) == fixed_only_width
    assert line.endswith("  ")  # notes collapsed to "", but the join's "  " separator stays


# --- sort_key_for_click: mapping a header click back to a sort key -------------------

def test_sort_key_for_click_maps_to_the_column_under_the_click(sample_rows):
    widths = ma.compact_row_widths(sample_rows)
    cols = list(range(ma.COL_LAST))  # every column visible, matches a wide terminal
    # Char index 1 is the first character of "  Name" itself.
    assert ma.sort_key_for_click(1, widths, cols) == "name"
    # Somewhere inside the Type column (after Name's 25-char width + 2-char gap).
    assert ma.sort_key_for_click(25 + 2 + 1, widths, cols) == "type"


def test_sort_key_for_click_in_the_gap_between_columns_counts_toward_the_left_one(sample_rows):
    widths = ma.compact_row_widths(sample_rows)
    cols = list(range(ma.COL_LAST))
    # Clicking exactly in the 2-space gap right after Name (chars 26-27, 1-indexed)
    # should still resolve to "name", not jump ahead to "type".
    assert ma.sort_key_for_click(26, widths, cols) == "name"
    assert ma.sort_key_for_click(27, widths, cols) == "name"


def test_sort_key_for_click_on_the_notes_column_is_not_sortable(sample_rows):
    widths = ma.compact_row_widths(sample_rows)
    cols = list(range(ma.COL_LAST))
    far_right = sum(widths[c] for c in cols) + 2 * len(cols) + 1
    assert ma.sort_key_for_click(far_right, widths, cols) is None


def test_sort_key_for_click_skips_columns_dropped_for_width(sample_rows):
    # With Drafter/Input/Tools/Released/Engine/Type all dropped (narrow terminal),
    # clicking where Type/Engine *would* have been (if they were still visible)
    # should resolve to whatever's now actually there -- not silently misattribute
    # the click to a column that isn't on screen.
    widths = ma.compact_row_widths(sample_rows)
    cols = ma.visible_columns(widths, 72)  # [Name, Size, Context, KV, Speed] only
    assert cols == [ma.COL_NAME, ma.COL_SIZE, ma.COL_CTX, ma.COL_KV, ma.COL_SPEED]
    capped = ma.cap_name_width(widths, cols, 72)
    # First char of the second visible column (Size), not of the long-gone Type column.
    click_at_size = capped[ma.COL_NAME] + 2 + 1
    assert ma.sort_key_for_click(click_at_size, capped, cols) == "size"


# --- color highlighting: keyed by original column index, not position in `cols` ----
# (this is exactly the bug introduced -- and caught by hand, not by any test -- when
# compact rows moved from always-eleven-fixed-columns to a variable-width `cols`
# subset; see format_model_compact_row's own comment on this)

def test_color_highlights_context_and_kv_green_when_unquantized(dense_model, monkeypatch):
    monkeypatch.setattr(_display, "use_color", lambda force=False: True)
    rows = ma.build_recommendations([dense_model], free_wired_gb=24, chip_name="Apple M3 Max", gpu_cores=40)
    widths = ma.compact_row_widths(rows)
    cols = list(range(ma.COL_LAST))
    line = ma.format_model_compact_row(rows[0], widths, cols, None)
    assert f"{ma.GREEN}262K tok" in line
    assert f"{ma.GREEN}16-bit{ma.RESET}" in line
    assert f"{ma.DIM}mlx_vlm" in line  # Engine is always dimmed, fit or not


def test_color_highlights_context_and_kv_yellow_when_kv_quantized(dense_model, monkeypatch):
    monkeypatch.setattr(_display, "use_color", lambda force=False: True)
    # free_wired_gb=9.5 is the verified fallback-to-8bit-KV case from
    # test_recommendation_math.py.
    rows = ma.build_recommendations([dense_model], free_wired_gb=9.5, chip_name="Apple M3 Max", gpu_cores=40)
    widths = ma.compact_row_widths(rows)
    cols = list(range(ma.COL_LAST))
    line = ma.format_model_compact_row(rows[0], widths, cols, None)
    assert f"{ma.YELLOW}8-bit{ma.RESET}" in line
    assert ma.GREEN not in line


def test_color_dims_the_entire_line_when_model_does_not_fit(dense_model, monkeypatch):
    monkeypatch.setattr(_display, "use_color", lambda force=False: True)
    rows = ma.build_recommendations([dense_model], free_wired_gb=1, chip_name="Apple M3 Max", gpu_cores=40)
    widths = ma.compact_row_widths(rows)
    cols = list(range(ma.COL_LAST))
    reason = ma.short_unavailable_reason(rows[0], 1, None)
    line = ma.format_model_compact_row(rows[0], widths, cols, reason)
    displayed = line.split("\t")[1]
    assert displayed.startswith(ma.DIM)
    assert displayed.endswith(ma.RESET)


def test_color_highlighting_does_not_crash_when_context_kv_engine_are_dropped(dense_model, monkeypatch):
    # The actual regression: COL_CTX/COL_KV/COL_ENGINE may not be in `cols` at all on
    # a narrow terminal -- highlighting must skip them gracefully, not KeyError.
    monkeypatch.setattr(_display, "use_color", lambda force=False: True)
    rows = ma.build_recommendations([dense_model], free_wired_gb=24, chip_name="Apple M3 Max", gpu_cores=40)
    widths = ma.compact_row_widths(rows)
    narrow_cols = [ma.COL_NAME, ma.COL_SIZE, ma.COL_CTX, ma.COL_KV, ma.COL_SPEED]
    line = ma.format_model_compact_row(rows[0], widths, narrow_cols, None)  # must not raise
    assert ma.DIM not in line.split("\t")[1].replace(f"{ma.DIM}×4 concurrent{ma.RESET}", "")
    assert f"{ma.GREEN}262K tok" in line


# --- use_color: force=True (--color) must not depend on stderr's own isatty status ----
# (the real bug: start_mlx_server.sh's fzf reload/bind commands all redirect this
# process's stderr to /dev/null to swallow system_profiler's unrelated noise, which
# makes sys.stderr.isatty() False on every single reload, on every machine -- not just
# a user's one-off non-tty invocation. --color/force=True is the fix: it must restore
# color even when stderr is provably not a terminal.)

def test_use_color_auto_detect_follows_stderr_isatty_when_not_forced(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(ma.sys.stderr, "isatty", lambda: False)
    assert ma.use_color() is False
    monkeypatch.setattr(ma.sys.stderr, "isatty", lambda: True)
    assert ma.use_color() is True


def test_use_color_force_true_wins_even_when_stderr_is_not_a_tty(monkeypatch):
    # This is exactly start_mlx_server.sh's reload-bind scenario: stderr redirected to
    # /dev/null, so isatty() is False, but --color was passed because the caller knows
    # the output is headed into fzf's --ansi screen regardless.
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(ma.sys.stderr, "isatty", lambda: False)
    assert ma.use_color(force=True) is True


def test_use_color_no_color_env_wins_even_when_forced(monkeypatch):
    # NO_COLOR is an explicit user accessibility opt-out, not a detection heuristic --
    # --color must never override it.
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr(ma.sys.stderr, "isatty", lambda: True)
    assert ma.use_color(force=True) is False
    assert ma.use_color() is False

