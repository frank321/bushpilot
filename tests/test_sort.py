"""Tests for the picker's sort machinery: next_sort's state machine (first open vs.
re-clicking vs. resize-triggered reload) and sort_rows' tested/untested/doesn't-fit
grouping.
"""
import model_advisor as ma

# --- next_sort: the state machine behind click-to-sort and resize-safe reloads -----

def test_next_sort_defaults_to_released_ascending_with_no_prior_state(tmp_path):
    state_path = tmp_path / "sort.state"
    key, reverse = ma.next_sort("released", str(state_path))
    assert (key, reverse) == ("released", False)
    assert state_path.read_text() == "released 0"


def test_next_sort_none_with_no_prior_state_also_defaults_to_released(tmp_path):
    # None means "keep whatever's active" -- with no prior state at all (e.g. the very
    # first render), that falls back to the same default a fresh "released" request
    # would give.
    state_path = tmp_path / "sort.state"
    key, reverse = ma.next_sort(None, str(state_path))
    assert (key, reverse) == ("released", False)


def test_next_sort_requesting_the_same_key_twice_flips_direction(tmp_path):
    state_path = tmp_path / "sort.state"
    ma.next_sort("speed", str(state_path))
    key, reverse = ma.next_sort("speed", str(state_path))
    assert (key, reverse) == ("speed", True)
    key, reverse = ma.next_sort("speed", str(state_path))
    assert (key, reverse) == ("speed", False)  # third click flips back


def test_next_sort_requesting_a_different_key_resets_direction(tmp_path):
    state_path = tmp_path / "sort.state"
    ma.next_sort("speed", str(state_path))
    ma.next_sort("speed", str(state_path))  # now speed/reversed
    key, reverse = ma.next_sort("size", str(state_path))
    assert (key, reverse) == ("size", False)  # new key always starts unreversed


def test_next_sort_none_preserves_whatever_is_currently_active(tmp_path):
    # This is the fix that makes a terminal resize (which re-renders the list with no
    # explicit --sort/--sort-col flag at all) not silently reset the user's chosen
    # sort back to the default -- see start_mlx_server.sh's `resize` binding.
    state_path = tmp_path / "sort.state"
    ma.next_sort("speed", str(state_path))
    ma.next_sort("speed", str(state_path))  # speed, reversed
    key, reverse = ma.next_sort(None, str(state_path))
    assert (key, reverse) == ("speed", True)


def test_next_sort_none_does_not_itself_flip_direction(tmp_path):
    # Preserving != re-requesting: asking for "whatever's active" repeatedly must not
    # look like clicking the same header twice.
    state_path = tmp_path / "sort.state"
    ma.next_sort("size", str(state_path))
    ma.next_sort(None, str(state_path))
    key, reverse = ma.next_sort(None, str(state_path))
    assert (key, reverse) == ("size", False)


def test_next_sort_without_a_state_path_still_returns_a_sort(tmp_path):
    # sort_state is None in some callers (e.g. a one-off --list-compact invocation
    # with no --sort-state flag) -- must not crash just because there's nowhere to
    # persist to.
    assert ma.next_sort("size", None) == ("size", False)
    assert ma.next_sort(None, None) == ("released", False)


def test_next_sort_corrupt_state_file_falls_back_to_default(tmp_path):
    state_path = tmp_path / "sort.state"
    state_path.write_text("garbage-not-a-valid-state-line")
    key, reverse = ma.next_sort(None, str(state_path))
    assert (key, reverse) == ("released", False)


# --- sort_rows: tested/untested/doesn't-fit grouping, independent of sort key -------

def _row(id_, status, fits):
    return {
        "model": {"id": id_, "status": status, "size_gb": 1.0, "released": "2026-01",
                  "type": "dense"},
        "fits": fits, "recommended_context": 1000 if fits else None,
        "estimated_tok_s": 10.0, "kv_bits": None, "concurrency": 1,
    }


def test_sort_rows_groups_tested_fitting_before_untested_fitting_before_non_fitting():
    rows = [
        _row("c-non-fitting", "tested", fits=False),
        _row("b-untested-fitting", "untested", fits=True),
        _row("a-tested-fitting", "tested", fits=True),
    ]
    sorted_rows, _ = ma.sort_rows(rows, "name")
    assert [r["model"]["id"] for r in sorted_rows] == [
        "a-tested-fitting", "b-untested-fitting", "c-non-fitting",
    ]


def test_sort_rows_group_order_is_unaffected_by_reverse(sample_rows):
    # moe_unknown doesn't fit; dense/sliding do -- reversing the sort must not pull
    # the non-fitting model ahead of the fitting ones.
    sorted_rows, _ = ma.sort_rows(sample_rows, "name", reverse=True)
    assert sorted_rows[-1]["model"]["id"] == "test-moe-unknown-active"


def test_sort_rows_within_group_orders_by_size_descending_by_default(sample_rows):
    sorted_rows, descending = ma.sort_rows(sample_rows, "size")
    assert descending is True
    fitting_ids = [r["model"]["id"] for r in sorted_rows if r["fits"]]
    assert fitting_ids == ["test-dense-4bit", "test-sliding-4bit"]  # 6.3GB before 4.5GB


def test_sort_rows_reverse_flips_within_group_order(sample_rows):
    sorted_rows, descending = ma.sort_rows(sample_rows, "size", reverse=True)
    assert descending is False
    fitting_ids = [r["model"]["id"] for r in sorted_rows if r["fits"]]
    assert fitting_ids == ["test-sliding-4bit", "test-dense-4bit"]  # 4.5GB before 6.3GB
