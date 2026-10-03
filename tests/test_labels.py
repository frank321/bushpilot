"""Tests for the small per-model display-label derivation functions: model_engine,
model_drafter_label, model_display_name, model_input_label, model_tools_label.

These are pure string/dict lookups with no hardware dependency -- every input here is
a hand-built model_entry dict, not something loaded from models.json, so the catalog
can change shape without these tests needing to track it."""
import model_advisor as ma


def test_model_engine_defaults_to_mlx_vlm_without_a_serving_block():
    assert ma.model_engine({"id": "plain-no-serving"}) == "mlx_vlm"


def test_model_engine_reads_serving_backend():
    assert ma.model_engine({"serving": {"backend": "mlx_dspark"}}) == "mlx_dspark"


def test_model_drafter_label_dash_when_no_drafter():
    assert ma.model_drafter_label({"id": "plain-model-4bit"}) == "—"
    assert ma.model_drafter_label({"serving": {"backend": "mlx_vlm"}}) == "—"


def test_model_drafter_label_known_kinds_mapped_to_display_names():
    mtp = {"serving": {"draft_model": "x", "draft_kind": "mtp"}}
    dflash = {"serving": {"drafter": "x", "mode": "dflash"}}
    eagle3 = {"serving": {"draft_model": "x", "draft_kind": "eagle3"}}
    assert ma.model_drafter_label(mtp) == "MTP"
    assert ma.model_drafter_label(dflash) == "DFlash"
    assert ma.model_drafter_label(eagle3) == "EAGLE3"


def test_model_drafter_label_unknown_kind_passes_through_verbatim():
    unknown = {"serving": {"draft_model": "x", "draft_kind": "zzz"}}
    assert ma.model_drafter_label(unknown) == "zzz"


def test_model_drafter_label_mlx_dspark_reads_mode_not_draft_kind():
    # mlx_dspark entries carry the method in `mode`, mlx_vlm entries in `draft_kind` --
    # confirm the fallback order doesn't silently prefer the wrong one.
    m = {"serving": {"drafter": "x", "mode": "dflash", "draft_kind": None}}
    assert ma.model_drafter_label(m) == "DFlash"


def test_model_display_name_unchanged_when_no_drafter():
    m = {"id": "plain-model-4bit"}
    assert ma.model_display_name(m) == "plain-model-4bit"


def test_model_display_name_strips_known_drafter_suffix():
    m = {"id": "test-model-mtp", "serving": {"draft_model": "x", "draft_kind": "mtp"}}
    assert ma.model_display_name(m) == "test-model"

    m2 = {"id": "test-model-dflash", "serving": {"drafter": "x", "mode": "dflash"}}
    assert ma.model_display_name(m2) == "test-model"

    m3 = {"id": "test-model-dflash-vlm", "serving": {"drafter": "x", "mode": "dflash"}}
    assert ma.model_display_name(m3) == "test-model"


def test_model_display_name_keeps_full_id_when_suffix_unrecognized():
    # Has a drafter, but the id doesn't end in any of DRAFTER_ID_SUFFIXES -- falls back
    # to the full id rather than guessing where to cut.
    m = {"id": "test-model-speculative", "serving": {"draft_model": "x", "draft_kind": "mtp"}}
    assert ma.model_display_name(m) == "test-model-speculative"


def test_model_input_label_text_only_default():
    assert ma.model_input_label({}) == "text"
    assert ma.model_input_label({"modalities": ["text"]}) == "text"


def test_model_input_label_adds_img_and_vid_in_order():
    m = {"modalities": ["text", "image", "video"]}
    assert ma.model_input_label(m) == "text+img+vid"


def test_model_input_label_image_only():
    m = {"modalities": ["text", "image"]}
    assert ma.model_input_label(m) == "text+img"


def test_model_input_label_narrowed_to_text_when_serving_drops_vision():
    # Every mlx_dspark entry: serving.multimodal_capable=False overrides the model's
    # own (broader) modalities list -- the serving variant actually running doesn't
    # have the vision tower attached.
    m = {"modalities": ["text", "image", "video"], "serving": {"multimodal_capable": False}}
    assert ma.model_input_label(m) == "text"


def test_model_tools_label_supported():
    assert ma.model_tools_label({"tool_calling": {"verdict": "SUPPORTED"}}) == "tools"


def test_model_tools_label_not_supported():
    assert ma.model_tools_label({"tool_calling": {"verdict": "NOT_SUPPORTED"}}) == "no tools"


def test_model_tools_label_untested_or_missing_is_unknown():
    assert ma.model_tools_label({}) == "tools?"
    assert ma.model_tools_label({"tool_calling": {}}) == "tools?"
    assert ma.model_tools_label({"tool_calling": {"verdict": "WEIRD"}}) == "tools?"
