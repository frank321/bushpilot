"""Tests for bushpilot/template_patches.py's known Gemma-4 chat-template bug detector
and patcher -- see that module's docstring for the full root-cause writeup. All tests
use a real temp directory shaped like the HF hub cache (hub/models--org--name/
snapshots/<hash>/chat_template.jinja) rather than mocking the filesystem, since the
glob pattern itself (models--/--  substitution, snapshots/*/) is part of what's under
test."""
import os

import pytest

from bushpilot.template_patches import (
    _BUGGY_SNIPPET,
    _GENERATION_PROMPT_BUG_SIGNATURE,
    _LEGACY_PATCHED_REPLACEMENT,
    _PATCH_MARKER,
    _PATCHED_REPLACEMENT,
    check_and_patch_chat_template,
    find_chat_template_path,
)

REPO_ID = "mlx-community/gemma-4-12B-it-4bit"


def _make_snapshot_dir(hf_home, repo_id):
    safe_repo = repo_id.replace("/", "--")
    snapshot_dir = os.path.join(hf_home, "hub", f"models--{safe_repo}", "snapshots", "deadbeef")
    os.makedirs(snapshot_dir, exist_ok=True)
    return snapshot_dir


def _write_template(hf_home, repo_id, content):
    snapshot_dir = _make_snapshot_dir(hf_home, repo_id)
    path = os.path.join(snapshot_dir, "chat_template.jinja")
    with open(path, "w") as f:
        f.write(content)
    return path


def test_find_chat_template_path_returns_none_when_not_downloaded(tmp_path):
    assert find_chat_template_path(str(tmp_path), REPO_ID) is None


def test_find_chat_template_path_returns_none_without_standalone_template(tmp_path):
    # Model is downloaded (snapshot dir exists) but has no chat_template.jinja --
    # e.g. a repo that embeds its template in tokenizer_config.json instead.
    _make_snapshot_dir(str(tmp_path), REPO_ID)
    assert find_chat_template_path(str(tmp_path), REPO_ID) is None


def test_find_chat_template_path_finds_real_file(tmp_path):
    path = _write_template(str(tmp_path), REPO_ID, "irrelevant content")
    assert find_chat_template_path(str(tmp_path), REPO_ID) == path


def test_check_and_patch_no_template_when_not_downloaded(tmp_path):
    assert check_and_patch_chat_template(str(tmp_path), REPO_ID) == "no_template"


def test_check_and_patch_not_affected_for_unrelated_template(tmp_path):
    _write_template(str(tmp_path), REPO_ID, "{%- for message in messages -%}{{ message['content'] }}{%- endfor -%}")
    assert check_and_patch_chat_template(str(tmp_path), REPO_ID) == "not_affected"


def test_check_and_patch_not_affected_when_only_history_snippet_present(tmp_path):
    # Regression test for a real false positive caught before shipping: an earlier
    # version of the detector matched on _BUGGY_SNIPPET alone. gemma-4-e4b-it-4bit's
    # real cached template has this exact 4-line history-serialization block (it's
    # shared boilerplate across the whole Gemma-4 family) but its add_generation_prompt
    # never injects the asymmetric marker in the first place -- so there's nothing for
    # a fix to correctly reproduce, and patching it would introduce a NEW asymmetry
    # instead of removing one. _GENERATION_PROMPT_BUG_SIGNATURE absent must mean
    # "not_affected", never "patched", regardless of _BUGGY_SNIPPET's presence.
    content = (
        "{%- set role = 'model' -%}\n"
        + _BUGGY_SNIPPET
        + "{%- if add_generation_prompt -%}\n"
        "    {%- if ns.prev_message_type != 'tool_response' and ns.prev_message_type != 'tool_call' -%}\n"
        "        {{- '<|turn>model\\n' -}}\n"
        "    {%- endif -%}\n"
        "{%- endif -%}\n"
    )
    path = _write_template(str(tmp_path), REPO_ID, content)

    result = check_and_patch_chat_template(str(tmp_path), REPO_ID)

    assert result == "not_affected"
    with open(path) as f:
        assert f.read() == content  # byte-for-byte untouched


def test_check_and_patch_not_affected_when_only_generation_prompt_signature_present(tmp_path):
    # Symmetric defensive case: the generation-prompt signature alone, without the
    # unpatched history-serialization block, also must not trigger a patch (e.g. a
    # template already using some other history-serialization strategy that happens
    # to still inject the marker at generation time).
    content = "{%- set role = 'model' -%}\n" + _GENERATION_PROMPT_BUG_SIGNATURE
    path = _write_template(str(tmp_path), REPO_ID, content)

    result = check_and_patch_chat_template(str(tmp_path), REPO_ID)

    assert result == "not_affected"
    with open(path) as f:
        assert f.read() == content


def _buggy_template(prefix="", suffix=""):
    """A minimal template containing BOTH required signatures -- the real shape of
    gemma-4-12b-it-4bit's/gemma-4-26b-a4b-it-4bit's actual cached templates."""
    return (
        prefix
        + _BUGGY_SNIPPET
        + "\n{%- if add_generation_prompt -%}\n"
        "    {%- if ns.prev_message_type != 'tool_response' and ns.prev_message_type != 'tool_call' -%}\n"
        + _GENERATION_PROMPT_BUG_SIGNATURE
        + "    {%- endif -%}\n"
        "{%- endif -%}\n"
        + suffix
    )


def test_check_and_patch_fixes_known_buggy_snippet(tmp_path):
    path = _write_template(str(tmp_path), REPO_ID, _buggy_template())

    result = check_and_patch_chat_template(str(tmp_path), REPO_ID)

    assert result == "patched"
    with open(path) as f:
        patched_content = f.read()
    assert _PATCH_MARKER in patched_content
    # The buggy snippet's literal text is no longer present verbatim (it's been
    # replaced by the elif-extended version) -- confirms this isn't a no-op append.
    assert _BUGGY_SNIPPET not in patched_content
    # The real fix line is present, covers tool-calling turns too (no
    # `not message.get('tool_calls')` exclusion), and the generation-prompt signature
    # this fix mirrors is left completely untouched.
    assert _PATCHED_REPLACEMENT in patched_content
    assert "not message.get('tool_calls')" not in patched_content
    assert "{{- '<|channel>thought\\n<channel|>' -}}" in patched_content
    assert _GENERATION_PROMPT_BUG_SIGNATURE in patched_content


def test_check_and_patch_upgrades_legacy_2026_10_05_patch(tmp_path):
    legacy_template = _buggy_template().replace(_BUGGY_SNIPPET, _LEGACY_PATCHED_REPLACEMENT, 1)
    path = _write_template(str(tmp_path), REPO_ID, legacy_template)

    first = check_and_patch_chat_template(str(tmp_path), REPO_ID)
    second = check_and_patch_chat_template(str(tmp_path), REPO_ID)

    assert first == "patched"
    assert second == "already_patched"
    with open(path) as f:
        upgraded = f.read()
    assert _LEGACY_PATCHED_REPLACEMENT not in upgraded
    assert _PATCHED_REPLACEMENT in upgraded


def test_check_and_patch_is_idempotent(tmp_path):
    _write_template(str(tmp_path), REPO_ID, _buggy_template())

    first = check_and_patch_chat_template(str(tmp_path), REPO_ID)
    second = check_and_patch_chat_template(str(tmp_path), REPO_ID)

    assert first == "patched"
    assert second == "already_patched"


def test_check_and_patch_recognizes_already_patched_file_without_rerunning(tmp_path):
    # A file that already contains the patch marker (e.g. hand-applied, or patched by
    # a previous bushpilot version) must be left alone, not re-patched or corrupted --
    # even if it no longer contains the exact original buggy snippet at all.
    content = f"{{#- {_PATCH_MARKER} already here -#}}\nsome other template content"
    path = _write_template(str(tmp_path), REPO_ID, content)

    result = check_and_patch_chat_template(str(tmp_path), REPO_ID)

    assert result == "already_patched"
    with open(path) as f:
        assert f.read() == content


def test_check_and_patch_preserves_content_around_the_buggy_snippet(tmp_path):
    before = "{%- set role = 'model' -%}\n"
    after = "{%- if other -%}{{- 'tail' -}}{%- endif -%}\n"
    path = _write_template(str(tmp_path), REPO_ID, _buggy_template(prefix=before, suffix=after))

    check_and_patch_chat_template(str(tmp_path), REPO_ID)

    with open(path) as f:
        patched = f.read()
    assert patched.startswith(before)
    assert patched.endswith(after)


def test_check_and_patch_only_replaces_first_occurrence(tmp_path):
    # Defensive: if a template somehow contained the buggy snippet twice (not expected
    # in practice), replace(..., 1) must touch only the first occurrence rather than
    # silently patching both or raising -- confirms the explicit count=1 is doing what
    # it's there for, not just happening to work on single-occurrence inputs.
    content = _buggy_template(suffix=_BUGGY_SNIPPET)
    path = _write_template(str(tmp_path), REPO_ID, content)

    result = check_and_patch_chat_template(str(tmp_path), REPO_ID)

    assert result == "patched"
    with open(path) as f:
        patched = f.read()
    assert patched.count(_PATCH_MARKER) == 1
    assert _BUGGY_SNIPPET in patched  # the second, untouched occurrence remains


def test_check_and_patch_never_raises_on_missing_hf_home(tmp_path):
    missing = os.path.join(str(tmp_path), "does-not-exist")
    assert check_and_patch_chat_template(missing, REPO_ID) == "no_template"


@pytest.mark.parametrize("repo_id", [
    "mlx-community/gemma-4-26b-a4b-it-4bit",
    "mlx-community/gemma-4-e4b-it-4bit",
    "org/weird..repo__name",
])
def test_find_chat_template_path_handles_various_repo_ids(tmp_path, repo_id):
    path = _write_template(str(tmp_path), repo_id, "content")
    assert find_chat_template_path(str(tmp_path), repo_id) == path


def test_find_chat_template_path_prefers_refs_main_when_multiple_snapshots(tmp_path):
    safe_repo = REPO_ID.replace("/", "--")
    model_dir = os.path.join(str(tmp_path), "hub", f"models--{safe_repo}")
    old_snap = os.path.join(model_dir, "snapshots", "old111")
    new_snap = os.path.join(model_dir, "snapshots", "new222")
    os.makedirs(old_snap, exist_ok=True)
    os.makedirs(new_snap, exist_ok=True)
    old_file = os.path.join(old_snap, "chat_template.jinja")
    new_file = os.path.join(new_snap, "chat_template.jinja")
    with open(old_file, "w") as f:
        f.write("old")
    with open(new_file, "w") as f:
        f.write("new")

    refs_dir = os.path.join(model_dir, "refs")
    os.makedirs(refs_dir, exist_ok=True)
    with open(os.path.join(refs_dir, "main"), "w") as f:
        f.write("new222\n")

    assert find_chat_template_path(str(tmp_path), REPO_ID) == new_file


def test_find_chat_template_path_picks_newest_mtime_without_refs(tmp_path):
    safe_repo = REPO_ID.replace("/", "--")
    model_dir = os.path.join(str(tmp_path), "hub", f"models--{safe_repo}")
    snap1 = os.path.join(model_dir, "snapshots", "snap1")
    snap2 = os.path.join(model_dir, "snapshots", "snap2")
    os.makedirs(snap1, exist_ok=True)
    os.makedirs(snap2, exist_ok=True)
    file1 = os.path.join(snap1, "chat_template.jinja")
    file2 = os.path.join(snap2, "chat_template.jinja")
    with open(file1, "w") as f:
        f.write("1")
    with open(file2, "w") as f:
        f.write("2")
    os.utime(file1, (1000, 1000))
    os.utime(file2, (2000, 2000))

    assert find_chat_template_path(str(tmp_path), REPO_ID) == file2

