"""Known chat-template bugs in specific HuggingFace model repos that silently break
mlx_vlm's Automatic Prefix Caching (APC) for growing multi-turn conversations -- NOT a
bug in mlx_vlm's or mlx-dspark's own code. See models.json's known_issues on
gemma-4-12b-it-4bit / gemma-4-26b-a4b-it-4bit for the full root-cause writeup (live-
tested 2026-10-05) and upstream mlx-vlm PR #1960 ("Redesign APC for dense and hybrid
model caches"), which independently found and fixed the identical bug for the
26b-a4b-it-5bit/31b-it-5bit siblings via a template-fix discussion on those HF repos --
a fix that was never applied to the 4-bit/8-bit repos this project actually serves.

start_mlx_server.sh calls check_and_patch_chat_template() right after confirming the
target model is downloaded and before launching the server (model_advisor.py's
--patch-chat-template flag is the entry point). Detection requires TWO separate
literal exact-string matches (see _BUGGY_SNIPPET and _GENERATION_PROMPT_BUG_SIGNATURE
below for why one alone is not enough -- a real false positive, caught before
shipping), never a fuzzy/regex guess, so a call on a model that doesn't have this
exact bug is always a safe no-op -- this is deliberately NOT gated on architecture/
model-id, so it stays correct (and inert) even if a future model's HF repo happens to
ship the same buggy pattern, or if this exact repo's template changes shape upstream
and no longer matches.
"""
import glob
import os

# The exact buggy snippet, verified byte-identical across gemma-4-12b-it-4bit and
# gemma-4-26b-a4b-it-4bit's cached chat_template.jinja on 2026-10-05: the per-turn
# history-serialization loop only re-renders a model/assistant turn's thinking channel
# when BOTH reasoning_content AND tool_calls are present on that message. But further
# down the same template, add_generation_prompt unconditionally injects an empty
# '<|channel>thought\n<channel|>' marker before every fresh non-thinking generation --
# so a plain completed turn (no tool_calls, no reasoning_text) never gets that marker
# reproduced when the conversation grows and this turn is re-serialized into history.
# That diverges the reconstructed prefix from what was actually prefilled, 4 tokens
# before the end. mlx_vlm's APC "checkpoint" strategy for Gemma-4's hybrid sliding-
# window cache requires an exact full-prefix hash match (all-or-nothing), so this makes
# it miss entirely on every turn of a growing conversation -- live-measured on
# gemma-4-12b-it-4bit/gemma-4-26b-a4b-it-4bit: cached_tokens 0/1540 with the bug, vs.
# 1516/1536 on gemma-4-e4b-it-4bit, whose template has no such marker at all.
_BUGGY_SNIPPET = (
    "    {%- set thinking_text = message.get('reasoning') or message.get('reasoning_content') -%}\n"
    "    {%- if thinking_text and loop.index0 > ns_turn.last_user_idx and message.get('tool_calls') -%}\n"
    "        {{- '<|channel>thought\\n' + thinking_text + '\\n<channel|>' -}}\n"
    "    {%- endif -%}\n"
)

# NOT, by itself, enough to confirm the bug: this exact 4-line reasoning-render block
# is shared boilerplate across every Gemma-4 chat_template.jinja checked so far,
# INCLUDING gemma-4-e4b-it-4bit's, which does NOT have this bug (confirmed live --
# its add_generation_prompt block never injects the marker in the first place, so
# there is nothing for history-serialization to fail to reproduce). Patching based on
# _BUGGY_SNIPPET alone was tried and caught before shipping: it falsely matched
# gemma-4-e4b-it-4bit's real cached template and would have INTRODUCED a new
# asymmetry there (history serialization emitting a marker that fresh generation never
# emits), which is worse than the original, correctly-symmetric, un-bugged state.
#
# The real differentiator is this second, separate signature: the add_generation_prompt
# block (elsewhere in the file) conditionally injecting the empty marker before a fresh
# non-thinking generation. Only templates with BOTH signatures present actually have the
# bug -- see check_and_patch_chat_template, which requires both before patching.
_GENERATION_PROMPT_BUG_SIGNATURE = (
    "        {{- '<|turn>model\\n' -}}\n"
    "        {%- if not enable_thinking | default(false) -%}\n"
    "            {{- '<|channel>thought\\n<channel|>' -}}\n"
    "        {%- endif -%}\n"
)


_PATCH_MARKER = "BUSHPILOT PATCH"

# Adds exactly one elif branch: when this is a model/assistant turn that just started
# (not continue_same_model_turn -- otherwise the marker would already have been
# emitted earlier in this same turn), has no tool_calls and no reasoning_text, and
# non-thinking mode is active, reproduce the identical empty marker add_generation_prompt
# would have injected when this turn was originally generated. Deliberately scoped to
# exactly this one verified case (plain conversational turns) -- NOT extended to cover
# tool-call-only turns or reasoning-without-tool_calls turns, since those paths were not
# directly tested and this patch should never be a bigger behavioral change than what
# was actually verified live (see the correctness check cited in models.json: identical
# secret-recall output with and without this patch, same chip, same prompt).
_PATCHED_REPLACEMENT = (
    "    {%- set thinking_text = message.get('reasoning') or message.get('reasoning_content') -%}\n"
    "    {%- if thinking_text and loop.index0 > ns_turn.last_user_idx and message.get('tool_calls') -%}\n"
    "        {{- '<|channel>thought\\n' + thinking_text + '\\n<channel|>' -}}\n"
    "    {%- elif role == 'model' and not continue_same_model_turn and not message.get('tool_calls') "
    "and not thinking_text and not (enable_thinking | default(false)) -%}\n"
    "        {#- " + _PATCH_MARKER + " 2026-10-05: add_generation_prompt (below) injects an empty\n"
    "            '<|channel>thought\\n<channel|>' marker before every fresh non-thinking generation,\n"
    "            but this history-serialization branch never reproduced it for a plain completed\n"
    "            turn -- so a growing conversation's reconstructed prefix diverged from what was\n"
    "            actually prefilled, and mlx_vlm's APC exact-checkpoint match (all-or-nothing) missed\n"
    "            on every turn. Scoped to exactly the case verified live: plain model turn, no\n"
    "            tool_calls, no reasoning_text, non-thinking mode. See models.json's known_issues on\n"
    "            this model id for the root-cause writeup and mlx-vlm PR #1960 (upstream fixed the\n"
    "            same bug for the 26b-a4b-it-5bit/31b-it-5bit repos via a template discussion; this\n"
    "            repo's template was not covered by that fix). -#}\n"
    "        {{- '<|channel>thought\\n<channel|>' -}}\n"
    "    {%- endif -%}\n"
)


def find_chat_template_path(hf_home, repo_id):
    """Returns the locally cached chat_template.jinja path for `repo_id` under
    `hf_home` (the HF_HOME this project exports, ~/.bushpilot/models), or None if the
    model isn't downloaded yet, or if its repo doesn't ship a standalone template file.

    Some repos embed chat_template directly in tokenizer_config.json instead of a
    standalone .jinja file -- out of scope here, since every repo this known bug has
    actually been found in ships a standalone chat_template.jinja; a future bug
    embedded the other way would need its own detector, not a guess bolted onto this
    one."""
    safe_repo = repo_id.replace("/", "--")
    model_dir = os.path.join(hf_home, "hub", f"models--{safe_repo}")

    # Check if refs/main (or refs/master) points to an active snapshot commit
    refs_dir = os.path.join(model_dir, "refs")
    if os.path.isdir(refs_dir):
        for ref_name in ("main", "master"):
            ref_path = os.path.join(refs_dir, ref_name)
            if os.path.isfile(ref_path):
                try:
                    with open(ref_path) as f:
                        commit = f.read().strip()
                    target = os.path.join(model_dir, "snapshots", commit, "chat_template.jinja")
                    if os.path.isfile(target):
                        return target
                except OSError:
                    pass

    pattern = os.path.join(model_dir, "snapshots", "*", "chat_template.jinja")
    matches = glob.glob(pattern)
    if not matches:
        return None
    matches.sort(key=os.path.getmtime, reverse=True)
    return matches[0]


def check_and_patch_chat_template(hf_home, repo_id):
    """Detects and fixes the known Gemma-4 growing-conversation APC cache-miss bug (see
    module docstring) in `repo_id`'s locally cached chat_template.jinja, if present.

    Detection requires BOTH _BUGGY_SNIPPET (the unpatched history-serialization block)
    AND _GENERATION_PROMPT_BUG_SIGNATURE (confirmation that add_generation_prompt
    actually injects the asymmetric marker this template needs reproduced) -- never a
    fuzzy/regex guess, and deliberately NOT just the first signature alone. An earlier
    version of this function matched on _BUGGY_SNIPPET by itself and was caught, before
    shipping, falsely flagging gemma-4-e4b-it-4bit (whose history-serialization block
    looks identical but whose add_generation_prompt never injects the marker in the
    first place) -- patching that model would have introduced a NEW asymmetry rather
    than fixing one. Requiring both signatures makes a false positive structurally
    impossible: a template without the second signature has nothing for the first
    signature's fix to correctly reproduce.

    This two-signature requirement, combined with literal exact-string containment
    (never fuzzy), means this is always either a precise, verified fix or a safe no-op,
    never a corrupting edit of an unrecognized template structure. Idempotent:
    re-running on an already-patched file is a no-op (checked via _PATCH_MARKER), safe
    to call on every launch without tracking patch state separately. Written atomically
    (tmp file + os.replace), same safety pattern as calibration.py's save_calibration,
    so a process killed mid-write never leaves a half-written, syntactically-broken
    template for the next launch (or the server itself) to trip over.

    Never raises on a missing/unreadable file or directory -- a model that fails this
    check (not downloaded yet, no standalone template, permission error) serves exactly
    as it did before this function existed; this is a pure opportunistic improvement,
    never a requirement for serving to work.

    Returns one of:
      "patched"         -- both signatures found, bug fixed just now
      "already_patched" -- this patch (or an identical one) was already applied
      "not_affected"     -- template exists but doesn't match both required signatures
      "no_template"      -- model not downloaded yet, or no standalone chat_template.jinja
    """
    try:
        path = find_chat_template_path(hf_home, repo_id)
        if path is None:
            return "no_template"
        with open(path) as f:
            content = f.read()
    except OSError:
        return "no_template"

    if _PATCH_MARKER in content:
        return "already_patched"
    if _BUGGY_SNIPPET not in content or _GENERATION_PROMPT_BUG_SIGNATURE not in content:
        return "not_affected"

    patched = content.replace(_BUGGY_SNIPPET, _PATCHED_REPLACEMENT, 1)
    tmp_path = f"{path}.bushpilot-tmp"
    try:
        with open(tmp_path, "w") as f:
            f.write(patched)
        os.replace(tmp_path, path)
    except OSError:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        return "not_affected"
    return "patched"
