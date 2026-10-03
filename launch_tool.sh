#!/usr/bin/env bash
set -euo pipefail

# launch_tool.sh — attaches a coding tool (Claude Code / OpenCode) to the already-
# running MLX supervisor (start_mlx_server.sh). See PHASE_PLAN.md Phase 3.
#
# This script does NOT start or manage the model server itself -- that's
# start_mlx_server.sh's job, and its dashboard is the thing keeping the shared
# server alive for every tool/terminal attached to it. If the server isn't up yet,
# this script offers to start it, but does so by `exec`-ing start_mlx_server.sh
# directly (replacing itself in this terminal) rather than launching a background
# copy -- the dashboard it takes over into already tells you to open another
# terminal and re-run launch_tool.sh there once it's ready.
#
# Usage:
#   launch_tool.sh                    → fzf tool picker (plain menu if fzf is missing)
#   launch_tool.sh claude   [args...] → skip the picker, launch Claude Code directly
#   launch_tool.sh opencode [args...] → skip the picker, launch OpenCode directly
#
# Antigravity is deliberately not offered here -- already researched and confirmed
# there's no way to point it at a custom local endpoint (see PHASE_PLAN.md).

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
START_PROXY_SCRIPT="$SCRIPT_DIR/start_mlx_server.sh"

# Must match start_mlx_server.sh's own BUSHPILOT_HOME/MLX_HOST/MLX_PORT -- this script
# never starts the server, only attaches to it, so these have to point at the exact
# same place that script uses, including reading the same optional config.json.
BUSHPILOT_HOME="$HOME/.bushpilot"
MLX_STATE_FILE="$BUSHPILOT_HOME/state/mlx-loaded.txt"
BUSHPILOT_CONFIG_FILE="$BUSHPILOT_HOME/config.json"
CONFIG_HOST=""
CONFIG_PORT=""
if [[ -f "$BUSHPILOT_CONFIG_FILE" ]]; then
    eval "$(python3 - "$BUSHPILOT_CONFIG_FILE" <<'PYEOF'
import json, shlex, sys
try:
    with open(sys.argv[1]) as f:
        cfg = json.load(f)
except Exception:
    cfg = {}
for var, key in (("CONFIG_HOST", "host"), ("CONFIG_PORT", "port")):
    val = cfg.get(key)
    if val is not None:
        print(f"{var}={shlex.quote(str(val))}")
PYEOF
)"
fi
MLX_HOST="${MLX_HOST:-${CONFIG_HOST:-127.0.0.1}}"
MLX_PORT="${MLX_PORT:-${CONFIG_PORT:-38472}}"

MLX_PYTHON="$BUSHPILOT_HOME/venv/bin/python"
[[ -x "$MLX_PYTHON" ]] || MLX_PYTHON="python3"

# Fallback floor if the live preload check below can't run for some reason -- keep in
# sync with model_advisor.py's own MIN_USABLE_CONTEXT (exposed in its --list-json
# output specifically so this constant doesn't have to be trusted blindly here).
MIN_USABLE_CONTEXT=12288

# Measured live (see PHASE_PLAN.md's live end-to-end test): Claude Code's own
# max-output-tokens is ~32,000. recommended_context/CONTEXT_WINDOW is a memory
# ceiling, not a usable prompt budget -- reserve this much off the top when setting
# CLAUDE_CODE_MAX_CONTEXT_TOKENS, or requests near the ceiling reproduce exactly the
# PromptTooLongError that live test hit.
CLAUDE_OUTPUT_TOKENS_RESERVE=32000

TOOLS=(claude opencode)

log_info() { echo "[INFO] $*"; }
log_error() { echo "[ERROR] $*" >&2; }
log_success() { echo "[✓] $*"; }

is_command_available() { command -v "$1" > /dev/null 2>&1; }

tool_display_name() {
    case "$1" in
        claude) echo "Claude Code" ;;
        opencode) echo "OpenCode" ;;
        *) echo "$1" ;;
    esac
}

tool_command() {
    case "$1" in
        claude) echo claude ;;
        opencode) echo opencode ;;
    esac
}

tool_install_hint() {
    case "$1" in
        claude) echo "npm install -g @anthropic-ai/claude-code   (or see https://claude.com/download)" ;;
        opencode) echo "curl -fsSL https://opencode.ai/install | bash   (or see https://opencode.ai)" ;;
    esac
}

check_tool_installed() { is_command_available "$(tool_command "$1")"; }

# Same "on writable $PATH" pattern as start_mlx_server.sh's ensure_symlinked_on_path.
ensure_symlinked_on_path() {
    local target_name="$(basename "$SCRIPT_PATH")"
    local candidate
    if is_command_available "$target_name"; then
        return 0
    fi
    for candidate in "$HOME/.local/bin" "/opt/homebrew/bin" "/usr/local/bin"; do
        if [[ -d "$candidate" && -w "$candidate" ]]; then
            ln -sf "$SCRIPT_PATH" "$candidate/$target_name"
            log_success "Symlinked $target_name into $candidate (now on \$PATH)"
            return 0
        fi
    done
    log_info "No writable \$PATH directory found to symlink $target_name into -- run it by full path instead."
}

# Tool selection uses fzf too (decided 2026-09-29), same UX family as Phase 6's model
# picker. Unlike the model picker, an unavailable tool is NOT hidden from the list --
# it's shown with a "(not installed)" note, and selecting it fails with a clear
# install hint instead of silently disappearing.
pick_tool() {
    local rows=() t label
    for t in "${TOOLS[@]}"; do
        label="$(tool_display_name "$t")"
        if ! check_tool_installed "$t"; then
            label="$label (not installed)"
        fi
        rows+=("$t"$'\t'"$label")
    done

    if is_command_available fzf; then
        local selection
        selection="$(printf '%s\n' "${rows[@]}" | fzf --delimiter=$'\t' --with-nth=2.. \
            --border=rounded --pointer="▶" \
            --header="↑↓ move · Enter select · Esc cancel" \
            --prompt="Launch which tool> " --height=~40% --layout=reverse)" || exit 1
        printf '%s' "$selection" | cut -f1
        return
    fi

    log_info "fzf not found -- falling back to a plain numbered menu."
    local i=1
    for t in "${TOOLS[@]}"; do
        label="$(tool_display_name "$t")"
        check_tool_installed "$t" || label="$label (not installed)"
        echo "  $i) $label" >&2
        i=$((i + 1))
    done
    local choice
    read -r -p "Select [1-${#TOOLS[@]}]: " choice
    if ! [[ "$choice" =~ ^[0-9]+$ ]] || (( choice < 1 || choice > ${#TOOLS[@]} )); then
        log_error "Invalid choice."
        exit 1
    fi
    echo "${TOOLS[$((choice - 1))]}"
}

proxy_is_healthy() {
    [[ -f "$MLX_STATE_FILE" ]] || return 1
    curl -fsS --max-time 5 "http://$MLX_HOST:$MLX_PORT/health" > /dev/null 2>&1
}

# Checks proxy/model health first; if down, confirms with the user, then `exec`s
# start_mlx_server.sh directly -- see the header comment for why this replaces the
# process instead of backgrounding it.
ensure_proxy_running() {
    if proxy_is_healthy; then
        return 0
    fi
    log_info "No healthy local model server detected on $MLX_HOST:$MLX_PORT."
    local confirm
    read -r -p "Start it now with $(basename "$START_PROXY_SCRIPT")? [y/N] " confirm || confirm=""
    case "$confirm" in
        y|Y|yes|YES) ;;
        *)
            log_error "Not starting the server. Run it yourself first: $START_PROXY_SCRIPT"
            exit 1
            ;;
    esac
    log_info "Starting the local model server in this terminal -- it becomes the live dashboard."
    log_info "Once it's ready, run $(basename "$SCRIPT_PATH") again in a different terminal."
    exec "$START_PROXY_SCRIPT" --interactive
}

# Populates MLX_MODEL / CONTEXT_WINDOW / KV_BITS from the supervisor's state file --
# MLX_MODEL here is already the real HF repo id (what start_mlx_server.sh writes),
# not a friendly models.json id.
read_loaded_state() {
    if [[ ! -f "$MLX_STATE_FILE" ]]; then
        log_error "No state file at $MLX_STATE_FILE -- the server doesn't look like it was started by start_mlx_server.sh."
        exit 1
    fi
    IFS='|' read -r MLX_MODEL CONTEXT_WINDOW KV_BITS < "$MLX_STATE_FILE"
}

# Validates preload fit at attach-time, not just liveness (PHASE_PLAN.md Phase 3):
# the supervisor is deliberately tool-agnostic now, so the "does this tool's preload
# leave enough usable context" question moves here, at the point a specific tool is
# actually about to attach.
check_preload_fit() {
    local tool="$1"
    local advisor_json
    if ! advisor_json="$("$MLX_PYTHON" "$SCRIPT_DIR/model_advisor.py" --list-json --tool "$tool" 2>/dev/null)"; then
        log_info "Could not estimate $(tool_display_name "$tool")'s context preload -- skipping the preload-fit check."
        return 0
    fi

    local preload_total
    preload_total="$("$MLX_PYTHON" -c "import json,sys; print(json.loads(sys.argv[1])['preload_overhead']['total'])" "$advisor_json")"
    MIN_USABLE_CONTEXT="$("$MLX_PYTHON" -c "import json,sys; print(json.loads(sys.argv[1])['min_usable_context'])" "$advisor_json")"

    local usable=$(( CONTEXT_WINDOW - preload_total ))
    if (( usable < MIN_USABLE_CONTEXT )); then
        log_error "The running server's loaded context doesn't leave enough room for $(tool_display_name "$tool"):"
        log_error "  loaded context:         ${CONTEXT_WINDOW} tokens"
        log_error "  $(tool_display_name "$tool") preload:      ~${preload_total} tokens (system prompt + tools + docs/skills/MCP in $(pwd))"
        log_error "  left for conversation:  ~${usable} tokens -- below the ${MIN_USABLE_CONTEXT}-token usable floor"
        log_error "Fix: restart the server with a bigger context, or pick the other tool."
        exit 1
    fi

    PRELOAD_TOTAL="$preload_total"
    log_success "Preload check: $(tool_display_name "$tool") needs ~${preload_total} tokens, leaving ~${usable} of ${CONTEXT_WINDOW} for the conversation."
}

launch_claude() {
    # Pointed straight at the MLX server, no /v1 suffix -- mlx_vlm.server registers
    # /v1/messages itself.
    export ANTHROPIC_BASE_URL="http://$MLX_HOST:$MLX_PORT"
    # Dummy credentials -- unconfirmed whether Claude Code behaves with none configured
    # at all against a keyless server, so keep these until a startup test proves
    # they're unnecessary (see PHASE_PLAN.md).
    export ANTHROPIC_API_KEY="local-token"
    export ANTHROPIC_AUTH_TOKEN="ollama"
    export ANTHROPIC_MODEL="$MLX_MODEL"
    export ANTHROPIC_SMALL_FAST_MODEL="$MLX_MODEL"
    # Claude Code has been observed sending a hardcoded claude-haiku-4-5-20251001 for
    # an internal background call regardless of ANTHROPIC_MODEL/ANTHROPIC_SMALL_FAST_MODEL
    # -- covering all three DEFAULT_* vars plus these is the confirmed workaround.
    export ANTHROPIC_DEFAULT_HAIKU_MODEL="$MLX_MODEL"
    export ANTHROPIC_DEFAULT_SONNET_MODEL="$MLX_MODEL"
    export ANTHROPIC_DEFAULT_OPUS_MODEL="$MLX_MODEL"

    local usable_context=$(( CONTEXT_WINDOW - CLAUDE_OUTPUT_TOKENS_RESERVE ))
    (( usable_context < MIN_USABLE_CONTEXT )) && usable_context=$MIN_USABLE_CONTEXT
    export CLAUDE_CODE_MAX_CONTEXT_TOKENS="$usable_context"

    log_success "Launching Claude Code against $MLX_MODEL (context cap ${usable_context} tokens, reserving ${CLAUDE_OUTPUT_TOKENS_RESERVE} for output)"
    exec claude --model "$MLX_MODEL" "$@"
}

# OpenCode v2 (a separate, parallel install track -- npm package @opencode/cli vs the
# 1.x line's opencode-ai; see opencode.ai/v2) dropped --model from the bare/default
# launch command entirely -- confirmed live against v2.0.22's own --help, which only
# documents --model under `opencode run`, never for the bare command (which just
# dumps top-level usage if you pass it anyway, rather than erroring). It also uses a
# different config schema: "providers" (plural, package-based) instead of v1's
# "provider" (singular, npm-based), and a root "model" field instead of
# "default_model". Detected by major version number from `--version` (v1 prints bare
# "1.18.34"; v2 prints "opencode v2.0.22" -- note the differing prefix); if that
# somehow fails to parse, falls back to what --help actually advertises for the bare
# command rather than guessing.
opencode_cli_is_v2() {
    local version major
    version="$(opencode --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
    major="${version%%.*}"
    if [[ "$major" =~ ^[0-9]+$ ]]; then
        [[ "$major" -ge 2 ]]
        return
    fi
    if opencode --help 2>&1 | grep -qE '(^|[[:space:]])(-m|--model)([[:space:]]|,|$)'; then
        return 1
    fi
    return 0
}

# Ports the old ensure_opencode_model_registered() to the mlx-direct provider block
# instead of the now-dead litellm one -- confirmed live that OpenCode still requires
# the model registered in opencode.jsonc even with LiteLLM removed (Provider.getModel
# does a plain object lookup and throws otherwise). Reserializes the whole file after
# stripping comments (rather than the old script's narrower text-surgery insertion):
# this edit also has to DELETE the litellm block, and a full reserialize is far more
# robust for that than brace-depth-scanning a deletion. The only comment this can lose
# is the litellm block's own now-moot rollback note -- everything else in the file
# survives, just reformatted.
#
# OpenCode v1 only -- see register_opencode_model_v2 for the v2 schema.
register_opencode_model() {
    local base_url="$1"
    local opencode_config="$HOME/.config/opencode/opencode.jsonc"
    mkdir -p "$(dirname "$opencode_config")"
    log_info "Checking OpenCode's mlx-direct provider registration ($opencode_config)..."

    local result
    result="$("$MLX_PYTHON" - "$opencode_config" "$MLX_MODEL" "$CONTEXT_WINDOW" "$base_url" "$SCRIPT_DIR/models.json" <<'PYEOF'
import json
import re
import sys

config_path, model_id, context_window, base_url, models_json_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]

# Display strings for the OpenCode model picker. Provider name is the project name;
# the model name is just its id (see the note at the assignment site for why the old
# "(no LiteLLM)" suffixes were removed).
MLX_DIRECT_PROVIDER_NAME = "BushPilot"
MLX_DIRECT_MODEL_NAME = lambda mid: mid


# Live-tested bug fix: without this, OpenCode's custom-provider model entries have no
# way to know a model accepts images, so it silently never encodes drag-and-dropped /
# -f attached images into the request at all -- the mlx_vlm.server backend and the
# model itself handle images perfectly fine (confirmed live via a raw /v1/chat/completions
# request with an image_url, which came back with a correct description), but OpenCode
# never sends the image bytes because the model lacks the "attachment"/"modalities"
# capability flags its own config schema (opencode.ai/config.json) exposes for exactly
# this. Reproduced live: identical request through `opencode run -f <image> ...` against
# a vision-capable model registered WITHOUT these flags got "I cannot process image
# input"; the IDENTICAL request after adding attachment:true + modalities.input
# containing "image" got a correct description, and the server's own request log
# confirmed images=1 only in the fixed case (images=0 before). Mirrors
# model_advisor.py's own model_input_label(): serving.multimodal_capable=False (every
# mlx_dspark/DFlash-text-only entry) narrows to text-only even if the underlying model
# has a vision tower; otherwise use the model's own `modalities` list.
def image_capable_modalities(models_json_path, hf_id):
    try:
        with open(models_json_path) as f:
            catalog = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    entries = [m for m in catalog.get("models", []) if m.get("huggingface") == hf_id]
    if not entries:
        return None
    # Known limitation: this script only has the resolved huggingface id, not the
    # catalog's own local model id for whichever serving mode is actually running --
    # the state file written by start_mlx_server.sh does not carry it. Several
    # catalog entries share one huggingface id across serving modes with DIFFERENT
    # multimodal capability (e.g. gemma-4-12b-it-4bit vs its mlx-dspark "dflash"
    # text-only sibling, which silently drops the vision tower). When those entries
    # disagree, guessing wrong in the image-capable direction would be worse than
    # todays no-image-support status quo -- OpenCode would then happily try to send
    # an image that the dflash backend silently ignores, a confusing new failure mode
    # instead of todays understood one. So: require unanimous agreement across every
    # matching entry before turning this on at all.
    capable_votes = set()
    all_mods = set()
    for entry in entries:
        if (entry.get("serving") or {}).get("multimodal_capable") is False:
            capable_votes.add(False)
        else:
            capable_votes.add(True)
            all_mods |= set(entry.get("modalities") or ["text"])
    if capable_votes != {True}:
        return None
    extra = [m for m in ("image", "video") if m in all_mods]
    if not extra:
        return None
    return ["text"] + extra


ATTACHMENT_MODALITIES = image_capable_modalities(models_json_path, model_id)


def strip_jsonc(s):
    out = []
    i = 0
    n = len(s)
    in_string = False
    while i < n:
        c = s[i]
        if in_string:
            out.append(c)
            if c == '\\' and i + 1 < n:
                out.append(s[i + 1])
                i += 2
                continue
            if c == '"':
                in_string = False
            i += 1
            continue
        if c == '"':
            in_string = True
            out.append(c)
            i += 1
            continue
        if c == '/' and i + 1 < n and s[i + 1] == '/':
            while i < n and s[i] != '\n':
                i += 1
            continue
        if c == '/' and i + 1 < n and s[i + 1] == '*':
            i += 2
            while i + 1 < n and not (s[i] == '*' and s[i + 1] == '/'):
                i += 1
            i += 2
            continue
        out.append(c)
        i += 1
    result = ''.join(out)
    result = re.sub(r',(\s*[}\]])', r'\1', result)
    return result


try:
    with open(config_path) as f:
        raw = f.read()
except FileNotFoundError:
    raw = '{"$schema": "https://opencode.ai/config.json", "provider": {}}'

try:
    parsed = json.loads(strip_jsonc(raw))
except Exception as e:
    print(f"PARSE_ERROR: {e}")
    sys.exit(2)

provider = parsed.setdefault("provider", {})
had_litellm = "litellm" in provider

existing = provider.get("mlx-direct", {})
# Also compare the display NAMES, not just the structural bits: OpenCode renders the
# provider name and model name together, so a stale/duplicated name would otherwise be
# left in place forever (the earlier "...(direct, no LiteLLM)" + "MLX direct (no
# LiteLLM)" pair concatenated into a stutter that this check never corrected).
existing_model = existing.get("models", {}).get(model_id, {})
already_correct = (
    not had_litellm
    and existing.get("options", {}).get("baseURL") == base_url
    and existing.get("name") == MLX_DIRECT_PROVIDER_NAME
    and model_id in existing.get("models", {})
    and existing_model.get("name") == MLX_DIRECT_MODEL_NAME(model_id)
    and existing_model.get("limit", {}).get("context") == context_window
    and existing_model.get("attachment") == (ATTACHMENT_MODALITIES is not None)
    and existing_model.get("modalities", {}).get("input") == ATTACHMENT_MODALITIES
)
if already_correct:
    print("ALREADY_REGISTERED")
    sys.exit(0)

# Drop the dead litellm provider -- pointed at the now-removed LiteLLM/Postgres
# stack (port 4100), nothing left listens there.
provider.pop("litellm", None)

mlx_direct = provider.setdefault("mlx-direct", {
    "npm": "@ai-sdk/openai-compatible",
})
# Display name is "BushPilot" (this project name), NOT "(no LiteLLM)" -- LiteLLM is
# long gone, so that qualifier was pure legacy noise, and having it on BOTH the provider
# and the model name made the combined OpenCode label read as a duplicated stutter. The
# model own name is just its id; the provider name already identifies the project, so
# OpenCode shows e.g. "mlx-community/gemma-4-26b-a4b-it-4bit  BushPilot".
mlx_direct["name"] = MLX_DIRECT_PROVIDER_NAME
# Hardcoded, not "{env:...}" -- confirmed nothing ever set that env var, and the port
# is already a fixed constant in the supervisor, so there is nothing dynamic to gain
# by routing this through the environment instead.
mlx_direct["options"] = {"baseURL": base_url, "apiKey": "sk-mlx-direct-local"}
models = mlx_direct.setdefault("models", {})
models[model_id] = {
    "name": MLX_DIRECT_MODEL_NAME(model_id),
    "limit": {"context": context_window, "output": 4096},
}
if ATTACHMENT_MODALITIES is not None:
    models[model_id]["attachment"] = True
    models[model_id]["modalities"] = {"input": ATTACHMENT_MODALITIES, "output": ["text"]}

with open(config_path, "w") as f:
    json.dump(parsed, f, indent=2)
    f.write("\n")
print("REGISTERED")
PYEOF
    )" || true

    case "$result" in
        ALREADY_REGISTERED)
            log_success "$MLX_MODEL already registered under mlx-direct -- no changes needed"
            ;;
        REGISTERED)
            log_success "Registered $MLX_MODEL in opencode.jsonc's mlx-direct provider (and removed the dead litellm provider, if it was present)"
            ;;
        *)
            log_error "Could not auto-register $MLX_MODEL in opencode.jsonc ($result)"
            log_error "Add it manually under provider.\"mlx-direct\".models, or OpenCode will fail with 'Model not found'."
            exit 1
            ;;
    esac
}

# v2 schema counterpart to register_opencode_model -- confirmed live against v2.0.22
# (see opencode_cli_is_v2): "providers" (plural) instead of "provider", a "package"
# field instead of "npm" (and a different package name: @opencode/ai/providers/
# openai-compatible, not @ai-sdk/openai-compatible), and each model entry needs its
# own "modelID" rather than using the dict key as the id directly. Critically, this
# ALSO sets the root "model" field -- v2 has no CLI flag to pick a model for a bare
# launch (only `opencode run --model`), so the only way to launch straight into our
# model is to make it the configured default (verified live: `opencode run` with no
# --model at all correctly picked up this field and attempted to reach our model).
# That default persists in opencode.jsonc after this script exits, unlike v1's
# --model flag which only ever affected the one launch -- see uninstall.sh, which
# already clears both the v1 and v2 shapes of this on removal.
register_opencode_model_v2() {
    local base_url="$1"
    local opencode_config="$HOME/.config/opencode/opencode.jsonc"
    mkdir -p "$(dirname "$opencode_config")"
    log_info "Checking OpenCode's mlx-direct provider registration ($opencode_config)..."

    local result
    result="$("$MLX_PYTHON" - "$opencode_config" "$MLX_MODEL" "$CONTEXT_WINDOW" "$base_url" "$SCRIPT_DIR/models.json" <<'PYEOF'
import json
import re
import sys

config_path, model_id, context_window, base_url, models_json_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]

MLX_DIRECT_PROVIDER_NAME = "BushPilot"
MLX_DIRECT_PACKAGE = "@opencode/ai/providers/openai-compatible"
MLX_DIRECT_MODEL_STRING = f"mlx-direct/{model_id}"


def image_capable_modalities(models_json_path, hf_id):
    try:
        with open(models_json_path) as f:
            catalog = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    entries = [m for m in catalog.get("models", []) if m.get("huggingface") == hf_id]
    if not entries:
        return None
    # Known limitation: this script only has the resolved huggingface id, not the
    # catalog's own local model id for whichever serving mode is actually running --
    # the state file written by start_mlx_server.sh does not carry it. Several
    # catalog entries share one huggingface id across serving modes with DIFFERENT
    # multimodal capability (e.g. gemma-4-12b-it-4bit vs its mlx-dspark "dflash"
    # text-only sibling, which silently drops the vision tower). When those entries
    # disagree, guessing wrong in the image-capable direction would be worse than
    # todays no-image-support status quo -- OpenCode would then happily try to send
    # an image that the dflash backend silently ignores, a confusing new failure mode
    # instead of todays understood one. So: require unanimous agreement across every
    # matching entry before turning this on at all.
    capable_votes = set()
    all_mods = set()
    for entry in entries:
        if (entry.get("serving") or {}).get("multimodal_capable") is False:
            capable_votes.add(False)
        else:
            capable_votes.add(True)
            all_mods |= set(entry.get("modalities") or ["text"])
    if capable_votes != {True}:
        return None
    extra = [m for m in ("image", "video") if m in all_mods]
    if not extra:
        return None
    return ["text"] + extra


ATTACHMENT_MODALITIES = image_capable_modalities(models_json_path, model_id)


# Same comment-stripping approach as the v1 register_opencode_model, kept in sync
# deliberately -- both edit the same jsonc file and must agree on how it's parsed.
def strip_jsonc(s):
    out = []
    i = 0
    n = len(s)
    in_string = False
    while i < n:
        c = s[i]
        if in_string:
            out.append(c)
            if c == '\\' and i + 1 < n:
                out.append(s[i + 1])
                i += 2
                continue
            if c == '"':
                in_string = False
            i += 1
            continue
        if c == '"':
            in_string = True
            out.append(c)
            i += 1
            continue
        if c == '/' and i + 1 < n and s[i + 1] == '/':
            while i < n and s[i] != '\n':
                i += 1
            continue
        if c == '/' and i + 1 < n and s[i + 1] == '*':
            i += 2
            while i + 1 < n and not (s[i] == '*' and s[i + 1] == '/'):
                i += 1
            i += 2
            continue
        out.append(c)
        i += 1
    result = ''.join(out)
    result = re.sub(r',(\s*[}\]])', r'\1', result)
    return result


try:
    with open(config_path) as f:
        raw = f.read()
except FileNotFoundError:
    raw = '{"$schema": "https://opencode.ai/config.json", "providers": {}}'

try:
    parsed = json.loads(strip_jsonc(raw))
except Exception as e:
    print(f"PARSE_ERROR: {e}")
    sys.exit(2)

# v1-schema leftovers, if this machine was ever set up against the 1.x line. Only
# clean up what THIS script owns -- its own old "mlx-direct" entry under the OLD
# singular "provider" key, and "default_model" if it specifically pointed there --
# never touch any OTHER provider the user configured under "provider" (confirmed live
# this matters: a real opencode.jsonc on this project's own dev machine has an
# unrelated "amazon-bedrock" entry sitting right next to mlx-direct under that same
# key). v2 doesn't read "provider"/"default_model" at all, so anything left behind
# here is inert either way -- but inert-and-preserved beats deleted-without-asking.
old_provider = parsed.get("provider") or {}
had_v1_mlx_direct = "mlx-direct" in old_provider
if had_v1_mlx_direct:
    del old_provider["mlx-direct"]
    if not old_provider:
        parsed.pop("provider", None)
had_v1_default_model = str(parsed.get("default_model", "")).startswith("mlx-direct/")
if had_v1_default_model:
    parsed.pop("default_model", None)

providers = parsed.setdefault("providers", {})
existing = providers.get("mlx-direct", {})
existing_model = existing.get("models", {}).get(model_id, {})
already_correct = (
    not had_v1_mlx_direct
    and not had_v1_default_model
    and existing.get("name") == MLX_DIRECT_PROVIDER_NAME
    and existing.get("package") == MLX_DIRECT_PACKAGE
    and existing.get("settings", {}).get("baseURL") == base_url
    and model_id in existing.get("models", {})
    and existing_model.get("modelID") == model_id
    and existing_model.get("limit", {}).get("context") == context_window
    and existing_model.get("attachment") == (ATTACHMENT_MODALITIES is not None)
    and existing_model.get("modalities", {}).get("input") == ATTACHMENT_MODALITIES
    and parsed.get("model") == MLX_DIRECT_MODEL_STRING
)
if already_correct:
    print("ALREADY_REGISTERED")
    sys.exit(0)

mlx_direct = providers.setdefault("mlx-direct", {})
mlx_direct["name"] = MLX_DIRECT_PROVIDER_NAME
mlx_direct["package"] = MLX_DIRECT_PACKAGE
mlx_direct["settings"] = {"baseURL": base_url, "apiKey": "sk-mlx-direct-local"}
models = mlx_direct.setdefault("models", {})
models[model_id] = {
    "modelID": model_id,
    "limit": {"context": context_window, "output": 4096},
}
if ATTACHMENT_MODALITIES is not None:
    models[model_id]["attachment"] = True
    models[model_id]["modalities"] = {"input": ATTACHMENT_MODALITIES, "output": ["text"]}
parsed["model"] = MLX_DIRECT_MODEL_STRING

with open(config_path, "w") as f:
    json.dump(parsed, f, indent=2)
    f.write("\n")
print("REGISTERED")
PYEOF
    )" || true

    case "$result" in
        ALREADY_REGISTERED)
            log_success "$MLX_MODEL already registered under mlx-direct -- no changes needed"
            ;;
        REGISTERED)
            log_success "Registered $MLX_MODEL in opencode.jsonc's mlx-direct provider and set it as your default model (v2 has no --model flag for a direct launch -- see $opencode_config to change the default back)"
            ;;
        *)
            log_error "Could not auto-register $MLX_MODEL in opencode.jsonc ($result)"
            log_error "Add it manually under providers.\"mlx-direct\".models and set root \"model\" to \"mlx-direct/$MLX_MODEL\", or OpenCode will fail with 'Model not found'."
            exit 1
            ;;
    esac
}

launch_opencode() {
    local base_url="http://$MLX_HOST:$MLX_PORT/v1"

    if opencode_cli_is_v2; then
        register_opencode_model_v2 "$base_url"
        log_success "Launching OpenCode against mlx-direct/$MLX_MODEL (context: ${CONTEXT_WINDOW} tokens)"
        exec opencode "$@"
    fi

    register_opencode_model "$base_url"
    log_success "Launching OpenCode against mlx-direct/$MLX_MODEL (context: ${CONTEXT_WINDOW} tokens)"
    exec opencode --model "mlx-direct/$MLX_MODEL" "$@"
}

main() {
    ensure_symlinked_on_path

    local tool="${1:-}"
    if [[ -n "$tool" ]]; then
        shift
    else
        tool="$(pick_tool)"
    fi

    case "$tool" in
        claude|opencode) ;;
        *)
            log_error "Unknown tool '$tool'. Supported: ${TOOLS[*]}"
            exit 1
            ;;
    esac

    if ! check_tool_installed "$tool"; then
        log_error "$(tool_display_name "$tool") is not installed -- install it first: $(tool_install_hint "$tool")"
        exit 1
    fi

    ensure_proxy_running
    read_loaded_state
    check_preload_fit "$tool"

    case "$tool" in
        claude) launch_claude "$@" ;;
        opencode) launch_opencode "$@" ;;
    esac
}

main "$@"
