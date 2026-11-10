#!/usr/bin/env bash
set -euo pipefail

# launch_tool.sh — attaches a coding tool (Claude Code / OpenCode / Aider / Cline / Pi)
# to the already-running MLX supervisor (start_mlx_server.sh). See docs/PHASE_PLAN.md Phase 3.
#
# This script does NOT manage the model server itself -- that's start_mlx_server.sh's
# job, and its dashboard is the thing keeping the shared server alive for every
# tool/terminal attached to it. If the server isn't up yet, this script offers to
# start it: when running inside a supported terminal (cmux, tmux, WezTerm, kitty,
# iTerm2, Ghostty, or macOS Terminal.app), it runs the interactive model picker right
# here, spawns start_mlx_server.sh's live dashboard in a split pane/tab, waits for it
# to pass its startup smoke check, and then launches the chosen coding tool in this
# original terminal. Otherwise it falls back to `exec`-ing start_mlx_server.sh here.
#
# Usage:
#   launch_tool.sh                    → fzf tool picker (plain menu if fzf is missing)
#   launch_tool.sh claude   [args...] → skip the picker, launch Claude Code directly
#   launch_tool.sh opencode [args...] → skip the picker, launch OpenCode directly
#   launch_tool.sh aider    [args...] → skip the picker, launch Aider directly
#   launch_tool.sh cline    [args...] → skip the picker, launch Cline directly
#   launch_tool.sh pi       [args...] → skip the picker, launch Pi directly
#
# Antigravity is deliberately not offered here -- already researched and confirmed
# there's no way to point it at a custom local endpoint (see docs/PHASE_PLAN.md). "Tau" and
# "Flash CLI" were also looked at and deliberately left out: neither name maps to a
# single established, reputable coding-agent CLI (searched npm/GitHub/PyPI -- the
# closest matches are either one-person projects with a few hundred downloads and no
# custom-endpoint support at all, or unrelated tools entirely -- e.g. the npm package
# literally named "flash-cli" is a terminal flashcards app, not an AI coding agent).
# Revisit if the user points at a specific repo for either name.

_resolve_script_path() {
    local src="${BASH_SOURCE[0]}" dir link
    while [[ -L "$src" ]]; do
        dir="$(cd -P "$(dirname "$src")" && pwd)"
        link="$(readlink "$src")"
        if [[ "$link" != /* ]]; then
            src="$dir/$link"
        else
            src="$link"
        fi
    done
    printf '%s/%s\n' "$(cd -P "$(dirname "$src")" && pwd)" "$(basename "$src")"
}
SCRIPT_PATH="$(_resolve_script_path)"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
START_PROXY_SCRIPT="$SCRIPT_DIR/start_mlx_server.sh"

# Must match start_mlx_server.sh's own BUSHPILOT_HOME/MLX_HOST/MLX_PORT -- this script
# never starts the server, only attaches to it, so these have to point at the exact
# same place that script uses, including reading the same optional config.json.
BUSHPILOT_HOME="$HOME/.bushpilot"
MLX_STATE_FILE="$BUSHPILOT_HOME/state/mlx-loaded.txt"
MLX_ROSTER_FILE="$BUSHPILOT_HOME/state/mlx-roster.json"
MLX_STARTING_PID_FILE="$BUSHPILOT_HOME/state/mlx-starting.pid"
BUSHPILOT_CONFIG_FILE="$BUSHPILOT_HOME/config.json"
CONFIG_HOST=""
CONFIG_PORT=""
if [[ -f "$BUSHPILOT_CONFIG_FILE" ]]; then
    # Same two-step pattern as register_opencode_model/_v2 below (write the heredoc to a
    # temp file by itself, then run it as a separate command with no heredoc inside the
    # $(...)) -- see the comment there for why combining a heredoc with $(...) in one
    # statement is unsafe on the stock macOS /bin/bash (3.2.57).
    _config_reader_py="$(mktemp)"
    cat <<'PYEOF' > "$_config_reader_py"
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
    eval "$(python3 "$_config_reader_py" "$BUSHPILOT_CONFIG_FILE")"
    rm -f "$_config_reader_py"
fi
MLX_HOST="${MLX_HOST:-${CONFIG_HOST:-127.0.0.1}}"
MLX_PORT="${MLX_PORT:-${CONFIG_PORT:-38472}}"

MLX_PYTHON="$BUSHPILOT_HOME/venv/bin/python"
[[ -x "$MLX_PYTHON" ]] || MLX_PYTHON="python3"

# Fallback floor if the live preload check below can't run for some reason -- keep in
# sync with model_advisor.py's own MIN_USABLE_CONTEXT (exposed in its --list-json
# output specifically so this constant doesn't have to be trusted blindly here).
MIN_USABLE_CONTEXT=12288

# Measured live (see docs/PHASE_PLAN.md's live end-to-end test): Claude Code's own
# max-output-tokens is ~32,000. recommended_context/CONTEXT_WINDOW is a memory
# ceiling, not a usable prompt budget -- reserve this much off the top when setting
# CLAUDE_CODE_MAX_CONTEXT_TOKENS, or requests near the ceiling reproduce exactly the
# PromptTooLongError that live test hit.
CLAUDE_OUTPUT_TOKENS_RESERVE=32000

TOOLS=(claude opencode aider cline pi)

log_info() { echo "[INFO] $*"; }
log_error() { echo "[ERROR] $*" >&2; }
log_success() { echo "[✓] $*"; }

is_command_available() { command -v "$1" > /dev/null 2>&1; }

tool_display_name() {
    case "$1" in
        claude) echo "Claude Code" ;;
        opencode) echo "OpenCode" ;;
        aider) echo "Aider" ;;
        cline) echo "Cline" ;;
        pi) echo "Pi" ;;
        *) echo "$1" ;;
    esac
}

tool_command() {
    case "$1" in
        claude) echo claude ;;
        opencode) echo opencode ;;
        aider) echo aider ;;
        cline) echo cline ;;
        pi) echo pi ;;
    esac
}

tool_install_hint() {
    case "$1" in
        claude) echo "npm install -g @anthropic-ai/claude-code   (or see https://claude.com/download)" ;;
        opencode) echo "curl -fsSL https://opencode.ai/install | bash   (or see https://opencode.ai)" ;;
        aider) echo "python -m pip install aider-install && aider-install   (or see https://aider.chat/docs/install.html)" ;;
        cline) echo "npm install -g cline   (or see https://cline.bot)" ;;
        pi) echo "curl -fsSL https://pi.dev/install.sh | sh   (or see https://pi.dev)" ;;
    esac
}

tool_description() {
    case "$1" in
        claude) echo "Anthropic agentic coding CLI" ;;
        opencode) echo "Open-source terminal coding agent" ;;
        aider) echo "Git-integrated AI pair programmer" ;;
        cline) echo "Autonomous terminal coding agent" ;;
        pi) echo "Minimal extensible coding agent" ;;
        *) echo "" ;;
    esac
}

check_tool_installed() { is_command_available "$(tool_command "$1")"; }

tool_picker_server_summary() {
    local bold="${1:-}" dim="${2:-}" green="${3:-}" yellow="${4:-}" cyan="${5:-}" reset="${6:-}"
    if [[ ! -f "$MLX_STATE_FILE" ]]; then
        printf '%sServer: not running on %s:%s (selecting a tool will offer to start it)%s' \
            "$dim" "$MLX_HOST" "$MLX_PORT" "$reset"
        return
    fi
    local st_model="" st_ctx="" st_kv="" st_id="" st_backend=""
    IFS='|' read -r st_model st_ctx st_kv st_id st_backend < "$MLX_STATE_FILE" || true
    if [[ -z "$st_model" ]]; then
        printf '%sServer: state file empty on %s:%s%s' "$dim" "$MLX_HOST" "$MLX_PORT" "$reset"
        return
    fi
    local ctx_label="${st_ctx} tok"
    if [[ "$st_ctx" =~ ^[0-9]+$ ]] && (( st_ctx >= 1024 && st_ctx % 1024 == 0 )); then
        ctx_label="$((st_ctx / 1024))k ctx"
    fi
    local line="${green}●${reset} Server: ${bold}${st_model}${reset} ${dim}(${ctx_label} · ${MLX_HOST}:${MLX_PORT})${reset}"

    local verdict=""
    if [[ -f "$SCRIPT_DIR/models.json" ]]; then
        verdict="$("$MLX_PYTHON" -c '
import json, sys
path, m_id, hf_id = sys.argv[1:4]
try:
    with open(path) as f:
        models = json.load(f).get("models", [])
    for m in models:
        if m.get("id") == m_id or m.get("hf_model_id") == hf_id:
            print((m.get("tool_calling") or {}).get("verdict") or "")
            break
except Exception:
    pass
' "$SCRIPT_DIR/models.json" "${st_id:-}" "$st_model" 2>/dev/null || true)"
    fi
    if [[ "$verdict" == "NOT_SUPPORTED" ]]; then
        line+=$'\n'"${yellow}⚠ Loaded model does not support tool calling — agentic coding tools may fail.${reset}"
    fi
    printf '%s' "$line"
}

# Same "on writable $PATH" pattern as start_mlx_server.sh's ensure_symlinked_on_path.
ensure_symlinked_on_path() {
    local target_name
    target_name="$(basename "$SCRIPT_PATH")"
    local candidate
    # Homebrew-formula install (frank321/homebrew-bushpilot): the formula owns the
    # bin wrappers and the Cellar files are read-only -- don't self-symlink. See the
    # fuller comment in start_mlx_server.sh's ensure_symlinked_on_path.
    if [[ "$SCRIPT_PATH" == */Cellar/bushpilot/* ]] \
        || { [[ -n "${HOMEBREW_PREFIX:-}" ]] && [[ "$SCRIPT_PATH" == "$HOMEBREW_PREFIX"/* ]]; }; then
        return 0
    fi
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
    local bold="" dim="" green="" yellow="" cyan="" reset=""
    if [[ -z "${NO_COLOR:-}" ]]; then
        bold=$'\033[1m'
        dim=$'\033[2m'
        green=$'\033[32m'
        yellow=$'\033[33m'
        cyan=$'\033[36m'
        reset=$'\033[0m'
    fi

    local srv_summary
    srv_summary="$(tool_picker_server_summary "$bold" "$dim" "$green" "$yellow" "$cyan" "$reset")"

    local rows=() t name desc label
    for t in "${TOOLS[@]}"; do
        name="$(tool_display_name "$t")"
        desc="$(tool_description "$t")"
        if check_tool_installed "$t"; then
            label="$(printf "%s✓%s %-13s  %s%s%s" "$green" "$reset" "$name" "$dim" "$desc" "$reset")"
        else
            label="$(printf "%s· %-13s  (not installed) — %s%s" "$dim" "$name" "$desc" "$reset")"
        fi
        rows+=("$t"$'\t'"$label")
    done

    if is_command_available fzf; then
        local header selection
        header="${srv_summary}"$'\n'"${dim}↑↓ move · Enter select · Esc cancel${reset}"
        selection="$(printf '%s\n' "${rows[@]}" | fzf --ansi --delimiter=$'\t' --with-nth=2.. \
            --border=rounded --pointer="▶" \
            --header="$header" \
            --prompt="Launch which tool> " --height=~45% --layout=reverse)" || exit 1
        printf '%s' "$selection" | cut -f1
        return
    fi

    log_info "fzf not found -- falling back to a plain numbered menu."
    echo "  $srv_summary" >&2
    local i=1
    for t in "${TOOLS[@]}"; do
        name="$(tool_display_name "$t")"
        desc="$(tool_description "$t")"
        if check_tool_installed "$t"; then
            printf "  %d) %-13s  %s%s%s\n" "$i" "$name" "$dim" "$desc" "$reset" >&2
        else
            printf "  %d) %s%-13s  (not installed) — %s%s\n" "$i" "$dim" "$name" "$desc" "$reset" >&2
        fi
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

resolve_cmux_bin() {
    if [[ -n "${CMUX_BUNDLED_CLI_PATH:-}" && -x "${CMUX_BUNDLED_CLI_PATH}" ]]; then
        echo "${CMUX_BUNDLED_CLI_PATH}"
        return 0
    fi
    if is_command_available cmux; then
        command -v cmux
        return 0
    fi
    if [[ -x "/Applications/cmux.app/Contents/Resources/bin/cmux" ]]; then
        echo "/Applications/cmux.app/Contents/Resources/bin/cmux"
        return 0
    fi
    return 1
}

# Detects whether the current terminal supports opening a split pane or tab for the
# server dashboard while keeping the current terminal free for the coding tool.
# Honors BUSHPILOT_SPLIT_MODE=none|off|false|0 to force single-terminal fallback, or an
# explicit splitter name (cmux|tmux|wezterm|kitty|iterm2|ghostty|apple_terminal).
detect_terminal_splitter() {
    local mode="${BUSHPILOT_SPLIT_MODE:-}"
    case "$mode" in
        none|off|false|0)
            echo ""
            return 0
            ;;
        cmux|tmux|wezterm|kitty|iterm2|ghostty|apple_terminal)
            echo "$mode"
            return 0
            ;;
    esac

    if [[ -n "${CMUX_SURFACE_ID:-}" || -n "${CMUX_WORKSPACE_ID:-}" ]] && resolve_cmux_bin >/dev/null 2>&1; then
        echo "cmux"
        return 0
    fi
    if [[ -n "${TMUX:-}" ]] && is_command_available tmux; then
        echo "tmux"
        return 0
    fi
    if [[ -n "${WEZTERM_PANE:-}" ]] && is_command_available wezterm; then
        echo "wezterm"
        return 0
    fi
    if [[ -n "${KITTY_LISTEN_ON:-}" ]] && is_command_available kitty; then
        echo "kitty"
        return 0
    fi
    if [[ "${TERM_PROGRAM:-}" == "iTerm.app" || -n "${ITERM_SESSION_ID:-}" ]] && is_command_available osascript; then
        echo "iterm2"
        return 0
    fi
    if [[ "${TERM_PROGRAM:-}" == "ghostty" ]] && [[ -d "/Applications/Ghostty.app" || -d "$HOME/Applications/Ghostty.app" ]]; then
        echo "ghostty"
        return 0
    fi
    if [[ "${TERM_PROGRAM:-}" == "Apple_Terminal" ]] && is_command_available osascript; then
        echo "apple_terminal"
        return 0
    fi
    echo ""
}

splitter_display_name() {
    case "$1" in
        cmux) echo "cmux right split" ;;
        tmux) echo "tmux split pane" ;;
        wezterm) echo "WezTerm right split" ;;
        kitty) echo "kitty window" ;;
        iterm2) echo "iTerm2 vertical split" ;;
        ghostty) echo "Ghostty window" ;;
        apple_terminal) echo "Terminal.app tab" ;;
        *) echo "$1" ;;
    esac
}

spawn_server_pane() {
    local splitter="$1"
    local cmd_str
    printf -v cmd_str 'exec %q --from-roster' "$START_PROXY_SCRIPT"

    case "$splitter" in
        cmux)
            local cmux_bin
            cmux_bin="$(resolve_cmux_bin)" || return 1
            "$cmux_bin" new-split right --focus false --command "$cmd_str" >/dev/null
            ;;
        tmux)
            tmux split-window -h -d "$START_PROXY_SCRIPT" --from-roster
            ;;
        wezterm)
            wezterm cli split-pane --right -- "$START_PROXY_SCRIPT" --from-roster >/dev/null
            ;;
        kitty)
            kitty @ launch --type=window --keep-focus "$START_PROXY_SCRIPT" --from-roster >/dev/null
            ;;
        iterm2)
            osascript - "$cmd_str" <<'APPLESCRIPT' >/dev/null
on run argv
    set cmd to item 1 of argv
    tell application "iTerm2"
        tell current window
            set origSession to current session of current tab
            tell origSession
                split vertically with default profile command cmd
            end tell
            select origSession
        end tell
    end tell
end run
APPLESCRIPT
            ;;
        ghostty)
            local ghostty_app="/Applications/Ghostty.app"
            [[ -d "$ghostty_app" ]] || ghostty_app="$HOME/Applications/Ghostty.app"
            [[ -d "$ghostty_app" ]] || return 1
            open -na "$ghostty_app" --args -e "$START_PROXY_SCRIPT" --from-roster
            ;;
        apple_terminal)
            osascript - "$cmd_str" <<'APPLESCRIPT' >/dev/null
on run argv
    set cmd to item 1 of argv
    tell application "Terminal"
        do script cmd
    end tell
end run
APPLESCRIPT
            ;;
        *)
            return 1
            ;;
    esac
}

wait_for_spawned_proxy() {
    local poll_sec="${BUSHPILOT_WAIT_POLL_SEC:-1}"
    local spawn_timeout_polls="${BUSHPILOT_SPAWN_TIMEOUT_POLLS:-15}"
    local polls=0
    # The trailing backslash is a literal spinner frame (the |/-\ rotation), not
    # an escape -- a double-quoted string makes that unambiguous to shellcheck.
    local spin_chars="|/-\\"
    local spin_idx=0

    while true; do
        if [[ ! -f "$MLX_STARTING_PID_FILE" ]] && proxy_is_healthy; then
            [[ -t 1 ]] && printf '\r\033[K'
            log_success "Local model server is healthy on $MLX_HOST:$MLX_PORT."
            return 0
        fi

        if [[ ! -f "$MLX_STARTING_PID_FILE" ]]; then
            if proxy_is_healthy; then
                [[ -t 1 ]] && printf '\r\033[K'
                log_success "Local model server is healthy on $MLX_HOST:$MLX_PORT."
                return 0
            fi
            [[ -t 1 ]] && printf '\r\033[K'
            log_error "Server startup in split pane exited before becoming healthy. Check $BUSHPILOT_HOME/logs/mlx-server.log."
            return 1
        fi

        local start_pid=""
        start_pid="$(tr -d ' \t\r\n' < "$MLX_STARTING_PID_FILE" 2>/dev/null || true)"
        if [[ "$start_pid" =~ ^[0-9]+$ ]]; then
            if ! kill -0 "$start_pid" 2>/dev/null; then
                if proxy_is_healthy; then
                    rm -f "$MLX_STARTING_PID_FILE"
                    [[ -t 1 ]] && printf '\r\033[K'
                    log_success "Local model server is healthy on $MLX_HOST:$MLX_PORT."
                    return 0
                fi
                rm -f "$MLX_STARTING_PID_FILE"
                [[ -t 1 ]] && printf '\r\033[K'
                log_error "Server process (PID $start_pid) exited during startup. Check $BUSHPILOT_HOME/logs/mlx-server.log."
                return 1
            fi
        elif [[ "$start_pid" == "spawning" && "$polls" -ge "$spawn_timeout_polls" ]]; then
            rm -f "$MLX_STARTING_PID_FILE"
            [[ -t 1 ]] && printf '\r\033[K'
            log_error "Timed out waiting for split pane to start $(basename "$START_PROXY_SCRIPT")."
            return 1
        fi

        if [[ -t 1 ]]; then
            local sc="${spin_chars:$((spin_idx % 4)):1}"
            printf '\r[INFO] %s Waiting for model server to finish loading in split pane (%ds)...' "$sc" "$polls"
            spin_idx=$((spin_idx + 1))
        fi

        sleep "$poll_sec"
        polls=$((polls + 1))
    done
}

# Checks proxy/model health first; if down, confirms with the user. When a supported
# terminal splitter is available, runs the model picker in the current terminal,
# spawns the server dashboard in a split pane/tab, waits for it to pass smoke checks,
# and returns so the chosen coding tool launches right here. Otherwise `exec`s
# start_mlx_server.sh directly in this terminal.
ensure_proxy_running() {
    local tool="${1:-}"
    if proxy_is_healthy; then
        return 0
    fi
    log_info "No healthy local model server detected on $MLX_HOST:$MLX_PORT."
    local splitter
    splitter="$(detect_terminal_splitter)"
    local prompt_msg="Start it now with $(basename "$START_PROXY_SCRIPT")? [y/N] "
    if [[ -n "$splitter" ]]; then
        prompt_msg="Start it now in a $(splitter_display_name "$splitter") and launch $(tool_display_name "${tool:-tool}") here? [y/N] "
    fi
    local confirm
    read -r -p "$prompt_msg" confirm || confirm=""
    case "$confirm" in
        y|Y|yes|YES) ;;
        *)
            log_error "Not starting the server. Run it yourself first: $START_PROXY_SCRIPT"
            exit 1
            ;;
    esac

    if [[ -n "$splitter" ]]; then
        local select_args=(--select-only)
        [[ -n "$tool" ]] && select_args+=(--tool "$tool")
        "$START_PROXY_SCRIPT" "${select_args[@]}" || exit 1

        mkdir -p "$(dirname "$MLX_STARTING_PID_FILE")"
        rm -f "$MLX_STATE_FILE"
        printf 'spawning\n' > "$MLX_STARTING_PID_FILE"

        log_info "Opening server dashboard in $(splitter_display_name "$splitter")..."
        if spawn_server_pane "$splitter"; then
            wait_for_spawned_proxy || exit 1
            return 0
        fi

        rm -f "$MLX_STARTING_PID_FILE"
        log_info "Could not open $(splitter_display_name "$splitter") -- falling back to starting the dashboard in this terminal."
        log_info "Once it's ready, run $(basename "$SCRIPT_PATH") again in a different terminal."
        exec "$START_PROXY_SCRIPT" --from-roster
    fi

    log_info "Starting the local model server in this terminal -- it becomes the live dashboard."
    log_info "Once it's ready, run $(basename "$SCRIPT_PATH") again in a different terminal."
    if [[ -n "$tool" ]]; then
        exec "$START_PROXY_SCRIPT" --interactive --tool "$tool"
    fi
    exec "$START_PROXY_SCRIPT" --interactive
}

# Populates MLX_MODEL / CONTEXT_WINDOW / KV_BITS (plus optional LOADED_LOCAL_MODEL /
# LOADED_BACKEND when present) from the supervisor's state file -- MLX_MODEL here is
# already the real HF repo id (what start_mlx_server.sh writes), not a friendly
# models.json id.
LOADED_LOCAL_MODEL=""
LOADED_BACKEND=""
read_loaded_state() {
    if [[ ! -f "$MLX_STATE_FILE" ]]; then
        log_error "No state file at $MLX_STATE_FILE -- the server doesn't look like it was started by start_mlx_server.sh."
        exit 1
    fi
    IFS='|' read -r MLX_MODEL CONTEXT_WINDOW KV_BITS LOADED_LOCAL_MODEL LOADED_BACKEND < "$MLX_STATE_FILE"
}

# Validates preload fit at attach-time, not just liveness (docs/PHASE_PLAN.md Phase 3):
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
    # they're unnecessary (see docs/PHASE_PLAN.md).
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

    local roster_file="${MLX_ROSTER_FILE:-$BUSHPILOT_HOME/state/mlx-roster.json}"
    if [[ -f "$roster_file" ]]; then
        # Route background Haiku checks through the bushpilot-active alias so switching
        # the primary model via `/model <id>` in Claude Code never causes background
        # probes to trigger a hot-swap back to the startup model.
        export ANTHROPIC_SMALL_FAST_MODEL="bushpilot-active"
        export ANTHROPIC_DEFAULT_HAIKU_MODEL="bushpilot-active"
    fi

    # Scale the output reserve proportionally for smaller context windows (1/4 of
    # CONTEXT_WINDOW, clamped to [4096, CLAUDE_OUTPUT_TOKENS_RESERVE]) and export
    # CLAUDE_CODE_MAX_OUTPUT_TOKENS to match, so a 32k context window doesn't
    # reserve 32k for output and trip mlx_vlm.server's prompt_tokens + max_tokens
    # <= MAX_KV_SIZE check.
    local output_reserve=$(( CONTEXT_WINDOW / 4 ))
    (( output_reserve < 4096 )) && output_reserve=4096
    (( output_reserve > CLAUDE_OUTPUT_TOKENS_RESERVE )) && output_reserve=$CLAUDE_OUTPUT_TOKENS_RESERVE

    local usable_context=$(( CONTEXT_WINDOW - output_reserve ))
    (( usable_context < MIN_USABLE_CONTEXT )) && usable_context=$MIN_USABLE_CONTEXT
    export CLAUDE_CODE_MAX_CONTEXT_TOKENS="$usable_context"
    export CLAUDE_CODE_MAX_OUTPUT_TOKENS="$output_reserve"

    log_success "Launching Claude Code against $MLX_MODEL (context cap ${usable_context} tokens, reserving ${output_reserve} for output)"
    exec claude --model "$MLX_MODEL" "$@"
}

# Aider (aider.chat / github.com/Aider-AI/aider) connects to any OpenAI-compatible
# endpoint via two env vars (OPENAI_API_BASE + OPENAI_API_KEY, both read by the
# litellm layer aider is built on -- see aider.chat/docs/llms/openai-compat.html) plus
# a --model flag prefixed "openai/" to select that provider path (aider also has a
# separate ollama_chat/ path, not used here since the MLX server speaks plain OpenAI,
# not Ollama's API). OPENAI_API_KEY must be non-empty -- litellm checks for its mere
# presence client-side before ever making a request -- but its value is never
# validated against the local server, so any dummy string works.
#
# --model-metadata-file populates aider's self.info (used for /tokens UI and repo-map
# sizing), while --model-settings-file populates self.extra_params so aider's
# send_completion() actually passes max_tokens in the /v1/chat/completions HTTP body
# (without extra_params.max_tokens, aider omits max_tokens from the request and
# mlx_vlm.server falls back to DEFAULT_MAX_TOKENS=2048, truncating large edits).
AIDER_OUTPUT_TOKENS_RESERVE=32768

launch_aider() {
    export OPENAI_API_BASE="http://$MLX_HOST:$MLX_PORT/v1"
    export OPENAI_API_KEY="sk-mlx-direct-local"

    local output_reserve=$(( CONTEXT_WINDOW / 4 ))
    (( output_reserve < 4096 )) && output_reserve=4096
    (( output_reserve > AIDER_OUTPUT_TOKENS_RESERVE )) && output_reserve=$AIDER_OUTPUT_TOKENS_RESERVE

    local usable_context=$(( CONTEXT_WINDOW - output_reserve ))
    (( usable_context < MIN_USABLE_CONTEXT )) && usable_context=$MIN_USABLE_CONTEXT
    mkdir -p "$BUSHPILOT_HOME/state"
    local metadata_file="$BUSHPILOT_HOME/state/aider-model-metadata.json"
    local settings_file="$BUSHPILOT_HOME/state/aider-model-settings.yml"
    local roster_file="${MLX_ROSTER_FILE:-$BUSHPILOT_HOME/state/mlx-roster.json}"
    local pyscript
    pyscript="$(mktemp)"
    cat <<'PYEOF' > "$pyscript"
import json
import sys

model_id = sys.argv[1]
max_input = int(sys.argv[2])
max_output = int(sys.argv[3])
metadata_path = sys.argv[4]
settings_path = sys.argv[5]
roster_path = sys.argv[6] if len(sys.argv) > 6 else ""
reserve_cap = int(sys.argv[7]) if len(sys.argv) > 7 else 32768
min_usable = int(sys.argv[8]) if len(sys.argv) > 8 else 12288

entries = [(f"openai/{model_id}", max_input, max_output)]

try:
    with open(roster_path) as f:
        roster = json.load(f)
    for item in roster.get("models", []):
        ctx = int(item.get("recommended_context") or 0)
        if ctx <= 0:
            continue
        out_res = min(reserve_cap, max(4096, ctx // 4))
        in_res = max(min_usable, ctx - out_res)
        for key in (item.get("huggingface"), item.get("id")):
            if key:
                full = f"openai/{key}"
                if all(existing[0] != full for existing in entries):
                    entries.append((full, in_res, out_res))
except (FileNotFoundError, json.JSONDecodeError, OSError):
    pass

meta_dict = {}
settings_lines = []
for full_name, in_tok, out_tok in entries:
    meta_dict[full_name] = {
        "max_tokens": out_tok,
        "max_input_tokens": in_tok,
        "max_output_tokens": out_tok,
        "litellm_provider": "openai",
        "mode": "chat",
    }
    settings_lines.append(
        f"- name: {json.dumps(full_name)}\n"
        f"  extra_params:\n"
        f"    max_tokens: {out_tok}\n"
    )

with open(metadata_path, "w") as f:
    json.dump(meta_dict, f)
with open(settings_path, "w") as f:
    f.write("".join(settings_lines))
PYEOF
    "$MLX_PYTHON" "$pyscript" "$MLX_MODEL" "$usable_context" "$output_reserve" "$metadata_file" "$settings_file" "$roster_file" "$AIDER_OUTPUT_TOKENS_RESERVE" "$MIN_USABLE_CONTEXT"
    rm -f "$pyscript"

    log_success "Launching Aider against openai/$MLX_MODEL (context cap ${usable_context} tokens, reserving ${output_reserve} for output)"
    exec aider --model "openai/$MLX_MODEL" --model-metadata-file "$metadata_file" --model-settings-file "$settings_file" "$@"
}

# Cline (cline.bot / github.com/cline/cline) CLI persists provider credentials via a
# `cline auth` subcommand rather than reading connection env vars directly -- confirmed
# live against the documented example in apps/cli/README.md:
#   cline auth --provider openai-native --apikey sk-... --modelid gpt-5 --baseurl https://api.example.com/v1
# "openai-native" is the provider id for a plain OpenAI-compatible endpoint (as
# opposed to "openai-codex", which is OAuth-only against the real ChatGPT subscription
# backend). Like Claude/OpenCode, this needs some API key value even for a keyless
# local server -- forwarded as the Authorization header, never validated against our
# side. Because `cline auth` mutates ~/.cline/data/settings/providers.json on disk
# (setting lastUsedProvider="openai-native" and overwriting providers["openai-native"]),
# we snapshot providers.json beforehand and restore the previous non-BushPilot state on
# exit (while preserving any provider switch the user made inside Cline during the session).
snapshot_cline_providers() {
    local providers_file="$1"
    local snapshot_file="$2"
    [[ -f "$providers_file" ]] || return 0
    mkdir -p "$(dirname "$snapshot_file")"
    local pyscript
    pyscript="$(mktemp)"
    cat <<'PYEOF' > "$pyscript"
import json
import os
import shutil
import sys

providers_path, snapshot_path = sys.argv[1], sys.argv[2]
BUSHPILOT_KEY = "sk-mlx-direct-local"

try:
    with open(providers_path) as f:
        current = json.load(f)
except (FileNotFoundError, json.JSONDecodeError, OSError):
    sys.exit(0)

if not isinstance(current, dict):
    sys.exit(0)

providers = current.get("providers")
cur_entry = providers.get("openai-native") if isinstance(providers, dict) else None
cur_settings = (cur_entry.get("settings") if isinstance(cur_entry, dict) else None) or {}
if cur_settings.get("apiKey") == BUSHPILOT_KEY and os.path.isfile(snapshot_path):
    # Another BushPilot Cline session is already active (or exited abnormally before
    # restoring); keep the original pre-BushPilot backup rather than overwriting it
    # with BushPilot's temporary credentials.
    sys.exit(0)

try:
    tmp_path = f"{snapshot_path}.tmp.{os.getpid()}"
    shutil.copyfile(providers_path, tmp_path)
    os.replace(tmp_path, snapshot_path)
except OSError:
    pass
PYEOF
    "$MLX_PYTHON" "$pyscript" "$providers_file" "$snapshot_file" 2>/dev/null || true
    rm -f "$pyscript"
}

restore_cline_providers() {
    local providers_file="$1"
    local snapshot_file="$2"
    local pyscript
    pyscript="$(mktemp)"
    cat <<'PYEOF' > "$pyscript"
import json
import os
import sys

providers_path, snapshot_path = sys.argv[1], sys.argv[2]
BUSHPILOT_KEY = "sk-mlx-direct-local"

try:
    with open(providers_path) as f:
        current = json.load(f)
except (FileNotFoundError, json.JSONDecodeError, OSError):
    sys.exit(0)

if not isinstance(current, dict):
    sys.exit(0)

snapshot = None
if snapshot_path and os.path.isfile(snapshot_path):
    try:
        with open(snapshot_path) as f:
            snapshot = json.load(f)
    except (json.JSONDecodeError, OSError):
        snapshot = None

providers = current.get("providers")
if not isinstance(providers, dict):
    sys.exit(0)

cur_entry = providers.get("openai-native")
cur_settings = (cur_entry.get("settings") if isinstance(cur_entry, dict) else None) or {}
if cur_settings.get("apiKey") != BUSHPILOT_KEY:
    # User reconfigured openai-native with their own credentials during the session.
    sys.exit(0)

prev_entry = None
prev_last = None
if isinstance(snapshot, dict):
    snap_providers = snapshot.get("providers")
    if isinstance(snap_providers, dict):
        cand = snap_providers.get("openai-native")
        cand_settings = (cand.get("settings") if isinstance(cand, dict) else None) or {}
        if isinstance(cand, dict) and cand_settings.get("apiKey") != BUSHPILOT_KEY:
            prev_entry = cand
    cand_last = snapshot.get("lastUsedProvider")
    if isinstance(cand_last, str) and cand_last:
        if cand_last != "openai-native" or prev_entry is not None:
            prev_last = cand_last

if prev_entry is not None:
    providers["openai-native"] = prev_entry
else:
    providers.pop("openai-native", None)

if current.get("lastUsedProvider") == "openai-native":
    if prev_last is not None:
        current["lastUsedProvider"] = prev_last
    elif prev_entry is None:
        fallback = next(iter(providers.keys()), None)
        if fallback:
            current["lastUsedProvider"] = fallback
        else:
            current.pop("lastUsedProvider", None)

try:
    tmp_path = f"{providers_path}.tmp.{os.getpid()}"
    with open(tmp_path, "w") as f:
        json.dump(current, f, indent=2)
        f.write("\n")
    os.replace(tmp_path, providers_path)
except OSError:
    pass
PYEOF
    "$MLX_PYTHON" "$pyscript" "$providers_file" "$snapshot_file" 2>/dev/null || true
    rm -f "$pyscript" "$snapshot_file"
}

launch_cline() {
    local base_url="http://$MLX_HOST:$MLX_PORT/v1"
    CLINE_PROVIDERS_FILE="$HOME/.cline/data/settings/providers.json"
    CLINE_SNAPSHOT_FILE="$BUSHPILOT_HOME/state/cline-providers-backup.json"
    snapshot_cline_providers "$CLINE_PROVIDERS_FILE" "$CLINE_SNAPSHOT_FILE"

    local rc=0
    trap 'restore_cline_providers "$CLINE_PROVIDERS_FILE" "$CLINE_SNAPSHOT_FILE"' EXIT INT TERM HUP
    cline auth --provider openai-native --apikey sk-mlx-direct-local \
        --modelid "$MLX_MODEL" --baseurl "$base_url" > /dev/null

    log_success "Launching Cline against $MLX_MODEL (openai-native provider, baseURL $base_url)"
    cline -P openai-native -m "$MLX_MODEL" -k sk-mlx-direct-local "$@" || rc=$?
    trap - EXIT INT TERM HUP
    restore_cline_providers "$CLINE_PROVIDERS_FILE" "$CLINE_SNAPSHOT_FILE"
    exit "$rc"
}

# Pi (pi.dev / github.com/earendil-works/pi) registers custom OpenAI-compatible
# endpoints in ~/.pi/agent/models.json -- confirmed live against 1.0.1's own docs
# (pi.dev/docs/latest/models#configure-a-compatible-endpoint, pi.dev/docs/latest/
# configuration#agent-directory):
#   {"providers": {"<name>": {"baseUrl", "api": "openai-completions", "apiKey",
#     "compat": {"maxTokensField": "max_tokens"},
#     "models": [{"id": "<model-id>", "contextWindow": ..., "maxTokens": ..., "input": [...]}]}}}
# Same provider-registration shape this project already uses for OpenCode's
# mlx-direct provider (register_opencode_model below), just a different file/schema.
# Explicitly sets compat.maxTokensField="max_tokens" because Pi's detectCompat()
# defaults custom providers to "max_completion_tokens", which mlx_vlm.server's
# /v1/chat/completions schema ignores (falling back to max_tokens=2048).
register_pi_model() {
    local base_url="$1"
    local pi_models_config="$HOME/.pi/agent/models.json"
    local roster_file="${MLX_ROSTER_FILE:-$BUSHPILOT_HOME/state/mlx-roster.json}"
    mkdir -p "$(dirname "$pi_models_config")"
    log_info "Checking Pi's mlx-direct provider registration ($pi_models_config)..."

    local result pyscript
    pyscript="$(mktemp)"
    # Written to a temp file FIRST, as its own plain `cat <<'PYEOF' > "$pyscript"` with no
    # surrounding $(...) -- see the longer comment above register_opencode_model for why
    # combining a heredoc with $(...) in one statement is unsafe on the stock macOS
    # /bin/bash (3.2.57): never repeated here, just referenced.
    cat <<'PYEOF' > "$pyscript"
import json
import sys

config_path, model_id, context_window, base_url, models_json_path = (
    sys.argv[1],
    sys.argv[2],
    int(sys.argv[3]),
    sys.argv[4],
    sys.argv[5],
)
local_model_id = sys.argv[6] if len(sys.argv) > 6 else ""
loaded_backend = sys.argv[7] if len(sys.argv) > 7 else ""
roster_path = sys.argv[8] if len(sys.argv) > 8 else ""


def pi_input_modalities(models_json_path, hf_id, local_id="", backend=""):
    try:
        with open(models_json_path) as f:
            catalog = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return ["text"]
    all_models = catalog.get("models", [])
    entries = []
    if local_id:
        entries = [m for m in all_models if m.get("id") == local_id]
    if not entries:
        entries = [m for m in all_models if m.get("huggingface") == hf_id]
        if backend and len(entries) > 1:
            filtered = [
                m for m in entries
                if ((m.get("serving") or {}).get("backend") or "mlx_vlm") == backend
            ]
            if filtered:
                entries = filtered
    if not entries:
        return ["text"]
    capable_votes = set()
    all_mods = set()
    for entry in entries:
        if (entry.get("serving") or {}).get("multimodal_capable") is False:
            capable_votes.add(False)
        else:
            capable_votes.add(True)
            all_mods |= set(entry.get("modalities") or ["text"])
    if capable_votes == {True} and "image" in all_mods:
        return ["text", "image"]
    return ["text"]


def build_pi_entry(mid, ctx, hf_id, loc_id, backend):
    out_lim = min(32768, max(4096, ctx // 4))
    return {
        "id": mid,
        "name": f"[BushPilot] {mid}",
        "contextWindow": ctx,
        "maxTokens": out_lim,
        "input": pi_input_modalities(models_json_path, hf_id, loc_id, backend),
    }


expected_models = []
try:
    with open(roster_path) as f:
        roster_data = json.load(f)
    for item in roster_data.get("models", []):
        ctx = int(item.get("recommended_context") or context_window)
        hf_id = item.get("huggingface") or model_id
        loc_id = item.get("id") or ""
        backend = item.get("backend") or "mlx_vlm"
        exposed_id = loc_id if item.get("expose_catalog_id") and loc_id else hf_id
        if all(m["id"] != exposed_id for m in expected_models):
            expected_models.append(build_pi_entry(exposed_id, ctx, hf_id, loc_id, backend))
except (FileNotFoundError, json.JSONDecodeError, OSError):
    pass

if not expected_models:
    expected_models = [build_pi_entry(model_id, context_window, model_id, local_model_id, loaded_backend)]
elif all(m["id"] not in (model_id, local_model_id) for m in expected_models):
    expected_models.insert(0, build_pi_entry(model_id, context_window, model_id, local_model_id, loaded_backend))

try:
    with open(config_path) as f:
        parsed = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    parsed = {}

providers = parsed.setdefault("providers", {})
existing = providers.get("mlx-direct", {})
existing_models = existing.get("models", [])
already_correct = (
    existing.get("baseUrl") == base_url
    and existing.get("api") == "openai-completions"
    and existing.get("compat", {}).get("maxTokensField") == "max_tokens"
    and existing_models == expected_models
)
if already_correct:
    print("ALREADY_REGISTERED")
    sys.exit(0)

providers["mlx-direct"] = {
    "baseUrl": base_url,
    "api": "openai-completions",
    "apiKey": "sk-mlx-direct-local",
    "compat": {"maxTokensField": "max_tokens"},
    "models": expected_models,
}

with open(config_path, "w") as f:
    json.dump(parsed, f, indent=2)
    f.write("\n")
print("REGISTERED")
PYEOF
    result="$("$MLX_PYTHON" "$pyscript" "$pi_models_config" "$MLX_MODEL" "$CONTEXT_WINDOW" "$base_url" "$SCRIPT_DIR/models.json" "${LOADED_LOCAL_MODEL:-}" "${LOADED_BACKEND:-}" "$roster_file")" || true
    rm -f "$pyscript"

    case "$result" in
        ALREADY_REGISTERED)
            log_success "$MLX_MODEL already registered under mlx-direct -- no changes needed"
            ;;
        REGISTERED)
            log_success "Registered $MLX_MODEL in ~/.pi/agent/models.json's mlx-direct provider"
            ;;
        *)
            log_error "Could not auto-register $MLX_MODEL in $pi_models_config ($result)"
            log_error "Add it manually under providers.\"mlx-direct\".models, or Pi won't find the model."
            exit 1
            ;;
    esac
}

launch_pi() {
    local base_url="http://$MLX_HOST:$MLX_PORT/v1"
    register_pi_model "$base_url"

    log_success "Launching Pi against mlx-direct/$MLX_MODEL (context: ${CONTEXT_WINDOW} tokens)"
    exec pi --model "mlx-direct/$MLX_MODEL" "$@"
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
    local roster_file="${MLX_ROSTER_FILE:-$BUSHPILOT_HOME/state/mlx-roster.json}"
    mkdir -p "$(dirname "$opencode_config")"
    log_info "Checking OpenCode's mlx-direct provider registration ($opencode_config)..."

    local result pyscript
    pyscript="$(mktemp)"
    # Written to a temp file FIRST, as its own plain `cat <<'PYEOF' > "$pyscript"` with no
    # surrounding $(...) -- never fed to python directly inside a command-substitution
    # heredoc (i.e. never `x="$(cmd <<'EOF' ... EOF)"`). Confirmed live on the stock macOS
    # /bin/bash (3.2.57, the last GPLv2 release -- Apple has shipped this exact build for
    # over a decade and will not update it): that combination has a real bug where finding
    # the matching ")" for the $(...) scans straight through the heredocs own body for
    # bare parentheses, even though the heredocs delimiter is quoted and the body is
    # supposed to be fully literal. On a long, comment-heavy Python payload like this one
    # (parens in comments, in dict literals, in regexes) that scan can lose track and
    # consume way more of the real script than the heredoc itself, running fragments of
    # register_opencode_model_v2 below as literal shell commands instead of feeding this
    # script to python. Splitting "write the heredoc" and "run the command" into two
    # separate statements (neither of which combines a heredoc with $(...) in the same
    # statement) sidesteps the bug entirely rather than relying on any fragile balance of
    # characters inside the payload below.
    cat <<'PYEOF' > "$pyscript"
import json
import re
import sys

config_path, model_id, context_window, base_url, models_json_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
local_model_id = sys.argv[6] if len(sys.argv) > 6 else ""
loaded_backend = sys.argv[7] if len(sys.argv) > 7 else ""
roster_path = sys.argv[8] if len(sys.argv) > 8 else ""
output_limit = min(32768, max(4096, context_window // 4))

# Display strings for the OpenCode model picker. Provider name is the project name;
# each model's display name is prefixed with "[BushPilot] " so local models are
# immediately recognizable in flat/recent/search views alongside other providers.
MLX_DIRECT_PROVIDER_NAME = "BushPilot"
MLX_DIRECT_MODEL_NAME = lambda mid: f"[BushPilot] {mid}"


# Live-tested bug fix: without this, OpenCode custom-provider model entries have no way
# to know a model accepts images, so it silently never encodes drag-and-dropped / -f
# attached images into the request at all -- the mlx_vlm.server backend and the model
# itself handle images perfectly fine (confirmed live via a raw /v1/chat/completions
# request with an image_url, which came back with a correct description), but OpenCode
# never sends the image bytes because the model lacks the "attachment"/"modalities"
# capability flags its own config schema (opencode.ai/config.json) exposes for exactly
# this. Reproduced live: identical request through "opencode run -f <image> ..." against
# a vision-capable model registered WITHOUT these flags got "I cannot process image
# input"; the IDENTICAL request after adding attachment:true + modalities.input
# containing "image" got a correct description, and the request log on the server side
# confirmed images=1 only in the fixed case (images=0 before). Mirrors the
# model_input_label() helper in model_advisor.py: serving.multimodal_capable=False
# (every mlx_dspark/DFlash-text-only entry) narrows to text-only even if the underlying
# model has a vision tower; otherwise use that models "modalities" list directly.
def image_capable_modalities(models_json_path, hf_id, local_id="", backend=""):
    try:
        with open(models_json_path) as f:
            catalog = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    all_models = catalog.get("models", [])
    entries = []
    if local_id:
        entries = [m for m in all_models if m.get("id") == local_id]
    if not entries:
        entries = [m for m in all_models if m.get("huggingface") == hf_id]
        if backend and len(entries) > 1:
            filtered = [
                m for m in entries
                if ((m.get("serving") or {}).get("backend") or "mlx_vlm") == backend
            ]
            if filtered:
                entries = filtered
    if not entries:
        return None
    # When local_id (or backend) is recorded in mlx-loaded.txt, entries narrows to the
    # exact catalog variant running. If neither was recorded (e.g. legacy 3-field state
    # file) and multiple catalog entries share one huggingface id across serving modes
    # with different multimodal capability, require unanimous agreement before enabling
    # image input.
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


def build_v1_model_entry(mid, ctx, hf_id, loc_id, backend):
    out_lim = min(32768, max(4096, ctx // 4))
    mods = image_capable_modalities(models_json_path, hf_id, loc_id, backend)
    entry = {
        "name": MLX_DIRECT_MODEL_NAME(mid),
        "limit": {"context": ctx, "output": out_lim},
        "attachment": mods is not None,
    }
    if mods is not None:
        entry["modalities"] = {"input": mods, "output": ["text"]}
    return entry


roster_models_dict = {}
try:
    with open(roster_path) as f:
        roster_data = json.load(f)
    for item in roster_data.get("models", []):
        ctx = int(item.get("recommended_context") or context_window)
        hf_id = item.get("huggingface") or model_id
        loc_id = item.get("id") or ""
        backend = item.get("backend") or "mlx_vlm"
        exposed_id = loc_id if item.get("expose_catalog_id") and loc_id else hf_id
        if exposed_id not in roster_models_dict:
            roster_models_dict[exposed_id] = build_v1_model_entry(exposed_id, ctx, hf_id, loc_id, backend)
except (FileNotFoundError, json.JSONDecodeError, OSError):
    pass

has_roster = bool(roster_models_dict)
target_model_id = (
    local_model_id
    if has_roster and local_model_id and local_model_id in roster_models_dict
    else model_id
)
if has_roster and target_model_id not in roster_models_dict:
    roster_models_dict[target_model_id] = build_v1_model_entry(target_model_id, context_window, model_id, local_model_id, loaded_backend)

ATTACHMENT_MODALITIES = image_capable_modalities(models_json_path, model_id, local_model_id, loaded_backend)


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
if has_roster:
    models_match = existing.get("models") == roster_models_dict
else:
    models_match = (
        model_id in existing.get("models", {})
        and existing_model.get("name") == MLX_DIRECT_MODEL_NAME(model_id)
        and existing_model.get("limit", {}).get("context") == context_window
        and existing_model.get("limit", {}).get("output") == output_limit
        and existing_model.get("attachment") == (ATTACHMENT_MODALITIES is not None)
        and existing_model.get("modalities", {}).get("input") == ATTACHMENT_MODALITIES
    )
already_correct = (
    not had_litellm
    and existing.get("options", {}).get("baseURL") == base_url
    and existing.get("name") == MLX_DIRECT_PROVIDER_NAME
    and models_match
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
if has_roster:
    mlx_direct["models"] = roster_models_dict
else:
    models = mlx_direct.setdefault("models", {})
    models[model_id] = build_v1_model_entry(model_id, context_window, model_id, local_model_id, loaded_backend)

with open(config_path, "w") as f:
    json.dump(parsed, f, indent=2)
    f.write("\n")
print("REGISTERED")
PYEOF
    result="$("$MLX_PYTHON" "$pyscript" "$opencode_config" "$MLX_MODEL" "$CONTEXT_WINDOW" "$base_url" "$SCRIPT_DIR/models.json" "${LOADED_LOCAL_MODEL:-}" "${LOADED_BACKEND:-}" "$roster_file")" || true
    rm -f "$pyscript"

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

# v2 schema counterpart to register_opencode_model -- confirmed live against v2.0.22+
# (see opencode_cli_is_v2): "providers" (plural) instead of "provider", a "package"
# field instead of "npm" (and a different package name: @opencode/ai/providers/
# openai-compatible, not @ai-sdk/openai-compatible), and each model entry needs its
# own "modelID" rather than using the dict key as the id directly.
#
# IMPORTANT: Do NOT set the root "model" field in opencode.jsonc (and remove any
# existing root "model" that points to "mlx-direct/..."). OpenCode v2's new-session
# model resolution (Me() in Local provider) checks opencode.jsonc's root "model"
# BEFORE ~/.local/state/opencode/model.json's "recent" list, whereas switching models
# in OpenCode's TUI only updates model.json's "recent" list. Setting root "model" in
# opencode.jsonc permanently overrides the user's last-selected model on every new
# session even after BushPilot is stopped. Instead, we prepend {"providerID":
# "mlx-direct", "modelID": <model_id>} to "recent" in model.json on launch so
# OpenCode opens on the local model while still remembering any model the user switches
# to afterward.
register_opencode_model_v2() {
    local base_url="$1"
    local opencode_config="$HOME/.config/opencode/opencode.jsonc"
    local opencode_model_state="${XDG_STATE_HOME:-$HOME/.local/state}/opencode/model.json"
    local roster_file="${MLX_ROSTER_FILE:-$BUSHPILOT_HOME/state/mlx-roster.json}"
    mkdir -p "$(dirname "$opencode_config")"
    log_info "Checking OpenCode's mlx-direct provider registration ($opencode_config)..."

    local result pyscript
    pyscript="$(mktemp)"
    # See the matching comment in register_opencode_model above: this heredoc is written
    # to a temp file by itself (no $(...) wrapping it) and then run as a separate, plain
    # command substitution with no heredoc inside it -- never combine the two in one
    # statement here, or stock macOS /bin/bash (3.2.57) can lose track of this heredocs
    # own boundary while scanning for the $(...)'s closing paren.
    cat <<'PYEOF' > "$pyscript"
import json
import os
import re
import sys

config_path, model_id, context_window, base_url, models_json_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
local_model_id = sys.argv[6] if len(sys.argv) > 6 else ""
loaded_backend = sys.argv[7] if len(sys.argv) > 7 else ""
roster_path = sys.argv[8] if len(sys.argv) > 8 else ""
model_state_path = (
    sys.argv[9]
    if len(sys.argv) > 9 and sys.argv[9]
    else os.path.join(
        os.environ.get("XDG_STATE_HOME") or os.path.expanduser("~/.local/state"),
        "opencode",
        "model.json",
    )
)
output_limit = min(32768, max(4096, context_window // 4))

MLX_DIRECT_PROVIDER_NAME = "BushPilot"
MLX_DIRECT_MODEL_NAME = lambda mid: f"[BushPilot] {mid}"
MLX_DIRECT_PACKAGE = "@opencode/ai/providers/openai-compatible"


def image_capable_modalities(models_json_path, hf_id, local_id="", backend=""):
    try:
        with open(models_json_path) as f:
            catalog = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    all_models = catalog.get("models", [])
    entries = []
    if local_id:
        entries = [m for m in all_models if m.get("id") == local_id]
    if not entries:
        entries = [m for m in all_models if m.get("huggingface") == hf_id]
        if backend and len(entries) > 1:
            filtered = [
                m for m in entries
                if ((m.get("serving") or {}).get("backend") or "mlx_vlm") == backend
            ]
            if filtered:
                entries = filtered
    if not entries:
        return None
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


def build_v2_model_entry(mid, ctx, hf_id, loc_id, backend):
    out_lim = min(32768, max(4096, ctx // 4))
    mods = image_capable_modalities(models_json_path, hf_id, loc_id, backend)
    entry = {
        "name": MLX_DIRECT_MODEL_NAME(mid),
        "modelID": mid,
        "limit": {"context": ctx, "output": out_lim},
        "attachment": mods is not None,
    }
    if mods is not None:
        entry["modalities"] = {"input": mods, "output": ["text"]}
    return entry


roster_models_dict = {}
try:
    with open(roster_path) as f:
        roster_data = json.load(f)
    for item in roster_data.get("models", []):
        ctx = int(item.get("recommended_context") or context_window)
        hf_id = item.get("huggingface") or model_id
        loc_id = item.get("id") or ""
        backend = item.get("backend") or "mlx_vlm"
        exposed_id = loc_id if item.get("expose_catalog_id") and loc_id else hf_id
        if exposed_id not in roster_models_dict:
            roster_models_dict[exposed_id] = build_v2_model_entry(exposed_id, ctx, hf_id, loc_id, backend)
except (FileNotFoundError, json.JSONDecodeError, OSError):
    pass

has_roster = bool(roster_models_dict)
target_model_id = (
    local_model_id
    if has_roster and local_model_id and local_model_id in roster_models_dict
    else model_id
)
if has_roster and target_model_id not in roster_models_dict:
    roster_models_dict[target_model_id] = build_v2_model_entry(target_model_id, context_window, model_id, local_model_id, loaded_backend)


def update_model_state(state_path, target_mid):
    if not state_path:
        return False
    entry = {"providerID": "mlx-direct", "modelID": target_mid}
    try:
        with open(state_path) as f:
            state = json.load(f)
        if not isinstance(state, dict):
            state = {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        state = {}

    recent = state.get("recent")
    if not isinstance(recent, list):
        recent = []
    if recent and recent[0] == entry:
        return False

    new_recent = [entry]
    for item in recent:
        if (
            isinstance(item, dict)
            and isinstance(item.get("providerID"), str)
            and item["providerID"]
            and isinstance(item.get("modelID"), str)
            and item["modelID"]
        ):
            if item["providerID"] == "mlx-direct" and item["modelID"] == target_mid:
                continue
            new_recent.append({"providerID": item["providerID"], "modelID": item["modelID"]})
    state["recent"] = new_recent[:10]
    if not isinstance(state.get("favorite"), list):
        state["favorite"] = []
    if not isinstance(state.get("variant"), dict):
        state["variant"] = {}

    try:
        state_dir = os.path.dirname(state_path)
        if state_dir:
            os.makedirs(state_dir, exist_ok=True)
        tmp_path = f"{state_path}.tmp.{os.getpid()}"
        with open(tmp_path, "w") as f:
            json.dump(state, f)
        os.replace(tmp_path, state_path)
        return True
    except OSError:
        return False


ATTACHMENT_MODALITIES = image_capable_modalities(models_json_path, model_id, local_model_id, loaded_backend)


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

# Always place the active local model at the front of OpenCode's recent-models state
# (even when opencode.jsonc itself is already up to date), so launching via BushPilot
# opens on this model without locking opencode.jsonc's root "model" setting.
update_model_state(model_state_path, target_model_id)

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

# Also remove any root "model" pointing to mlx-direct written by earlier BushPilot
# versions -- leaving it in opencode.jsonc causes OpenCode v2 to override model.json's
# recent[0] on every new session even after the user switches models in the TUI.
raw_root_model = parsed.get("model")
had_v2_mlx_direct_model = (
    (isinstance(raw_root_model, str) and raw_root_model.startswith("mlx-direct/"))
    or (isinstance(raw_root_model, dict) and raw_root_model.get("providerID") == "mlx-direct")
)
if had_v2_mlx_direct_model:
    parsed.pop("model", None)

providers = parsed.setdefault("providers", {})
existing = providers.get("mlx-direct", {})
existing_model = existing.get("models", {}).get(model_id, {})
if has_roster:
    models_match = existing.get("models") == roster_models_dict
else:
    models_match = (
        model_id in existing.get("models", {})
        and existing_model.get("name") == MLX_DIRECT_MODEL_NAME(model_id)
        and existing_model.get("modelID") == model_id
        and existing_model.get("limit", {}).get("context") == context_window
        and existing_model.get("limit", {}).get("output") == output_limit
        and existing_model.get("attachment") == (ATTACHMENT_MODALITIES is not None)
        and existing_model.get("modalities", {}).get("input") == ATTACHMENT_MODALITIES
    )
already_correct = (
    not had_v1_mlx_direct
    and not had_v1_default_model
    and not had_v2_mlx_direct_model
    and existing.get("name") == MLX_DIRECT_PROVIDER_NAME
    and existing.get("package") == MLX_DIRECT_PACKAGE
    and existing.get("settings", {}).get("baseURL") == base_url
    and existing.get("compatibility", {}).get("maxTokensField") == "max_tokens"
    and models_match
)
if already_correct:
    print("ALREADY_REGISTERED")
    sys.exit(0)

mlx_direct = providers.setdefault("mlx-direct", {})
mlx_direct["name"] = MLX_DIRECT_PROVIDER_NAME
mlx_direct["package"] = MLX_DIRECT_PACKAGE
mlx_direct["settings"] = {"baseURL": base_url, "apiKey": "sk-mlx-direct-local"}
# OpenCode v2's @opencode/ai/providers/openai-compatible defaults maxTokensField to
# "max_completion_tokens" for custom providers, which mlx_vlm.server's ChatCompletionRequest
# ignores (falling back to DEFAULT_MAX_TOKENS=2048 and truncating large code responses
# mid-turn). Explicitly request the classic "max_tokens" wire field.
mlx_direct["compatibility"] = {"maxTokensField": "max_tokens"}
if has_roster:
    mlx_direct["models"] = roster_models_dict
else:
    models = mlx_direct.setdefault("models", {})
    models[model_id] = build_v2_model_entry(model_id, context_window, model_id, local_model_id, loaded_backend)

with open(config_path, "w") as f:
    json.dump(parsed, f, indent=2)
    f.write("\n")
print("REGISTERED")
PYEOF
    result="$("$MLX_PYTHON" "$pyscript" "$opencode_config" "$MLX_MODEL" "$CONTEXT_WINDOW" "$base_url" "$SCRIPT_DIR/models.json" "${LOADED_LOCAL_MODEL:-}" "${LOADED_BACKEND:-}" "$roster_file" "$opencode_model_state")" || true
    rm -f "$pyscript"

    case "$result" in
        ALREADY_REGISTERED)
            log_success "$MLX_MODEL already registered under mlx-direct -- selected as active model in $opencode_model_state"
            ;;
        REGISTERED)
            if is_command_available opencode; then
                opencode reload > /dev/null 2>&1 || true
            fi
            log_success "Registered $MLX_MODEL in opencode.jsonc's mlx-direct provider and selected it in $opencode_model_state"
            ;;
        *)
            log_error "Could not auto-register $MLX_MODEL in opencode.jsonc ($result)"
            log_error "Add it manually under providers.\"mlx-direct\".models, or OpenCode will fail with 'Model not found'."
            exit 1
            ;;
    esac
}

launch_opencode() {
    local base_url="http://$MLX_HOST:$MLX_PORT/v1"

    if opencode_cli_is_v2; then
        register_opencode_model_v2 "$base_url"
        log_success "Launching OpenCode against mlx-direct/$MLX_MODEL (context: ${CONTEXT_WINDOW} tokens)"
        # In OpenCode v2, bare `opencode` takes no --model flag (reads recent[0] from
        # ~/.local/state/opencode/model.json), whereas `opencode run` and `opencode mini`
        # still accept --model on the CLI.
        if [[ "${1:-}" == "run" || "${1:-}" == "mini" ]]; then
            local subcmd="$1"
            shift
            local has_model_flag=0 arg
            for arg in "$@"; do
                case "$arg" in
                    -m|--model|--model=*) has_model_flag=1; break ;;
                esac
            done
            if (( ! has_model_flag )); then
                exec opencode "$subcmd" --model "mlx-direct/$MLX_MODEL" "$@"
            fi
            exec opencode "$subcmd" "$@"
        fi
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
        claude|opencode|aider|cline|pi) ;;
        *)
            log_error "Unknown tool '$tool'. Supported: ${TOOLS[*]}"
            exit 1
            ;;
    esac

    if ! check_tool_installed "$tool"; then
        log_error "$(tool_display_name "$tool") is not installed -- install it first: $(tool_install_hint "$tool")"
        exit 1
    fi

    ensure_proxy_running "$tool"
    read_loaded_state
    check_preload_fit "$tool"

    case "$tool" in
        claude) launch_claude "$@" ;;
        opencode) launch_opencode "$@" ;;
        aider) launch_aider "$@" ;;
        cline) launch_cline "$@" ;;
        pi) launch_pi "$@" ;;
    esac
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
