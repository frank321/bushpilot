#!/usr/bin/env bats
# Tests for launch_tool.sh's provider registration and output-token scaling across
# OpenCode v2, Pi, Aider, and Claude Code.

load helpers.bash

LAUNCH_TOOL_SH="$REPO_ROOT/launch_tool.sh"

extract_launch_function() {
    local name="$1"
    awk -v fn="$name" '$0 == fn "() {" { p=1 } p { print } /<<'\''PYEOF'\''/ { hd=1 } /^PYEOF$/ { hd=0 } p && !hd && $0 == "}" { exit }' "$LAUNCH_TOOL_SH"
}

source_launch_functions() {
    local name tmpf
    tmpf="$(mktemp)"
    for name in "$@"; do
        extract_launch_function "$name" > "$tmpf"
        if [ ! -s "$tmpf" ]; then
            rm -f "$tmpf"
            echo "extract_launch_function: '$name' not found in $LAUNCH_TOOL_SH" >&2
            return 1
        fi
        # shellcheck disable=SC1090
        source "$tmpf"
    done
    rm -f "$tmpf"
}

setup() {
    TEST_TMPDIR="$(mktemp -d)"
    HOME="$TEST_TMPDIR/home"
    BUSHPILOT_HOME="$HOME/.bushpilot"
    mkdir -p "$HOME" "$BUSHPILOT_HOME"
    SCRIPT_DIR="$REPO_ROOT"
    MLX_PYTHON="python3"
    MLX_HOST="127.0.0.1"
    MLX_PORT="38472"
    MLX_MODEL="mlx-community/gemma-4-26b-a4b-it-4bit"
    CONTEXT_WINDOW=262144
    MIN_USABLE_CONTEXT=12288
    CLAUDE_OUTPUT_TOKENS_RESERVE=32000
    AIDER_OUTPUT_TOKENS_RESERVE=32768
    LOADED_LOCAL_MODEL="gemma-4-26b-a4b-it-4bit"
    LOADED_BACKEND="mlx_vlm"

    log_info() { :; }
    log_error() { echo "[ERROR] $*" >&2; }
    log_success() { :; }
}

teardown() {
    rm -rf "$TEST_TMPDIR"
}

@test "register_opencode_model_v2 sets compatibility.maxTokensField=max_tokens and upgrades existing entries" {
    source_launch_functions register_opencode_model_v2
    local cfg="$HOME/.config/opencode/opencode.jsonc"
    mkdir -p "$(dirname "$cfg")"

    # Seed an older v2 config that lacks compatibility.maxTokensField
    cat <<'EOF' > "$cfg"
{
  "$schema": "https://opencode.ai/config.json",
  "providers": {
    "mlx-direct": {
      "name": "BushPilot",
      "package": "@opencode/ai/providers/openai-compatible",
      "settings": {
        "baseURL": "http://127.0.0.1:38472/v1",
        "apiKey": "sk-mlx-direct-local"
      },
      "models": {
        "mlx-community/gemma-4-26b-a4b-it-4bit": {
          "modelID": "mlx-community/gemma-4-26b-a4b-it-4bit",
          "limit": {
            "context": 262144,
            "output": 32768
          },
          "attachment": true,
          "modalities": {
            "input": ["text", "image"],
            "output": ["text"]
          }
        }
      }
    }
  },
  "model": "mlx-direct/mlx-community/gemma-4-26b-a4b-it-4bit"
}
EOF

    call register_opencode_model_v2 "http://127.0.0.1:38472/v1"
    [ "$status" -eq 0 ]

    python3 -c '
import json, sys
with open(sys.argv[1]) as f:
    d = json.load(f)
p = d["providers"]["mlx-direct"]
assert p["compatibility"] == {"maxTokensField": "max_tokens"}, p
m = p["models"]["mlx-community/gemma-4-26b-a4b-it-4bit"]
assert m["name"] == "[BushPilot] mlx-community/gemma-4-26b-a4b-it-4bit", m
assert m["limit"] == {"context": 262144, "output": 32768}, m
' "$cfg"

    # Second call should be a no-op (ALREADY_REGISTERED)
    call register_opencode_model_v2 "http://127.0.0.1:38472/v1"
    [ "$status" -eq 0 ]
}

@test "register_pi_model sets compat.maxTokensField=max_tokens, contextWindow, maxTokens, and input modalities" {
    source_launch_functions register_pi_model
    local cfg="$HOME/.pi/agent/models.json"
    mkdir -p "$(dirname "$cfg")"

    # Seed an older Pi config that only had {"id": model_id}
    cat <<'EOF' > "$cfg"
{
  "providers": {
    "mlx-direct": {
      "baseUrl": "http://127.0.0.1:38472/v1",
      "api": "openai-completions",
      "apiKey": "sk-mlx-direct-local",
      "models": [
        {
          "id": "mlx-community/gemma-4-26b-a4b-it-4bit"
        }
      ]
    }
  }
}
EOF

    call register_pi_model "http://127.0.0.1:38472/v1"
    [ "$status" -eq 0 ]

    python3 -c '
import json, sys
with open(sys.argv[1]) as f:
    d = json.load(f)
p = d["providers"]["mlx-direct"]
assert p["compat"] == {"maxTokensField": "max_tokens"}, p
assert p["models"] == [{
    "id": "mlx-community/gemma-4-26b-a4b-it-4bit",
    "name": "[BushPilot] mlx-community/gemma-4-26b-a4b-it-4bit",
    "contextWindow": 262144,
    "maxTokens": 32768,
    "input": ["text", "image"],
}], p["models"]
' "$cfg"
}

@test "launch_aider writes both metadata and settings files with extra_params.max_tokens" {
    source_launch_functions launch_aider
    CONTEXT_WINDOW=65536
    # Replace `exec aider` in a subshell by shadowing aider and stripping `exec `
    local body
    body="$(extract_launch_function launch_aider | sed 's/exec aider/aider_stub/')"
    eval "$body"

    aider_args=""
    aider_stub() { aider_args="$*"; }

    call launch_aider
    [ "$status" -eq 0 ]
    [[ "$aider_args" == *"--model-metadata-file $BUSHPILOT_HOME/state/aider-model-metadata.json"* ]]
    [[ "$aider_args" == *"--model-settings-file $BUSHPILOT_HOME/state/aider-model-settings.yml"* ]]

    python3 -c '
import json, sys
with open(sys.argv[1]) as f:
    meta = json.load(f)
entry = meta["openai/mlx-community/gemma-4-26b-a4b-it-4bit"]
assert entry["max_tokens"] == 16384, entry
assert entry["max_output_tokens"] == 16384, entry
assert entry["max_input_tokens"] == 49152, entry
with open(sys.argv[2]) as f:
    settings = f.read()
assert "max_tokens: 16384" in settings, settings
' "$BUSHPILOT_HOME/state/aider-model-metadata.json" "$BUSHPILOT_HOME/state/aider-model-settings.yml"
}

@test "launch_claude scales output reserve on small context windows and exports CLAUDE_CODE_MAX_OUTPUT_TOKENS" {
    local body
    body="$(extract_launch_function launch_claude | sed 's/exec claude/claude_stub/')"
    eval "$body"

    claude_stub() { :; }

    CONTEXT_WINDOW=32768
    call launch_claude
    [ "$status" -eq 0 ]
    [ "$CLAUDE_CODE_MAX_OUTPUT_TOKENS" = "8192" ]
    [ "$CLAUDE_CODE_MAX_CONTEXT_TOKENS" = "24576" ]

    CONTEXT_WINDOW=262144
    call launch_claude
    [ "$status" -eq 0 ]
    [ "$CLAUDE_CODE_MAX_OUTPUT_TOKENS" = "32000" ]
    [ "$CLAUDE_CODE_MAX_CONTEXT_TOKENS" = "230144" ]
    [ "$ANTHROPIC_SMALL_FAST_MODEL" = "$MLX_MODEL" ]

    mkdir -p "$BUSHPILOT_HOME/state"
    echo '{"models": [{"id": "gemma-4-26b-a4b-it-4bit", "huggingface": "mlx-community/gemma-4-26b-a4b-it-4bit", "recommended_context": 262144}]}' > "$BUSHPILOT_HOME/state/mlx-roster.json"
    call launch_claude
    [ "$status" -eq 0 ]
    [ "$ANTHROPIC_SMALL_FAST_MODEL" = "bushpilot-active" ]
    [ "$ANTHROPIC_DEFAULT_HAIKU_MODEL" = "bushpilot-active" ]
}

@test "register_opencode_model_v2 and register_pi_model register all models from mlx-roster.json" {
    source_launch_functions register_opencode_model_v2 register_pi_model
    mkdir -p "$BUSHPILOT_HOME/state"
    cat <<'EOF' > "$BUSHPILOT_HOME/state/mlx-roster.json"
{
  "models": [
    {
      "id": "gemma-4-26b-a4b-it-4bit",
      "huggingface": "mlx-community/gemma-4-26b-a4b-it-4bit",
      "backend": "mlx_vlm",
      "recommended_context": 262144,
      "expose_catalog_id": false
    },
    {
      "id": "qwen3.5-9b-mlx-8bit",
      "huggingface": "mlx-community/Qwen3.5-9B-MLX-8bit",
      "backend": "mlx_vlm",
      "recommended_context": 131072,
      "expose_catalog_id": false
    }
  ]
}
EOF

    call register_opencode_model_v2 "http://127.0.0.1:38472/v1"
    [ "$status" -eq 0 ]

    call register_pi_model "http://127.0.0.1:38472/v1"
    [ "$status" -eq 0 ]

    python3 -c '
import json, sys
with open(sys.argv[1]) as f:
    oc = json.load(f)
oc_models = oc["providers"]["mlx-direct"]["models"]
assert set(oc_models.keys()) == {
    "mlx-community/gemma-4-26b-a4b-it-4bit",
    "mlx-community/Qwen3.5-9B-MLX-8bit",
}, oc_models
assert oc_models["mlx-community/gemma-4-26b-a4b-it-4bit"]["name"] == "[BushPilot] mlx-community/gemma-4-26b-a4b-it-4bit"
assert oc_models["mlx-community/Qwen3.5-9B-MLX-8bit"]["name"] == "[BushPilot] mlx-community/Qwen3.5-9B-MLX-8bit"

with open(sys.argv[2]) as f:
    pi = json.load(f)
pi_ids = [m["id"] for m in pi["providers"]["mlx-direct"]["models"]]
assert pi_ids == [
    "mlx-community/gemma-4-26b-a4b-it-4bit",
    "mlx-community/Qwen3.5-9B-MLX-8bit",
], pi_ids
pi_names = [m["name"] for m in pi["providers"]["mlx-direct"]["models"]]
assert pi_names == [
    "[BushPilot] mlx-community/gemma-4-26b-a4b-it-4bit",
    "[BushPilot] mlx-community/Qwen3.5-9B-MLX-8bit",
], pi_names
' "$HOME/.config/opencode/opencode.jsonc" "$HOME/.pi/agent/models.json"
}

@test "detect_terminal_splitter prioritizes cmux, tmux, wezterm, kitty, iterm2, apple_terminal, and honors BUSHPILOT_SPLIT_MODE" {
    source_launch_functions resolve_cmux_bin detect_terminal_splitter
    is_command_available() { command -v "$1" > /dev/null 2>&1; }

    local fake_cmux="$TEST_TMPDIR/fake-cmux"
    printf '#!/usr/bin/env bash\nexit 0\n' > "$fake_cmux"
    chmod +x "$fake_cmux"

    # Explicit disable via BUSHPILOT_SPLIT_MODE=none
    BUSHPILOT_SPLIT_MODE="none" CMUX_SURFACE_ID="surface:1" CMUX_BUNDLED_CLI_PATH="$fake_cmux" \
        run detect_terminal_splitter
    [ "$status" -eq 0 ]
    [ "$output" = "" ]

    # Explicit override via BUSHPILOT_SPLIT_MODE=tmux
    BUSHPILOT_SPLIT_MODE="tmux" CMUX_SURFACE_ID="" TMUX="" \
        run detect_terminal_splitter
    [ "$status" -eq 0 ]
    [ "$output" = "tmux" ]

    # cmux detection via CMUX_SURFACE_ID + CMUX_BUNDLED_CLI_PATH
    BUSHPILOT_SPLIT_MODE="" CMUX_SURFACE_ID="surface:1" CMUX_BUNDLED_CLI_PATH="$fake_cmux" \
        TMUX="" WEZTERM_PANE="" KITTY_LISTEN_ON="" TERM_PROGRAM="ghostty" \
        run detect_terminal_splitter
    [ "$status" -eq 0 ]
    [ "$output" = "cmux" ]

    # tmux detection when not in cmux
    is_command_available() { [[ "$1" == "tmux" ]]; }
    BUSHPILOT_SPLIT_MODE="" CMUX_SURFACE_ID="" CMUX_WORKSPACE_ID="" TMUX="/tmp/tmux-1000/default,123,0" \
        WEZTERM_PANE="" KITTY_LISTEN_ON="" TERM_PROGRAM="" \
        run detect_terminal_splitter
    [ "$status" -eq 0 ]
    [ "$output" = "tmux" ]

    # wezterm detection
    is_command_available() { [[ "$1" == "wezterm" ]]; }
    BUSHPILOT_SPLIT_MODE="" CMUX_SURFACE_ID="" CMUX_WORKSPACE_ID="" TMUX="" WEZTERM_PANE="1" \
        KITTY_LISTEN_ON="" TERM_PROGRAM="WezTerm" \
        run detect_terminal_splitter
    [ "$status" -eq 0 ]
    [ "$output" = "wezterm" ]

    # kitty detection
    is_command_available() { [[ "$1" == "kitty" ]]; }
    BUSHPILOT_SPLIT_MODE="" CMUX_SURFACE_ID="" CMUX_WORKSPACE_ID="" TMUX="" WEZTERM_PANE="" \
        KITTY_LISTEN_ON="unix:/tmp/kitty" TERM_PROGRAM="" \
        run detect_terminal_splitter
    [ "$status" -eq 0 ]
    [ "$output" = "kitty" ]

    # iTerm2 detection
    is_command_available() { [[ "$1" == "osascript" ]]; }
    BUSHPILOT_SPLIT_MODE="" CMUX_SURFACE_ID="" CMUX_WORKSPACE_ID="" TMUX="" WEZTERM_PANE="" \
        KITTY_LISTEN_ON="" TERM_PROGRAM="iTerm.app" ITERM_SESSION_ID="" \
        run detect_terminal_splitter
    [ "$status" -eq 0 ]
    [ "$output" = "iterm2" ]

    # Apple Terminal detection
    BUSHPILOT_SPLIT_MODE="" CMUX_SURFACE_ID="" CMUX_WORKSPACE_ID="" TMUX="" WEZTERM_PANE="" \
        KITTY_LISTEN_ON="" TERM_PROGRAM="Apple_Terminal" ITERM_SESSION_ID="" \
        run detect_terminal_splitter
    [ "$status" -eq 0 ]
    [ "$output" = "apple_terminal" ]
}

@test "wait_for_spawned_proxy succeeds when startup PID file clears and proxy_is_healthy passes, and fails fast if startup PID dies" {
    source_launch_functions wait_for_spawned_proxy
    MLX_STARTING_PID_FILE="$BUSHPILOT_HOME/state/mlx-starting.pid"
    START_PROXY_SCRIPT="$REPO_ROOT/start_mlx_server.sh"
    BUSHPILOT_WAIT_POLL_SEC="0.05"
    BUSHPILOT_SPAWN_TIMEOUT_POLLS="3"
    mkdir -p "$BUSHPILOT_HOME/state"

    # Case 1: Startup PID is alive on poll 1, then clears and server is healthy on poll 2
    local poll_count=0
    printf '%s\n' "$$" > "$MLX_STARTING_PID_FILE"
    proxy_is_healthy() {
        [[ ! -f "$MLX_STARTING_PID_FILE" ]]
    }
    (
        sleep 0.08
        rm -f "$MLX_STARTING_PID_FILE"
    ) &
    local bg_pid=$!
    call wait_for_spawned_proxy
    wait "$bg_pid" 2>/dev/null || true
    [ "$status" -eq 0 ]

    # Case 2: Startup PID is a dead process and proxy_is_healthy is false -> immediate failure
    printf '999999\n' > "$MLX_STARTING_PID_FILE"
    proxy_is_healthy() { return 1; }
    run wait_for_spawned_proxy
    [ "$status" -eq 1 ]
    [[ "$output" == *"Server process (PID 999999) exited during startup"* ]]

    # Case 3: Stuck in 'spawning' past timeout -> failure
    printf 'spawning\n' > "$MLX_STARTING_PID_FILE"
    proxy_is_healthy() { return 1; }
    run wait_for_spawned_proxy
    [ "$status" -eq 1 ]
    [[ "$output" == *"Timed out waiting for split pane"* ]]
}

@test "ensure_proxy_running auto-splits: runs --select-only in current terminal, spawns split pane with --from-roster, and aborts before split if picker is cancelled" {
    source_launch_functions splitter_display_name tool_display_name ensure_proxy_running
    MLX_STATE_FILE="$BUSHPILOT_HOME/state/mlx-loaded.txt"
    MLX_STARTING_PID_FILE="$BUSHPILOT_HOME/state/mlx-starting.pid"
    mkdir -p "$BUSHPILOT_HOME/state"

    local calls_log="$TEST_TMPDIR/calls.log"
    START_PROXY_SCRIPT="$TEST_TMPDIR/fake_start_mlx_server.sh"
    cat <<'EOF' > "$START_PROXY_SCRIPT"
#!/usr/bin/env bash
echo "start_mlx_server:$*" >> "$CALLS_LOG"
exit "${FAKE_SELECT_EXIT:-0}"
EOF
    chmod +x "$START_PROXY_SCRIPT"
    export CALLS_LOG="$calls_log"

    proxy_is_healthy() { return 1; }
    detect_terminal_splitter() { echo "cmux"; }
    spawn_server_pane() {
        echo "spawn_server_pane:$1" >> "$calls_log"
        rm -f "$MLX_STARTING_PID_FILE"
        return 0
    }
    wait_for_spawned_proxy() {
        echo "wait_for_spawned_proxy" >> "$calls_log"
        return 0
    }
    ensure_proxy_running_with_yes() {
        ensure_proxy_running "$1" <<<"y"
    }

    # Happy path: user confirms 'y', --select-only succeeds, split pane spawns, wait succeeds
    export FAKE_SELECT_EXIT=0
    run ensure_proxy_running_with_yes "claude"
    [ "$status" -eq 0 ]
    grep -q "^start_mlx_server:--select-only --tool claude$" "$calls_log"
    grep -q "^spawn_server_pane:cmux$" "$calls_log"
    grep -q "^wait_for_spawned_proxy$" "$calls_log"

    # Cancel path: if --select-only exits non-zero, spawn_server_pane must NOT be called
    : > "$calls_log"
    export FAKE_SELECT_EXIT=1
    run ensure_proxy_running_with_yes "claude"
    [ "$status" -eq 1 ]
    grep -q "^start_mlx_server:--select-only --tool claude$" "$calls_log"
    ! grep -q "^spawn_server_pane:" "$calls_log"
}

@test "load_primary_from_roster_file in start_mlx_server.sh loads resolved primary model settings" {
    source_backstop_functions apply_resolved_model_json load_primary_from_roster_file
    mkdir -p "$BUSHPILOT_HOME/state"
    MLX_ROSTER_FILE="$BUSHPILOT_HOME/state/mlx-roster.json"
    cat <<'EOF' > "$MLX_ROSTER_FILE"
{
  "primary_id": "qwen3.5-9b-mlx-8bit",
  "explicit_selection": true,
  "models": [
    {
      "id": "qwen3.5-9b-mlx-8bit",
      "huggingface": "mlx-community/Qwen3.5-9B-MLX-8bit",
      "recommended_context": 131072,
      "kv_bits": 8,
      "kv_bits_apc_caveat": false,
      "apc_num_blocks": 1024,
      "concurrency": 2,
      "prefill_step_size": 4096,
      "backend": "mlx_vlm"
    }
  ]
}
EOF

    call load_primary_from_roster_file
    [ "$status" -eq 0 ]
    [ "$LOCAL_MODEL" = "qwen3.5-9b-mlx-8bit" ]
    [ "$MLX_MODEL" = "mlx-community/Qwen3.5-9B-MLX-8bit" ]
    [ "$CONTEXT_WINDOW" = "131072" ]
    [ "$KV_BITS" = "8" ]
    [ "$APC_NUM_BLOCKS" = "1024" ]
    [ "$CONCURRENCY" = "2" ]
    [ "$PREFILL_STEP_SIZE" = "4096" ]
}
