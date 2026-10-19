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
}
