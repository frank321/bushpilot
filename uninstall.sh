#!/usr/bin/env bash
set -euo pipefail

# uninstall.sh — removes everything Bushpilot installed on this machine:
#   1. Stops the model server if one is currently running
#   2. Removes the OpenCode "mlx-direct" provider registration (see
#      launch_tool.sh's register_opencode_model, which this reverses)
#   3. Removes the PATH symlinks start_mlx_server.sh/launch_tool.sh may have
#      created (see both scripts' ensure_symlinked_on_path)
#   4. Deletes ~/.bushpilot (downloaded model weights, Python venv, logs, state --
#      see start_mlx_server.sh's own header comment on why everything lives there)
#   5. Deletes this project folder itself, last
#
# Distributed inside the release zip (see docs/build_release.sh) -- deliberately
# self-contained (no sourcing of other files in this folder, no non-stdlib Python),
# since steps 4-5 remove the ground this script and everything around it is
# standing on. Safe to run from an extracted copy that's missing other files, and
# safe to run from a double-click/Terminal in any cwd.
#
# Usage: ./uninstall.sh [-y|--yes]
#   -y, --yes   skip the confirmation prompt (for scripted/unattended uninstalls)

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
BUSHPILOT_HOME="$HOME/.bushpilot"
OPENCODE_CONFIG="$HOME/.config/opencode/opencode.jsonc"

# Read the configured port from ~/.bushpilot/config.json when MLX_PORT wasn't
# explicitly overridden in the environment, falling back to start_mlx_server.sh's
# default (38472).
if [[ -z "${MLX_PORT:-}" ]] && [[ -f "$BUSHPILOT_HOME/config.json" ]] && command -v python3 > /dev/null 2>&1; then
    MLX_PORT="$(python3 -c '
import json, sys
try:
    with open(sys.argv[1]) as f:
        p = json.load(f).get("port")
    if isinstance(p, int) and p > 0:
        print(p)
except Exception:
    pass
' "$BUSHPILOT_HOME/config.json" 2>/dev/null || true)"
fi
MLX_PORT="${MLX_PORT:-38472}"
MLX_BACKEND_PORT="${MLX_BACKEND_PORT:-$((MLX_PORT + 1))}"
MLX_PROC_PATTERN="mlx_vlm\.server|mlx-dspark serve|bushpilot\.router"

log_info() { echo "[INFO] $*"; }
log_error() { echo "[ERROR] $*" >&2; }
log_success() { echo "[✓] $*"; }
is_command_available() { command -v "$1" > /dev/null 2>&1; }

ASSUME_YES=0
for arg in "$@"; do
    case "$arg" in
        -y|--yes) ASSUME_YES=1 ;;
        -h|--help)
            echo "Usage: $(basename "$0") [-y|--yes]"
            echo
            echo "Removes ~/.bushpilot, the OpenCode/Pi mlx-direct providers, this"
            echo "project's PATH symlinks, and this project folder."
            exit 0
            ;;
        *)
            log_error "Unknown argument: $arg"
            exit 1
            ;;
    esac
done

# Sanity guard before anything destructive below: SCRIPT_DIR is always wherever this
# file's own symlink-resolved path says it is, which should never be $HOME or /, but
# this is a recursive delete shipped to run on machines this project has no control
# over -- refuse rather than assume that always holds.
if [[ -z "$SCRIPT_DIR" || "$SCRIPT_DIR" == "$HOME" || "$SCRIPT_DIR" == "/" ]]; then
    log_error "Refusing to continue -- unexpected install location ($SCRIPT_DIR)."
    exit 1
fi

bushpilot_home_size="not present"
if [[ -d "$BUSHPILOT_HOME" ]]; then
    bushpilot_home_size="$(du -sh "$BUSHPILOT_HOME" 2>/dev/null | cut -f1)"
fi

echo "This will remove:"
echo "  - Any running Bushpilot model server"
echo "  - $BUSHPILOT_HOME (downloaded models, logs, state -- $bushpilot_home_size)"
echo "  - The mlx-direct provider entry in $OPENCODE_CONFIG, if present"
echo "  - The mlx-direct provider entry in ~/.pi/agent/models.json, if present"
echo "  - PATH symlinks to start_mlx_server.sh/launch_tool.sh, if any"
echo "  - This folder: $SCRIPT_DIR"
echo

if [[ "$ASSUME_YES" != "1" ]]; then
    reply=""
    read -r -p "Continue? [y/N] " reply </dev/tty || reply=""
    case "$reply" in
        y|Y|yes|YES) ;;
        *)
            log_info "Cancelled -- nothing was removed."
            exit 1
            ;;
    esac
fi

# --- 1. Stop a running server, if any ----------------------------------------------
stop_running_server() {
    local pids=""
    if is_command_available lsof; then
        pids="$(lsof -ti "tcp:$MLX_PORT" -ti "tcp:$MLX_BACKEND_PORT" 2>/dev/null | tr '\n' ' ' || true)"
    fi
    if [[ -z "${pids// /}" ]] && is_command_available pgrep; then
        pids="$(pgrep -f "$MLX_PROC_PATTERN" 2>/dev/null | tr '\n' ' ' || true)"
    fi
    if [[ -z "${pids// /}" ]]; then
        log_info "No running model server found"
        return 0
    fi

    log_info "Stopping running model server (PID(s): $pids)..."
    kill -TERM $pids > /dev/null 2>&1 || true
    local attempts=0
    while is_command_available pgrep && pgrep -f "$MLX_PROC_PATTERN" > /dev/null 2>&1; do
        attempts=$((attempts + 1))
        if [[ "$attempts" -ge 10 ]]; then
            log_info "Still running after 10s -- sending SIGKILL..."
            pkill -KILL -f "$MLX_PROC_PATTERN" > /dev/null 2>&1 || true
            break
        fi
        sleep 1
    done
    log_success "Model server stopped"
}

stop_running_server

# --- 2. Remove PATH symlinks --------------------------------------------------------
remove_symlinks() {
    local name candidate target resolved
    for name in start_mlx_server.sh launch_tool.sh; do
        for candidate in "$HOME/.local/bin" "/opt/homebrew/bin" "/usr/local/bin"; do
            target="$candidate/$name"
            [[ -L "$target" ]] || continue
            # Only remove it if it's actually OUR symlink -- never delete a real file,
            # and never delete a same-named symlink some other install put there.
            resolved="$(readlink "$target" 2>/dev/null || true)"
            if [[ "$resolved" == "$SCRIPT_DIR/$name" ]]; then
                rm -f "$target"
                log_success "Removed symlink $target"
            fi
        done
    done
}
remove_symlinks

# --- 3. Remove the OpenCode mlx-direct provider -------------------------------------
remove_opencode_provider() {
    if [[ ! -f "$OPENCODE_CONFIG" ]]; then
        log_info "No OpenCode config at $OPENCODE_CONFIG -- nothing to remove"
        return 0
    fi
    if ! is_command_available python3; then
        log_error "python3 not found -- could not check $OPENCODE_CONFIG for the mlx-direct provider"
        log_error "Remove provider.\"mlx-direct\" from it by hand if present."
        return 0
    fi

    local pyscript result
    # Two-step temp-file pattern rather than `$(python3 - ... <<'PYEOF')`: stock macOS
    # /bin/bash is 3.2, whose parser mis-counts single quotes inside heredocs nested
    # in $(...) command substitutions.
    pyscript="$(mktemp)"
    cat <<'PYEOF' > "$pyscript"
import json
import re
import sys

config_path = sys.argv[1]


# Same comment-stripping approach as launch_tool.sh's register_opencode_model, kept
# in sync deliberately -- both scripts edit the same jsonc file and must agree on
# how it's parsed.
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
    print("NO_CONFIG")
    sys.exit(0)

try:
    parsed = json.loads(strip_jsonc(raw))
except Exception as e:
    print(f"PARSE_ERROR: {e}")
    sys.exit(2)

provider = parsed.get("provider", {})
providers = parsed.get("providers", {})
had_v1 = "mlx-direct" in provider
had_v2 = "mlx-direct" in providers
if not had_v1 and not had_v2:
    print("NOT_REGISTERED")
    sys.exit(0)

# v1 schema ("provider" singular, "default_model") and v2 schema ("providers"
# plural, root "model") -- see opencode_cli_is_v2 in launch_tool.sh for why both
# exist. Checked independently (not elif) in case a machine has leftovers of both,
# e.g. from switching between OpenCode generations without ever uninstalling.
if had_v1:
    del provider["mlx-direct"]
    if not provider:
        parsed.pop("provider", None)
    if str(parsed.get("default_model", "")).startswith("mlx-direct/"):
        parsed.pop("default_model", None)
if had_v2:
    del providers["mlx-direct"]
    if not providers:
        parsed.pop("providers", None)
    if str(parsed.get("model", "")).startswith("mlx-direct/"):
        parsed.pop("model", None)

with open(config_path, "w") as f:
    json.dump(parsed, f, indent=2)
    f.write("\n")
print("REMOVED")
PYEOF
    result="$(python3 "$pyscript" "$OPENCODE_CONFIG" 2>/dev/null)" || true
    rm -f "$pyscript"

    case "$result" in
        REMOVED) log_success "Removed mlx-direct provider from $OPENCODE_CONFIG" ;;
        NOT_REGISTERED) log_info "No mlx-direct provider registered in OpenCode -- nothing to remove" ;;
        NO_CONFIG) log_info "No OpenCode config at $OPENCODE_CONFIG -- nothing to remove" ;;
        *)
            log_error "Could not clean up $OPENCODE_CONFIG ($result)"
            log_error "Remove \"mlx-direct\" from it by hand if present (under \"provider\" or \"providers\", depending on your OpenCode version)."
            ;;
    esac
}
remove_opencode_provider

# --- 4. Remove the Pi mlx-direct provider -------------------------------------------
remove_pi_provider() {
    local pi_config="$HOME/.pi/agent/models.json"
    if [[ ! -f "$pi_config" ]]; then
        return 0
    fi
    if ! is_command_available python3; then
        log_error "python3 not found -- could not check $pi_config for the mlx-direct provider"
        return 0
    fi

    local pyscript result
    pyscript="$(mktemp)"
    cat <<'PYEOF' > "$pyscript"
import json
import sys

config_path = sys.argv[1]
try:
    with open(config_path) as f:
        parsed = json.load(f)
except Exception:
    sys.exit(0)

providers = parsed.get("providers", {})
if "mlx-direct" not in providers:
    sys.exit(0)

del providers["mlx-direct"]
if not providers:
    parsed.pop("providers", None)

with open(config_path, "w") as f:
    json.dump(parsed, f, indent=2)
    f.write("\n")
print("REMOVED")
PYEOF
    result="$(python3 "$pyscript" "$pi_config" 2>/dev/null)" || true
    rm -f "$pyscript"

    if [[ "$result" == "REMOVED" ]]; then
        log_success "Removed mlx-direct provider from $pi_config"
    fi
}
remove_pi_provider

# --- 5. Remove ~/.bushpilot ----------------------------------------------------------
if [[ -d "$BUSHPILOT_HOME" ]]; then
    rm -rf "$BUSHPILOT_HOME"
    log_success "Removed $BUSHPILOT_HOME"
else
    log_info "$BUSHPILOT_HOME doesn't exist -- nothing to remove"
fi

# --- 6. Remove this project folder itself, last --------------------------------------
# Deleting the directory a running script's file lives in is safe on macOS/Linux --
# the shell already has this file open for reading, and an open file descriptor
# keeps working after its directory entry is unlinked. Nothing below this line reads
# another file from $SCRIPT_DIR, so there's nothing left to break.
log_success "Bushpilot uninstalled."
log_info "Removing $SCRIPT_DIR..."
cd "$HOME"
rm -rf "$SCRIPT_DIR"
