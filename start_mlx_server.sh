#!/usr/bin/env bash
set -euo pipefail

# start_mlx_server.sh — MLX-only model-serving supervisor.
#
# Renamed from start_litellm_proxy.sh (PHASE_PLAN.md Phase 4) — this script no longer
# starts or manages LiteLLM (or Postgres) at all, see PHASE_PLAN.md's "drop LiteLLM +
# Postgres entirely" decision. `mlx_vlm.server` natively serves both the OpenAI
# (/v1/chat/completions) and Anthropic (/v1/messages) wire formats, so there is no
# translation-layer proxy process needed anymore.
#
# This script's ONLY job: bring up mlx_vlm.server safely (memory-aware, won't crash
# the machine), show a live status dashboard in the foreground, and tear everything
# down cleanly on Ctrl-C / `q` / a closed terminal / a memory emergency. Picking a
# coding tool (Claude Code / OpenCode) and pointing it at this server is
# launch_tool.sh's job (PHASE_PLAN.md Phase 3) — multiple tools/terminals are meant to
# attach to the one shared server this starts.
#
# Usage:
#   start_mlx_server.sh [--interactive] [model_id] [huggingface_id] [context]
#
#   --interactive (default when no model_id is given): runs model_advisor.py, which
#     detects free wired memory + chip, shows every models.json entry (greying out
#     ones that don't currently fit) with a computed recommended context size and an
#     estimated tok/s, and lets you pick.
#
#   Fixed-model form (for scripting/restarts without the interactive picker):
#     start_mlx_server.sh qwen3.5-9b-mlx-8bit mlx-community/Qwen3.5-9B-MLX-8bit 131072
#
# Automatic Prefix Caching (APC) is always enabled — see HARDWARE_SPEED_ESTIMATION.md
# §13. KV-cache quantization is applied only when the interactively-recommended
# context doesn't already fit free memory unquantized (see model_advisor.py); the
# fixed-model form has no such computed decision and defaults to unquantized.

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"

INTERACTIVE=false
ARGS=()
for arg in "$@"; do
    case "$arg" in
        --interactive) INTERACTIVE=true ;;
        *) ARGS+=("$arg") ;;
    esac
done

LOCAL_MODEL="${ARGS[0]:-}"
MLX_MODEL="${ARGS[1]:-mlx-community/Qwen3.5-9B-MLX-8bit}"
CONTEXT_WINDOW="${ARGS[2]:-131072}"
# Static fallback only for the fixed-model path (model_advisor.py never runs there, so
# there's no computed decision to use). --interactive overwrites this with
# model_advisor.py's own per-model decision, which is empty (no quantization) whenever
# the recommended context already fits free memory unquantized.
KV_BITS=""

if [[ "$INTERACTIVE" == "false" && -z "$LOCAL_MODEL" ]]; then
    INTERACTIVE=true
fi

# ---------------------------------------------------------------------------
# Phase 1: single-home directory layout. Everything (venv, downloaded models, APC
# disk cache, state, logs) lives under ~/.local-model-setup/ so the whole install is
# removable by deleting one directory. Migrating an EXISTING scattered install
# (~/pip_env/mlx_lm, ~/mlx_models/~/MLX_Models, ~/.cache/mlx-vlm/caching,
# /tmp/mlx-llm-proxy-*) into this layout is a deliberate one-time manual step on this
# machine, not scripted here (see PHASE_PLAN.md's "Open / still deciding" resolution)
# — this script always uses the new paths below, full stop.
# ---------------------------------------------------------------------------
LMS_HOME="$HOME/.local-model-setup"
LMS_VENV="$LMS_HOME/venv"
LMS_STATE_DIR="$LMS_HOME/state"
LMS_LOG_DIR="$LMS_HOME/logs"
LMS_APC_DIR="$LMS_HOME/apc-cache"
export HF_HOME="$LMS_HOME/models"
mkdir -p "$LMS_VENV" "$LMS_STATE_DIR" "$LMS_LOG_DIR" "$LMS_APC_DIR" "$HF_HOME"

MLX_STATE_FILE="$LMS_STATE_DIR/mlx-loaded.txt"
MLX_LOG_FILE="$LMS_LOG_DIR/mlx-server.log"

MLX_PYTHON="$LMS_VENV/bin/python"
MLX_SERVER_MODULE="mlx_vlm.server"
MLX_HOST=127.0.0.1
MLX_PORT=11435

# Panel finding (GLM, high): the LiteLLM-removal decision rests on this specific
# installed mlx-vlm version's behavior (native /v1/messages routes, /metrics's
# ServerMetricsStore fields, /v1/cache/stats's APCStats fields). Pin a minimum verified
# version rather than installing unbounded.
MLX_VLM_MIN_VERSION="0.7.0"

KV_GROUP_SIZE=64
QUANTIZED_KV_START=0
PREFILL_STEP_SIZE=2048

# Automatic Prefix Caching (APC) — see HARDWARE_SPEED_ESTIMATION.md §13.
APC_ENABLED=1
APC_NUM_BLOCKS=4096
APC_DISK_PATH="$LMS_APC_DIR"
APC_DISK_MAX_GB=5

# Memory-pressure backstop thresholds — same values/rationale as the now-retired
# monitor_memory.sh (see MONITORING_ARCHITECTURE.md): kern.memorystatus_level is the
# kernel's own real-time percent-free figure, machine-size-normalized by construction.
MEM_ABORT_BELOW_PCT=10
MEM_WARN_BELOW_PCT=18
MEM_CAUTION_BELOW_PCT=25
# Require this many CONSECUTIVE polls in the abort zone before tearing down, to absorb
# a single noisy sample — except the discrete CRITICAL pressure-level override below,
# which is treated as unambiguous and acts immediately (same as monitor_memory.sh did).
MEM_ABORT_CONSECUTIVE_POLLS=3

DASHBOARD_POLL_INTERVAL=5

# Use Python's certifi CA bundle if available, fall back to system default.
CERT_PATH=$(python3 -c "import certifi; print(certifi.where())" 2>/dev/null) || true
if [[ -z "$CERT_PATH" || ! -f "$CERT_PATH" ]]; then
    for cert_file in /etc/ssl/cert.pem /usr/local/etc/openssl/cert.pem /opt/homebrew/etc/openssl@3/cert.pem; do
        if [ -f "$cert_file" ]; then
            CERT_PATH="$cert_file"
            break
        fi
    done
fi
if [[ -n "$CERT_PATH" ]]; then
    export SSL_CERT_FILE="$CERT_PATH"
    export REQUESTS_CA_BUNDLE="$CERT_PATH"
    export CURL_CA_BUNDLE="$CERT_PATH"
fi

# Logging functions
log_info() { echo "[INFO] $*"; }
log_error() { echo "[ERROR] $*" >&2; }
log_success() { echo "[✓] $*"; }

is_command_available() { command -v "$1" > /dev/null 2>&1; }

# ---------------------------------------------------------------------------
# Phase 1: symlink this script into a writable $PATH directory so it can be run from
# anywhere without typing a full path. Detects one that already exists rather than
# assuming — checked in the order a user is most likely to already have on $PATH.
# ---------------------------------------------------------------------------
ensure_symlinked_on_path() {
    local target_name="$(basename "$SCRIPT_PATH")"
    local candidate

    # Already reachable by name on $PATH (whether via this exact symlink or otherwise)
    # -> nothing to do.
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

    log_info "No writable \$PATH directory found to symlink $target_name into (checked ~/.local/bin, /opt/homebrew/bin, /usr/local/bin) — run it by full path instead."
}

wait_for_url() {
    local url="$1"
    local timeout="${2:-30}"
    local attempts=0
    while [ $attempts -lt "$timeout" ]; do
        if curl -fsS --max-time 5 "$url" > /dev/null 2>&1; then
            return 0
        fi
        attempts=$((attempts + 1))
        sleep 1
    done
    return 1
}

wait_for_process_gone() {
    local pattern="$1"
    local label="$2"
    local timeout="${3:-20}"
    local attempts=0

    if ! pgrep -f "$pattern" > /dev/null 2>&1; then
        return 0
    fi

    pkill -TERM -f "$pattern" > /dev/null 2>&1 || true
    log_info "Sent stop signal to $label, waiting for it to exit..."

    while pgrep -f "$pattern" > /dev/null 2>&1; do
        attempts=$((attempts + 1))
        if [ "$attempts" -ge "$timeout" ]; then
            log_info "$label still running after ${timeout}s, sending SIGKILL..."
            pkill -KILL -f "$pattern" > /dev/null 2>&1 || true
            sleep 1
            break
        fi
        sleep 1
    done

    if pgrep -f "$pattern" > /dev/null 2>&1; then
        log_error "$label did not exit"
        return 1
    fi
    log_success "$label stopped"
    return 0
}

# wait_for_process_gone only kills processes matching a COMMAND-LINE pattern -- but
# what actually determines whether a fresh bind() will succeed is whether anything is
# bound to the PORT, which can be a process that doesn't match the expected pattern at
# all. This checks (and if needed, kills) whatever is ACTUALLY bound to the port,
# independent of process name/command line.
wait_for_port_free() {
    local port="$1"
    local timeout="${2:-15}"
    local attempts=0
    local pids

    pids="$(lsof -ti "tcp:$port" 2>/dev/null || true)"
    if [[ -z "$pids" ]]; then
        return 0
    fi

    log_info "Port $port still held by PID(s): $pids -- sending stop signal..."
    kill -TERM $pids > /dev/null 2>&1 || true

    while true; do
        pids="$(lsof -ti "tcp:$port" 2>/dev/null || true)"
        if [[ -z "$pids" ]]; then
            return 0
        fi
        attempts=$((attempts + 1))
        if [ "$attempts" -ge "$timeout" ]; then
            log_info "Port $port still held after ${timeout}s -- sending SIGKILL to PID(s): $pids..."
            kill -KILL $pids > /dev/null 2>&1 || true
            sleep 1
            break
        fi
        sleep 1
    done

    pids="$(lsof -ti "tcp:$port" 2>/dev/null || true)"
    if [[ -n "$pids" ]]; then
        log_error "Port $port is still in use by PID(s): $pids"
        return 1
    fi
    return 0
}

check_system_dependencies() {
    log_info "Checking system dependencies..."
    if ! is_command_available python3; then
        log_error "python3 is not installed"
        exit 1
    fi
    if ! is_command_available curl; then
        log_error "curl is not installed"
        exit 1
    fi
    log_success "System dependencies OK"
}

# mlx-vlm>=0.7.0 requires Python >=3.10, but the system `python3` on this machine is
# 3.9.6 (confirmed live). Rather than a hardcoded version list that goes stale every
# time a new Python ships (confirmed live: mlx/mlx-vlm already publish real cp314
# wheels, not just cp310-cp313, so pinning to e.g. 3.12 would give up a real, already-
# available newer interpreter for no reason) -- this scans for whatever python3.X
# binaries actually exist on this machine and picks the highest-versioned one that
# qualifies (>=3.10). Newest-available is preferred on the theory that it's the
# best-maintained/best-supported choice going forward, not because newer is
# inherently better; if a future mlx/mlx-vlm release lags a very new Python and this
# ever picks one without real wheels yet, install_packages_if_needed's install step
# will fail loudly and visibly rather than silently falling back to a source build.
find_suitable_python() {
    local dir bin candidates=()
    for dir in /opt/homebrew/bin /usr/local/bin; do
        [ -d "$dir" ] || continue
        for bin in "$dir"/python3.[0-9]*; do
            [[ -x "$bin" && "$bin" != *-config ]] && candidates+=("$bin")
        done
    done
    is_command_available python3 && candidates+=("$(command -v python3)")

    local c ver best="" best_ver=""
    for c in "${candidates[@]}"; do
        ver="$("$c" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)" || continue
        "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null || continue
        if [[ -z "$best_ver" ]] || printf '%s\n%s\n' "$best_ver" "$ver" | sort -V | tail -1 | grep -qx "$ver"; then
            best="$c"
            best_ver="$ver"
        fi
    done
    [[ -n "$best" ]] && echo "$best" && return 0
    return 1
}

create_venv_if_needed() {
    if [ -d "$LMS_VENV" ] && [ -x "$MLX_PYTHON" ] \
        && "$MLX_PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
        log_info "MLX venv already exists"
        return 0
    fi

    if [ -d "$LMS_VENV" ]; then
        log_info "Existing MLX venv is built on a Python older than 3.10 (required by mlx-vlm>=${MLX_VLM_MIN_VERSION}) — rebuilding it..."
        rm -rf "$LMS_VENV"
    fi

    local py
    py="$(find_suitable_python)" || {
        log_error "No Python >=3.10 found (checked /opt/homebrew/bin, /usr/local/bin, and python3 on \$PATH)."
        log_error "Install one, e.g.: brew install python@3.12"
        exit 1
    }
    log_info "Creating MLX virtual environment at $LMS_VENV using $py ($("$py" --version 2>&1))..."
    "$py" -m venv "$LMS_VENV"
    log_success "MLX venv created"
}

# Fixed panel finding (Kimi, confirmed live): the old check ran
# `python -c "import $(echo $pkg | cut -d= -f1)"`, which for pkg="mlx-lm" literally
# executes `import mlx-lm` -- a Python SyntaxError, since hyphens aren't valid in a bare
# import identifier. That fast path NEVER succeeded, so every invocation silently
# re-ran a full pip install/upgrade even when already installed. Fixed by mapping the
# pip distribution name to its real import name explicitly.
dist_to_import_name() {
    case "$1" in
        mlx-lm) echo "mlx_lm" ;;
        mlx-vlm) echo "mlx_vlm" ;;
        *) echo "${1//-/_}" ;;
    esac
}

installed_version() {
    local import_name="$1"
    "$MLX_PYTHON" -c "import importlib.metadata as m; print(m.version('$import_name'))" 2>/dev/null || true
}

version_at_least() {
    local have="$1" want="$2"
    [[ -z "$have" ]] && return 1
    "$MLX_PYTHON" -c "
import sys
from packaging.version import Version
sys.exit(0 if Version('$have') >= Version('$want') else 1)
" 2>/dev/null
}

install_packages_if_needed() {
    local pkg import_name have

    for pkg in "mlx-lm" "mlx-vlm>=${MLX_VLM_MIN_VERSION}"; do
        local dist="${pkg%%[<>=]*}"
        import_name="$(dist_to_import_name "$dist")"
        log_info "Checking if $dist is installed in the MLX venv..."
        if "$MLX_PYTHON" -c "import $import_name" 2>/dev/null; then
            have="$(installed_version "$import_name")"
            if [[ "$dist" == "mlx-vlm" ]] && ! version_at_least "$have" "$MLX_VLM_MIN_VERSION"; then
                log_info "mlx-vlm $have is older than required minimum $MLX_VLM_MIN_VERSION -- upgrading..."
                "$LMS_VENV/bin/pip" install --upgrade "$pkg" > /dev/null
                log_success "mlx-vlm upgraded to $(installed_version mlx_vlm)"
            else
                log_info "$dist already installed (version $have)"
            fi
        else
            log_info "Installing $pkg..."
            "$LMS_VENV/bin/pip" install --upgrade pip setuptools wheel > /dev/null
            "$LMS_VENV/bin/pip" install "$pkg" > /dev/null
            log_success "$pkg installed"
        fi
    done

    # packaging is used by version_at_least above; certifi for the CA bundle lookup.
    if ! "$MLX_PYTHON" -c "import packaging" 2>/dev/null; then
        "$LMS_VENV/bin/pip" install packaging > /dev/null
    fi
}

bootstrap_mlx() {
    log_info "Bootstrapping MLX environment at $LMS_HOME..."
    create_venv_if_needed
    install_packages_if_needed
    if [ ! -x "$MLX_PYTHON" ]; then
        log_error "MLX Python executable not found: $MLX_PYTHON"
        exit 1
    fi
    log_success "MLX bootstrap complete"
}

# Phase 6: arrow-key model picker with a live detail pane, using fzf rather than
# hand-rolled TUI code -- see PHASE_PLAN.md for why fzf over stdlib curses. Prints
# model_advisor.py's own --resolve output (the same JSON contract the plain-menu path
# already produces) on success, or returns non-zero on cancel/failure.
#
# NOTE: deliberately NOT using --with-nth to hide the id column. Confirmed live
# (fzf 0.74.4) that --with-nth restricts fuzzy MATCHING to only the displayed fields,
# not just their presentation -- hiding the id column would make "type to filter" not
# match on model name at all, the opposite of the locked design's own mockup (which
# shows the id as the first visible column and explicitly promises "Type to filter").
# Showing all 4 tab fields (id/size/context/speed) keeps both display and search
# matching the mockup; `cut -f1` below extracts the id from the accepted line without
# depending on a newer fzf feature (--accept-nth) that may not exist on an older
# install.
run_fzf_model_picker() {
    local selection selected_id
    local preview_cmd="\"$MLX_PYTHON\" \"$SCRIPT_DIR/model_advisor.py\" --detail {1}"

    selection="$("$MLX_PYTHON" "$SCRIPT_DIR/model_advisor.py" --list-compact | fzf \
        --delimiter=$'\t' \
        --preview "$preview_cmd" --preview-window down:50%:wrap \
        --prompt="Model> " --height=~80% --layout=reverse)"
    if [[ -z "$selection" ]]; then
        log_error "Model selection cancelled or no model available."
        return 1
    fi

    selected_id="$(printf '%s' "$selection" | cut -f1)"
    if ! "$MLX_PYTHON" "$SCRIPT_DIR/model_advisor.py" --resolve "$selected_id"; then
        log_error "Could not resolve '$selected_id' -- free memory may have changed since the list was shown. Try again."
        return 1
    fi
}

# Interactive model selection (Phase 2: no --tool factored in here anymore -- the
# supervisor is deliberately tool-agnostic now; a tool-aware preload check moves to
# Phase 3's launch_tool at attach-time instead).
run_interactive_selection() {
    log_info "Detecting hardware and free memory for model recommendations..."

    local advisor_output
    if is_command_available fzf; then
        advisor_output="$(run_fzf_model_picker)" || exit 1
    elif ! advisor_output=$("$MLX_PYTHON" "$SCRIPT_DIR/model_advisor.py"); then
        log_error "Model selection cancelled or failed."
        exit 1
    fi

    MLX_MODEL=$("$MLX_PYTHON" -c "import json,sys; print(json.loads(sys.argv[1])['huggingface'])" "$advisor_output")
    LOCAL_MODEL=$("$MLX_PYTHON" -c "import json,sys; print(json.loads(sys.argv[1])['id'])" "$advisor_output")
    CONTEXT_WINDOW=$("$MLX_PYTHON" -c "import json,sys; print(json.loads(sys.argv[1])['recommended_context'])" "$advisor_output")
    KV_BITS=$("$MLX_PYTHON" -c "import json,sys; v=json.loads(sys.argv[1])['kv_bits']; print(v if v is not None else '')" "$advisor_output")

    local kv_bits_log="unquantized (full decode speed — recommended context already fits free memory)"
    if [[ -n "$KV_BITS" ]]; then
        kv_bits_log="quantized to ${KV_BITS}-bit (memory-constrained — needed to fit this context)"
    fi
    log_success "Selected $LOCAL_MODEL ($MLX_MODEL) — context set to ${CONTEXT_WINDOW} tokens (computed from your current free memory)"
    log_success "KV cache: $kv_bits_log"
}

# Pre-downloads the model via snapshot_download() BEFORE the server ever starts.
# mlx_vlm.server's own lazy on-demand download has no resume support when the process
# is killed mid-transfer; snapshot_download() run standalone does.
ensure_model_downloaded() {
    log_info "Checking whether $MLX_MODEL is already cached..."
    local already_cached
    already_cached=$("$MLX_PYTHON" -c "
import os, glob
pattern = os.path.expanduser('$HF_HOME/hub/models--${MLX_MODEL//\//--}/snapshots/*/config.json')
print('yes' if glob.glob(pattern) else 'no')
")
    if [[ "$already_cached" == "yes" ]]; then
        log_success "$MLX_MODEL already cached — no download needed"
        return 0
    fi

    log_info "Model not cached locally. Downloading $MLX_MODEL now (this can take several minutes for larger models)..."
    if ! "$MLX_PYTHON" -c "
from huggingface_hub import snapshot_download
snapshot_download('$MLX_MODEL')
"; then
        log_error "Download failed for $MLX_MODEL."
        exit 1
    fi
    log_success "Download complete: $MLX_MODEL"
}

# Returns 0 only if the process actually LISTENING on $MLX_PORT is our expected
# mlx_vlm.server process. /health alone is not sufficient -- confirmed live, a
# foreign, unmanaged server (e.g. a stray mlx_lm.server) can hold the port while
# answering /health and hang forever on real inference.
mlx_port_owner_is_expected() {
    local pids
    pids="$(lsof -ti "tcp:$MLX_PORT" 2>/dev/null || true)"
    [[ -z "$pids" ]] && return 1

    local pid
    for pid in $pids; do
        if ps -p "$pid" -o command= 2>/dev/null | grep -q "$MLX_SERVER_MODULE"; then
            return 0
        fi
    done
    return 1
}

# Returns 0 only if the server actually completes a real (tiny) inference request.
mlx_inference_responds() {
    local probe_model="${1:-$MLX_MODEL}"
    local response
    response="$(curl -fsS --max-time 30 "http://$MLX_HOST:$MLX_PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$probe_model\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":1}" \
        2>/dev/null)" || return 1

    [[ -n "$response" ]] || return 1
    grep -q '"choices"' <<<"$response"
}

# Startup smoke check (panel finding, GLM): after the server reports healthy, actually
# call /v1/messages, /metrics, and /v1/cache/stats once and verify the expected shape
# before declaring the supervisor ready -- catches behavior drift even within the
# allowed mlx-vlm version range, not just "too old".
run_startup_smoke_check() {
    log_info "Running startup smoke check against /v1/messages, /metrics, /v1/cache/stats..."

    local messages_resp
    messages_resp="$(curl -fsS --max-time 30 "http://$MLX_HOST:$MLX_PORT/v1/messages" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$MLX_MODEL\",\"max_tokens\":1,\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}]}" \
        2>/dev/null)" || { log_error "Smoke check: /v1/messages did not respond"; return 1; }
    if ! grep -q '"content"' <<<"$messages_resp"; then
        log_error "Smoke check: /v1/messages response missing expected 'content' field"
        return 1
    fi

    local metrics_resp
    metrics_resp="$(curl -fsS --max-time 10 "http://$MLX_HOST:$MLX_PORT/metrics" 2>/dev/null)" \
        || { log_error "Smoke check: /metrics did not respond"; return 1; }
    if ! grep -q '"summary"' <<<"$metrics_resp"; then
        log_error "Smoke check: /metrics response missing expected 'summary' field"
        return 1
    fi

    local cache_resp
    cache_resp="$(curl -fsS --max-time 10 "http://$MLX_HOST:$MLX_PORT/v1/cache/stats" 2>/dev/null)" \
        || { log_error "Smoke check: /v1/cache/stats did not respond"; return 1; }
    if ! grep -q '"token_hit_rate"' <<<"$cache_resp"; then
        log_error "Smoke check: /v1/cache/stats response missing expected 'token_hit_rate' field"
        return 1
    fi

    log_success "Smoke check passed — native /v1/messages, /metrics, /v1/cache/stats all behave as expected"
}

ensure_mlx_running() {
    log_info "Checking MLX server on $MLX_HOST:$MLX_PORT..."

    local desired_state="$MLX_MODEL|$CONTEXT_WINDOW|$KV_BITS"
    local server_up=false

    if curl -fsS --max-time 5 "http://$MLX_HOST:$MLX_PORT/health" > /dev/null 2>&1; then
        if mlx_port_owner_is_expected; then
            local loaded_state=""
            [[ -f "$MLX_STATE_FILE" ]] && loaded_state="$(cat "$MLX_STATE_FILE" 2>/dev/null)"
            local probe_model="${loaded_state%%|*}"
            [[ -z "$probe_model" || "$probe_model" == "$loaded_state" ]] && probe_model="$MLX_MODEL"

            if mlx_inference_responds "$probe_model"; then
                server_up=true
            else
                log_error "MLX server owns port $MLX_PORT and answers /health, but did NOT return a completion within the inference probe timeout:"
                log_error "  probed model: $probe_model -- the process is hung, not healthy"
                log_info "Restarting the hung MLX server..."
                wait_for_process_gone "$MLX_SERVER_MODULE" "MLX-VLM" 30
                rm -f "$MLX_STATE_FILE"
            fi
        else
            log_error "Port $MLX_PORT is answering /health, but it is NOT this script's mlx_vlm.server process:"
            local owner_pids
            owner_pids="$(lsof -ti "tcp:$MLX_PORT" 2>/dev/null || true)"
            if [[ -n "$owner_pids" ]]; then
                ps -p $owner_pids -o pid=,command= 2>/dev/null | sed 's/^/  /' >&2 || true
            fi
            log_info "Reclaiming port $MLX_PORT from the foreign process..."
            wait_for_port_free "$MLX_PORT" 15
        fi
    fi

    if [[ "$server_up" == "true" ]]; then
        local loaded_state=""
        [[ -f "$MLX_STATE_FILE" ]] && loaded_state="$(cat "$MLX_STATE_FILE" 2>/dev/null)"

        if [[ "$loaded_state" == "$desired_state" ]]; then
            log_success "MLX server already healthy with $MLX_MODEL (context: ${CONTEXT_WINDOW} tokens)"
            return 0
        fi

        log_error "A different model is already running on $MLX_HOST:$MLX_PORT:"
        if [[ -n "$loaded_state" ]]; then
            log_error "  currently loaded: ${loaded_state%%|*} (recorded state: $loaded_state)"
        else
            log_error "  currently loaded: unknown (server is up, but no recorded state -- started outside this script?)"
        fi
        log_error "  requested:         $MLX_MODEL (context: ${CONTEXT_WINDOW} tokens, kv-bits: ${KV_BITS:-unquantized})"

        if [[ "$INTERACTIVE" == "true" ]]; then
            local reply
            printf "Stop the running one and start %s instead? [y/N]: " "$MLX_MODEL" >&2
            read -r reply
            if [[ "$reply" =~ ^[Yy]$ ]]; then
                wait_for_process_gone "$MLX_SERVER_MODULE" "MLX-VLM" 30
                rm -f "$MLX_STATE_FILE"
            else
                log_error "Not stopping the running model. Stop it manually first (Ctrl-C / q in its dashboard)."
                exit 1
            fi
        else
            log_error "This script only supports one model at a time. Stop the running one first (Ctrl-C / q in its dashboard)."
            exit 1
        fi
    fi

    if [[ "$server_up" != "true" ]]; then
        log_info "Starting MLX server with model $MLX_MODEL..."
        wait_for_process_gone "$MLX_SERVER_MODULE" "MLX-VLM" 30
        rm -f "$MLX_STATE_FILE"
    fi

    wait_for_port_free "$MLX_PORT" 15

    local mlx_args=(
        --model "$MLX_MODEL"
        --host "$MLX_HOST"
        --port "$MLX_PORT"
        --max-kv-size "$CONTEXT_WINDOW"
        --prefill-step-size "$PREFILL_STEP_SIZE"
    )
    if [[ -n "$KV_BITS" ]]; then
        mlx_args+=(
            --kv-bits "$KV_BITS"
            --kv-group-size "$KV_GROUP_SIZE"
            --quantized-kv-start "$QUANTIZED_KV_START"
            --kv-quant-scheme uniform
        )
    fi

    mkdir -p "$APC_DISK_PATH"
    export APC_ENABLED APC_NUM_BLOCKS APC_DISK_PATH APC_DISK_MAX_GB

    "$MLX_PYTHON" -m "$MLX_SERVER_MODULE" "${mlx_args[@]}" \
        > "$MLX_LOG_FILE" 2>&1 &
    MLX_PID=$!

    if ! wait_for_url "http://$MLX_HOST:$MLX_PORT/health" 600; then
        log_error "MLX server failed to start. Check $MLX_LOG_FILE"
        tail -20 "$MLX_LOG_FILE" >&2
        exit 1
    fi

    if ! run_startup_smoke_check; then
        log_error "Startup smoke check failed — tearing down. Check $MLX_LOG_FILE"
        wait_for_process_gone "$MLX_SERVER_MODULE" "MLX-VLM" 30
        exit 1
    fi

    echo "$desired_state" > "$MLX_STATE_FILE"
    log_success "MLX server started and healthy"
}

# ---------------------------------------------------------------------------
# Live status dashboard + memory-pressure backstop + full teardown on
# Ctrl-C/q/HUP/TERM. See PHASE_PLAN.md Phase 2's locked dashboard layout.
# ---------------------------------------------------------------------------
DASHBOARD_START_TS=$(date +%s)
MEM_ABORT_STREAK=0
TEARDOWN_REASON=""

full_teardown() {
    local reason="${1:-}"
    [[ -n "$reason" ]] && echo -e "\n[!] Stopping: $reason"
    stty sane 2>/dev/null || true
    wait_for_process_gone "$MLX_SERVER_MODULE" "MLX-VLM" 20 || true
    rm -f "$MLX_STATE_FILE"
    exit 0
}

# kern.memorystatus_level: kernel's own real-time percent-free figure (0-100).
get_percent_free() { sysctl -n kern.memorystatus_level 2>/dev/null || echo ""; }
# kern.memorystatus_vm_pressure_level: 1=NORMAL, 2=WARN, 4=CRITICAL.
get_pressure_level() { sysctl -n kern.memorystatus_vm_pressure_level 2>/dev/null || echo ""; }
pressure_level_name() {
    case "$1" in
        1) echo "normal" ;;
        2) echo "warn" ;;
        4) echo "critical" ;;
        *) echo "unknown" ;;
    esac
}

get_mlx_rss_gb() {
    local pid="$1" rss_kb
    rss_kb=$(ps -p "$pid" -o rss= 2>/dev/null | tr -d ' ')
    [[ -z "$rss_kb" ]] && { echo ""; return; }
    awk -v k="$rss_kb" 'BEGIN { printf "%.1f", k/1024/1024 }'
}

fmt_num() { "$MLX_PYTHON" -c "print(f'{int(sys.argv[1]):,}')" "$1" 2>/dev/null || echo "$1"; }

# mlx_vlm.server (third-party) doesn't tag requests with any client/tool identity in
# its /metrics response, so we can't learn this from the API. What we CAN see
# ourselves: every tool talks to $MLX_PORT over a local TCP socket, so lsof on that
# port surfaces the PIDs holding a connection to it. Exclude the server's own PID
# (identified by matching $MLX_SERVER_MODULE in its command line, since $MLX_PID
# isn't set when we attached to an already-running server) and resolve the rest to
# process names -- that's "what tool is connected", best-effort, not request-level.
get_connected_clients() {
    local pids server_pid pid nm names
    pids="$(lsof -ti "tcp:$MLX_PORT" 2>/dev/null || true)"
    [[ -z "$pids" ]] && { echo ""; return; }

    server_pid=""
    for pid in $pids; do
        if ps -p "$pid" -o command= 2>/dev/null | grep -q "$MLX_SERVER_MODULE"; then
            server_pid="$pid"
            break
        fi
    done

    names=""
    for pid in $pids; do
        [[ "$pid" == "$server_pid" ]] && continue
        nm="$(ps -p "$pid" -o comm= 2>/dev/null)" || continue
        [[ -z "$nm" ]] && continue
        names+="$(basename "$nm")"$'\n'
    done
    [[ -z "$names" ]] && { echo ""; return; }
    printf '%s' "$names" | sort | uniq -c | awk '{count=$1; $1=""; sub(/^ /,""); printf "%s (%s), ", $0, count}' | sed 's/, $//'
}

render_dashboard() {
    local metrics_json cache_json
    metrics_json="$(curl -fsS --max-time 3 "http://$MLX_HOST:$MLX_PORT/metrics" 2>/dev/null || echo '{}')"
    cache_json="$(curl -fsS --max-time 3 "http://$MLX_HOST:$MLX_PORT/v1/cache/stats" 2>/dev/null || echo '{}')"

    local pct_free pressure pressure_name mlx_rss now_ts elapsed elapsed_fmt connected_clients
    pct_free="$(get_percent_free)"
    pressure="$(get_pressure_level)"
    pressure_name="$(pressure_level_name "${pressure:-0}")"
    mlx_rss="$(get_mlx_rss_gb "${MLX_PID:-}")"
    connected_clients="$(get_connected_clients)"
    now_ts=$(date +%s)
    elapsed=$((now_ts - DASHBOARD_START_TS))
    elapsed_fmt=$(printf "%02d:%02d:%02d" $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60)))

    clear
    "$MLX_PYTHON" - "$metrics_json" "$cache_json" "$MLX_MODEL" "$MLX_PORT" "$CONTEXT_WINDOW" \
        "$elapsed_fmt" "${APC_ENABLED}" "${pct_free:-?}" "$pressure_name" "${mlx_rss:-?}" "$TEARDOWN_REASON" \
        "$SCRIPT_DIR/launch_tool.sh" "${connected_clients:-}" <<'PYEOF'
import json, sys

metrics_raw, cache_raw, model, port, context, uptime, apc_on, pct_free, pressure, rss, teardown_reason, launch_cmd, connected = sys.argv[1:14]

def safe_json(s):
    try:
        return json.loads(s)
    except Exception:
        return {}

metrics = safe_json(metrics_raw)
cache = safe_json(cache_raw)
summary = metrics.get("summary", {})
latest = metrics.get("latest", {})

def num(v, default="—"):
    return f"{v:,}" if isinstance(v, (int, float)) else default

def rate(v, default="—"):
    return f"{v:.1f}" if isinstance(v, (int, float)) else default

W = 90
def line(s=""):
    s = s[:W - 4]
    print("│ " + s.ljust(W - 4) + " │")
def rule(ch="─"):
    print("├" + ch * (W - 2) + "┤")
def gb(v):
    return f"{v/1e9:.1f}" if isinstance(v, (int, float)) else "—"

print("┌" + "─" * (W - 2) + "┐")
line(f"local-model-setup  {model}  up {uptime}")
line(f"port {port}   context {context}   APC: {'on' if apc_on == '1' else 'off'}")
rule()
line(f"REQUESTS        completed {num(summary.get('requests_completed'), 0)}   "
     f"failed {num(summary.get('requests_failed'), 0)}   in-flight {num(summary.get('in_flight'), 0)}")
line(f"LIFETIME        prompt {num(summary.get('prompt_tokens_total'), 0)} tok   "
     f"generated {num(summary.get('generated_tokens_total'), 0)} tok   "
     f"avg decode {rate(summary.get('avg_decode_tok_s'), '—')} tok/s")
rule()
line(f"LATEST REQUEST  prefill {rate(latest.get('prefill_tok_s'))} tok/s   "
     f"decode {rate(latest.get('decode_tok_s'))} tok/s   ttft {rate(latest.get('ttft_s'))}s")
line(f"                elapsed {rate(latest.get('request_elapsed_s'))}s   "
     f"finish_reason: {latest.get('finish_reason', '—')}")
rule()
hits = cache.get("lookups_hit")
misses = cache.get("lookups_miss")
hit_rate = cache.get("token_hit_rate")
hit_rate_s = f"{hit_rate*100:.0f}%" if isinstance(hit_rate, (int, float)) else "—"
line(f"APC CACHE       hit rate {hit_rate_s}   hits {num(hits, 0)} / misses {num(misses, 0)}   "
     f"evictions {num(cache.get('evictions'), 0)}")
line(f"                mem pool {num(cache.get('pool_used'), 0)}/{num(cache.get('num_blocks'), 0)} blocks   "
     f"disk {gb(cache.get('disk_bytes'))}/{gb(cache.get('disk_max_bytes'))} GB")
rule()
line(f"SYSTEM          free mem {pct_free}%   pressure: {pressure}   MLX RSS {rss} GB   "
     f"peak {rate(latest.get('peak_memory_gb'))} GB")
rule()
line(f"CONNECTED       {connected if connected else 'no tool connected yet'}")
rule()
err = summary.get("last_error") or "none"
line(f"last error: {err}")
if teardown_reason:
    rule()
    line(f"⚠ {teardown_reason}")
print("└" + "─" * (W - 2) + "┘")
print(f"  To connect a tool: open another terminal and run {launch_cmd}")
print("  Ctrl-C / q : stop everything")
PYEOF
}

# Folds monitor_memory.sh's real-time backstop directly into this poll loop (Phase 2
# decision) — same abort/warn/caution percent-free thresholds, same discrete CRITICAL
# pressure-level override, applying the exact same full-teardown path as Ctrl-C/q.
check_memory_backstop() {
    local pct_free pressure
    pct_free="$(get_percent_free)"
    pressure="$(get_pressure_level)"

    if [[ -n "$pressure" && "$pressure" -eq 4 ]]; then
        TEARDOWN_REASON="Memory pressure CRITICAL — stopping to protect the machine."
        return 1
    fi

    if [[ -n "$pct_free" ]] && awk -v p="$pct_free" -v a="$MEM_ABORT_BELOW_PCT" 'BEGIN{exit !(p<a)}'; then
        MEM_ABORT_STREAK=$((MEM_ABORT_STREAK + 1))
        if [[ "$MEM_ABORT_STREAK" -ge "$MEM_ABORT_CONSECUTIVE_POLLS" ]]; then
            TEARDOWN_REASON="Free memory below ${MEM_ABORT_BELOW_PCT}% for ${MEM_ABORT_STREAK} consecutive checks — stopping to protect the machine."
            return 1
        fi
    else
        MEM_ABORT_STREAK=0
    fi
    return 0
}

run_dashboard() {
    trap 'full_teardown "Ctrl-C"' INT
    trap 'full_teardown "terminal closed (SIGHUP)"' HUP
    trap 'full_teardown "terminated"' TERM

    local old_stty
    old_stty=$(stty -g 2>/dev/null) || old_stty=""
    [[ -n "$old_stty" ]] && stty -icanon -echo min 0 time 0 2>/dev/null || true

    while true; do
        if ! check_memory_backstop; then
            render_dashboard
            full_teardown "$TEARDOWN_REASON"
        fi

        render_dashboard

        local key=""
        read -r -t "$DASHBOARD_POLL_INTERVAL" -n 1 key || true
        if [[ "$key" == "q" || "$key" == "Q" ]]; then
            full_teardown "stopped by user (q)"
        fi
    done
}

main() {
    check_system_dependencies
    ensure_symlinked_on_path
    bootstrap_mlx

    if [[ "$INTERACTIVE" == "true" ]]; then
        run_interactive_selection
    fi

    log_info "Local model server starting with $LOCAL_MODEL ($MLX_MODEL), context ${CONTEXT_WINDOW} tokens"

    ensure_model_downloaded
    ensure_mlx_running

    run_dashboard
}

main "$@"
