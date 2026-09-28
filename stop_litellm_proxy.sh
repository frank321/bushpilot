#!/usr/bin/env bash
set -euo pipefail

# Stops everything start_litellm_proxy.sh starts: the MLX-VLM server, the LiteLLM
# proxy, and its embedded PostgreSQL cluster. Constants below are mirrored from
# start_litellm_proxy.sh — keep them in sync if either script's ports/paths change.
#
# Unlike the older stop_llm_proxy.sh, this does NOT filter the MLX kill by model name.
# start_litellm_proxy.sh can launch any of the models in models.json (fixed arg or
# interactive pick), so a hardcoded model-name filter here would silently fail to match
# whatever model actually got loaded — pkill's `|| true` would swallow that mismatch,
# leave the MLX server (and its model's memory) running, while this script still
# reported success. Only one MLX-VLM server should ever be bound to this module at a
# time, so matching on the module name alone is both simpler and actually correct.

LITELLM_DIR="$HOME/pip_env/litellm"
LITELLM_VENV="$LITELLM_DIR/venv"
LITELLM_CONFIG="start-litellm-proxy-config.yaml"
LITELLM_PORT=4100
PG_DATA_DIR="$LITELLM_DIR/pgdata"
PG_HOST="127.0.0.1"
PG_PORT=5433
PG_BIN="$(brew --prefix postgresql@15 2>/dev/null)/bin"

MLX_SERVER_MODULE="mlx_vlm.server"
MLX_PORT=11435
MLX_STATE_FILE="/tmp/mlx-llm-proxy-loaded.txt"

log_info() { echo "[INFO] $*"; }
log_success() { echo "[✓] $*"; }
log_error() { echo "[ERROR] $*" >&2; }

# Sends SIGTERM to every process matching $pattern, then actually waits for them to
# exit (polling pgrep) instead of firing-and-forgetting -- pkill returning does NOT
# mean the process has exited yet. Without this wait, a start run launched right after
# a stop run can find the old process still alive (still holding its port / still
# "healthy") because it's mid-shutdown -- e.g. still unloading a model's weights from
# unified memory -- which is exactly the race that caused start_litellm_proxy.sh to
# refuse with "a different model is already running" immediately after a stop.
wait_for_process_gone() {
    local pattern="$1"
    local label="$2"
    local timeout="${3:-20}"
    local attempts=0

    if ! pgrep -f "$pattern" > /dev/null 2>&1; then
        log_info "$label was not running"
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

# Stop LiteLLM, but only the proxy launched with this configuration.
wait_for_process_gone "litellm.*$LITELLM_CONFIG" "LiteLLM" 20

# Stop the embedded PostgreSQL cluster used by LiteLLM.
if [ -x "$PG_BIN/pg_ctl" ] && "$PG_BIN/pg_ctl" -D "$PG_DATA_DIR" status > /dev/null 2>&1; then
    log_info "Stopping embedded PostgreSQL on $PG_HOST:$PG_PORT..."
    "$PG_BIN/pg_ctl" -D "$PG_DATA_DIR" -w stop > /tmp/pgserver-llm-proxy-stop.log 2>&1 || true
    log_success "PostgreSQL stopped"
else
    log_info "Embedded PostgreSQL is not running"
fi

# Stop the MLX-VLM server (whichever model it loaded) and release its KV cache.
# Longer timeout than LiteLLM's -- unloading a model's weights from unified memory can
# take a few seconds, especially for the larger models in models.json.
wait_for_process_gone "$MLX_SERVER_MODULE" "MLX-VLM" 30
# Clear the "what's loaded" state file start_litellm_proxy.sh uses to decide whether
# it can reuse an already-running server -- otherwise a stale record could make a
# later run wrongly believe the (now-stopped) old model is still loaded.
rm -f "$MLX_STATE_FILE"

log_success "LiteLLM proxy services stopped."
