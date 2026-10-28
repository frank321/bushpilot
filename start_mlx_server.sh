#!/usr/bin/env bash
set -euo pipefail

# start_mlx_server.sh — MLX-only model-serving supervisor.
#
# Renamed from start_litellm_proxy.sh (docs/PHASE_PLAN.md Phase 4) — this script no longer
# starts or manages LiteLLM (or Postgres) at all, see docs/PHASE_PLAN.md's "drop LiteLLM +
# Postgres entirely" decision. `mlx_vlm.server` natively serves both the OpenAI
# (/v1/chat/completions) and Anthropic (/v1/messages) wire formats, so there is no
# translation-layer proxy process needed anymore.
#
# This script's ONLY job: bring up mlx_vlm.server safely (memory-aware, won't crash
# the machine), show a live status dashboard in the foreground, and tear everything
# down cleanly on Ctrl-C / `q` / a closed terminal / a memory emergency. Picking a
# coding tool (Claude Code / OpenCode) and pointing it at this server is
# launch_tool.sh's job (docs/PHASE_PLAN.md Phase 3) — multiple tools/terminals are meant to
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
# Automatic Prefix Caching (APC) is always enabled — see docs/HARDWARE_SPEED_ESTIMATION.md
# §13. KV-cache quantization is applied only when the interactively-recommended
# context doesn't already fit free memory unquantized (see model_advisor.py); the
# fixed-model form has no such computed decision and defaults to unquantized.

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"

INTERACTIVE=false
SELECT_ONLY=false
FROM_ROSTER=false
TOOL_FOR_ADVISOR=""
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --interactive) INTERACTIVE=true ;;
        --select-only) SELECT_ONLY=true; INTERACTIVE=true ;;
        --from-roster) FROM_ROSTER=true ;;
        --tool)
            shift
            TOOL_FOR_ADVISOR="${1:-}"
            ;;
        --tool=*)
            TOOL_FOR_ADVISOR="${1#--tool=}"
            ;;
        *) ARGS+=("$1") ;;
    esac
    shift
done

LOCAL_MODEL="${ARGS[0]:-}"
MLX_MODEL="${ARGS[1]:-mlx-community/Qwen3.5-9B-MLX-8bit}"
CONTEXT_WINDOW="${ARGS[2]:-131072}"
# Static fallback only for the fixed-model path (model_advisor.py never runs there, so
# there's no computed decision to use). --interactive overwrites this with
# model_advisor.py's own per-model decision, which is empty (no quantization) whenever
# the recommended context already fits free memory unquantized.
KV_BITS=""

if [[ "$FROM_ROSTER" == "true" ]]; then
    INTERACTIVE=false
elif [[ "$INTERACTIVE" == "false" && -z "$LOCAL_MODEL" ]]; then
    INTERACTIVE=true
fi

# ---------------------------------------------------------------------------
# Phase 1: single-home directory layout. Everything (venv, downloaded models, APC
# disk cache, state, logs) lives under ~/.bushpilot/ so the whole install is
# removable by deleting one directory. Migrating an EXISTING scattered install
# (~/pip_env/mlx_lm, ~/mlx_models/~/MLX_Models, ~/.cache/mlx-vlm/caching,
# /tmp/mlx-llm-proxy-*) into this layout is a deliberate one-time manual step on this
# machine, not scripted here (see docs/PHASE_PLAN.md's "Open / still deciding" resolution)
# — this script always uses the new paths below, full stop.
# ---------------------------------------------------------------------------
BUSHPILOT_HOME="$HOME/.bushpilot"
BUSHPILOT_VENV="$BUSHPILOT_HOME/venv"
BUSHPILOT_STATE_DIR="$BUSHPILOT_HOME/state"
BUSHPILOT_LOG_DIR="$BUSHPILOT_HOME/logs"
BUSHPILOT_APC_DIR="$BUSHPILOT_HOME/apc-cache"
export HF_HOME="$BUSHPILOT_HOME/models"
mkdir -p "$BUSHPILOT_VENV" "$BUSHPILOT_STATE_DIR" "$BUSHPILOT_LOG_DIR" "$BUSHPILOT_APC_DIR" "$HF_HOME"

MLX_STATE_FILE="$BUSHPILOT_STATE_DIR/mlx-loaded.txt"
MLX_ROSTER_FILE="$BUSHPILOT_STATE_DIR/mlx-roster.json"
MLX_STARTING_PID_FILE="$BUSHPILOT_STATE_DIR/mlx-starting.pid"
MLX_SWAP_REQUEST_FILE="$BUSHPILOT_STATE_DIR/mlx-swap-request.txt"
MLX_SWAP_ERROR_FILE="$BUSHPILOT_STATE_DIR/mlx-swap-error.txt"
# Written best-effort after every FRESH launch (never on an attach-to-already-healthy
# path, which never calls wait_for_url at all) with the measured process-start → first-
# healthy-response seconds -- "load time" per docs/NEXT_SESSION_HANDOFF_PREFILL_SWEEP.md's
# Tier 1 metrics table ("not tracked anywhere today" before this). sweep_model.py reads
# this file (if freshly written) to fill its own "load.load_time_s" field without
# needing to manage this script's own process lifecycle -- see that script's docstring.
MLX_LOAD_TIME_FILE="$BUSHPILOT_STATE_DIR/mlx-load-time.txt"
MLX_LOG_FILE="$BUSHPILOT_LOG_DIR/mlx-server.log"
# The log is truncated ONCE per run by start_run_log() (called from main), then every
# writer--this script's own profile/launch/teardown lines AND the engine's redirected
# stdout/stderr--APPENDS. This replaced a `> "$MLX_LOG_FILE"` on the engine launch,
# which would have erased anything logged before the process started.
start_run_log() {
    : > "$MLX_LOG_FILE"
    printf '%s - ===== BushPilot run start (pid $$) =====\n' "$(date '+%Y-%m-%d %H:%M:%S')" >> "$MLX_LOG_FILE"
}

MLX_PYTHON="$BUSHPILOT_VENV/bin/python"
MLX_SERVER_MODULE="mlx_vlm.server"
MLX_ROUTER_MODULE="bushpilot.router"

# Config file (optional): $BUSHPILOT_HOME/config.json can override host/port/preferred_model/
# concurrency/max_context_tokens below the built-in defaults, plus the mlx-dspark
# prefix-cache tuning keys (CONFIG_DSPARK_PC_*, documented at their use site).
# Precedence: MLX_HOST/MLX_PORT env var (if
# exported by the caller) > config.json > hardcoded default here. 38472 is deliberately
# NOT 8000/3000/5000/8080 (common dev-server ports) or 11434 (Ollama's default, one off
# from this project's old 11435) -- picked to avoid colliding with other local software.
#
# "concurrency" is left UNSET (empty CONFIG_CONCURRENCY, auto mode) by default rather
# than defaulting to 1 here -- auto mode still maximizes context for one slot first,
# then adds free bonus slots from tier-rounding leftover (see model_advisor.py's
# compute_bonus_concurrency), which is strictly better than a hardcoded 1 whenever
# there's slack. Set an explicit integer to force that many slots instead, trading
# context size for guaranteed concurrency.
#
# "max_context_tokens" is an optional persistent UPPER BOUND on the interactive
# picker's computed context -- the memory-fit-safe alternative to the fixed-model
# CLI form's 3rd positional arg (which bypasses that check entirely). The advisor's
# own free-memory math still runs underneath the cap, which can only shrink the
# result, never grow it past what the host can afford; like all computed contexts
# it rounds down to the nearest clean tier. Invalid values are ignored (the advisor
# falls back to no cap) rather than erroring the launch.
BUSHPILOT_CONFIG_FILE="$BUSHPILOT_HOME/config.json"
CONFIG_HOST=""
CONFIG_PORT=""
CONFIG_PREFERRED_MODEL=""
CONFIG_CONCURRENCY=""
CONFIG_MAX_CONTEXT=""
CONFIG_DSPARK_PC_SLOTS=""
CONFIG_DSPARK_PC_RUNGS=""
CONFIG_DSPARK_PC_DIR=""
CONFIG_DSPARK_PC_MAX_RAM_MB=""
if [[ -f "$BUSHPILOT_CONFIG_FILE" ]]; then
    eval "$(python3 - "$BUSHPILOT_CONFIG_FILE" <<'PYEOF'
import json, shlex, sys
try:
    with open(sys.argv[1]) as f:
        cfg = json.load(f)
except Exception:
    cfg = {}
for var, key in (("CONFIG_HOST", "host"), ("CONFIG_PORT", "port"),
                 ("CONFIG_PREFERRED_MODEL", "preferred_model"), ("CONFIG_CONCURRENCY", "concurrency"),
                 ("CONFIG_MAX_CONTEXT", "max_context_tokens"),
                 ("CONFIG_DSPARK_PC_SLOTS", "dspark_prefix_cache_slots"),
                 ("CONFIG_DSPARK_PC_RUNGS", "dspark_prefix_cache_rungs"),
                 ("CONFIG_DSPARK_PC_DIR", "dspark_prefix_cache_dir"),
                 ("CONFIG_DSPARK_PC_MAX_RAM_MB", "dspark_prefix_cache_max_ram_mb")):
    val = cfg.get(key)
    if val is not None:
        print(f"{var}={shlex.quote(str(val))}")
PYEOF
)"
fi
MLX_HOST="${MLX_HOST:-${CONFIG_HOST:-127.0.0.1}}"
MLX_PORT="${MLX_PORT:-${CONFIG_PORT:-38472}}"
MLX_BACKEND_PORT="${MLX_BACKEND_PORT:-$((MLX_PORT + 1))}"
# Resolved to an actual integer once model selection completes (interactive path) or
# right below (fixed-model path) -- 1 is the safe fallback if nothing ever sets it.
CONCURRENCY=1
# Every model_advisor.py invocation in this script (fzf picker, detail preview,
# --resolve, preferred_model) gets these flags, so config.json's advisor-affecting
# keys apply uniformly across every path that consults the advisor.
#
# NOTE on expanding this array: this script runs `set -u` and macOS's /bin/bash is
# 3.2.57, where "${ADVISOR_FLAGS[@]}" on an EMPTY array is an "unbound variable"
# error (fixed upstream only in bash 4.4). This bit for real before the fix: any
# interactive launch whose config.json set preferred_model but NOT concurrency
# crashed with 'CONCURRENCY_FLAG[@]: unbound variable' (reproduced against the
# pre-fix script). Every call site therefore uses the empty-safe idiom
# ${ADVISOR_FLAGS[@]+"${ADVISOR_FLAGS[@]}"} -- copy that form for any new use.
ADVISOR_FLAGS=()
if [[ -n "$CONFIG_CONCURRENCY" ]]; then
    ADVISOR_FLAGS+=(--concurrency "$CONFIG_CONCURRENCY")
    [[ "$INTERACTIVE" == "false" ]] && CONCURRENCY="$CONFIG_CONCURRENCY"
fi
[[ -n "$CONFIG_MAX_CONTEXT" ]] && ADVISOR_FLAGS+=(--max-context-tokens "$CONFIG_MAX_CONTEXT")
# model_advisor.py's context-sizing math needs to know mlx-dspark's own prefix-cache
# slot count too (see bushpilot/kv.py's prefix_cache_cost_multiplier) -- otherwise it
# would size context for a single live KV cache while the actual mlx-dspark launch
# below keeps up to this many MORE full-size cached conversation snapshots resident at
# once, understating real peak memory (confirmed live: this exact gap is what drove the
# repeated memory-guard sheds documented at DSPARK_DEFAULT_PREFIX_CACHE_SLOTS). Unset
# here means "let the advisor assume mlx-dspark's own default," matching what actually
# gets launched below when CONFIG_DSPARK_PC_SLOTS is also unset.
[[ -n "$CONFIG_DSPARK_PC_SLOTS" ]] && ADVISOR_FLAGS+=(--prefix-cache-slots "$CONFIG_DSPARK_PC_SLOTS")
[[ -n "$TOOL_FOR_ADVISOR" ]] && ADVISOR_FLAGS+=(--tool "$TOOL_FOR_ADVISOR")

# Panel finding (GLM, high): the LiteLLM-removal decision rests on this specific
# installed mlx-vlm version's behavior (native /v1/messages routes, /metrics's
# ServerMetricsStore fields, /v1/cache/stats's APCStats fields). Pin a minimum verified
# version rather than installing unbounded.
MLX_VLM_MIN_VERSION="0.7.0"
# Same principle for the second backend: every mlx-dspark behavior this script relies
# on (flag set via its own --help, /metrics shape, dflash mode, prefix-cache defaults)
# was verified against this exact version on the reference machine. Checked only when
# a resolved model actually needs the mlx_dspark backend (ensure_backend_installed),
# unlike mlx-vlm which the unconditional bootstrap always installs.
MLX_DSPARK_MIN_VERSION="0.20.1"

KV_GROUP_SIZE=64
QUANTIZED_KV_START=0
# Static fallback for the fixed-model path and for any (chip, model) pair with no
# sweep calibration yet. The --interactive path overrides this per-launch with
# model_advisor.py's recommend_prefill_step_size() -- a real measurement from
# sweep_model.py's per-(chip, model) sweep when one exists on this machine, falling
# back to this same global default otherwise. See
# docs/NEXT_SESSION_HANDOFF_PREFILL_SWEEP.md for why a single hardcoded global constant
# here (derived from 2 Gemma-4 models on this one M5 Pro) doesn't generalize across
# chip architectures, and `python3 sweep_model.py --model-id <id>` to calibrate one.
PREFILL_STEP_SIZE=2048

# Automatic Prefix Caching (APC) — see docs/HARDWARE_SPEED_ESTIMATION.md §13.
# APC_NUM_BLOCKS is paired with MLX_VLM_APC_NUM_BLOCKS in bushpilot/kv.py; the
# interactive advisor path overrides this with MLX_VLM_APC_CONSTRAINED_NUM_BLOCKS
# (1024) when free memory is too tight for the full 4096-block pool.
APC_ENABLED=1
APC_NUM_BLOCKS=4096
APC_DISK_PATH="$BUSHPILOT_APC_DIR"
APC_DISK_MAX_GB=5

# Memory-pressure backstop thresholds — same values/rationale as the now-retired
# monitor_memory.sh (see docs/MONITORING_ARCHITECTURE.md): kern.memorystatus_level is the
# kernel's own real-time percent-free figure, machine-size-normalized by construction.
MEM_ABORT_BELOW_PCT=10
MEM_WARN_BELOW_PCT=18
MEM_CAUTION_BELOW_PCT=25
# Require this many CONSECUTIVE polls in the abort zone before tearing down, to absorb
# a single noisy sample. The discrete CRITICAL pressure-level override below applies
# the SAME debounce (it originally acted on a single sample): kern.memorystatus_vm_
# pressure_level is a momentary high-water signal under active memory use, not a
# stable state, so one transient CRITICAL reading shouldn't kill a session on its
# own -- sustained CRITICAL (3 consecutive 5s polls) still tears down.
MEM_ABORT_CONSECUTIVE_POLLS=3
MEM_CRITICAL_CONSECUTIVE_POLLS=3
# When tearing down due to a memory-backstop emergency, only wait this many seconds
# after SIGTERM before escalating to SIGKILL so wired GPU memory is freed before macOS
# jetsam or heavy swap thrashing locks the system.
MEM_EMERGENCY_KILL_TIMEOUT=3

# Backend-liveness backstop: distinct from the memory-pressure backstop above --
# this catches the backend process (mlx_vlm.server / mlx-dspark) dying or getting its
# port stolen for reasons that have NOTHING to do with memory (an unhandled exception,
# a signal from outside this script, etc.). Found live (2026-10-03): a server crashed
# silently with no traceback in the log, and the dashboard loop kept polling and
# rendering happily for 2.5+ hours afterward because check_memory_backstop() only ever
# looks at system memory pressure, never at whether the backend itself is still there --
# render_dashboard's own /metrics curl already fails silently in that case (falls back
# to '{}', just renders dashes), so there was no visible signal at all that every tool
# pointed at this server was failing to connect. Same consecutive-polls debounce
# pattern as the memory checks, to absorb one slow/transient health check rather than
# tearing down on a single blip.
BACKEND_DOWN_CONSECUTIVE_POLLS=2
# When the expected MLX process still owns $MLX_PORT and is alive, a /health HTTP
# timeout (curl exit 28) usually means Python's single-threaded asyncio loop is blocked
# inside a synchronous Metal prefill or vision-encoder pass -- not that the server
# crashed. Allow up to 18 consecutive busy polls (~90s at 5s interval) before declaring
# the inference engine hung.
BACKEND_HANG_CONSECUTIVE_POLLS=18

DASHBOARD_POLL_INTERVAL=5

# Respect an already-set SSL_CERT_FILE (e.g. a corporate TLS-interception-aware
# bundle exported in ~/.zshrc) -- overriding it with certifi's bundle would drop
# the intercepting root CA and break every HTTPS download behind such a proxy.
# Otherwise use Python's certifi CA bundle if available, fall back to system default.
if [[ -n "${SSL_CERT_FILE:-}" && -f "${SSL_CERT_FILE:-}" ]]; then
    CERT_PATH="$SSL_CERT_FILE"
else
    CERT_PATH=$(python3 -c "import certifi; print(certifi.where())" 2>/dev/null) || true
    if [[ -z "$CERT_PATH" || ! -f "$CERT_PATH" ]]; then
        for cert_file in /etc/ssl/cert.pem /usr/local/etc/openssl/cert.pem /opt/homebrew/etc/openssl@3/cert.pem; do
            if [ -f "$cert_file" ]; then
                CERT_PATH="$cert_file"
                break
            fi
        done
    fi
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
    local target_name
    target_name="$(basename "$SCRIPT_PATH")"
    local candidate

    # Installed via a Homebrew formula (frank321/homebrew-bushpilot): the formula
    # already places a wrapper for this script in $(brew --prefix)/bin, and the
    # runtime files live read-only under the Cellar. Self-symlinking here would
    # either clobber the formula's own bin entries or try to write into a Cellar
    # path Homebrew owns -- so defer entirely to the formula. Matched two ways
    # because HOMEBREW_PREFIX isn't guaranteed to be exported into the wrapper's
    # environment, while the Cellar path layout always holds for a formula install.
    # Only a Cellar install matches; the git-clone/zip paths below are unchanged.
    if [[ "$SCRIPT_PATH" == */Cellar/bushpilot/* ]] \
        || { [[ -n "${HOMEBREW_PREFIX:-}" ]] && [[ "$SCRIPT_PATH" == "$HOMEBREW_PREFIX"/* ]]; }; then
        return 0
    fi

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
    local watch_pid="${3:-}"
    local attempts=0
    while [ $attempts -lt "$timeout" ]; do
        if curl -fsS --max-time 5 "$url" > /dev/null 2>&1; then
            return 0
        fi
        # If we're waiting on a freshly-launched child and it already exited,
        # stop polling immediately -- there is nothing left to come up. Without
        # this, a server that crashes at startup (bad flag, missing download,
        # OOM) leaves the supervisor silently burning the full timeout before
        # reporting anything (reproduced: crashed mlx-dspark + 600s health wait).
        if [[ -n "$watch_pid" ]] && ! kill -0 "$watch_pid" 2>/dev/null; then
            return 1
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
    local target_pid="${4:-${MLX_PID:-}}"
    local attempts=0

    if ! pgrep -f "$pattern" > /dev/null 2>&1 \
        && { [[ -z "$target_pid" ]] || ! kill -0 "$target_pid" 2>/dev/null; }; then
        return 0
    fi

    if [[ -n "$target_pid" ]]; then
        pkill -TERM -P "$target_pid" > /dev/null 2>&1 || true
        kill -TERM "$target_pid" > /dev/null 2>&1 || true
    fi
    pkill -TERM -f "$pattern" > /dev/null 2>&1 || true
    log_info "Sent stop signal to $label, waiting for it to exit..."

    while pgrep -f "$pattern" > /dev/null 2>&1 \
        || { [[ -n "$target_pid" ]] && kill -0 "$target_pid" 2>/dev/null; }; do
        attempts=$((attempts + 1))
        if [ "$attempts" -ge "$timeout" ]; then
            log_info "$label still running after ${timeout}s, sending SIGKILL..."
            if [[ -n "$target_pid" ]]; then
                pkill -KILL -P "$target_pid" > /dev/null 2>&1 || true
                kill -KILL "$target_pid" > /dev/null 2>&1 || true
            fi
            pkill -KILL -f "$pattern" > /dev/null 2>&1 || true
            sleep 1
            break
        fi
        sleep 1
    done

    if pgrep -f "$pattern" > /dev/null 2>&1 \
        || { [[ -n "$target_pid" ]] && kill -0 "$target_pid" 2>/dev/null; }; then
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
    if [ -d "$BUSHPILOT_VENV" ] && [ -x "$MLX_PYTHON" ] \
        && "$MLX_PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
        log_info "MLX venv already exists"
        return 0
    fi

    if [ -d "$BUSHPILOT_VENV" ]; then
        log_info "Existing MLX venv is built on a Python older than 3.10 (required by mlx-vlm>=${MLX_VLM_MIN_VERSION}) — rebuilding it..."
        rm -rf "$BUSHPILOT_VENV"
    fi

    local py
    py="$(find_suitable_python)" || {
        log_error "No Python >=3.10 found (checked /opt/homebrew/bin, /usr/local/bin, and python3 on \$PATH)."
        log_error "Install one, e.g.: brew install python@3.12"
        exit 1
    }
    log_info "Creating MLX virtual environment at $BUSHPILOT_VENV using $py ($("$py" --version 2>&1))..."
    "$py" -m venv "$BUSHPILOT_VENV"
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
                "$BUSHPILOT_VENV/bin/pip" install --upgrade "$pkg" > /dev/null
                log_success "mlx-vlm upgraded to $(installed_version mlx_vlm)"
            else
                log_info "$dist already installed (version $have)"
            fi
        else
            log_info "Installing $pkg..."
            "$BUSHPILOT_VENV/bin/pip" install --upgrade pip setuptools wheel > /dev/null
            "$BUSHPILOT_VENV/bin/pip" install "$pkg" > /dev/null
            log_success "$pkg installed"
        fi
    done

    # packaging is used by version_at_least above; certifi for the CA bundle lookup.
    if ! "$MLX_PYTHON" -c "import packaging" 2>/dev/null; then
        "$BUSHPILOT_VENV/bin/pip" install packaging > /dev/null
    fi
}

bootstrap_mlx() {
    log_info "Bootstrapping MLX environment at $BUSHPILOT_HOME..."
    create_venv_if_needed
    install_packages_if_needed
    if [ ! -x "$MLX_PYTHON" ]; then
        log_error "MLX Python executable not found: $MLX_PYTHON"
        exit 1
    fi
    log_success "MLX bootstrap complete"
}

# ---------------------------------------------------------------------------
# Serving backend resolution (models.json's optional "serving" block -- see its
# _schema_notes.serving for the full field reference). Reads models.json DIRECTLY by
# $LOCAL_MODEL id here, rather than threading this through model_advisor.py's own
# --resolve JSON, so this is a SINGLE code path that works identically whether
# LOCAL_MODEL came from the interactive picker or the fixed-model CLI form -- those
# two paths can't drift out of sync on which backend a given model id uses. Sets
# defaults (plain mlx_vlm, no drafter) when a model has no "serving" block at all,
# which is every model in this catalog before this feature existed and is the correct
# behavior for any future model added without one.
# ---------------------------------------------------------------------------
MLX_BACKEND="mlx_vlm"
MLX_DSPARK_MODE=""
MLX_DRAFTER=""
MLX_DRAFT_MODEL=""
MLX_DRAFT_KIND=""
MLX_DRAFT_BLOCK_SIZE=""
MLX_PROC_PATTERN="$MLX_SERVER_MODULE"

resolve_serving_config() {
    [[ -z "$LOCAL_MODEL" ]] && return 0

    # Two-step (capture into a local var, THEN eval it) rather than the single-step
    # `eval "$(cmd <<'EOF' ... EOF)"` form the top-of-file config.json parsing uses --
    # confirmed live that form breaks with a "syntax error near unexpected token `('"
    # on macOS's shipped /bin/bash (3.2.57, pre-GPLv3, arm64-apple-darwin) SPECIFICALLY
    # when it's nested inside a function body; bash prints the substituted PYTHON
    # SOURCE TEXT itself back as the "output" rather than running it, i.e. the heredoc
    # never actually reaches python's stdin in that nesting. The top-of-file usage
    # works because it runs at the script's top level, not inside a function -- this
    # is the first use of that pattern inside one. Two-step form is unaffected on the
    # same bash.
    local resolved
    resolved="$("$MLX_PYTHON" - "$SCRIPT_DIR/models.json" "$LOCAL_MODEL" <<'PYEOF'
import json, shlex, sys
models_path, model_id = sys.argv[1], sys.argv[2]
with open(models_path) as f:
    data = json.load(f)
entry = next((m for m in data["models"] if m["id"] == model_id), None)
serving = (entry or {}).get("serving") or {}
for var, key in (
    ("MLX_BACKEND", "backend"), ("MLX_DSPARK_MODE", "mode"),
    ("MLX_DRAFTER", "drafter"), ("MLX_DRAFT_MODEL", "draft_model"),
    ("MLX_DRAFT_KIND", "draft_kind"),
):
    print(f"{var}={shlex.quote(str(serving.get(key, '') or ''))}")
block_size = serving.get("draft_block_size", "")
print(f"MLX_DRAFT_BLOCK_SIZE={shlex.quote(str(block_size) if block_size != '' else '')}")
PYEOF
)"
    eval "$resolved"
    [[ -z "$MLX_BACKEND" ]] && MLX_BACKEND="mlx_vlm"

    if [[ "$MLX_BACKEND" == "mlx_dspark" ]]; then
        # Deliberately more specific than just "mlx-dspark" -- that string alone would
        # also match `mlx-dspark generate`/`mlx-dspark doctor` invocations (e.g. a
        # manual debugging session in another terminal), which this script has no
        # business killing or mistaking for its own managed server.
        MLX_PROC_PATTERN="mlx-dspark serve"
    else
        MLX_PROC_PATTERN="$MLX_SERVER_MODULE"
    fi
}

# Only pip-installs mlx-dspark when a resolved model actually needs it -- most models
# in this catalog don't, so this stays a no-op for them (bootstrap_mlx's own
# install_packages_if_needed already guarantees mlx-lm/mlx-vlm unconditionally,
# since those are needed for model_advisor.py's own picker regardless of which
# backend the eventually-selected model turns out to use).
ensure_backend_installed() {
    [[ "$MLX_BACKEND" != "mlx_dspark" ]] && return 0

    log_info "Checking if mlx-dspark >= ${MLX_DSPARK_MIN_VERSION} is installed in the MLX venv..."
    if "$MLX_PYTHON" -c "import mlx_dspark" 2>/dev/null; then
        local have
        have="$(installed_version mlx_dspark)"
        if ! version_at_least "$have" "$MLX_DSPARK_MIN_VERSION"; then
            # Same upgrade-if-older pattern install_packages_if_needed uses for
            # mlx-vlm (see MLX_VLM_MIN_VERSION there) -- helpers are shared.
            log_info "mlx-dspark $have is older than required minimum $MLX_DSPARK_MIN_VERSION -- upgrading..."
            "$BUSHPILOT_VENV/bin/pip" install --upgrade "mlx-dspark>=${MLX_DSPARK_MIN_VERSION}" > /dev/null
            log_success "mlx-dspark upgraded to $(installed_version mlx_dspark)"
        else
            log_info "mlx-dspark already installed (version $have)"
        fi
        return 0
    fi

    log_info "Installing mlx-dspark>=${MLX_DSPARK_MIN_VERSION} (required by $LOCAL_MODEL's serving.backend)..."
    "$BUSHPILOT_VENV/bin/pip" install "mlx-dspark>=${MLX_DSPARK_MIN_VERSION}" > /dev/null
    log_success "mlx-dspark installed"
}

# Phase 6: arrow-key model picker with a live detail pane, using fzf rather than
# hand-rolled TUI code -- see docs/PHASE_PLAN.md for why fzf over stdlib curses. Prints
# model_advisor.py's own --resolve output (the same JSON contract the plain-menu path
# already produces) on success, or returns non-zero on cancel/failure.
#
# NOTE: rows are `id<TAB>display<TAB>reason`. --with-nth=2 hides field 1 (the bare id,
# used for {1} in the preview and `cut -f1` below) and field 3 from view AND from
# matching (confirmed live on fzf 0.74.4), so model_advisor.py's display text in field 2
# carries everything worth typing a filter against (name, engine, drafter, ...).
# Field 3 is empty for models that fit and the why-not text for the dimmed ones; fzf
# can't make a row unselectable, so picking a dimmed row explains why and reopens.
# Ctrl-R/S/T/O re-run the list command with a different --sort via fzf's reload.
run_fzf_model_picker() {
    local selection selected_id reason
    # preview_cmd is a double-quoted STRING re-parsed by fzf, so the empty-safe
    # array idiom can't be used inline here -- build it conditionally instead
    # (the ${#arr[@]} length expansion is safe on empty arrays under set -u even
    # on bash 3.2).
    local preview_cmd="\"$MLX_PYTHON\" \"$SCRIPT_DIR/model_advisor.py\" --detail {1} --color"
    [[ ${#ADVISOR_FLAGS[@]} -gt 0 ]] && preview_cmd+=" ${ADVISOR_FLAGS[*]}"
    # system_profiler prints a hw.cpufamily line to stderr on every call; fzf would show it.
    preview_cmd+=" 2>/dev/null"

    # Our own best-effort guess at the terminal width, used only for the single frame
    # between this script's own list-generating call below and fzf's `start` event (see
    # the "start:reload" bind) replacing it with fzf's own, authoritative $FZF_COLUMNS --
    # `tput cols`/$COLUMNS can't always be trusted (wrapped/embedded terminals, no real
    # tty on stdout, etc.), so this is deliberately a placeholder, not the source of
    # truth. Falls back to 80 if neither COLUMNS nor `tput cols` resolves to a number.
    local term_width
    term_width="${COLUMNS:-$(tput cols 2>/dev/null)}"
    [[ "$term_width" =~ ^[0-9]+$ ]] || term_width=80
    term_width=$((term_width - 4))

    # Shell-quoted list command for fzf's reload bindings (no banner: the banner goes
    # to stderr and would scribble over fzf's screen).
    local list_cmd
    # --sort-state remembers the current sort between reloads so asking for the same
    # column twice (key or header click) reverses it.
    local sort_state="$BUSHPILOT_STATE_DIR/picker-sort.state"
    # --color: this process's own stderr-isatty color auto-detection (model_advisor.py's
    # use_color()) can't be trusted here -- every reload/bind command below redirects
    # this process's stderr to /dev/null to swallow system_profiler's unrelated
    # hw.cpufamily noise, which defeats that auto-detection on every single reload, on
    # every machine (not just a lower-memory one -- it's just far more NOTICEABLE there,
    # since that's where rows actually need to be dimmed). We know for certain this is
    # headed into fzf's --ansi screen below, so assert it explicitly instead.
    list_cmd="$(printf '%q ' "$MLX_PYTHON" "$SCRIPT_DIR/model_advisor.py" --list-compact --color --no-banner --sort-state "$sort_state" ${ADVISOR_FLAGS[@]+"${ADVISOR_FLAGS[@]}"})"
    # Deliberately NOT inside the %q-escaped block above: ${FZF_COLUMNS:-...} must stay a
    # literal, unexpanded shell parameter reference in list_cmd's value, so that both the
    # `start` event (fires the instant fzf launches -- corrects for $term_width above
    # possibly being wrong) and the `resize` event (fires on every terminal resize while
    # the picker is open, exactly like the live-adjusting --detail preview pane below)
    # re-run this with whatever width is current AT THAT MOMENT. $FZF_COLUMNS is exported
    # by fzf to every reload/bind command it runs and is authoritative -- it's what the
    # preview pane's own live wrapping is based on -- so it's used ahead of our own
    # guessed $term_width, which only backstops the one invocation below that runs before
    # fzf exists (and gets immediately superseded by the `start` reload anyway).
    list_cmd+="--term-width \"\${FZF_COLUMNS:-$term_width}\""

    local b="" d="" g="" y="" c="" m="" r=""
    if [[ -z "${NO_COLOR:-}" ]]; then
        b=$'\033[1m'
        d=$'\033[2m'
        g=$'\033[32m'
        y=$'\033[33m'
        c=$'\033[36m'
        m=$'\033[35m'
        r=$'\033[0m'
    fi
    local header_text
    header_text="${d}Context size & speed are calculated live for your chip and current free memory.${r}"$'\n'
    header_text+="${b}${c}Keys:${r}   ${b}↑↓${r} move · ${b}type${r} filter · ${b}${g}Enter${r} launch · ${b}${m}Tab${r} mark ${m}●${r} (up to 5) · ${b}Esc${r} cancel"$'\n'
    header_text+="${b}${m}Roster:${r} ${b}${g}Enter${r} alone = picked model + up to 4 cached models in tool · ${b}${m}Tab${r} 2–5 = only marked models (1st loads now)"$'\n'
    header_text+="${b}${y}Sort:${r}   click column header (again to reverse) · ${b}^R${r} newest · ${b}^S${r} fastest · ${b}^T${r} largest · ${b}^O${r} name"$'\n'
    header_text+="${b}Legend:${r} ${g}■ fits as-is${r} · ${y}■ KV-quantized${r} · ${d}■ won't fit${r} · ${b}71 tok/s${r} measured · ${d}~71 tok/s est${r} · ${g}✓ tested${r} · ${d}○ untested${r}"
    # The column-header row sits right below fzf's --header lines; a click-header event
    # reports which line was clicked, and only that row should trigger a re-sort.
    local header_row=$(( $(printf '%s\n' "$header_text" | wc -l) + 1 ))

    : > "$sort_state"   # start from the default sort, preserved if reopened after picking a dimmed row
    while true; do
        selection="$("$MLX_PYTHON" "$SCRIPT_DIR/model_advisor.py" --list-compact --color --term-width "$term_width" --sort-state "$sort_state" ${ADVISOR_FLAGS[@]+"${ADVISOR_FLAGS[@]}"} | fzf \
            --multi=5 \
            --delimiter=$'\t' --with-nth=2 \
            --preview "$preview_cmd" --preview-window down:50%:wrap \
            --bind "start:reload($list_cmd 2>/dev/null)" \
            --bind "resize:reload($list_cmd 2>/dev/null)" \
            --bind "ctrl-r:reload($list_cmd --sort released 2>/dev/null)" \
            --bind "ctrl-s:reload($list_cmd --sort speed 2>/dev/null)" \
            --bind "ctrl-t:reload($list_cmd --sort size 2>/dev/null)" \
            --bind "ctrl-o:reload($list_cmd --sort name 2>/dev/null)" \
            --bind "click-header:transform:[ \"\$FZF_CLICK_HEADER_LINE\" = $header_row ] && echo \"reload($list_cmd --sort-col \$FZF_CLICK_HEADER_COLUMN 2>/dev/null)\"" \
            --header-lines=1 \
            --ansi --border=rounded --pointer="▶" --marker="●" \
            --header="$header_text" \
            --prompt="Model> " --height=~80% --layout=reverse)"
        if [[ -z "$selection" ]]; then
            log_error "Model selection cancelled or no model available."
            return 1
        fi

        local line bad_id="" bad_reason="" selected_ids_csv=""
        selected_id=""
        while IFS= read -r line; do
            [[ -z "$line" ]] && continue
            local row_id row_reason
            row_id="$(printf '%s' "$line" | cut -f1)"
            row_reason="$(printf '%s' "$line" | cut -f3)"
            if [[ -n "$row_reason" ]]; then
                bad_id="$row_id"
                bad_reason="$row_reason"
                break
            fi
            [[ -z "$selected_id" ]] && selected_id="$row_id"
            selected_ids_csv+="${selected_ids_csv:+,}${row_id}"
        done <<<"$selection"

        if [[ -n "$bad_reason" ]]; then
            echo "[INFO] $bad_id doesn't fit this machine right now: $bad_reason" >&2
            read -r -p "Press Enter to pick another model... " _ </dev/tty || true
            continue
        fi

        if ! "$MLX_PYTHON" "$SCRIPT_DIR/model_advisor.py" --build-roster "$selected_ids_csv" ${ADVISOR_FLAGS[@]+"${ADVISOR_FLAGS[@]}"} > "${MLX_ROSTER_FILE}.tmp" 2>/dev/null; then
            rm -f "${MLX_ROSTER_FILE}.tmp"
        else
            mv -f "${MLX_ROSTER_FILE}.tmp" "$MLX_ROSTER_FILE"
        fi

        if ! "$MLX_PYTHON" "$SCRIPT_DIR/model_advisor.py" --resolve "$selected_id" ${ADVISOR_FLAGS[@]+"${ADVISOR_FLAGS[@]}"}; then
            log_error "Could not resolve '$selected_id' -- free memory may have changed since the list was shown. Try again."
            return 1
        fi
        return 0
    done
}

# Builds ~/.bushpilot/state/mlx-roster.json for non-fzf or single-model paths before the
# primary model loads (while full free memory is available).
ensure_roster_file_built() {
    local primary_id="${1:-$LOCAL_MODEL}"
    [[ -z "$primary_id" ]] && return 0
    if "$MLX_PYTHON" "$SCRIPT_DIR/model_advisor.py" --build-roster "$primary_id" ${ADVISOR_FLAGS[@]+"${ADVISOR_FLAGS[@]}"} > "${MLX_ROSTER_FILE}.tmp" 2>/dev/null; then
        mv -f "${MLX_ROSTER_FILE}.tmp" "$MLX_ROSTER_FILE"
    else
        rm -f "${MLX_ROSTER_FILE}.tmp"
        # Fallback for fixed-model CLI invocations where primary_id does not pass the
        # advisor fit check or isn't in models.json: synthesize a 1-entry roster so the
        # router and coding tools still see the active model.
        "$MLX_PYTHON" - "$MLX_ROSTER_FILE" "$primary_id" "$MLX_MODEL" "$CONTEXT_WINDOW" "${KV_BITS:-}" "${PREFILL_STEP_SIZE:-2048}" "${APC_NUM_BLOCKS:-4096}" "${CONCURRENCY:-1}" "${MLX_BACKEND:-mlx_vlm}" <<'PYEOF' 2>/dev/null || true
import json, sys
path, mid, hf_id, ctx, kv, prefill, apc, conc, backend = sys.argv[1:10]
with open(path, "w") as f:
    json.dump({
        "primary_id": mid,
        "explicit_selection": False,
        "models": [{
            "id": mid,
            "huggingface": hf_id,
            "recommended_context": int(ctx) if ctx.isdigit() else 131072,
            "kv_bits": int(kv) if kv.isdigit() else None,
            "prefill_step_size": int(prefill) if prefill.isdigit() else 2048,
            "apc_num_blocks": int(apc) if apc.isdigit() else 4096,
            "concurrency": int(conc) if conc.isdigit() else 1,
            "backend": backend or "mlx_vlm",
            "multimodal_capable": backend != "mlx_dspark",
            "modalities": ["text"],
            "expose_catalog_id": False,
        }],
    }, f)
PYEOF
    fi
}

# Applies a single model resolution JSON (from `--resolve` or a roster entry) to the
# supervisor's global launch variables.
apply_resolved_model_json() {
    local advisor_output="$1"
    MLX_MODEL=$("$MLX_PYTHON" -c "import json,sys; print(json.loads(sys.argv[1])['huggingface'])" "$advisor_output")
    LOCAL_MODEL=$("$MLX_PYTHON" -c "import json,sys; print(json.loads(sys.argv[1])['id'])" "$advisor_output")
    CONTEXT_WINDOW=$("$MLX_PYTHON" -c "import json,sys; print(json.loads(sys.argv[1])['recommended_context'])" "$advisor_output")
    KV_BITS=$("$MLX_PYTHON" -c "import json,sys; v=json.loads(sys.argv[1]).get('kv_bits'); print(v if v is not None else '')" "$advisor_output")
    KV_BITS_APC_CAVEAT=$("$MLX_PYTHON" -c "import json,sys; print('1' if json.loads(sys.argv[1]).get('kv_bits_apc_caveat') else '')" "$advisor_output")
    CONCURRENCY=$("$MLX_PYTHON" -c "import json,sys; print(json.loads(sys.argv[1]).get('concurrency', 1))" "$advisor_output")
    # Overrides this script's own static PREFILL_STEP_SIZE=2048 default (set near the
    # top of this file) with model_advisor.py's recommend_prefill_step_size() decision
    # -- a real per-(chip, model) sweep measurement when one exists on this machine
    # (see docs/NEXT_SESSION_HANDOFF_PREFILL_SWEEP.md / sweep_model.py), the SAME global
    # default otherwise. Only the --interactive path goes through model_advisor.py at
    # all, so the fixed-model form (scripting/restarts) keeps today's static default
    # unchanged, exactly like KV_BITS above.
    local prefill_step_size_val
    prefill_step_size_val=$("$MLX_PYTHON" -c "import json,sys; print(json.loads(sys.argv[1]).get('prefill_step_size', 2048))" "$advisor_output" 2>/dev/null) || prefill_step_size_val=""
    [[ -n "$prefill_step_size_val" ]] && PREFILL_STEP_SIZE="$prefill_step_size_val"

    local apc_num_blocks_val
    apc_num_blocks_val=$("$MLX_PYTHON" -c "import json,sys; v=json.loads(sys.argv[1]).get('apc_num_blocks'); print(v if v is not None else '')" "$advisor_output" 2>/dev/null) || apc_num_blocks_val=""
    if [[ -n "$apc_num_blocks_val" ]]; then
        APC_NUM_BLOCKS="$apc_num_blocks_val"
    else
        APC_NUM_BLOCKS=4096
    fi
}

# Loads the primary model entry from ~/.bushpilot/state/mlx-roster.json (populated by a
# prior `--select-only` run in launch_tool.sh's auto-split flow) and applies its
# resolved parameters without re-prompting.
load_primary_from_roster_file() {
    if [[ ! -f "$MLX_ROSTER_FILE" ]]; then
        log_error "No model roster found at $MLX_ROSTER_FILE for --from-roster launch."
        return 1
    fi
    local entry_json
    entry_json="$("$MLX_PYTHON" -c '
import json, sys
try:
    with open(sys.argv[1]) as f:
        data = json.load(f)
    models = data.get("models") or []
    primary_id = data.get("primary_id")
    chosen = next((m for m in models if m.get("id") == primary_id), None)
    if chosen is None and models:
        chosen = models[0]
    if chosen:
        print(json.dumps(chosen))
except Exception:
    pass
' "$MLX_ROSTER_FILE" 2>/dev/null || true)"
    if [[ -z "$entry_json" ]]; then
        log_error "Could not read primary model from $MLX_ROSTER_FILE."
        return 1
    fi
    apply_resolved_model_json "$entry_json"
    log_success "Loaded selected model from roster: $LOCAL_MODEL ($MLX_MODEL) — context ${CONTEXT_WINDOW} tokens"
    return 0
}

# Interactive model selection (Phase 2: no --tool factored in here anymore -- the
# supervisor is deliberately tool-agnostic now; a tool-aware preload check moves to
# Phase 3's launch_tool at attach-time instead).
run_interactive_selection() {
    log_info "Detecting hardware and free memory for model recommendations..."

    rm -f "$MLX_ROSTER_FILE"
    local advisor_output
    if [[ -n "$CONFIG_PREFERRED_MODEL" ]] && advisor_output=$("$MLX_PYTHON" "$SCRIPT_DIR/model_advisor.py" --resolve "$CONFIG_PREFERRED_MODEL" ${ADVISOR_FLAGS[@]+"${ADVISOR_FLAGS[@]}"} 2>/dev/null); then
        log_success "Using preferred_model from config.json: $CONFIG_PREFERRED_MODEL"
    elif is_command_available fzf; then
        advisor_output="$(run_fzf_model_picker)" || exit 1
    elif ! advisor_output=$("$MLX_PYTHON" "$SCRIPT_DIR/model_advisor.py" ${ADVISOR_FLAGS[@]+"${ADVISOR_FLAGS[@]}"}); then
        log_error "Model selection cancelled or failed."
        exit 1
    fi

    apply_resolved_model_json "$advisor_output"
    [[ -f "$MLX_ROSTER_FILE" ]] || ensure_roster_file_built "$LOCAL_MODEL"

    local kv_bits_log="unquantized (full decode speed — recommended context already fits free memory)"
    if [[ -n "$KV_BITS" ]]; then
        kv_bits_log="quantized to ${KV_BITS}-bit (memory-constrained — needed to fit this context)"
    fi
    local ctx_source_log="computed from your current free memory"
    [[ -n "$CONFIG_MAX_CONTEXT" ]] && ctx_source_log+=" and capped by config.json's max_context_tokens (rounds down to the nearest tier)"
    log_success "Selected $LOCAL_MODEL ($MLX_MODEL) — context set to ${CONTEXT_WINDOW} tokens ($ctx_source_log)"
    log_success "KV cache: $kv_bits_log"
    if [[ "$APC_NUM_BLOCKS" != "4096" ]]; then
        log_success "APC pool scaled to $APC_NUM_BLOCKS blocks ($((APC_NUM_BLOCKS * 16)) cached tokens) to fit memory"
    fi
    if [[ "$PREFILL_STEP_SIZE" != "2048" ]]; then
        log_success "Prefill step size: $PREFILL_STEP_SIZE (measured on this machine for this model — see ~/.bushpilot/calibration)"
    fi
    if [[ -n "$KV_BITS_APC_CAVEAT" ]]; then
        # Real, upstream-confirmed gap (mlx-vlm issue #1559, "APC + kv-bits full
        # batch/SWA coverage") -- see model_advisor.py's recommend_kv_bits_and_context
        # docstring. Non-blocking: prefix reuse still works per upstream's own smoke
        # note, this is a heads-up, not a reason this launch was refused.
        log_info "⚠ This model's sliding-window layers + quantized KV cache + prefix caching (APC, always on) is a known upstream gap (mlx-vlm #1559) — sliding layers likely won't cache on concurrent requests until it's fixed. Prefix reuse itself still works."
    fi
    if [[ "$CONCURRENCY" -gt 1 ]]; then
        log_success "Concurrency: $CONCURRENCY slots (free bonus from tier-rounding leftover, or your config.json override)"
    fi
    if [[ -f "$MLX_ROSTER_FILE" ]]; then
        local roster_summary
        roster_summary="$("$MLX_PYTHON" -c '
import json, sys
try:
    with open(sys.argv[1]) as f:
        models = json.load(f).get("models", [])
    if len(models) > 1:
        print(", ".join(m["id"] for m in models))
except Exception:
    pass
' "$MLX_ROSTER_FILE" 2>/dev/null || true)"
        if [[ -n "$roster_summary" ]]; then
            log_success "Hot-swap roster (selectable inside coding tools): $roster_summary"
        fi
    fi
}

# Pre-downloads a repo via snapshot_download() BEFORE the server ever starts.
# mlx_vlm.server's/mlx-dspark's own lazy on-demand download has no resume support
# when the process is killed mid-transfer; snapshot_download() run standalone does.
# Shared by the target model and (when the resolved serving config has one) the
# drafter -- same download semantics apply to both, they're just two separate repos.
ensure_repo_downloaded() {
    local repo="$1" label="$2"
    log_info "Checking whether $label ($repo) is already cached..."
    local already_cached
    already_cached=$("$MLX_PYTHON" -c "
import os, glob
pattern = os.path.expanduser('$HF_HOME/hub/models--${repo//\//--}/snapshots/*/config.json')
print('yes' if glob.glob(pattern) else 'no')
")
    if [[ "$already_cached" == "yes" ]]; then
        log_success "$repo already cached — no download needed"
        return 0
    fi

    log_info "$label not cached locally. Downloading $repo now (this can take several minutes for larger models)..."
    if ! "$MLX_PYTHON" -c "
from huggingface_hub import snapshot_download
snapshot_download('$repo')
"; then
        log_error "Download failed for $repo."
        return 1
    fi
    log_success "Download complete: $repo"
}

ensure_model_downloaded() {
    ensure_repo_downloaded "$MLX_MODEL" "target model" || return 1
    # MLX_DRAFTER (mlx_dspark backend) and MLX_DRAFT_MODEL (mlx_vlm backend) are
    # mutually exclusive per resolve_serving_config -- at most one is ever set.
    local drafter_repo="${MLX_DRAFTER:-$MLX_DRAFT_MODEL}"
    if [[ -n "$drafter_repo" ]]; then
        ensure_repo_downloaded "$drafter_repo" "drafter" || return 1
    fi
}

# Opportunistic, model_advisor.py-delegated fix for a real chat-template bug found in
# specific HuggingFace repos (gemma-4-12b-it-4bit, gemma-4-26b-a4b-it-4bit, and
# siblings -- see those entries' known_issues in models.json for the full root-cause
# writeup): a 4-token asymmetry between how a fresh generation prompt and a completed
# turn's history-serialization render the same turn, which breaks mlx_vlm's APC
# prefix-cache reuse on every turn of a growing conversation. NOT an mlx_vlm/
# mlx-dspark code bug -- purely in the model repo's own chat_template.jinja; upstream
# mlx-vlm PR #1960 fixed the identical bug for the 5-bit siblings via an HF repo
# template discussion, a fix never applied to the 4-bit/8-bit repos this project
# actually serves. See bushpilot/template_patches.py for the patch itself.
#
# Called only for the target model, after it's confirmed downloaded (there's a real
# local file to inspect) and before the server starts (so a freshly-applied patch is
# already in place for the very first request) -- never for a drafter repo, since
# drafters aren't invoked through chat-template serialization. Detection inside
# model_advisor.py is purely content-based (two required exact-string signatures, see
# that module's docstring for a real false positive this caught before shipping), not
# gated on this model id, so this call is a safe, silent no-op for every model that
# isn't affected -- intentionally unconditional rather than an allowlist of known-
# affected ids, so a sibling quantization with the identical bug gets fixed too
# without this script needing its own copy of that list.
patch_known_chat_template_bugs() {
    "$MLX_PYTHON" "$SCRIPT_DIR/model_advisor.py" --patch-chat-template "$MLX_MODEL"
}

# Returns 0 only if the process actually LISTENING on the backend port ($MLX_BACKEND_PORT,
# falling back to $MLX_PORT when $MLX_BACKEND_PORT is unset in isolated shell tests) is
# our expected server process (mlx_vlm.server, or `mlx-dspark serve` for an
# mlx_dspark-backed model -- see MLX_PROC_PATTERN in resolve_serving_config). /health
# alone is not sufficient -- confirmed live, a foreign, unmanaged server (e.g. a stray
# mlx_lm.server) can hold the port while answering /health and hang forever on real
# inference.
mlx_port_owner_is_expected() {
    local check_port="${MLX_BACKEND_PORT:-$MLX_PORT}"
    local pids
    pids="$(lsof -ti "tcp:$check_port" 2>/dev/null || true)"
    [[ -z "$pids" ]] && return 1

    local pid
    for pid in $pids; do
        if ps -p "$pid" -o command= 2>/dev/null | grep -q "$MLX_PROC_PATTERN"; then
            return 0
        fi
    done
    return 1
}

router_port_owner_is_expected() {
    local pids
    pids="$(lsof -ti "tcp:$MLX_PORT" 2>/dev/null || true)"
    [[ -z "$pids" ]] && return 1

    local pid
    for pid in $pids; do
        if ps -p "$pid" -o command= 2>/dev/null | grep -q "$MLX_ROUTER_MODULE"; then
            return 0
        fi
    done
    return 1
}

ensure_router_running() {
    if router_port_owner_is_expected; then
        return 0
    fi
    wait_for_port_free "$MLX_PORT" 15
    PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}" "$MLX_PYTHON" -m "$MLX_ROUTER_MODULE" \
        --host "$MLX_HOST" \
        --port "$MLX_PORT" \
        --backend-host "$MLX_HOST" \
        --backend-port "$MLX_BACKEND_PORT" \
        --state-dir "$BUSHPILOT_STATE_DIR" >> "$MLX_LOG_FILE" 2>&1 &
    ROUTER_PID=$!
    if ! wait_for_url "http://$MLX_HOST:$MLX_PORT/v1/models" 15 "$ROUTER_PID"; then
        log_error "Bushpilot HTTP router failed to start on $MLX_HOST:$MLX_PORT"
        return 1
    fi
    return 0
}

stop_router() {
    wait_for_process_gone "$MLX_ROUTER_MODULE" "Bushpilot router" 10 || true
}

stop_all_mlx_backends() {
    local timeout="${1:-30}"
    wait_for_process_gone "$MLX_SERVER_MODULE" "MLX server" "$timeout" || true
    wait_for_process_gone "mlx-dspark serve" "mlx-dspark server" "$timeout" || true
    wait_for_port_free "$MLX_BACKEND_PORT" 15 || true
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
# call a few endpoints once and verify the expected shape before declaring the
# supervisor ready -- catches behavior drift even within the allowed version range,
# not just "too old". /v1/messages is checked identically on both backends (both
# speak Anthropic's Messages API with a "content" field -- confirmed live against a
# real mlx-dspark serve instance, not assumed from its docs). /metrics and
# /v1/cache/stats are mlx_vlm-specific (its own APC feature); mlx-dspark's /metrics
# has a different shape (no "summary" field, confirmed live) and it has no
# /v1/cache/stats endpoint at all (its own prefix caching is a different feature,
# always on by default, with no equivalent stats route), so those two checks only
# run on the mlx_vlm backend.
run_startup_smoke_check() {
    local messages_resp
    messages_resp="$(curl -fsS --max-time 30 "http://$MLX_HOST:$MLX_PORT/v1/messages" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$MLX_MODEL\",\"max_tokens\":1,\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}]}" \
        2>/dev/null)" || { log_error "Smoke check: /v1/messages did not respond"; return 1; }
    if ! grep -q '"content"' <<<"$messages_resp"; then
        log_error "Smoke check: /v1/messages response missing expected 'content' field"
        return 1
    fi

    if [[ "$MLX_BACKEND" == "mlx_dspark" ]]; then
        log_info "Running startup smoke check against /v1/messages, /metrics..."
        local metrics_resp
        metrics_resp="$(curl -fsS --max-time 10 "http://$MLX_HOST:$MLX_PORT/metrics" 2>/dev/null)" \
            || { log_error "Smoke check: /metrics did not respond"; return 1; }
        if ! grep -q '"requests"' <<<"$metrics_resp"; then
            log_error "Smoke check: /metrics response missing expected 'requests' field"
            return 1
        fi
        log_success "Smoke check passed — mlx-dspark's /v1/messages and /metrics behave as expected"
        return 0
    fi

    log_info "Running startup smoke check against /v1/messages, /metrics, /v1/cache/stats..."
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

launch_backend_engine() {
    local mode="${1:-startup}"
    local desired_state="$MLX_MODEL|$CONTEXT_WINDOW|$KV_BITS|${LOCAL_MODEL:-}|${MLX_BACKEND:-mlx_vlm}"
    local mlx_args=()
    local -a launch_cmd=()

    if [[ "$MLX_BACKEND" == "mlx_dspark" ]]; then
        mlx_args=(
            --model "$MLX_MODEL"
            --host "$MLX_HOST"
            --port "$MLX_BACKEND_PORT"
            --context-window "$CONTEXT_WINDOW"
            --mode "${MLX_DSPARK_MODE:-dflash}"
        )
        [[ -n "$MLX_DRAFTER" ]] && mlx_args+=(--drafter "$MLX_DRAFTER")
        if [[ -n "$CONFIG_DSPARK_PC_MAX_RAM_MB" && -z "$CONFIG_DSPARK_PC_DIR" ]]; then
            log_error "config.json: dspark_prefix_cache_max_ram_mb requires dspark_prefix_cache_dir (per mlx-dspark's own --prefix-cache-max-ram-mb contract) — set both or neither."
            return 1
        fi
        [[ -n "$CONFIG_DSPARK_PC_SLOTS" ]] && mlx_args+=(--prefix-cache-slots "$CONFIG_DSPARK_PC_SLOTS")
        [[ -n "$CONFIG_DSPARK_PC_RUNGS" ]] && mlx_args+=(--prefix-cache-rungs "$CONFIG_DSPARK_PC_RUNGS")
        [[ -n "$CONFIG_DSPARK_PC_DIR" ]] && mlx_args+=(--prefix-cache-dir "$CONFIG_DSPARK_PC_DIR")
        [[ -n "$CONFIG_DSPARK_PC_MAX_RAM_MB" ]] && mlx_args+=(--prefix-cache-max-ram-mb "$CONFIG_DSPARK_PC_MAX_RAM_MB")
        if [[ -n "$KV_BITS" ]]; then
            mlx_args+=(--kv-bits "$KV_BITS")
        else
            mlx_args+=(--kv-bits 0)
            [[ "$CONCURRENCY" -gt 1 ]] && mlx_args+=(--max-batch "$CONCURRENCY")
        fi
        launch_cmd=("$BUSHPILOT_VENV/bin/mlx-dspark" serve "${mlx_args[@]}")
    else
        mlx_args=(
            --model "$MLX_MODEL"
            --host "$MLX_HOST"
            --port "$MLX_BACKEND_PORT"
            --max-kv-size "$CONTEXT_WINDOW"
            --prefill-step-size "$PREFILL_STEP_SIZE"
            --max-num-seqs "$CONCURRENCY"
        )
        if [[ -n "$KV_BITS" ]]; then
            mlx_args+=(
                --kv-bits "$KV_BITS"
                --kv-group-size "$KV_GROUP_SIZE"
                --quantized-kv-start "$QUANTIZED_KV_START"
                --kv-quant-scheme uniform
            )
        fi
        if [[ -n "$MLX_DRAFT_MODEL" ]]; then
            mlx_args+=(--draft-model "$MLX_DRAFT_MODEL")
            [[ -n "$MLX_DRAFT_KIND" ]] && mlx_args+=(--draft-kind "$MLX_DRAFT_KIND")
            [[ -n "$MLX_DRAFT_BLOCK_SIZE" ]] && mlx_args+=(--draft-block-size "$MLX_DRAFT_BLOCK_SIZE")
        fi

        mkdir -p "$APC_DISK_PATH"
        export APC_ENABLED APC_NUM_BLOCKS APC_DISK_PATH APC_DISK_MAX_GB

        launch_cmd=("$MLX_PYTHON" -m "$MLX_SERVER_MODULE" "${mlx_args[@]}")
    fi

    if [[ "$mode" == "startup" ]]; then
        start_run_log
        log_hardware_profile
        flush_context_warning
    else
        local ts; ts="$(date '+%Y-%m-%d %H:%M:%S')"
        printf '\n%s - === HOT-SWAP TO %s (%s) ===\n' "$ts" "${LOCAL_MODEL:-?}" "$MLX_MODEL" >> "$MLX_LOG_FILE"
        log_hardware_profile
    fi
    log_launch_config "${launch_cmd[@]}"

    "${launch_cmd[@]}" >> "$MLX_LOG_FILE" 2>&1 &
    MLX_PID=$!
    local load_start_ts; load_start_ts=$(date +%s)

    if ! wait_for_url "http://$MLX_HOST:$MLX_BACKEND_PORT/health" 600 "$MLX_PID"; then
        log_error "MLX backend failed to start on $MLX_HOST:$MLX_BACKEND_PORT. Check $MLX_LOG_FILE"
        tail -20 "$MLX_LOG_FILE" >&2 || true
        return 1
    fi
    printf '%s\n' "$(( $(date +%s) - load_start_ts ))" > "$MLX_LOAD_TIME_FILE" 2>/dev/null || true

    # Write MLX_STATE_FILE before running the smoke check through the router on $MLX_PORT
    # so the router sees the newly loaded model as active and forwards /v1/messages
    # directly rather than treating the smoke check as a new swap request.
    echo "$desired_state" > "$MLX_STATE_FILE"

    if ! ensure_router_running; then
        rm -f "$MLX_STATE_FILE"
        stop_all_mlx_backends 20
        return 1
    fi

    if ! run_startup_smoke_check; then
        log_error "Startup smoke check failed — tearing down. Check $MLX_LOG_FILE"
        rm -f "$MLX_STATE_FILE"
        stop_all_mlx_backends 20
        return 1
    fi

    return 0
}

ensure_mlx_running() {
    log_info "Checking MLX server on $MLX_HOST:$MLX_PORT..."

    local desired_state="$MLX_MODEL|$CONTEXT_WINDOW|$KV_BITS|${LOCAL_MODEL:-}|${MLX_BACKEND:-mlx_vlm}"
    local legacy_desired_state="$MLX_MODEL|$CONTEXT_WINDOW|$KV_BITS"
    local server_up=false

    if curl -fsS --max-time 5 "http://$MLX_HOST:$MLX_PORT/health" > /dev/null 2>&1; then
        if router_port_owner_is_expected && mlx_port_owner_is_expected; then
            local loaded_state=""
            [[ -f "$MLX_STATE_FILE" ]] && loaded_state="$(cat "$MLX_STATE_FILE" 2>/dev/null)"
            local probe_model="${loaded_state%%|*}"
            [[ -z "$probe_model" || "$probe_model" == "$loaded_state" ]] && probe_model="$MLX_MODEL"

            if mlx_inference_responds "$probe_model"; then
                server_up=true
            else
                log_error "MLX server answers /health, but did NOT return a completion within the inference probe timeout:"
                log_error "  probed model: $probe_model -- the process is hung, not healthy"
                log_info "Restarting the hung MLX server..."
                stop_all_mlx_backends 30
                rm -f "$MLX_STATE_FILE"
            fi
        else
            log_error "Port $MLX_PORT is answering /health, but it is NOT this script's managed router/backend ($MLX_PROC_PATTERN):"
            local owner_pids
            owner_pids="$(lsof -ti "tcp:$MLX_PORT" 2>/dev/null || true)"
            if [[ -n "$owner_pids" ]]; then
                ps -p $owner_pids -o pid=,command= 2>/dev/null | sed 's/^/  /' >&2 || true
            fi
            log_info "Reclaiming port $MLX_PORT and $MLX_BACKEND_PORT..."
            stop_router
            stop_all_mlx_backends 15
            wait_for_port_free "$MLX_PORT" 15
        fi
    fi

    if [[ "$server_up" == "true" ]]; then
        local loaded_state=""
        [[ -f "$MLX_STATE_FILE" ]] && loaded_state="$(cat "$MLX_STATE_FILE" 2>/dev/null)"

        if [[ "$loaded_state" == "$desired_state" || "$loaded_state" == "$legacy_desired_state" ]]; then
            log_success "MLX server already healthy with $MLX_MODEL (context: ${CONTEXT_WINDOW} tokens)"
            local ts; ts="$(date '+%Y-%m-%d %H:%M:%S')"
            printf '%s - ATTACHED to already-running server (state=%s)\n' "$ts" "$loaded_state" >> "$MLX_LOG_FILE"
            log_hardware_profile
            flush_context_warning
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
                stop_all_mlx_backends 30
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
        stop_all_mlx_backends 30
        rm -f "$MLX_STATE_FILE" "$MLX_SWAP_REQUEST_FILE" "$MLX_SWAP_ERROR_FILE"
    fi

    if ! launch_backend_engine "startup"; then
        exit 1
    fi

    log_success "MLX server started and healthy"
}

# Orchestrates a live model swap requested either by the HTTP router
# ($MLX_SWAP_REQUEST_FILE) or by pressing `m` in the dashboard. Stops the outgoing
# backend FIRST so its wired GPU memory is completely reclaimed before the incoming
# model loads. On failure, writes $MLX_SWAP_ERROR_FILE and restores the previous model.
perform_model_swap() {
    local target_id="$1"
    [[ -z "$target_id" ]] && { rm -f "$MLX_SWAP_REQUEST_FILE"; return 1; }

    local prev_local_model="$LOCAL_MODEL"
    local prev_mlx_model="$MLX_MODEL"
    local prev_ctx="$CONTEXT_WINDOW"
    local prev_kv="$KV_BITS"
    local prev_kv_caveat="${KV_BITS_APC_CAVEAT:-}"
    local prev_prefill="$PREFILL_STEP_SIZE"
    local prev_apc="$APC_NUM_BLOCKS"
    local prev_conc="$CONCURRENCY"

    log_info "Switching model from ${LOCAL_MODEL:-$MLX_MODEL} to $target_id..."
    save_calibration_snapshot || true

    # Free the outgoing model's wired RAM before resolving or loading the new model.
    stop_all_mlx_backends 20
    rm -f "$MLX_STATE_FILE" "$MLX_SWAP_ERROR_FILE"

    local resolved_json=""
    if [[ -f "$MLX_ROSTER_FILE" ]]; then
        resolved_json="$("$MLX_PYTHON" - "$MLX_ROSTER_FILE" "$target_id" <<'PYEOF' 2>/dev/null || true
import json, sys
roster_path, target = sys.argv[1], sys.argv[2]
try:
    with open(roster_path) as f:
        models = json.load(f).get("models", [])
    for m in models:
        if m.get("id") == target or m.get("huggingface") == target:
            print(json.dumps(m))
            break
except Exception:
    pass
PYEOF
)"
    fi

    if [[ -z "$resolved_json" ]]; then
        resolved_json="$("$MLX_PYTHON" "$SCRIPT_DIR/model_advisor.py" --resolve "$target_id" ${ADVISOR_FLAGS[@]+"${ADVISOR_FLAGS[@]}"} 2>/dev/null || true)"
    fi

    local swap_ok=false
    if [[ -n "$resolved_json" ]]; then
        apply_resolved_model_json "$resolved_json"
        resolve_serving_config
        ensure_backend_installed
        if ensure_model_downloaded; then
            patch_known_chat_template_bugs
            if launch_backend_engine "swap"; then
                swap_ok=true
            fi
        fi
    fi

    if [[ "$swap_ok" == "true" ]]; then
        # Ensure the newly active model is also present in mlx-roster.json so /v1/models
        # and newly-launched coding tools see it.
        if [[ -f "$MLX_ROSTER_FILE" && -n "$resolved_json" ]]; then
            "$MLX_PYTHON" - "$MLX_ROSTER_FILE" "$resolved_json" <<'PYEOF' 2>/dev/null || true
import json, sys
roster_path, raw_entry = sys.argv[1], sys.argv[2]
try:
    with open(roster_path) as f:
        data = json.load(f)
    entry = json.loads(raw_entry)
    models = data.get("models", [])
    if not any(m.get("id") == entry.get("id") for m in models):
        models.insert(0, entry)
        data["models"] = models[:5]
    data["primary_id"] = entry.get("id", data.get("primary_id"))
    with open(roster_path, "w") as f:
        json.dump(data, f, indent=2)
except Exception:
    pass
PYEOF
        fi
        BACKEND_DOWN_STREAK=0
        BACKEND_BUSY_STREAK=0
        MEM_ABORT_STREAK=0
        MEM_CRITICAL_STREAK=0
        MEM_WARNING_STATUS=""
        DASHBOARD_START_TS=$(date +%s)
        rm -f "$MLX_SWAP_REQUEST_FILE"
        log_success "Hot-swap complete: now serving $LOCAL_MODEL ($MLX_MODEL) at ${CONTEXT_WINDOW} context"
        return 0
    fi

    local err_msg="Failed to hot-swap to '$target_id'; restoring previous model ($prev_local_model)."
    log_error "$err_msg"
    printf '%s\n' "$err_msg" > "$MLX_SWAP_ERROR_FILE"
    rm -f "$MLX_SWAP_REQUEST_FILE"

    LOCAL_MODEL="$prev_local_model"
    MLX_MODEL="$prev_mlx_model"
    CONTEXT_WINDOW="$prev_ctx"
    KV_BITS="$prev_kv"
    KV_BITS_APC_CAVEAT="$prev_kv_caveat"
    PREFILL_STEP_SIZE="$prev_prefill"
    APC_NUM_BLOCKS="$prev_apc"
    CONCURRENCY="$prev_conc"
    resolve_serving_config
    launch_backend_engine "swap" || true
    return 1
}

# ---------------------------------------------------------------------------
# Live status dashboard + memory-pressure backstop + full teardown on
# Ctrl-C/q/HUP/TERM. See docs/PHASE_PLAN.md Phase 2's locked dashboard layout.
# ---------------------------------------------------------------------------
DASHBOARD_START_TS=$(date +%s)
MEM_ABORT_STREAK=0
MEM_CRITICAL_STREAK=0
MEM_WARNING_STATUS=""
BACKEND_DOWN_STREAK=0
BACKEND_BUSY_STREAK=0
TEARDOWN_REASON=""
CONTEXT_WARNING_PENDING=""

# Appends a timestamped line to the server log so a teardown's REASON survives after
# the terminal is gone -- previously full_teardown only echoed to the terminal, so an
# unattended memory-backstop stop left the log ending in a bare uvicorn "Shutting down"
# with no explanation (real incident: had to reconstruct the cause from sysctls/log show).
log_teardown() {
    local reason="$1"
    [[ -z "$reason" ]] && return 0
    local ts pct pressure level sw rss
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    pct="$(get_percent_free 2>/dev/null)"
    pressure="$(get_pressure_level 2>/dev/null)"
    level="$(pressure_level_name "${pressure:-0}")"
    sw="$(sysctl -n vm.swapusage 2>/dev/null | sed -n 's/.*used = \([0-9.]*M\).*/\1/p')"
    rss="$(get_mlx_rss_gb "${MLX_PID:-}" 2>/dev/null)"
    printf '%s - TEARDOWN: %s (free=%s%% pressure=%s swap_used=%s MLX_RSS=%sGB)\n' \
        "$ts" "$reason" "${pct:-?}" "${level:-unknown}" "${sw:-?}" "${rss:-?}" >> "$MLX_LOG_FILE"
}

# Records the hardware profile that model_advisor.py evaluated every model against, so
# a later bad recommendation (context too large, model wrongly excluded) can be traced
# back to the inputs: which chip, total RAM, how much was considered "used"/"free", and
# the current kernel free%/pressure. Pulled from `model_advisor.py --list-json` (the one
# command that computes all of these consistently) rather than re-deriving them in bash.
# Appends, so it's safe whether or not the engine has written to the log yet.
log_hardware_profile() {
    [[ -x "$MLX_PYTHON" ]] || return 0
    local ts info
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    info="$("$MLX_PYTHON" "$SCRIPT_DIR/model_advisor.py" --list-json 2>/dev/null \
        | "$MLX_PYTHON" -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit(0)
print("chip=%s gpu_cores=%s total_ram=%sGB used=%sGB free=%sGB" % (
    d.get("chip"), d.get("gpu_cores"),
    round(d["total_ram_gb"], 1) if d.get("total_ram_gb") else "?",
    round(d["current_wired_gb"], 2) if d.get("current_wired_gb") is not None else "?",
    round(d["free_wired_gb"], 2) if d.get("free_wired_gb") is not None else "?",
))
' 2>/dev/null)" || true
    [[ -z "$info" ]] && info="(hardware profile unavailable)"
    local pct pressure
    pct="$(get_percent_free 2>/dev/null)"
    pressure="$(pressure_level_name "$(get_pressure_level 2>/dev/null || echo 0)")"
    printf '%s - HARDWARE PROFILE: %s | kernel_free=%s%% pressure=%s\n' \
        "$ts" "$info" "${pct:-?}" "${pressure:-unknown}" >> "$MLX_LOG_FILE"
}

# Records what was selected and the EXACT argv the engine was launched with -- the one
# place to see, after the fact, how context size / kv-bits / backend / drafter / APC /
# concurrency actually resolved for this run. Appended just before launch.
log_launch_config() {
    local ts
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    {
        printf '%s - SELECTED MODEL: %s (%s)  backend=%s  context=%s  kv_bits=%s  concurrency=%s\n' \
            "$ts" "${LOCAL_MODEL:-?}" "${MLX_MODEL:-?}" "${MLX_BACKEND:-mlx_vlm}" \
            "${CONTEXT_WINDOW:-?}" "${KV_BITS:-unquantized}" "${CONCURRENCY:-?}"
        local drafter="${MLX_DRAFTER:-${MLX_DRAFT_MODEL:-}}"
        local mode="${MLX_DSPARK_MODE:-}"
        printf '%s - SPECULATION: drafter=%s  mode=%s  draft_kind=%s  draft_block_size=%s\n' \
            "$ts" "${drafter:-none}" "${mode:-none}" "${MLX_DRAFT_KIND:-none}" "${MLX_DRAFT_BLOCK_SIZE:-none}"
        printf '%s - APC: enabled=%s  num_blocks=%s  disk_path=%s  disk_max_gb=%s\n' \
            "$ts" "${APC_ENABLED:-0}" "${APC_NUM_BLOCKS:-?}" "${APC_DISK_PATH:-?}" "${APC_DISK_MAX_GB:-?}"
        printf '%s - LAUNCH ARGV: %s\n' "$ts" "$*"
    } >> "$MLX_LOG_FILE"
}

# Best-effort: write this session's real decode speed/TTFT/peak-memory to
# model_advisor.py's calibration store (~/.bushpilot/calibration/) so a FUTURE launch
# of this exact model on this exact chip can report a MEASURED tok/s instead of
# estimate_speed_tok_s's physics-based guess -- see model_advisor.py's
# save_calibration docstring, and the conversation history behind this feature: one
# global SPEED_CORRECTION_FACTOR is already confirmed wrong by up to ~1.7x across
# model families (docs/HARDWARE_SPEED_ESTIMATION.md §17 item #2), so a real per-model,
# per-chip measurement is strictly more trustworthy than the formula once one exists.
#
# Deliberately never allowed to block or fail teardown: a curl timeout, a server
# that's already dead, a parse error, or model_advisor.py itself erroring all just
# mean "nothing saved this time" -- every step below is `|| true`/`2>/dev/null`'d, and
# the whole function is called with `|| true` at its one call site in full_teardown.
# Called BEFORE wait_for_process_gone kills the server, since it needs the server
# alive to answer one last /metrics poll.
save_calibration_snapshot() {
    [[ -z "${LOCAL_MODEL:-}" ]] && return 0

    local metrics_json
    metrics_json="$(curl -fsS --max-time 3 "http://$MLX_HOST:$MLX_PORT/metrics" 2>/dev/null)" || return 0
    [[ -z "$metrics_json" ]] && return 0

    local parsed decode_tok_s context_tokens ttft_s peak_mem_gb
    parsed="$("$MLX_PYTHON" - "$MLX_BACKEND" "$metrics_json" "${CONTEXT_WINDOW:-}" <<'PYEOF' 2>/dev/null
import json, sys

backend, metrics_raw, context_window = sys.argv[1:4]

def safe_json(s):
    try:
        return json.loads(s)
    except Exception:
        return {}

def num(v):
    return v if isinstance(v, (int, float)) else ""

m = safe_json(metrics_raw)
if backend == "mlx_dspark":
    decode_tok_s = num(m.get("mean_decode_tokens_per_sec"))
    ttft_s = ""  # mlx-dspark does not report TTFT in its own /metrics
    peak_bytes = m.get("memory", {}).get("peak_bytes")
    peak_mem_gb = num(peak_bytes / 1e9) if isinstance(peak_bytes, (int, float)) else ""
else:
    decode_tok_s = num(m.get("summary", {}).get("avg_decode_tok_s"))
    ttft_s = num(m.get("latest", {}).get("ttft_s"))
    peak_mem_gb = num(m.get("latest", {}).get("peak_memory_gb"))

print(decode_tok_s)
print(context_window)
print(ttft_s)
print(peak_mem_gb)
PYEOF
)" || return 0

    { read -r decode_tok_s; read -r context_tokens; read -r ttft_s; read -r peak_mem_gb; } <<<"$parsed"
    [[ -z "$decode_tok_s" ]] && return 0

    local save_args=(--save-calibration "$LOCAL_MODEL" --decode-tok-s "$decode_tok_s")
    [[ -n "$context_tokens" ]] && save_args+=(--context-tokens "$context_tokens")
    [[ -n "$ttft_s" ]] && save_args+=(--ttft-s "$ttft_s")
    [[ -n "$peak_mem_gb" ]] && save_args+=(--peak-memory-gb "$peak_mem_gb")
    local active_drafter="${MLX_DRAFTER:-${MLX_DRAFT_MODEL:-}}"
    [[ -n "$active_drafter" ]] && save_args+=(--drafter "$active_drafter")

    "$MLX_PYTHON" "$SCRIPT_DIR/model_advisor.py" "${save_args[@]}" 2>/dev/null || true
}

full_teardown() {
    local reason="${1:-}"
    local mode="${2:-normal}"
    [[ -t 1 ]] && printf '\033[?25h' 2>/dev/null || true
    [[ -n "$reason" ]] && echo -e "\n[!] Stopping: $reason"
    log_teardown "$reason"
    if [[ "$mode" == "emergency" ]]; then
        # Under severe memory pressure (CRITICAL or <10% free), free the model's
        # wired GPU memory IMMEDIATELY before doing any dashboard render or
        # spawning Python subprocesses for calibration snapshots.
        wait_for_process_gone "$MLX_PROC_PATTERN" "MLX server" "${MEM_EMERGENCY_KILL_TIMEOUT:-3}" || true
        declare -F stop_router > /dev/null 2>&1 && stop_router
        rm -f "$MLX_STATE_FILE" "${MLX_ROSTER_FILE:-}" "${MLX_STARTING_PID_FILE:-}" "${MLX_SWAP_REQUEST_FILE:-}" "${MLX_SWAP_ERROR_FILE:-}"
        render_dashboard || true
        [[ -t 1 ]] && printf '\033[?25h' 2>/dev/null || true
        stty sane 2>/dev/null || true
    else
        save_calibration_snapshot || true
        [[ -t 1 ]] && printf '\033[?25h' 2>/dev/null || true
        stty sane 2>/dev/null || true
        wait_for_process_gone "$MLX_PROC_PATTERN" "MLX server" 20 || true
        declare -F stop_router > /dev/null 2>&1 && stop_router
        rm -f "$MLX_STATE_FILE" "${MLX_ROSTER_FILE:-}" "${MLX_STARTING_PID_FILE:-}" "${MLX_SWAP_REQUEST_FILE:-}" "${MLX_SWAP_ERROR_FILE:-}"
    fi
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

# mlx_vlm.server (third-party) doesn't tag requests with any client/tool identity in
# its /metrics response, so we can't learn this from the API. What we CAN see
# ourselves: every tool talks to $MLX_PORT over a local TCP socket, so lsof on that
# port surfaces the PIDs holding a connection to it. Exclude the server's/router's own
# PIDs and resolve the rest to process names -- that's "what tool is connected",
# best-effort, not request-level.
get_connected_clients() {
    local pids pid nm names cmdline
    pids="$(lsof -ti "tcp:$MLX_PORT" 2>/dev/null || true)"
    [[ -z "$pids" ]] && { echo ""; return; }

    names=""
    for pid in $pids; do
        cmdline="$(ps -p "$pid" -o command= 2>/dev/null || true)"
        if grep -q "$MLX_PROC_PATTERN" <<<"$cmdline" || grep -q "$MLX_ROUTER_MODULE" <<<"$cmdline"; then
            continue
        fi
        nm="$(ps -p "$pid" -o comm= 2>/dev/null)" || continue
        [[ -z "$nm" ]] && continue
        names+="$(basename "$nm")"$'\n'
    done
    [[ -z "$names" ]] && { echo ""; return; }
    printf '%s' "$names" | sort | uniq -c | awk '{count=$1; $1=""; sub(/^ /,""); printf "%s (%s), ", $0, count}' | sed 's/, $//'
}

# mlx-dspark's own /metrics already includes everything mlx_vlm needs a SEPARATE
# /v1/cache/stats call for (its prefix-cache stats are nested right in /metrics), and
# it has no /v1/cache/stats endpoint at all -- so the dspark branch below only ever
# fetches /metrics, skipping a call that would otherwise fail every single poll.
render_dashboard() {
    local metrics_json='{}' cache_json='{}' metrics_ok=0
    if metrics_json="$(curl -fsS --max-time 1 "http://$MLX_HOST:$MLX_PORT/metrics" 2>/dev/null)"; then
        metrics_ok=1
    else
        metrics_json='{}'
    fi
    if [[ "$MLX_BACKEND" != "mlx_dspark" && "$metrics_ok" -eq 1 ]]; then
        cache_json="$(curl -fsS --max-time 1 "http://$MLX_HOST:$MLX_PORT/v1/cache/stats" 2>/dev/null || echo '{}')"
    fi

    local pct_free pressure pressure_name mlx_rss now_ts elapsed elapsed_fmt connected_clients term_cols
    pct_free="$(get_percent_free)"
    pressure="$(get_pressure_level)"
    pressure_name="$(pressure_level_name "${pressure:-0}")"
    mlx_rss="$(get_mlx_rss_gb "${MLX_PID:-}")"
    connected_clients="$(get_connected_clients)"
    now_ts=$(date +%s)
    elapsed=$((now_ts - DASHBOARD_START_TS))
    elapsed_fmt=$(printf "%02d:%02d:%02d" $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60)))
    term_cols="${COLUMNS:-$(tput cols 2>/dev/null || echo 90)}"

    # mlx-dspark's /metrics shape is genuinely different from mlx_vlm's, not just a
    # field-renaming exercise -- confirmed live: no separate "latest request" vs.
    # "lifetime summary" split (only running means), no requests_failed/in_flight
    # counters, but real speculative-decoding stats (mean_accept_len, draft
    # acceptance rate, PER-POSITION acceptance showing where in the draft block
    # accuracy decays) mlx_vlm has no equivalent of, a real memory.peak_bytes/
    # active_bytes/cache_bytes breakdown, and real system.swap_used_bytes -- none of
    # which mlx_vlm's own dashboard can get from its API either. Deliberately
    # OMITTED: "auto_cap" (metrics field only populates when --max-draft auto is
    # passed; this script never passes it, so it's always null here -- not worth a
    # permanent "—"), and mlx-dspark's own "verdict.headline" (live-tested and found
    # it can say "No decode measured yet" even while mean_decode_tokens_per_sec and
    # mean_accept_len are already populated in the same response, so it isn't
    # trustworthy as a live-updating signal).
    "$MLX_PYTHON" - "$MLX_BACKEND" "$metrics_json" "$cache_json" "$MLX_MODEL" "$MLX_PORT" \
        "$CONTEXT_WINDOW" "$elapsed_fmt" "${APC_ENABLED}" "${pct_free:-?}" "$pressure_name" \
        "${mlx_rss:-?}" "$TEARDOWN_REASON" "$SCRIPT_DIR/launch_tool.sh" "${connected_clients:-}" \
        "$MLX_HOST" "$CONCURRENCY" "${MLX_DRAFTER:-}" "${MLX_DSPARK_MODE:-dflash}" \
        "${MEM_WARNING_STATUS:-}" "$term_cols" "${MLX_ROSTER_FILE:-}" "${LOCAL_MODEL:-}" <<'PYEOF'
import json, os, re, sys

(
    backend, metrics_raw, cache_raw, model, port, context, uptime, apc_on,
    pct_free, pressure, rss, teardown_reason, launch_cmd, connected, host,
    concurrency, drafter, mode, mem_warning, term_cols_raw, roster_path, active_local_id
) = sys.argv[1:23]

def safe_json(s):
    try:
        return json.loads(s)
    except Exception:
        return {}

metrics = safe_json(metrics_raw)
cache = safe_json(cache_raw)

roster_models = []
if roster_path and os.path.isfile(roster_path):
    try:
        with open(roster_path) as rf:
            roster_models = json.load(rf).get("models", [])
    except Exception:
        roster_models = []

def num(v, default="—"):
    return f"{v:,}" if isinstance(v, (int, float)) else default

def rate(v, default="—"):
    return f"{v:.1f}" if isinstance(v, (int, float)) else default

def pct(v, default="—"):
    return f"{v*100:.0f}%" if isinstance(v, (int, float)) else default

# Unlike pct() above (which formats an already-computed 0-1 fraction straight from
# /metrics), this derives the fraction itself from two raw counts -- mlx-dspark's
# /metrics has no precomputed cache-hit-rate field the way mlx_vlm's /v1/cache/stats
# does (token_hit_rate), and for mlx_vlm deriving it from the same two raw counts
# displayed on that line (prompt/reused) avoids trusting a separately-computed field.
def pct_ratio(numerator, denominator, default="—"):
    if not isinstance(numerator, (int, float)) or not isinstance(denominator, (int, float)) or denominator <= 0:
        return default
    return f"{numerator / denominator * 100:.1f}%"

def gb(v):
    return f"{v/1e9:.1f}" if isinstance(v, (int, float)) else "—"

def pos_accept(vals):
    if not isinstance(vals, list) or not vals:
        return "—"
    return "/".join(f"{v*100:.0f}%" for v in vals)

try:
    tc = int(term_cols_raw)
except ValueError:
    tc = 90
W = max(72, min(tc - 2, 110))

is_tty = sys.stdout.isatty()
use_color = is_tty and not os.environ.get("NO_COLOR")
BOLD = "\033[1m" if use_color else ""
DIM = "\033[2m" if use_color else ""
GREEN = "\033[32m" if use_color else ""
YELLOW = "\033[33m" if use_color else ""
RED = "\033[31m" if use_color else ""
CYAN = "\033[36m" if use_color else ""
RESET = "\033[0m" if use_color else ""

ANSI_RE = re.compile(r"\033\[[0-9;]*m")

def vis_len(s):
    return len(ANSI_RE.sub("", s))

def vis_trunc(s, max_vis):
    if vis_len(s) <= max_vis:
        return s
    out = []
    vis = 0
    pos = 0
    for m in ANSI_RE.finditer(s):
        chunk = s[pos:m.start()]
        take = min(len(chunk), max_vis - vis)
        out.append(chunk[:take])
        vis += take
        out.append(m.group(0))
        pos = m.end()
        if vis >= max_vis:
            break
    if vis < max_vis and pos < len(s):
        chunk = s[pos:]
        out.append(chunk[:max_vis - vis])
    if RESET and "\033[" in s:
        out.append(RESET)
    return "".join(out)

out_lines = []
eol = "\033[K" if is_tty else ""

def emit(raw):
    out_lines.append(raw + eol)

def line(s=""):
    inner_w = W - 4
    s_fit = vis_trunc(s, inner_w)
    pad = max(0, inner_w - vis_len(s_fit))
    emit(f"{DIM}│{RESET} {s_fit}{' ' * pad} {DIM}│{RESET}")

def wrap_line(s="", indent="                ", style=""):
    # For free-text that must not be lost (error messages, teardown reasons): wraps
    # onto as many boxed lines as needed instead of line()'s silent hard truncation.
    # Continuation lines are prefixed with `indent` to line up under the value column.
    width = W - 4
    plain = ANSI_RE.sub("", s)
    if not plain:
        line("")
        return
    first = True
    while plain:
        avail = width - (0 if first else len(indent))
        chunk, plain = plain[:avail], plain[avail:]
        text = chunk if first else indent + chunk
        line(f"{style}{text}{RESET}" if style else text)
        first = False

def rule(ch="─"):
    emit(f"{DIM}├{ch * (W - 2)}┤{RESET}")

def fmt_mem_status(pct_s, press_s):
    try:
        pf = int(pct_s)
    except ValueError:
        pf = -1
    bar_w = 10
    if 0 <= pf <= 100:
        used_pct = 100 - pf
        filled = max(0, min(bar_w, round(used_pct * bar_w / 100)))
        bar = "█" * filled + "░" * (bar_w - filled)
    else:
        bar = "░" * bar_w
    if press_s == "critical" or (0 <= pf < 15):
        c = f"{BOLD}{RED}"
    elif press_s == "warn" or (0 <= pf < 25):
        c = YELLOW
    else:
        c = GREEN
    return f"{c}[{bar}] {pct_s}%{RESET}", f"{c}{press_s}{RESET}"

mem_gauge, press_styled = fmt_mem_status(pct_free, pressure)
conn_styled = f"{GREEN}● {connected}{RESET}" if connected else f"{DIM}no tool connected yet{RESET}"

roster_styled = ""
if len(roster_models) > 1:
    parts = []
    for rm in roster_models:
        rid = rm.get("id") or rm.get("huggingface") or "?"
        if rid == active_local_id or rm.get("huggingface") == model:
            parts.append(f"{GREEN}{BOLD}● {rid}{RESET}")
        else:
            parts.append(f"{DIM}○ {rid}{RESET}")
    roster_styled = "   ".join(parts)

emit(f"{DIM}┌{'─' * (W - 2)}┐{RESET}")

if backend == "mlx_dspark":
    rounds = metrics.get("rounds", {})
    prefix_cache = metrics.get("prefix_cache", {})
    mem = metrics.get("memory", {})
    system = metrics.get("system", {})
    guard = metrics.get("memory_guard", {})

    line(f"{BOLD}{CYAN}BushPilot{RESET}  {BOLD}{model}{RESET}  {DIM}up {uptime}{RESET}   {CYAN}[mlx-dspark]{RESET}")
    line(f"{DIM}API endpoint{RESET}  {CYAN}http://{host}:{port}/v1{RESET}   {DIM}(OpenAI-compatible -- point any tool here){RESET}")
    line(f"{DIM}context{RESET} {context}   {DIM}concurrency{RESET} {concurrency}")
    rule()
    line(f"{BOLD}REQUESTS{RESET}        completed {num(metrics.get('requests'), 0)}   {DIM}(failed/in-flight not reported by mlx-dspark){RESET}")
    line(f"{BOLD}LIFETIME{RESET}        generated {num(metrics.get('completion_tokens'), 0)} tok   "
         f"overall {rate(metrics.get('mean_tokens_per_sec'))} tok/s   decode {BOLD}{rate(metrics.get('mean_decode_tokens_per_sec'))} tok/s{RESET}")
    rule()
    line(f"{BOLD}SPECULATION{RESET}     mode {mode or '—'}   drafter {drafter or '—'}")
    line(f"                accept_len {BOLD}{rate(metrics.get('mean_accept_len'))}{RESET}   draft_acceptance {pct(rounds.get('draft_acceptance'))}   "
         f"rounds {num(rounds.get('rounds'), 0)}   lookup_rounds {num(rounds.get('lookup_rounds'), 0)}")
    line(f"                position accept: {pos_accept(rounds.get('position_acceptance'))}")
    rule()
    line(f"{BOLD}PREFIX CACHE{RESET}    mode {prefix_cache.get('mode', '—')}   l2 {'on' if prefix_cache.get('l2') else 'off'}")
    line(f"                prompt {num(metrics.get('prompt_tokens'), 0)} tok   "
         f"reused {num(prefix_cache.get('reused_tokens'), 0)} tok   "
         f"cached {pct_ratio(prefix_cache.get('reused_tokens'), metrics.get('prompt_tokens'))}")
    line(f"                cached_tokens {num(prefix_cache.get('cached_tokens'), 0)}   hits {num(prefix_cache.get('hits'), 0)}   "
         f"partial_hits {num(prefix_cache.get('partial_hits'), 0)}")
    rule()
    line(f"{BOLD}MEMORY{RESET}          active {gb(mem.get('active_bytes'))} GB   cache {gb(mem.get('cache_bytes'))} GB   peak {gb(mem.get('peak_bytes'))} GB")
    line(f"{BOLD}SYSTEM{RESET}          free mem {mem_gauge}   pressure: {press_styled}   MLX RSS {rss} GB   swap {gb(system.get('swap_used_bytes'))} GB")
    rule()
    line(f"{BOLD}CONNECTED{RESET}       {conn_styled}")
    if roster_styled:
        line(f"{BOLD}ROSTER{RESET}          {roster_styled}")
    sheds = guard.get("sheds")
    if isinstance(sheds, (int, float)) and sheds > 0:
        rule()
        line(f"{YELLOW}⚠ memory guard has shed cache {int(sheds)} time(s) — last: {guard.get('last_shed') or 'unknown'}{RESET}")
    rule()
    line(f"{DIM}last error: none{RESET}")
else:
    summary = metrics.get("summary", {})
    latest = metrics.get("latest", {})
    apc_styled = f"{GREEN}on{RESET}" if apc_on == "1" else f"{DIM}off{RESET}"

    line(f"{BOLD}{CYAN}BushPilot{RESET}  {BOLD}{model}{RESET}  {DIM}up {uptime}{RESET}   {CYAN}[mlx-vlm]{RESET}")
    line(f"{DIM}API endpoint{RESET}  {CYAN}http://{host}:{port}/v1{RESET}   {DIM}(OpenAI-compatible -- point any tool here){RESET}")
    line(f"{DIM}context{RESET} {context}   {DIM}concurrency{RESET} {concurrency}   {DIM}APC:{RESET} {apc_styled}")
    rule()
    failed_n = summary.get("requests_failed")
    failed_s = f"{RED}{num(failed_n, 0)}{RESET}" if isinstance(failed_n, (int, float)) and failed_n > 0 else num(failed_n, 0)
    inflight_n = summary.get("in_flight")
    inflight_s = f"{CYAN}{num(inflight_n, 0)}{RESET}" if isinstance(inflight_n, (int, float)) and inflight_n > 0 else num(inflight_n, 0)
    line(f"{BOLD}REQUESTS{RESET}        completed {num(summary.get('requests_completed'), 0)}   "
         f"failed {failed_s}   in-flight {inflight_s}")
    line(f"{BOLD}LIFETIME{RESET}        generated {num(summary.get('generated_tokens_total'), 0)} tok   "
         f"avg decode {BOLD}{rate(summary.get('avg_decode_tok_s'), '—')} tok/s{RESET}")
    rule()
    line(f"{BOLD}LATEST REQUEST{RESET}  prefill {rate(latest.get('prefill_tok_s'))} tok/s   "
         f"decode {BOLD}{rate(latest.get('decode_tok_s'))} tok/s{RESET}   ttft {rate(latest.get('ttft_s'))}s")
    line(f"                elapsed {rate(latest.get('request_elapsed_s'))}s   "
         f"finish_reason: {latest.get('finish_reason', '—')}")
    rule()
    hits = cache.get("lookups_hit")
    misses = cache.get("lookups_miss")
    hit_rate = cache.get("token_hit_rate")
    hit_rate_s = f"{hit_rate*100:.0f}%" if isinstance(hit_rate, (int, float)) else "—"
    line(f"{BOLD}APC CACHE{RESET}       hit rate {hit_rate_s}   hits {num(hits, 0)} / misses {num(misses, 0)}   "
         f"evictions {num(cache.get('evictions'), 0)}")
    line(f"                prompt {num(summary.get('prompt_tokens_total'), 0)} tok   "
         f"reused {num(cache.get('matched_tokens'), 0)} tok   "
         f"cached {pct_ratio(cache.get('matched_tokens'), summary.get('prompt_tokens_total'))}")
    line(f"                mem pool {num(cache.get('pool_used'), 0)}/{num(cache.get('num_blocks'), 0)} blocks   "
         f"disk {gb(cache.get('disk_bytes'))}/{gb(cache.get('disk_max_bytes'))} GB")
    rule()
    line(f"{BOLD}SYSTEM{RESET}          free mem {mem_gauge}   pressure: {press_styled}   MLX RSS {rss} GB   "
         f"peak {rate(latest.get('peak_memory_gb'))} GB")
    rule()
    line(f"{BOLD}CONNECTED{RESET}       {conn_styled}")
    if roster_styled:
        line(f"{BOLD}ROSTER{RESET}          {roster_styled}")
    rule()
    err = summary.get("last_error") or "none"
    if err == "none":
        line(f"{DIM}last error: none{RESET}")
    else:
        wrap_line(f"last error: {err}", indent="            ", style=RED)

if mem_warning and not teardown_reason:
    rule()
    wrap_line(f"⚠ {mem_warning}", indent="  ", style=YELLOW)
if teardown_reason:
    rule()
    wrap_line(f"⚠ {teardown_reason}", indent="  ", style=f"{BOLD}{RED}")

emit(f"{DIM}└{'─' * (W - 2)}┘{RESET}")
emit(f"  {DIM}To connect a tool: open another terminal and run{RESET} {BOLD}{launch_cmd}{RESET}")
emit(f"  {DIM}m : switch model · Ctrl-C / q : stop everything{RESET}")

prefix = "\033[H" if is_tty else ""
suffix = "\033[J\n" if is_tty else "\n"
sys.stdout.write(prefix + "\n".join(out_lines) + suffix)
sys.stdout.flush()
PYEOF
}

# Folds monitor_memory.sh's real-time backstop directly into this poll loop (Phase 2
# decision) — same abort/warn/caution percent-free thresholds, same discrete CRITICAL
# pressure-level override, applying the exact same full-teardown path as Ctrl-C/q.
check_memory_backstop() {
    local pct_free pressure warn_pct caution_pct
    pct_free="$(get_percent_free)"
    pressure="$(get_pressure_level)"
    warn_pct="${MEM_WARN_BELOW_PCT:-18}"
    caution_pct="${MEM_CAUTION_BELOW_PCT:-25}"
    MEM_WARNING_STATUS=""

    if [[ -n "$pressure" && "$pressure" -eq 4 ]]; then
        MEM_CRITICAL_STREAK=$((MEM_CRITICAL_STREAK + 1))
        if [[ "$MEM_CRITICAL_STREAK" -ge "$MEM_CRITICAL_CONSECUTIVE_POLLS" ]]; then
            TEARDOWN_REASON="Memory pressure CRITICAL for ${MEM_CRITICAL_STREAK} consecutive checks — stopping to protect the machine."
            return 1
        fi
    else
        MEM_CRITICAL_STREAK=0
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

    if [[ "$MEM_CRITICAL_STREAK" -gt 0 ]]; then
        MEM_WARNING_STATUS="Memory pressure CRITICAL (${MEM_CRITICAL_STREAK}/${MEM_CRITICAL_CONSECUTIVE_POLLS} checks before auto-stop)"
    elif [[ "$MEM_ABORT_STREAK" -gt 0 ]]; then
        MEM_WARNING_STATUS="Free memory ${pct_free}% < ${MEM_ABORT_BELOW_PCT}% abort threshold (${MEM_ABORT_STREAK}/${MEM_ABORT_CONSECUTIVE_POLLS} checks before auto-stop)"
    elif [[ -n "$pressure" && "$pressure" -eq 2 ]]; then
        MEM_WARNING_STATUS="Kernel memory pressure WARN (free memory ${pct_free:-?}%)"
    elif [[ -n "$pct_free" ]] && awk -v p="$pct_free" -v w="$warn_pct" 'BEGIN{exit !(p<w)}'; then
        MEM_WARNING_STATUS="Low free memory: ${pct_free}% (<${warn_pct}% warning threshold)"
    elif [[ -n "$pct_free" ]] && awk -v p="$pct_free" -v c="$caution_pct" 'BEGIN{exit !(p<c)}'; then
        MEM_WARNING_STATUS="Memory caution: ${pct_free}% free (<${caution_pct}% caution threshold)"
    fi
    return 0
}

# Catches the backend dying (or its port getting stolen) for any reason OTHER than
# memory pressure -- see the comment at BACKEND_DOWN_CONSECUTIVE_POLLS above for the
# live incident this fixes. Deliberately checks BOTH /health reachability and that the
# expected process still owns the port: a dead backend fails the curl; a backend that
# got its port stolen out from under it (e.g. something else bound to $MLX_PORT after
# ours exited) would still make SOME /health call succeed, just not against our process.
# Also distinguishes curl HTTP timeout (exit 28) while the expected process still owns
# the port (which happens during heavy synchronous Metal prefill / vision passes) from
# connection refused / port loss, using BACKEND_HANG_CONSECUTIVE_POLLS (90s) for the
# former so legitimate long prefills are never killed after 10 seconds.
check_backend_alive() {
    local check_port="${MLX_BACKEND_PORT:-$MLX_PORT}"
    local curl_rc=0
    curl -fsS --max-time 2 "http://$MLX_HOST:$check_port/health" > /dev/null 2>&1 || curl_rc=$?

    if [[ "$curl_rc" -eq 0 ]] && mlx_port_owner_is_expected; then
        BACKEND_DOWN_STREAK=0
        BACKEND_BUSY_STREAK=0
        return 0
    fi

    if [[ "$curl_rc" -eq 28 ]] && mlx_port_owner_is_expected; then
        BACKEND_DOWN_STREAK=0
        BACKEND_BUSY_STREAK=$(( ${BACKEND_BUSY_STREAK:-0} + 1 ))
        if [[ "$BACKEND_BUSY_STREAK" -ge "${BACKEND_HANG_CONSECUTIVE_POLLS:-18}" ]]; then
            TEARDOWN_REASON="Backend on $MLX_HOST:$MLX_PORT timed out on /health for ${BACKEND_BUSY_STREAK} consecutive checks (inference hung) -- stopping. Check $MLX_LOG_FILE for the last thing it logged before this."
            return 1
        fi
        return 0
    fi

    BACKEND_BUSY_STREAK=0
    BACKEND_DOWN_STREAK=$((BACKEND_DOWN_STREAK + 1))
    if [[ "$BACKEND_DOWN_STREAK" -ge "$BACKEND_DOWN_CONSECUTIVE_POLLS" ]]; then
        TEARDOWN_REASON="Backend not responding on $MLX_HOST:$MLX_PORT for ${BACKEND_DOWN_STREAK} consecutive checks (process died or port was taken over) -- stopping. Check $MLX_LOG_FILE for the last thing it logged before this."
        return 1
    fi
    return 0
}

warn_if_context_exceeds_recommendation() {
    # Only meaningful in the fixed-model form (interactive mode derived CONTEXT_WINDOW
    # FROM the advisor, so it can't exceed it). No LOCAL_MODEL id means we can't look
    # the model up; no advisor/python yet means skip silently rather than block launch.
    [[ "$INTERACTIVE" == "true" ]] && return 0
    [[ -z "$LOCAL_MODEL" ]] && return 0
    [[ -x "$MLX_PYTHON" ]] || return 0

    local recommended
    recommended="$("$MLX_PYTHON" "$SCRIPT_DIR/model_advisor.py" --resolve "$LOCAL_MODEL" 2>/dev/null \
        | "$MLX_PYTHON" -c "import json,sys; print(json.load(sys.stdin).get('recommended_context') or '')" 2>/dev/null)" || true
    # Advisor exits non-zero (and prints nothing) when the model doesn't fit at all --
    # that's exactly the case worth warning about, so don't bail on empty here.
    if [[ -z "$recommended" ]]; then
        log_error "⚠ Context check: model_advisor.py reports '$LOCAL_MODEL' does not fit this machine's free memory at any context."
        log_error "  You requested ${CONTEXT_WINDOW} tokens. Continuing because this is the explicit fixed-model form, but the memory backstop may stop the server."
        _log_context_warning "$LOCAL_MODEL" "$CONTEXT_WINDOW" "does_not_fit"
        return 0
    fi

    if [[ "$CONTEXT_WINDOW" -gt "$recommended" ]]; then
        log_error "⚠ Context check: you requested ${CONTEXT_WINDOW} tokens, but model_advisor.py recommends at most ${recommended} for $LOCAL_MODEL on this machine's free memory."
        log_error "  A larger context grows the KV cache and can trip the memory backstop mid-session. Continuing since you asked explicitly."
        _log_context_warning "$LOCAL_MODEL" "$CONTEXT_WINDOW" "$recommended"
    fi
}

_log_context_warning() {
    # BUFFERS instead of appending: the fresh-launch path truncates the log in
    # ensure_mlx_running AFTER this runs (warn happens early in main(), before a
    # potentially long model download -- the terminal warning must not wait for
    # that), so an immediate append would be erased by start_run_log. flush_context_
    # warning() writes the buffered line(s) after the truncation point; on the
    # attach path (no truncation) they're flushed there instead.
    local model="$1" requested="$2" recommended="$3"
    local ts; ts="$(date '+%Y-%m-%d %H:%M:%S')"
    CONTEXT_WARNING_PENDING+="${CONTEXT_WARNING_PENDING:+$'\n'}${ts} - CONTEXT WARNING: requested=${requested} recommended=${recommended} model=${model} (fixed-model CLI form bypasses model_advisor.py)"
}

flush_context_warning() {
    [[ -n "${CONTEXT_WARNING_PENDING:-}" ]] || return 0
    printf '%s\n' "$CONTEXT_WARNING_PENDING" >> "$MLX_LOG_FILE"
    CONTEXT_WARNING_PENDING=""
}

switch_model_from_dashboard() {
    local old_stty="$1"
    [[ -t 1 ]] && printf '\033[?25h\033[2J\033[H' 2>/dev/null || true
    if [[ -n "$old_stty" ]]; then
        stty "$old_stty" 2>/dev/null || stty sane 2>/dev/null || true
    else
        stty sane 2>/dev/null || true
    fi

    local current_rss
    current_rss="$(get_mlx_rss_gb "${MLX_PID:-}")"
    export BUSHPILOT_RECLAIMABLE_RSS_GB="${current_rss:-0}"

    local advisor_output=""
    if is_command_available fzf; then
        advisor_output="$(run_fzf_model_picker)" || advisor_output=""
    else
        advisor_output="$("$MLX_PYTHON" "$SCRIPT_DIR/model_advisor.py" ${ADVISOR_FLAGS[@]+"${ADVISOR_FLAGS[@]}"})" || advisor_output=""
    fi
    unset BUSHPILOT_RECLAIMABLE_RSS_GB

    if [[ -n "$advisor_output" ]]; then
        local new_id
        new_id="$("$MLX_PYTHON" -c "import json,sys; print(json.loads(sys.argv[1])['id'])" "$advisor_output" 2>/dev/null || true)"
        if [[ -n "$new_id" && "$new_id" != "$LOCAL_MODEL" ]]; then
            perform_model_swap "$new_id" || true
        fi
    fi

    [[ -n "$old_stty" ]] && stty -icanon -echo min 0 time 0 2>/dev/null || true
    [[ -t 1 ]] && printf '\033[2J\033[H\033[?25l' 2>/dev/null || true
}

check_pending_swap_request() {
    [[ -f "$MLX_SWAP_REQUEST_FILE" ]] || return 1
    local req_model
    req_model="$(cat "$MLX_SWAP_REQUEST_FILE" 2>/dev/null | tr -d '\r\n' || true)"
    if [[ -z "$req_model" ]]; then
        rm -f "$MLX_SWAP_REQUEST_FILE"
        return 1
    fi
    perform_model_swap "$req_model" || true
    return 0
}

run_dashboard() {
    trap 'full_teardown "Ctrl-C"' INT
    trap 'full_teardown "terminal closed (SIGHUP)"' HUP
    trap 'full_teardown "terminated"' TERM

    local old_stty
    old_stty=$(stty -g 2>/dev/null) || old_stty=""
    [[ -n "$old_stty" ]] && stty -icanon -echo min 0 time 0 2>/dev/null || true
    [[ -t 1 ]] && printf '\033[2J\033[H\033[?25l' 2>/dev/null || true

    while true; do
        local loop_start_ts loop_elapsed sleep_rem
        loop_start_ts=$(date +%s)

        if check_pending_swap_request; then
            [[ -t 1 ]] && printf '\033[2J\033[H\033[?25l' 2>/dev/null || true
        fi

        if ! check_memory_backstop; then
            full_teardown "$TEARDOWN_REASON" "emergency"
        fi

        if ! check_backend_alive; then
            render_dashboard
            full_teardown "$TEARDOWN_REASON"
        fi

        render_dashboard

        loop_elapsed=$(( $(date +%s) - loop_start_ts ))
        sleep_rem=$(( DASHBOARD_POLL_INTERVAL - loop_elapsed ))
        [[ "$sleep_rem" -lt 1 ]] && sleep_rem=1

        local tick=0
        while [[ "$tick" -lt "$sleep_rem" ]]; do
            if check_pending_swap_request; then
                [[ -t 1 ]] && printf '\033[2J\033[H\033[?25l' 2>/dev/null || true
                break
            fi
            local key=""
            read -r -t 1 -n 1 key || true
            if [[ "$key" == "q" || "$key" == "Q" ]]; then
                full_teardown "stopped by user (q)"
            elif [[ "$key" == "m" || "$key" == "M" ]]; then
                switch_model_from_dashboard "$old_stty"
                break
            fi
            tick=$((tick + 1))
        done
    done
}

main() {
    check_system_dependencies
    ensure_symlinked_on_path
    bootstrap_mlx

    if [[ "$SELECT_ONLY" == "true" ]]; then
        run_interactive_selection
        return 0
    elif [[ "$FROM_ROSTER" == "true" ]]; then
        load_primary_from_roster_file || exit 1
    elif [[ "$INTERACTIVE" == "true" ]]; then
        run_interactive_selection
    fi

    # Record our startup PID so a waiting launch_tool.sh (auto-split mode) can track
    # startup progress, detect if this process dies before completing smoke checks, and
    # wait until ensure_mlx_running finishes before attaching the coding tool.
    printf '%s\n' "$$" > "$MLX_STARTING_PID_FILE"
    trap 'rm -f "${MLX_STARTING_PID_FILE:-}"' EXIT
    trap 'rm -f "${MLX_STARTING_PID_FILE:-}"; exit 130' INT HUP TERM

    # Runs AFTER model selection (not folded into bootstrap_mlx above) because the
    # interactive path doesn't know which model id it's serving until
    # run_interactive_selection returns -- bootstrap_mlx's own unconditional mlx-lm/
    # mlx-vlm install still has to happen before that, since model_advisor.py's picker
    # itself needs them regardless of which backend gets selected.
    resolve_serving_config

    if [[ "$INTERACTIVE" != "true" && "$FROM_ROSTER" != "true" ]]; then
        ensure_roster_file_built "$LOCAL_MODEL"
    fi

    # Fixed-model CLI form: the 3rd positional arg sets CONTEXT_WINDOW directly and
    # never consults model_advisor.py (see the ARGS[2] default near the top). That's a
    # deliberate escape hatch, but it silently allows a context far above what this
    # machine can hold -- a real incident: `... <id> <hf-id> 262144` on a 24GB Mac put
    # the KV cache at ~10GB and tripped the memory backstop. Warn (and log the same
    # line into the server log) when the requested context exceeds the advisor's own
    # memory-fit recommendation for this model; never override the user's choice.
    if [[ "$FROM_ROSTER" != "true" ]]; then
        warn_if_context_exceeds_recommendation
    fi

    ensure_backend_installed

    log_info "Local model server starting with $LOCAL_MODEL ($MLX_MODEL), context ${CONTEXT_WINDOW} tokens"
    if [[ "$MLX_BACKEND" == "mlx_dspark" ]]; then
        log_info "Serving backend: mlx-dspark (mode: ${MLX_DSPARK_MODE:-dflash}) -- text-only, no image/video support"
    elif [[ -n "$MLX_DRAFT_MODEL" ]]; then
        log_info "Serving backend: mlx_vlm with $MLX_DRAFT_KIND drafter ($MLX_DRAFT_MODEL) -- multimodal-capable"
    fi

    ensure_model_downloaded || exit 1
    patch_known_chat_template_bugs
    ensure_mlx_running
    rm -f "$MLX_STARTING_PID_FILE"

    run_dashboard
}

# Guarded so tests/shell/'s bats suite can `source` this file to unit-test the
# memory/backend-liveness backstop functions in isolation, without starting a real
# server (main's full lifecycle: venv bootstrap, model download, mlx_vlm.server
# launch, the live dashboard loop). Executing this file directly (./start_mlx_server.sh
# or `bash start_mlx_server.sh`) behaves exactly as before -- this only changes
# behavior when sourced.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
