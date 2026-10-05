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
    local preview_cmd="\"$MLX_PYTHON\" \"$SCRIPT_DIR/model_advisor.py\" --detail {1}"
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

    local header_text=$'Models fit, context size and speed are calculated live for your hardware (chip + free memory); speed is an estimate.\n↑↓ move · type to filter · Enter select · Esc cancel\nSort: click a column header (again to reverse), or ^R newest · ^S fastest · ^T largest · ^O name\ngreen = fits as-is · yellow = KV-quantized to fit · dim = does not fit · ✓ tested · ○ untested'
    # The column-header row sits right below fzf's --header lines; a click-header event
    # reports which line was clicked, and only that row should trigger a re-sort.
    local header_row=$(( $(printf '%s\n' "$header_text" | wc -l) + 1 ))

    while true; do
        : > "$sort_state"   # every (re)opened picker starts from the default sort
        selection="$("$MLX_PYTHON" "$SCRIPT_DIR/model_advisor.py" --list-compact --color --term-width "$term_width" --sort-state "$sort_state" ${ADVISOR_FLAGS[@]+"${ADVISOR_FLAGS[@]}"} | fzf \
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
            --ansi --border=rounded --pointer="▶" \
            --header="$header_text" \
            --prompt="Model> " --height=~80% --layout=reverse)"
        if [[ -z "$selection" ]]; then
            log_error "Model selection cancelled or no model available."
            return 1
        fi

        selected_id="$(printf '%s' "$selection" | cut -f1)"
        reason="$(printf '%s' "$selection" | cut -f3)"
        if [[ -n "$reason" ]]; then
            echo "[INFO] $selected_id doesn't fit this machine right now: $reason" >&2
            read -r -p "Press Enter to pick another model... " _ </dev/tty || true
            continue
        fi

        if ! "$MLX_PYTHON" "$SCRIPT_DIR/model_advisor.py" --resolve "$selected_id" ${ADVISOR_FLAGS[@]+"${ADVISOR_FLAGS[@]}"}; then
            log_error "Could not resolve '$selected_id' -- free memory may have changed since the list was shown. Try again."
            return 1
        fi
        return 0
    done
}

# Interactive model selection (Phase 2: no --tool factored in here anymore -- the
# supervisor is deliberately tool-agnostic now; a tool-aware preload check moves to
# Phase 3's launch_tool at attach-time instead).
run_interactive_selection() {
    log_info "Detecting hardware and free memory for model recommendations..."

    local advisor_output
    if [[ -n "$CONFIG_PREFERRED_MODEL" ]] && advisor_output=$("$MLX_PYTHON" "$SCRIPT_DIR/model_advisor.py" --resolve "$CONFIG_PREFERRED_MODEL" ${ADVISOR_FLAGS[@]+"${ADVISOR_FLAGS[@]}"} 2>/dev/null); then
        log_success "Using preferred_model from config.json: $CONFIG_PREFERRED_MODEL"
    elif is_command_available fzf; then
        advisor_output="$(run_fzf_model_picker)" || exit 1
    elif ! advisor_output=$("$MLX_PYTHON" "$SCRIPT_DIR/model_advisor.py" ${ADVISOR_FLAGS[@]+"${ADVISOR_FLAGS[@]}"}); then
        log_error "Model selection cancelled or failed."
        exit 1
    fi

    MLX_MODEL=$("$MLX_PYTHON" -c "import json,sys; print(json.loads(sys.argv[1])['huggingface'])" "$advisor_output")
    LOCAL_MODEL=$("$MLX_PYTHON" -c "import json,sys; print(json.loads(sys.argv[1])['id'])" "$advisor_output")
    CONTEXT_WINDOW=$("$MLX_PYTHON" -c "import json,sys; print(json.loads(sys.argv[1])['recommended_context'])" "$advisor_output")
    KV_BITS=$("$MLX_PYTHON" -c "import json,sys; v=json.loads(sys.argv[1])['kv_bits']; print(v if v is not None else '')" "$advisor_output")
    KV_BITS_APC_CAVEAT=$("$MLX_PYTHON" -c "import json,sys; print('1' if json.loads(sys.argv[1])['kv_bits_apc_caveat'] else '')" "$advisor_output")
    CONCURRENCY=$("$MLX_PYTHON" -c "import json,sys; print(json.loads(sys.argv[1])['concurrency'])" "$advisor_output")
    # Overrides this script's own static PREFILL_STEP_SIZE=2048 default (set near the
    # top of this file) with model_advisor.py's recommend_prefill_step_size() decision
    # -- a real per-(chip, model) sweep measurement when one exists on this machine
    # (see docs/NEXT_SESSION_HANDOFF_PREFILL_SWEEP.md / sweep_model.py), the SAME global
    # default otherwise. Only the --interactive path goes through model_advisor.py at
    # all, so the fixed-model form (scripting/restarts) keeps today's static default
    # unchanged, exactly like KV_BITS above.
    local prefill_step_size_val
    prefill_step_size_val=$("$MLX_PYTHON" -c "import json,sys; print(json.loads(sys.argv[1])['prefill_step_size'])" "$advisor_output" 2>/dev/null) || prefill_step_size_val=""
    [[ -n "$prefill_step_size_val" ]] && PREFILL_STEP_SIZE="$prefill_step_size_val"

    local kv_bits_log="unquantized (full decode speed — recommended context already fits free memory)"
    if [[ -n "$KV_BITS" ]]; then
        kv_bits_log="quantized to ${KV_BITS}-bit (memory-constrained — needed to fit this context)"
    fi
    local ctx_source_log="computed from your current free memory"
    [[ -n "$CONFIG_MAX_CONTEXT" ]] && ctx_source_log+=" and capped by config.json's max_context_tokens (rounds down to the nearest tier)"
    log_success "Selected $LOCAL_MODEL ($MLX_MODEL) — context set to ${CONTEXT_WINDOW} tokens ($ctx_source_log)"
    log_success "KV cache: $kv_bits_log"
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
        exit 1
    fi
    log_success "Download complete: $repo"
}

ensure_model_downloaded() {
    ensure_repo_downloaded "$MLX_MODEL" "target model"
    # MLX_DRAFTER (mlx_dspark backend) and MLX_DRAFT_MODEL (mlx_vlm backend) are
    # mutually exclusive per resolve_serving_config -- at most one is ever set.
    local drafter_repo="${MLX_DRAFTER:-$MLX_DRAFT_MODEL}"
    if [[ -n "$drafter_repo" ]]; then
        ensure_repo_downloaded "$drafter_repo" "drafter"
    fi
}

# Returns 0 only if the process actually LISTENING on $MLX_PORT is our expected
# server process (mlx_vlm.server, or `mlx-dspark serve` for an mlx_dspark-backed
# model -- see MLX_PROC_PATTERN in resolve_serving_config). /health alone is not
# sufficient -- confirmed live, a foreign, unmanaged server (e.g. a stray
# mlx_lm.server) can hold the port while answering /health and hang forever on real
# inference.
mlx_port_owner_is_expected() {
    local pids
    pids="$(lsof -ti "tcp:$MLX_PORT" 2>/dev/null || true)"
    [[ -z "$pids" ]] && return 1

    local pid
    for pid in $pids; do
        if ps -p "$pid" -o command= 2>/dev/null | grep -q "$MLX_PROC_PATTERN"; then
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
                wait_for_process_gone "$MLX_PROC_PATTERN" "MLX server" 30
                rm -f "$MLX_STATE_FILE"
            fi
        else
            log_error "Port $MLX_PORT is answering /health, but it is NOT this script's managed server process ($MLX_PROC_PATTERN):"
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
            # Append-only (do NOT truncate -- the running server's own log must survive,
            # and its fd keeps writing to this file). Notes the attach plus a fresh
            # hardware snapshot, since free memory at attach time is what any FUTURE
            # advisor consult in this terminal would see.
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
                wait_for_process_gone "$MLX_PROC_PATTERN" "MLX server" 30
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
        wait_for_process_gone "$MLX_PROC_PATTERN" "MLX server" 30
        rm -f "$MLX_STATE_FILE"
    fi

    wait_for_port_free "$MLX_PORT" 15

    local mlx_args=()
    local -a launch_cmd=()

    if [[ "$MLX_BACKEND" == "mlx_dspark" ]]; then
        # mlx-dspark's own flag set -- deliberately different names/semantics than
        # mlx_vlm's below (--context-window not --max-kv-size, --kv-bits {0,4,8} not
        # a separate --kv-group-size/--quantized-kv-start/--kv-quant-scheme trio,
        # --max-batch not --max-num-seqs). No --prefill-step-size equivalent exists
        # (mlx-dspark auto-calibrates its own prefill path per machine+model+quant --
        # that self-tuning is the whole reason this backend is used at all for the
        # models that have one, see models.json's serving.measured_speedup notes).
        mlx_args=(
            --model "$MLX_MODEL"
            --host "$MLX_HOST"
            --port "$MLX_PORT"
            --context-window "$CONTEXT_WINDOW"
            --mode "${MLX_DSPARK_MODE:-dflash}"
        )
        [[ -n "$MLX_DRAFTER" ]] && mlx_args+=(--drafter "$MLX_DRAFTER")
        # Prefix-cache tuning from config.json (all optional; unset = mlx-dspark's
        # own defaults, which are fine -- this is a power-user tier, e.g. the L2 SSD
        # spill via dspark_prefix_cache_dir). Flag names/semantics confirmed against
        # `mlx-dspark serve --help`, including max-ram-mb's requires-dir rule, which
        # is enforced here rather than leaving the user to decode mlx-dspark's own
        # startup-error wording for it.
        if [[ -n "$CONFIG_DSPARK_PC_MAX_RAM_MB" && -z "$CONFIG_DSPARK_PC_DIR" ]]; then
            log_error "config.json: dspark_prefix_cache_max_ram_mb requires dspark_prefix_cache_dir (per mlx-dspark's own --prefix-cache-max-ram-mb contract) — set both or neither."
            exit 1
        fi
        [[ -n "$CONFIG_DSPARK_PC_SLOTS" ]] && mlx_args+=(--prefix-cache-slots "$CONFIG_DSPARK_PC_SLOTS")
        [[ -n "$CONFIG_DSPARK_PC_RUNGS" ]] && mlx_args+=(--prefix-cache-rungs "$CONFIG_DSPARK_PC_RUNGS")
        [[ -n "$CONFIG_DSPARK_PC_DIR" ]] && mlx_args+=(--prefix-cache-dir "$CONFIG_DSPARK_PC_DIR")
        [[ -n "$CONFIG_DSPARK_PC_MAX_RAM_MB" ]] && mlx_args+=(--prefix-cache-max-ram-mb "$CONFIG_DSPARK_PC_MAX_RAM_MB")
        if [[ -n "$KV_BITS" ]]; then
            # --kv-bits disables --max-batch on mlx-dspark's own side (confirmed in
            # `mlx-dspark serve --help`) -- omit --max-batch entirely in this branch
            # rather than passing a combination it would reject.
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
            --port "$MLX_PORT"
            --max-kv-size "$CONTEXT_WINDOW"
            --prefill-step-size "$PREFILL_STEP_SIZE"
            # Server's own default is UNBOUNDED concurrent sequences -- since
            # recommend_max_context already sizes CONTEXT_WINDOW's memory footprint for
            # exactly $CONCURRENCY slots (see model_advisor.py), leaving this unset would
            # let an (N+1)th concurrent request allocate a KV cache with zero memory
            # headroom reserved for it -- the exact crash this project exists to prevent.
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
        # --draft-model/--draft-kind/--draft-block-size are additive to the same
        # mlx_vlm.server process -- this is the multimodal-capable "keep vision, some
        # speedup" tier (see models.json's serving.multimodal_capable), not a
        # different server or a different admission/health contract.
        if [[ -n "$MLX_DRAFT_MODEL" ]]; then
            mlx_args+=(--draft-model "$MLX_DRAFT_MODEL")
            [[ -n "$MLX_DRAFT_KIND" ]] && mlx_args+=(--draft-kind "$MLX_DRAFT_KIND")
            [[ -n "$MLX_DRAFT_BLOCK_SIZE" ]] && mlx_args+=(--draft-block-size "$MLX_DRAFT_BLOCK_SIZE")
        fi

        mkdir -p "$APC_DISK_PATH"
        export APC_ENABLED APC_NUM_BLOCKS APC_DISK_PATH APC_DISK_MAX_GB

        launch_cmd=("$MLX_PYTHON" -m "$MLX_SERVER_MODULE" "${mlx_args[@]}")
    fi

    # Fresh launch coming up: reset the log ONCE here (not at script start -- an
    # attach to an already-healthy server returns above without reaching this, and
    # truncating then would have destroyed the running server's own log), then
    # record this run's hardware profile and launch decision before the engine
    # starts appending, so context/kv-bits/backend/drafter/APC/concurrency are
    # always reconstructable from the log alone.
    start_run_log
    log_hardware_profile
    flush_context_warning
    log_launch_config "${launch_cmd[@]}"

    "${launch_cmd[@]}" >> "$MLX_LOG_FILE" 2>&1 &
    MLX_PID=$!
    local load_start_ts; load_start_ts=$(date +%s)

    if ! wait_for_url "http://$MLX_HOST:$MLX_PORT/health" 600 "$MLX_PID"; then
        log_error "MLX server failed to start. Check $MLX_LOG_FILE"
        tail -20 "$MLX_LOG_FILE" >&2
        exit 1
    fi
    # Best-effort, never blocks the launch -- a stale or missing file just means
    # sweep_model.py's "load.load_time_s" field reports null, same as if this script
    # predated the feature entirely.
    printf '%s\n' "$(( $(date +%s) - load_start_ts ))" > "$MLX_LOAD_TIME_FILE" 2>/dev/null || true

    if ! run_startup_smoke_check; then
        log_error "Startup smoke check failed — tearing down. Check $MLX_LOG_FILE"
        wait_for_process_gone "$MLX_PROC_PATTERN" "MLX server" 30
        exit 1
    fi

    echo "$desired_state" > "$MLX_STATE_FILE"
    log_success "MLX server started and healthy"
}

# ---------------------------------------------------------------------------
# Live status dashboard + memory-pressure backstop + full teardown on
# Ctrl-C/q/HUP/TERM. See docs/PHASE_PLAN.md Phase 2's locked dashboard layout.
# ---------------------------------------------------------------------------
DASHBOARD_START_TS=$(date +%s)
MEM_ABORT_STREAK=0
MEM_CRITICAL_STREAK=0
BACKEND_DOWN_STREAK=0
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

    "$MLX_PYTHON" "$SCRIPT_DIR/model_advisor.py" "${save_args[@]}" 2>/dev/null || true
}

full_teardown() {
    local reason="${1:-}"
    [[ -n "$reason" ]] && echo -e "\n[!] Stopping: $reason"
    log_teardown "$reason"
    save_calibration_snapshot || true
    stty sane 2>/dev/null || true
    wait_for_process_gone "$MLX_PROC_PATTERN" "MLX server" 20 || true
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

# mlx-dspark's own /metrics already includes everything mlx_vlm needs a SEPARATE
# /v1/cache/stats call for (its prefix-cache stats are nested right in /metrics), and
# it has no /v1/cache/stats endpoint at all -- so the dspark branch below only ever
# fetches /metrics, skipping a call that would otherwise fail every single poll.
render_dashboard() {
    local metrics_json cache_json=""
    metrics_json="$(curl -fsS --max-time 3 "http://$MLX_HOST:$MLX_PORT/metrics" 2>/dev/null || echo '{}')"
    [[ "$MLX_BACKEND" != "mlx_dspark" ]] && \
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

    if [[ "$MLX_BACKEND" == "mlx_dspark" ]]; then
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
        "$MLX_PYTHON" - "$metrics_json" "$MLX_MODEL" "$MLX_PORT" "$CONTEXT_WINDOW" \
            "$elapsed_fmt" "${pct_free:-?}" "$pressure_name" "${mlx_rss:-?}" "$TEARDOWN_REASON" \
            "$SCRIPT_DIR/launch_tool.sh" "${connected_clients:-}" "$MLX_HOST" "$CONCURRENCY" \
            "$MLX_DRAFTER" "${MLX_DSPARK_MODE:-dflash}" <<'PYEOF-DSPARK'
import json, sys

metrics_raw, model, port, context, uptime, pct_free, pressure, rss, teardown_reason, launch_cmd, connected, host, concurrency, drafter, mode = sys.argv[1:16]

def safe_json(s):
    try:
        return json.loads(s)
    except Exception:
        return {}

metrics = safe_json(metrics_raw)
rounds = metrics.get("rounds", {})
prefix_cache = metrics.get("prefix_cache", {})
mem = metrics.get("memory", {})
system = metrics.get("system", {})
guard = metrics.get("memory_guard", {})

def num(v, default="—"):
    return f"{v:,}" if isinstance(v, (int, float)) else default

def rate(v, default="—"):
    return f"{v:.1f}" if isinstance(v, (int, float)) else default

def pct(v, default="—"):
    return f"{v*100:.0f}%" if isinstance(v, (int, float)) else default

def gb(v):
    return f"{v/1e9:.1f}" if isinstance(v, (int, float)) else "—"

def pos_accept(vals):
    if not isinstance(vals, list) or not vals:
        return "—"
    return "/".join(f"{v*100:.0f}%" for v in vals)

W = 90
def line(s=""):
    s = s[:W - 4]
    print("│ " + s.ljust(W - 4) + " │")
def wrap_line(s="", indent="                "):
    # For free-text that must not be lost (teardown reasons): wraps onto as many
    # boxed lines as needed instead of line()'s silent hard truncation.
    width = W - 4
    if not s:
        line("")
        return
    first = True
    while s:
        avail = width - (0 if first else len(indent))
        chunk, s = s[:avail], s[avail:]
        line(chunk if first else indent + chunk)
        first = False
def rule(ch="─"):
    print("├" + ch * (W - 2) + "┤")

print("┌" + "─" * (W - 2) + "┐")
line(f"BushPilot  {model}  up {uptime}   [mlx-dspark]")
line(f"API endpoint  http://{host}:{port}/v1   (OpenAI-compatible -- point any tool here)")
line(f"context {context}   concurrency {concurrency}")
rule()
line(f"REQUESTS        completed {num(metrics.get('requests'), 0)}   (failed/in-flight not reported by mlx-dspark)")
line(f"LIFETIME        prompt {num(metrics.get('prompt_tokens'), 0)} tok   generated {num(metrics.get('completion_tokens'), 0)} tok")
line(f"                overall {rate(metrics.get('mean_tokens_per_sec'))} tok/s   decode {rate(metrics.get('mean_decode_tokens_per_sec'))} tok/s")
rule()
line(f"SPECULATION     mode {mode or '—'}   drafter {drafter or '—'}")
line(f"                accept_len {rate(metrics.get('mean_accept_len'))}   draft_acceptance {pct(rounds.get('draft_acceptance'))}   "
     f"rounds {num(rounds.get('rounds'), 0)}   lookup_rounds {num(rounds.get('lookup_rounds'), 0)}")
line(f"                position accept: {pos_accept(rounds.get('position_acceptance'))}")
rule()
line(f"PREFIX CACHE    mode {prefix_cache.get('mode', '—')}   l2 {'on' if prefix_cache.get('l2') else 'off'}")
line(f"                cached_tokens {num(prefix_cache.get('cached_tokens'), 0)}   hits {num(prefix_cache.get('hits'), 0)}   "
     f"partial_hits {num(prefix_cache.get('partial_hits'), 0)}   reused {num(prefix_cache.get('reused_tokens'), 0)}")
rule()
line(f"MEMORY          active {gb(mem.get('active_bytes'))} GB   cache {gb(mem.get('cache_bytes'))} GB   peak {gb(mem.get('peak_bytes'))} GB")
line(f"SYSTEM          free mem {pct_free}%   pressure: {pressure}   MLX RSS {rss} GB   swap {gb(system.get('swap_used_bytes'))} GB")
rule()
line(f"CONNECTED       {connected if connected else 'no tool connected yet'}")
sheds = guard.get("sheds")
if isinstance(sheds, (int, float)) and sheds > 0:
    rule()
    line(f"⚠ memory guard has shed cache {int(sheds)} time(s) — last: {guard.get('last_shed') or 'unknown'}")
rule()
line("last error: none")
if teardown_reason:
    rule()
    wrap_line(f"⚠ {teardown_reason}", indent="  ")
print("└" + "─" * (W - 2) + "┘")
print(f"  To connect a tool: open another terminal and run {launch_cmd}")
print("  Ctrl-C / q : stop everything")
PYEOF-DSPARK
        return
    fi

    "$MLX_PYTHON" - "$metrics_json" "$cache_json" "$MLX_MODEL" "$MLX_PORT" "$CONTEXT_WINDOW" \
        "$elapsed_fmt" "${APC_ENABLED}" "${pct_free:-?}" "$pressure_name" "${mlx_rss:-?}" "$TEARDOWN_REASON" \
        "$SCRIPT_DIR/launch_tool.sh" "${connected_clients:-}" "$MLX_HOST" "$CONCURRENCY" <<'PYEOF'
import json, sys

metrics_raw, cache_raw, model, port, context, uptime, apc_on, pct_free, pressure, rss, teardown_reason, launch_cmd, connected, host, concurrency = sys.argv[1:16]

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
def wrap_line(s="", indent="                "):
    # For free-text that must not be lost (error messages, teardown reasons): wraps
    # onto as many boxed lines as needed instead of line()'s silent hard truncation.
    # Continuation lines are prefixed with `indent` to line up under the value column.
    width = W - 4
    if not s:
        line("")
        return
    first = True
    while s:
        avail = width - (0 if first else len(indent))
        chunk, s = s[:avail], s[avail:]
        line(chunk if first else indent + chunk)
        first = False
def rule(ch="─"):
    print("├" + ch * (W - 2) + "┤")
def gb(v):
    return f"{v/1e9:.1f}" if isinstance(v, (int, float)) else "—"

print("┌" + "─" * (W - 2) + "┐")
line(f"BushPilot  {model}  up {uptime}")
line(f"API endpoint  http://{host}:{port}/v1   (OpenAI-compatible -- point any tool here)")
line(f"context {context}   concurrency {concurrency}   APC: {'on' if apc_on == '1' else 'off'}")
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
wrap_line(f"last error: {err}", indent="            ")
if teardown_reason:
    rule()
    wrap_line(f"⚠ {teardown_reason}", indent="  ")
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
    return 0
}

# Catches the backend dying (or its port getting stolen) for any reason OTHER than
# memory pressure -- see the comment at BACKEND_DOWN_CONSECUTIVE_POLLS above for the
# live incident this fixes. Deliberately checks BOTH /health reachability and that the
# expected process still owns the port: a dead backend fails the curl; a backend that
# got its port stolen out from under it (e.g. something else bound to $MLX_PORT after
# ours exited) would still make SOME /health call succeed, just not against our process.
check_backend_alive() {
    if curl -fsS --max-time 3 "http://$MLX_HOST:$MLX_PORT/health" > /dev/null 2>&1 \
        && mlx_port_owner_is_expected; then
        BACKEND_DOWN_STREAK=0
        return 0
    fi

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

        if ! check_backend_alive; then
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

    # Runs AFTER model selection (not folded into bootstrap_mlx above) because the
    # interactive path doesn't know which model id it's serving until
    # run_interactive_selection returns -- bootstrap_mlx's own unconditional mlx-lm/
    # mlx-vlm install still has to happen before that, since model_advisor.py's picker
    # itself needs them regardless of which backend gets selected.
    resolve_serving_config

    # Fixed-model CLI form: the 3rd positional arg sets CONTEXT_WINDOW directly and
    # never consults model_advisor.py (see the ARGS[2] default near the top). That's a
    # deliberate escape hatch, but it silently allows a context far above what this
    # machine can hold -- a real incident: `... <id> <hf-id> 262144` on a 24GB Mac put
    # the KV cache at ~10GB and tripped the memory backstop. Warn (and log the same
    # line into the server log) when the requested context exceeds the advisor's own
    # memory-fit recommendation for this model; never override the user's choice.
    warn_if_context_exceeds_recommendation

    ensure_backend_installed

    log_info "Local model server starting with $LOCAL_MODEL ($MLX_MODEL), context ${CONTEXT_WINDOW} tokens"
    if [[ "$MLX_BACKEND" == "mlx_dspark" ]]; then
        log_info "Serving backend: mlx-dspark (mode: ${MLX_DSPARK_MODE:-dflash}) -- text-only, no image/video support"
    elif [[ -n "$MLX_DRAFT_MODEL" ]]; then
        log_info "Serving backend: mlx_vlm with $MLX_DRAFT_KIND drafter ($MLX_DRAFT_MODEL) -- multimodal-capable"
    fi

    ensure_model_downloaded
    ensure_mlx_running

    run_dashboard
}

main "$@"
