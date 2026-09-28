#!/usr/bin/env bash
set -euo pipefail

# Usage: start_litellm_proxy.sh [--interactive] [mode] [model] [model_id]
# Modes: claude, opencode
# Example (fixed model):  start_litellm_proxy.sh claude qwen3.5-9b-mlx-local
# Example (interactive):  start_litellm_proxy.sh --interactive
# Example (also interactive, same as above): start_litellm_proxy.sh
#
# Running with no mode at all (e.g. just `start_litellm_proxy.sh`) defaults to
# --interactive rather than erroring out demanding a mode be named.
#
# --interactive runs model_advisor.py: detects free wired memory + chip, shows every
# models.json entry (greying out ones that don't currently fit) with a
# computed-not-guessed recommended context size and an estimated (labeled as such)
# tok/s, and lets the user pick. See HARDWARE_SPEED_ESTIMATION.md and model_advisor.py
# for the methodology behind both numbers — neither is fabricated, but the context
# number is an exact calculation while the speed number is a real-but-approximate
# estimate. model_advisor.py also now decides, per model, whether the KV cache gets
# quantized at all (HARDWARE_SPEED_ESTIMATION.md §14): unquantized (full decode speed)
# whenever the recommended context already fits free memory, only falling back to
# --kv-bits 8 when it doesn't — confirmed real KV-quant costs real decode speed with
# zero memory benefit at short context, so it's no longer applied unconditionally.
#
# Automatic Prefix Caching (APC, HARDWARE_SPEED_ESTIMATION.md §13) is always enabled on
# every launch — a real mlx_vlm feature, off by default upstream, that reuses cached KV
# blocks across requests sharing a prefix (measured ~97.7% TTFT reduction on the
# growing-conversation pattern OpenCode/Claude Code produce every turn) and persists
# that cache to disk (~/.cache/mlx-vlm/caching) across this script's own model-switch
# restarts of mlx_vlm.server.
#
# If <mode> is omitted, --interactive also prompts for which tool to launch (Claude
# Code or OpenCode), greying out whichever of those isn't detected on this system.
#
# Google Antigravity is deliberately NOT supported here (researched and dropped, not
# just unimplemented): there is no way to point it at this local/custom model.
# Antigravity's desktop IDE custom-endpoint settings are reported non-functional
# (Google AI Developer forum thread, unresolved as of this writing). Its CLI (agy)
# only lets you override the endpoint for its "gemini" provider via
# GOOGLE_GEMINI_BASE_URL, which expects a Gemini-API-shaped server, not the
# OpenAI-shaped one LiteLLM serves here — and even that path only reaches agy's own
# fixed hosted model roster (Gemini/Claude/GPT-OSS) via --model, not an arbitrary
# local model name. The only bridges found were unofficial third-party proxy/wrapper
# projects patching Google's CLI, which isn't something to wire in silently. Same
# situation as Devin CLI, which also has no proxy override at all.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTERACTIVE=false
MONITOR=false
ARGS=()
for arg in "$@"; do
    case "$arg" in
        --interactive) INTERACTIVE=true ;;
        --monitor) MONITOR=true ;;
        *) ARGS+=("$arg") ;;
    esac
done

MODE="${ARGS[0]:-}"
LOCAL_MODEL="${ARGS[1]:-}"
MLX_MODEL="${ARGS[2]:-mlx-community/Qwen3.5-9B-MLX-8bit}"

# No mode given at all (e.g. running the script with zero args, or just --monitor) ->
# default to interactive rather than erroring out demanding a mode be named.
if [[ "$INTERACTIVE" == "false" && -z "$MODE" ]]; then
    INTERACTIVE=true
fi

# Configuration
#
# Model download directory: reuse whatever's already there (case-insensitively) rather
# than creating a second, differently-cased directory next to an existing one full of
# already-downloaded models — e.g. an existing ~/MLX_Models must be reused, not
# shadowed by a fresh empty ~/mlx_models. `find -iname` matches case-insensitively
# regardless of the underlying filesystem's own case-sensitivity, so this is correct
# both on macOS's default case-insensitive APFS and on a case-sensitive APFS volume.
EXISTING_MLX_HOME="$(find "$HOME" -maxdepth 1 -iname "mlx_models" -type d 2>/dev/null | head -1)"
export HF_HOME="${EXISTING_MLX_HOME:-$HOME/mlx_models}"
mkdir -p "$HF_HOME"

# Use Python's certifi CA bundle if available, fall back to system default.
# `|| true`: if the system python3 doesn't have certifi installed, this exits nonzero
# and -- under `set -e` above -- a bare `x=$(...)` assignment would otherwise kill the
# WHOLE script right here with no error message, before ever reaching the fallback
# logic below that's specifically designed to handle exactly this case.
CERT_PATH=$(python3 -c "import certifi; print(certifi.where())" 2>/dev/null) || true
if [[ -z "$CERT_PATH" || ! -f "$CERT_PATH" ]]; then
    # Fall back to macOS system CA bundle locations
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

MLX_DIR="$HOME/pip_env/mlx_lm"
MLX_VENV="$MLX_DIR/venv"
MLX_PYTHON="$MLX_VENV/bin/python"
MLX_SERVER_MODULE="mlx_vlm.server"
MLX_PORT=11435
# Static fallback only for the non-interactive fixed-model path (model_advisor.py never
# runs there, so there's no computed decision to use) — see HARDWARE_SPEED_ESTIMATION.md
# §14. --interactive overwrites this with model_advisor.py's own per-model decision,
# which is empty/unset (no quantization) whenever the model's recommended context
# already fits free memory unquantized — only falling back to 8 when it doesn't. Kept
# as 8 here, not empty, so the older non-interactive invocation keeps its prior
# always-quantized behavior rather than silently changing on a code path that has no
# free-memory-driven decision behind it either way.
KV_BITS=8
KV_GROUP_SIZE=64
QUANTIZED_KV_START=0
PREFILL_STEP_SIZE=2048
CONTEXT_WINDOW=131072

# Automatic Prefix Caching (APC) — see HARDWARE_SPEED_ESTIMATION.md §13. Real mlx_vlm
# 0.7.0 feature (mlx_vlm/apc.py), off by default upstream, turned on here after live
# measurement: ~97.7% TTFT reduction on repeated/growing-prefix requests (exactly the
# pattern OpenCode/Claude Code produce every turn — same system prompt + growing
# history, one new message appended), and the disk tier specifically survives THIS
# script's own model-switch restarts of mlx_vlm.server, which would otherwise cold-lose
# the in-memory cache every time. APC_NUM_BLOCKS=4096 (double the mlx_vlm default of
# 2048) is a lazy-allocated ceiling, not an eager memory cost (confirmed in apc.py's own
# pool implementation) — see §13 for the measured numbers and what's NOT yet verified
# (only tested on one model, one synthetic prefix, one prefix length).
APC_ENABLED=1
APC_NUM_BLOCKS=4096
APC_DISK_PATH="$HOME/.cache/mlx-vlm/caching"
APC_DISK_MAX_GB=5

MONITOR_LOG="/tmp/monitor.log"
MONITOR_PID=""

LITELLM_DIR="$HOME/pip_env/litellm"
LITELLM_VENV="$LITELLM_DIR/venv"
# Dedicated to this script -- generated from models.json, never hand-edited, never the
# shared opencode-router-config.yaml (which has unrelated real Bedrock Claude entries
# and may be used by other tools). See ensure_litellm_config().
LITELLM_CONFIG="start-litellm-proxy-config.yaml"
LITELLM_SHARED_CONFIG_TO_BACK_UP="opencode-router-config.yaml"
LITELLM_PORT=4100
PG_DATA_DIR="$LITELLM_DIR/pgdata"
PG_HOST="127.0.0.1"
PG_PORT=5433
PG_DB="litellm"
PG_USER="llmproxy"
PG_BIN="$(brew --prefix postgresql@15)/bin"
DATABASE_URL="postgresql://${PG_USER}:@${PG_HOST}:${PG_PORT}/${PG_DB}"

# Logging functions
log_info() {
    echo "[INFO] $*"
}

log_error() {
    echo "[ERROR] $*" >&2
}

log_success() {
    echo "[✓] $*"
}

# Utility functions
validate_args() {
    if [[ "$INTERACTIVE" == "false" && -z "$MODE" ]]; then
        log_error "Usage: $0 [--interactive] [--monitor] <claude|opencode> [model_name] [model_id]"
        log_error "Example: $0 claude qwen3.5-9b-mlx-local mlx-community/Qwen3.5-9B-MLX-8bit"
        log_error "Example: $0 --interactive --monitor"
        exit 2
    fi

    if [[ "$INTERACTIVE" == "false" && -z "$LOCAL_MODEL" ]]; then
        log_error "Usage: $0 <claude|opencode> <model_name> [model_id]"
        log_error "Example: $0 claude qwen3.5-9b-mlx-local mlx-community/Qwen3.5-9B-MLX-8bit"
        log_error "(or pass --interactive to pick a tool and a model instead of naming them)"
        exit 2
    fi

    if [[ -n "$MODE" ]]; then
        case "$MODE" in
            claude|opencode)
                ;;
            *)
                log_error "Mode must be 'claude' or 'opencode', got: $MODE"
                exit 2
                ;;
        esac
    fi
}

# Tool detection — used by the interactive tool-selection screen to grey out
# whichever of the supported tools isn't installed on this system yet.
is_tool_available() {
    case "$1" in
        claude) is_command_available claude ;;
        opencode) is_command_available opencode ;;
        *)
            return 1
            ;;
    esac
}

# Interactive tool selection — shown after model selection when no mode was passed on
# the command line. Lists all supported tools always (never hides one), greying out
# whichever aren't detected so the user can see what this script supports and go
# install what's missing, same spirit as model_advisor.py's greyed-out models.
run_interactive_tool_selection() {
    local tool_ids=(claude opencode)
    local tool_labels=("Claude Code" "OpenCode")
    local selectable_ids=()

    log_info "Checking which supported tools are installed on this system..."
    {
        echo ""
        echo "===================================================================="
        echo "  Select a tool to launch"
        echo "===================================================================="
    } >&2

    local n=1 idx id label
    for idx in "${!tool_ids[@]}"; do
        id="${tool_ids[$idx]}"
        label="${tool_labels[$idx]}"
        if is_tool_available "$id"; then
            selectable_ids+=("$id")
            echo "  ${n}) ${label}" >&2
            n=$((n + 1))
        else
            printf "\033[2m  —) %s  (not found — install it to enable this option)\033[0m\n" "$label" >&2
        fi
    done
    echo "" >&2

    if [ ${#selectable_ids[@]} -eq 0 ]; then
        log_error "None of the supported tools (Claude Code, OpenCode) were found on this system."
        log_error "Install one of them and re-run this script."
        exit 1
    fi

    echo "q) Quit" >&2
    local choice
    while true; do
        printf "Select [1-%d] or 'q': " "${#selectable_ids[@]}" >&2
        read -r choice
        choice="$(echo "$choice" | tr -d '[:space:]')"
        if [[ "$choice" == "q" || "$choice" == "Q" ]]; then
            exit 1
        fi
        if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le "${#selectable_ids[@]}" ]; then
            MODE="${selectable_ids[$((choice - 1))]}"
            break
        fi
        log_error "Invalid choice, try again."
    done

    log_success "Selected tool: $MODE"
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
# all (e.g. a leftover/unrelated process squatting on it, or one started completely
# outside this script). Confirmed live: a stale MLX server surviving on 11435 across
# runs caused exactly this kind of port conflict even with the pattern-based cleanup in
# place. This checks (and if needed, kills) whatever is ACTUALLY bound to the port,
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

is_command_available() {
    command -v "$1" > /dev/null 2>&1
}

# Dependency checks and bootstrapping
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

create_venv_if_needed() {
    local venv_path="$1"
    local venv_name="$2"

    if [ -d "$venv_path" ]; then
        log_info "$venv_name venv already exists"
        return 0
    fi

    log_info "Creating $venv_name virtual environment at $venv_path..."
    mkdir -p "$(dirname "$venv_path")"
    python3 -m venv "$venv_path"
    log_success "$venv_name venv created"
}

install_packages_if_needed() {
    local venv_path="$1"
    local venv_name="$2"
    shift 2
    local packages=("$@")

    local python_bin="$venv_path/bin/python"

    if [ ! -x "$python_bin" ]; then
        log_error "$venv_name venv Python not found: $python_bin"
        exit 1
    fi

    for pkg in "${packages[@]}"; do
        log_info "Checking if $pkg is installed in $venv_name..."
        if "$python_bin" -c "import $(echo $pkg | cut -d= -f1)" 2>/dev/null; then
            log_info "$pkg already installed"
        else
            log_info "Installing $pkg..."
            "$venv_path/bin/pip" install --upgrade pip setuptools wheel > /dev/null
            "$venv_path/bin/pip" install "$pkg" > /dev/null
            log_success "$pkg installed"
        fi
    done
}

# Interactive model selection
run_interactive_selection() {
    log_info "Detecting hardware and free memory for model recommendations..."

    # advisor_output captures ONLY stdout (the final JSON selection line, by design —
    # see model_advisor.py's own module docstring). stderr is intentionally left
    # un-redirected here so the interactive menu and prompts render live on the
    # terminal instead of being silently captured/hidden.
    # --tool lets model_advisor.py factor in that tool's own context preload (system
    # prompt, built-in tools, this directory's CLAUDE.md/AGENTS.md, installed skills,
    # configured MCP servers) when deciding which models are actually usable, not just
    # which ones fit in memory. Requires MODE to already be known — see main()'s
    # ordering (tool is selected before the model for exactly this reason).
    local advisor_output
    if ! advisor_output=$("$MLX_PYTHON" "$SCRIPT_DIR/model_advisor.py" --tool "$MODE"); then
        log_error "Model selection cancelled or failed."
        exit 1
    fi

    MLX_MODEL=$("$MLX_PYTHON" -c "import json,sys; print(json.loads(sys.argv[1])['huggingface'])" "$advisor_output")
    LOCAL_MODEL=$("$MLX_PYTHON" -c "import json,sys; print(json.loads(sys.argv[1])['id'])" "$advisor_output")
    CONTEXT_WINDOW=$("$MLX_PYTHON" -c "import json,sys; print(json.loads(sys.argv[1])['recommended_context'])" "$advisor_output")
    # kv_bits is JSON null (Python None) whenever model_advisor.py's own dynamic
    # decision found the recommended context already fits free memory unquantized —
    # see HARDWARE_SPEED_ESTIMATION.md §14. Printing that directly would give the
    # bash string "None", which is not a valid --kv-bits value and would either error
    # or (worse) get treated as a truthy, non-empty flag value — must become an empty
    # string here so ensure_mlx_running's `[[ -n "$KV_BITS" ]]` check below correctly
    # omits the whole KV-quantization flag group instead of trying to pass "None" to it.
    KV_BITS=$("$MLX_PYTHON" -c "import json,sys; v=json.loads(sys.argv[1])['kv_bits']; print(v if v is not None else '')" "$advisor_output")

    local kv_bits_log="unquantized (full decode speed — recommended context already fits free memory)"
    if [[ -n "$KV_BITS" ]]; then
        kv_bits_log="quantized to ${KV_BITS}-bit (memory-constrained — needed to fit this context)"
    fi
    log_success "Selected $LOCAL_MODEL ($MLX_MODEL) — context set to ${CONTEXT_WINDOW} tokens (computed from your current free memory)"
    log_success "KV cache: $kv_bits_log"
}

# Pre-downloads the model via snapshot_download() BEFORE the server ever starts.
# Root cause of a real, reproducible bug found during model verification: mlx_vlm.server's
# own lazy on-demand download has no resume support when the process is killed
# mid-transfer — a slow/interrupted download left abandoned, never-completed .incomplete
# blob fragments on disk. snapshot_download() run standalone, never killed mid-transfer,
# has real resume support. See ../devx-agent-eval verify_max_kv_size.sh's own
# predownload_model() function for the first fix of this exact bug — this mirrors it.
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

bootstrap_mlx() {
    log_info "Bootstrapping MLX environment..."

    create_venv_if_needed "$MLX_VENV" "MLX"

    # Install MLX packages
    install_packages_if_needed "$MLX_VENV" "MLX" \
        "mlx-lm" \
        "mlx-vlm"

    if [ ! -x "$MLX_PYTHON" ]; then
        log_error "MLX Python executable not found: $MLX_PYTHON"
        exit 1
    fi

    log_success "MLX bootstrap complete"
}

bootstrap_litellm() {
    log_info "Bootstrapping LiteLLM environment..."

    create_venv_if_needed "$LITELLM_VENV" "LiteLLM"

    # Install LiteLLM with its proxy AND extra_proxy extras. Not handled by
    # install_packages_if_needed above (which only checks `import litellm`, and the bare
    # package imports fine even without either extras group).
    #
    # Two separate gaps confirmed live, one after the other:
    # 1. `litellm` alone installs and imports fine, but `litellm --config ...` fails at
    #    startup with "No module named 'websockets'" -- the proxy server's own deps
    #    (websockets, fastapi, uvicorn, etc.) live behind the 'proxy' extra.
    # 2. `litellm[proxy]` is NOT enough either: this script sets DATABASE_URL (for the
    #    Postgres-backed key/usage tracking), and LiteLLM's DB layer needs the `prisma`
    #    package -- but prisma lives behind a DIFFERENT, separate extra called
    #    'extra_proxy' (checked directly against litellm 1.103.0's own metadata:
    #    `prisma>=0.11.0,<1.0 ; extra == 'extra-proxy'`), not 'proxy'. Without it,
    #    startup fails with "ModuleNotFoundError: No module named 'prisma'" only once it
    #    actually tries to connect to the DB -- well after the "installed" check would
    #    otherwise have looked satisfied, which is why this checks for `prisma`
    #    specifically rather than just `litellm, websockets`.
    local litellm_python="$LITELLM_VENV/bin/python"
    if [ ! -x "$litellm_python" ]; then
        log_error "LiteLLM venv Python not found: $litellm_python"
        exit 1
    fi
    if "$litellm_python" -c "import litellm, websockets, prisma" 2>/dev/null; then
        log_info "litellm[proxy,extra_proxy] already installed"
    else
        log_info "Installing litellm[proxy,extra_proxy]..."
        "$LITELLM_VENV/bin/pip" install --upgrade pip setuptools wheel > /dev/null
        "$LITELLM_VENV/bin/pip" install "litellm[proxy,extra_proxy]" > /dev/null
        log_success "litellm[proxy,extra_proxy] installed"
    fi

    # A THIRD, separate gap on top of the two above: installing the `prisma` pip
    # package does NOT generate its Python client -- that's a one-time codegen step
    # (`prisma generate`) that writes generated code into site-packages/prisma/.
    # Confirmed live: `import prisma` succeeds even when this hasn't been run (the
    # RuntimeError only fires on actually touching the generated `Prisma` attribute),
    # so the check above can't catch it -- this checks `from prisma import Prisma`
    # specifically instead. `prisma generate` also spawns a `prisma-client-py` generator
    # binary AS A SUBPROCESS LOOKED UP BY NAME ON PATH (not via sys.executable's
    # directory) -- it lives in $LITELLM_VENV/bin, which isn't on PATH when this script
    # invokes litellm by absolute path, so PATH is extended just for this one command.
    # LiteLLM's own runtime never runs this step for you (unlike migrations, which its
    # toolchain does apply automatically on every startup) -- it just expects the
    # generated client to already exist and raises if it doesn't.
    if ! "$litellm_python" -c "from prisma import Prisma" 2>/dev/null; then
        log_info "Generating Prisma Python client (one-time codegen)..."
        local litellm_schema
        litellm_schema="$("$litellm_python" -c "import litellm_proxy_extras, os; print(os.path.join(os.path.dirname(litellm_proxy_extras.__file__), 'schema.prisma'))")"
        if ! PATH="$LITELLM_VENV/bin:$PATH" "$litellm_python" -m prisma generate --schema="$litellm_schema" > /tmp/litellm-prisma-generate.log 2>&1; then
            log_error "prisma generate failed. Check /tmp/litellm-prisma-generate.log"
            tail -20 /tmp/litellm-prisma-generate.log >&2
            exit 1
        fi
        log_success "Prisma Python client generated"
    fi

    # Check if PostgreSQL is installed via Homebrew
    if ! command -v brew > /dev/null 2>&1; then
        log_error "Homebrew is not installed. Please install Homebrew first: https://brew.sh"
        exit 1
    fi

    if ! brew list postgresql@15 > /dev/null 2>&1; then
        log_error "PostgreSQL 15 is not installed via Homebrew"
        log_error "Install it with: brew install postgresql@15"
        exit 1
    fi

    if [ ! -x "$PG_BIN/pg_isready" ] || [ ! -x "$PG_BIN/pg_ctl" ]; then
        log_error "PostgreSQL binaries not found under $PG_BIN"
        exit 1
    fi

    ensure_litellm_config

    log_success "LiteLLM bootstrap complete"
}

# Generates $LITELLM_DIR/$LITELLM_CONFIG (our own dedicated LiteLLM config) from
# models.json: one model_list entry per model, all pointing at the MLX server. This is
# what makes a freshly-selected model in the interactive picker actually reachable
# through LiteLLM -- without it, LiteLLM has no idea any of these model names exist and
# rejects them outright (confirmed live: a models.json id gave a hard 400 "Invalid
# model name", while an alias that WAS registered failed differently -- connection
# refused, because only the alias registration was being checked). Regenerated fresh
# every run so it can never drift out of sync with models.json; safe to regenerate
# because nothing hand-edits this file (the header says so, and only this function
# writes to it).
#
# This is intentionally a SEPARATE file from opencode-router-config.yaml, not a
# rewrite of it -- that file has unrelated real Bedrock Claude model entries and may be
# used by other tools/routing setups outside this script, so it's left alone. On the
# first run where it's found, it's copied aside as a timestamped backup purely as an
# extra safety net (this script never touches it either way).
ensure_litellm_config() {
    local shared_config="$LITELLM_DIR/$LITELLM_SHARED_CONFIG_TO_BACK_UP"
    local our_config="$LITELLM_DIR/$LITELLM_CONFIG"

    if [ -f "$shared_config" ]; then
        local backup="${shared_config}.bak-$(date +%Y%m%d%H%M%S)"
        if ! ls "${shared_config}.bak-"* > /dev/null 2>&1; then
            log_info "Backing up existing shared LiteLLM config to $backup (this script no longer reads or writes it)..."
            cp "$shared_config" "$backup"
            log_success "Backup saved: $backup"
        fi
    fi

    log_info "Generating $LITELLM_CONFIG from models.json..."
    python3 - "$SCRIPT_DIR/models.json" "$our_config" "$MLX_PORT" <<'PYEOF'
import json
import sys

models_path, out_path, mlx_port = sys.argv[1], sys.argv[2], sys.argv[3]
with open(models_path) as f:
    catalog = json.load(f)

lines = [
    "# AUTO-GENERATED by start_litellm_proxy.sh's ensure_litellm_config() from",
    "# models.json -- regenerated (and overwritten) on every run. Do not hand-edit;",
    "# edits here will not survive the next run. Dedicated to start_litellm_proxy.sh",
    "# only; see LITELLM_CONFIG's definition for why this is a separate file from",
    "# opencode-router-config.yaml.",
    "model_list:",
]
for m in catalog["models"]:
    lines += [
        f"  - model_name: {m['id']}",
        "    litellm_params:",
        f"      model: openai/{m['huggingface']}",
        f"      api_base: http://127.0.0.1:{mlx_port}/v1",
        "      api_key: local-token",
    ]
lines += [
    "",
    "litellm_settings:",
    "  drop_params: true",
    "",
    "general_settings:",
    "  master_key: sk-opencode-router-local",
]

with open(out_path, "w") as f:
    f.write("\n".join(lines) + "\n")
PYEOF
    log_success "LiteLLM config ready: $our_config"
}

# Service management
#
# Tracks which model/context/kv-bits combo is actually loaded in the running MLX
# server. A bare /health check can't tell you that -- it only says SOMETHING is
# listening, not which model. Since model selection now changes per run (interactive
# picker, possibly a different free-memory-driven context size even for the same
# model), reusing a stale server that's serving the WRONG model would silently point
# Claude Code/OpenCode at the wrong weights while claiming success.
MLX_STATE_FILE="/tmp/mlx-llm-proxy-loaded.txt"

# Returns 0 only if the process actually LISTENING on $MLX_PORT is our expected
# mlx_vlm.server process. /health alone is not sufficient: it only proves SOMETHING is
# answering on the port, and confirmed live, a foreign, unmanaged server can hold the
# port while answering /health. A `python -m mlx_lm.server` process (note: mlx_LM, not
# mlx_VLM -- a different module entirely, started outside this script by another tool)
# squatted on 11435 for 17h, answered /health with {"status":"ok"}, but hung forever on
# any real /v1/chat/completions request (0 bytes after 60s). Trusting /health alone made
# this script treat that hung stranger as "already healthy" and hand the client a dead
# model. This correlates the port's actual owner PID with the mlx_vlm.server process so
# a foreign/hung server is never mistaken for ours.
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

# Returns 0 only if the server actually completes a real (tiny) inference request. Even
# a genuine mlx_vlm.server can be alive, own the port, and answer /health while hung on
# inference -- the same failure mode seen live with the foreign mlx_lm.server squatter,
# and /health can't distinguish it either way. This sends a 1-token chat completion and
# only accepts the server if a well-formed completion comes back within the timeout.
# The model id sent is the one the state file records as loaded; when there's no state
# file the server is one this script just started (or is about to be rejected by the
# owner check anyway), so $MLX_MODEL is the correct id to probe with in that case.
mlx_inference_responds() {
    local probe_model="${1:-$MLX_MODEL}"
    local response
    response="$(curl -fsS --max-time 30 "http://127.0.0.1:$MLX_PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$probe_model\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":1}" \
        2>/dev/null)" || return 1

    [[ -n "$response" ]] || return 1
    # A real completion has at least one choice; an error body (e.g. an unknown-model
    # 500), a truncated body, or an empty response does not -- so this checks for
    # "choices" specifically rather than just HTTP 200, which curl -f would accept.
    grep -q '"choices"' <<<"$response"
}

ensure_mlx_running() {
    log_info "Checking MLX server on 127.0.0.1:$MLX_PORT..."

    local desired_state="$MLX_MODEL|$CONTEXT_WINDOW|$KV_BITS"
    local server_up=false

    # --max-time: a process that's alive and holding the port but stuck/unresponsive
    # (not dead, not healthy) would otherwise hang this curl indefinitely with no
    # timeout, which stalls the whole script before it ever reaches the cleanup logic
    # below -- confirmed as one real cause of the "port conflict" symptom.
    #
    # /health, the port's owner being our expected mlx_vlm.server process (see
    # mlx_port_owner_is_expected), AND a real inference probe (mlx_inference_responds)
    # are ALL required to call the server "up". If /health answers but a foreign process
    # owns the port, this deliberately treats it as NOT ours and reclaims the port; if
    # OUR process owns the port but its inference is hung, the probe below fails and the
    # server is restarted rather than handed to the client as healthy.
    if curl -fsS --max-time 5 "http://127.0.0.1:$MLX_PORT/health" > /dev/null 2>&1; then
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

        # A DIFFERENT model (or the same model at a different context/kv-bits) is
        # already loaded -- this stack only supports one model at a time (one MLX
        # server, one fixed port, and every LiteLLM alias for the *-mlx-local models
        # points at that same port). Killing it out from under whatever session is
        # using it would silently redirect that session to different weights with no
        # error -- so this only happens on an explicit yes, and only when there's a
        # human present to answer (INTERACTIVE); a non-interactive/scripted invocation
        # has no one to confirm with, so it keeps refusing outright.
        log_error "A different model is already running on 127.0.0.1:$MLX_PORT:"
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
                log_error "Not stopping the running model. Stop it manually first:"
                log_error "  $SCRIPT_DIR/stop_litellm_proxy.sh"
                exit 1
            fi
        else
            log_error "This script only supports one model at a time. Stop the running one first:"
            log_error "  $SCRIPT_DIR/stop_litellm_proxy.sh"
            exit 1
        fi
    fi

    if [[ "$server_up" != "true" ]]; then
        log_info "Starting MLX server with model $MLX_MODEL..."
        # Nothing answered /health above, so there's no live session to protect here --
        # this is just cleaning up a stuck/crashed process that's still holding the port
        # (not the "different model already serving" case above, which prompts/exits
        # instead).
        wait_for_process_gone "$MLX_SERVER_MODULE" "MLX-VLM" 30
        rm -f "$MLX_STATE_FILE"
    fi

    # Final safety net regardless of which path above ran: wait_for_process_gone only
    # kills processes matching the mlx_vlm.server command-line pattern, but what
    # actually determines whether the bind() below succeeds is the PORT, which can
    # still be held by something that pattern never matched (confirmed live as the
    # actual "port conflict" case). Cheap no-op when the port's already free.
    wait_for_port_free "$MLX_PORT" 15

    # --max-kv-size is the real, verified hard context cap (see
    # devx-agent-eval/LITELLM_ROUTING_PLAN.md and mlx_verification_results.json —
    # confirmed live across 14 models that this rejects an oversized request with a
    # clean HTTP 400 rather than silently ignoring it, unlike Ollama's MLX runner).
    # CONTEXT_WINDOW must be passed here, not just to the client — a client-side
    # context limit alone (CLAUDE_CODE_MAX_CONTEXT_TOKENS below) does not bound what
    # the MLX server itself will accept.
    # CAVEAT not yet closed: verify_max_kv_size.sh tested --max-kv-size ALONE (no
    # --kv-bits/--kv-group-size/--quantized-kv-start/--kv-quant-scheme). The
    # combination below, with KV quantization also enabled, has not been separately
    # re-verified — the admission-gate check operates on token counts rather than
    # byte sizes so quantization shouldn't interact with it, but that's reasoning by
    # analogy, not a live measurement, and this session has repeatedly found that
    # distinction matters.
    local mlx_args=(
        --model "$MLX_MODEL"
        --host 127.0.0.1
        --port "$MLX_PORT"
        --max-kv-size "$CONTEXT_WINDOW"
        --prefill-step-size "$PREFILL_STEP_SIZE"
    )
    # KV-cache quantization is now OPTIONAL, not unconditional — see
    # HARDWARE_SPEED_ESTIMATION.md §14. KV_BITS is empty whenever the recommended
    # context already fits free memory unquantized (the common case; full decode
    # speed, confirmed zero memory cost at short context in §10/§12's own
    # measurements), and only set to a real bit-width when model_advisor.py's dynamic
    # decision found it necessary to fit the context at all. The non-interactive
    # fixed-model path has no such decision behind it and keeps KV_BITS=8 from its
    # static default above, so this still quantizes unconditionally on that path,
    # same as before this change.
    if [[ -n "$KV_BITS" ]]; then
        mlx_args+=(
            --kv-bits "$KV_BITS"
            --kv-group-size "$KV_GROUP_SIZE"
            --quantized-kv-start "$QUANTIZED_KV_START"
            --kv-quant-scheme uniform
        )
    fi

    # Automatic Prefix Caching (APC) — see HARDWARE_SPEED_ESTIMATION.md §13. Real
    # mlx_vlm feature, configured via environment variables read by
    # mlx_vlm/server/runtime_config.py, not CLI flags — hence exported here rather
    # than added to mlx_args. mkdir -p: mlx_vlm's own disk-tier code expects the
    # directory to exist; it does not create it.
    mkdir -p "$APC_DISK_PATH"
    export APC_ENABLED APC_NUM_BLOCKS APC_DISK_PATH APC_DISK_MAX_GB

    "$MLX_PYTHON" -m "$MLX_SERVER_MODULE" "${mlx_args[@]}" \
        > /tmp/mlx-llm-proxy.log 2>&1 &

    if ! wait_for_url "http://127.0.0.1:$MLX_PORT/health" 600; then
        log_error "MLX server failed to start. Check /tmp/mlx-llm-proxy.log"
        tail -20 /tmp/mlx-llm-proxy.log >&2
        exit 1
    fi

    echo "$desired_state" > "$MLX_STATE_FILE"
    log_success "MLX server started and healthy"
}

ensure_postgres_running() {
    log_info "Checking PostgreSQL on $PG_HOST:$PG_PORT..."

    if "$PG_BIN/pg_isready" -h "$PG_HOST" -p "$PG_PORT" > /dev/null 2>&1; then
        log_success "PostgreSQL already running"
    else
        log_info "Starting PostgreSQL..."

        # Stop any existing instance
        if "$PG_BIN/pg_ctl" -D "$PG_DATA_DIR" status > /dev/null 2>&1; then
            log_info "Stopping existing PostgreSQL instance..."
            "$PG_BIN/pg_ctl" -D "$PG_DATA_DIR" -w stop > /tmp/pgserver-llm-proxy.log 2>&1 || true
            sleep 1
        fi

        # Initialize data directory if needed
        if [ ! -d "$PG_DATA_DIR" ]; then
            log_info "Initializing PostgreSQL data directory..."
            mkdir -p "$PG_DATA_DIR"
            # --username=postgres: without it, plain initdb names the superuser after
            # whoever runs it (confirmed live) -- fine on its own, but this repo's
            # existing production pgdata cluster already has its superuser named
            # "postgres" (set up before this script tracked that requirement), so a
            # fresh cluster needs the SAME fixed name for ensure_litellm_role_and_db's
            # admin connection to work identically on a new machine as it does here.
            "$PG_BIN/initdb" -D "$PG_DATA_DIR" --username=postgres > /tmp/pgserver-llm-proxy.log 2>&1
        fi

        # Start PostgreSQL
        "$PG_BIN/pg_ctl" -D "$PG_DATA_DIR" \
            -o "-h $PG_HOST -p $PG_PORT -k $PG_DATA_DIR" \
            -l "$PG_DATA_DIR/log" \
            -w start >> /tmp/pgserver-llm-proxy.log 2>&1

        if ! "$PG_BIN/pg_isready" -h "$PG_HOST" -p "$PG_PORT" > /dev/null 2>&1; then
            log_error "PostgreSQL failed to start. Check /tmp/pgserver-llm-proxy.log"
            tail -20 /tmp/pgserver-llm-proxy.log >&2
            exit 1
        fi

        log_success "PostgreSQL started and ready"
    fi

    ensure_litellm_role_and_db
}

# Creates the role/database LiteLLM's DATABASE_URL expects (PG_USER/PG_DB), if they
# don't already exist. Without --username=postgres above, plain initdb creates its
# default superuser named after whoever RUNS it (confirmed live) -- not "postgres",
# and not PG_USER either. So on a truly fresh machine, PG_USER/PG_DB never existed
# until this function creates them; without it, LiteLLM's Prisma connection fails
# outright. Idempotent: skips whichever of role/db already exists.
ensure_litellm_role_and_db() {
    local admin_user="postgres"

    if ! "$PG_BIN/psql" -h "$PG_HOST" -p "$PG_PORT" -U "$admin_user" -d postgres -tAc \
        "SELECT 1 FROM pg_roles WHERE rolname='$PG_USER'" 2>/dev/null | grep -q 1; then
        log_info "Creating PostgreSQL role '$PG_USER' (for LiteLLM's DATABASE_URL)..."
        if ! "$PG_BIN/psql" -h "$PG_HOST" -p "$PG_PORT" -U "$admin_user" -d postgres \
            -c "CREATE ROLE $PG_USER WITH LOGIN;" > /dev/null 2>/tmp/pgserver-llm-proxy-role.log; then
            log_error "Failed to create role '$PG_USER'. Check /tmp/pgserver-llm-proxy-role.log"
            exit 1
        fi
        log_success "Role '$PG_USER' created"
    fi

    if ! "$PG_BIN/psql" -h "$PG_HOST" -p "$PG_PORT" -U "$admin_user" -d postgres -tAc \
        "SELECT 1 FROM pg_database WHERE datname='$PG_DB'" 2>/dev/null | grep -q 1; then
        log_info "Creating PostgreSQL database '$PG_DB' (owner: $PG_USER)..."
        if ! "$PG_BIN/psql" -h "$PG_HOST" -p "$PG_PORT" -U "$admin_user" -d postgres \
            -c "CREATE DATABASE $PG_DB OWNER $PG_USER;" > /dev/null 2>/tmp/pgserver-llm-proxy-db.log; then
            log_error "Failed to create database '$PG_DB'. Check /tmp/pgserver-llm-proxy-db.log"
            exit 1
        fi
        log_success "Database '$PG_DB' created"
    fi
}

ensure_litellm_running() {
    log_info "Checking LiteLLM on 127.0.0.1:$LITELLM_PORT..."

    if curl -fsS --max-time 5 "http://127.0.0.1:$LITELLM_PORT/health/readiness" > /dev/null 2>&1; then
        log_success "LiteLLM already healthy"
        return 0
    fi

    log_info "Starting LiteLLM..."
    wait_for_process_gone "litellm.*$LITELLM_CONFIG" "LiteLLM" 20

    (
        cd "$LITELLM_DIR"
        export DATABASE_URL
        exec "$LITELLM_VENV/bin/litellm" \
            --config "./$LITELLM_CONFIG" \
            --host 127.0.0.1 \
            --port "$LITELLM_PORT"
    ) > /tmp/litellm-llm-proxy.log 2>&1 &

    # /health/readiness, not /health/liveliness: liveliness only confirms the HTTP
    # server process itself is up and accepting connections, which Uvicorn does as
    # soon as the ASGI app is constructed -- BEFORE FastAPI's startup event (Prisma
    # client setup, DB connection, model list registration) has actually finished.
    # Confirmed live: a client (OpenCode) could reach LiteLLM immediately after
    # liveliness passed but before startup truly completed, causing intermittent
    # failures downstream. Readiness explicitly checks DB connectivity and only
    # returns healthy once that's actually ready.
    #
    # 120s, not 30s: on a machine that's never run the LiteLLM proxy before, this first
    # startup also bootstraps its Prisma/Node toolchain (downloads a Node.js binary +
    # Prisma query-engine binary over the network -- see litellm_proxy_extras'
    # ensure_prisma_toolchain) before the DB connection is ready. That easily exceeds
    # 30s on a fresh machine even though it would have succeeded given more time.
    if ! wait_for_url "http://127.0.0.1:$LITELLM_PORT/health/readiness" 120; then
        log_error "LiteLLM failed to start. Check /tmp/litellm-llm-proxy.log"
        tail -20 /tmp/litellm-llm-proxy.log >&2
        exit 1
    fi

    log_success "LiteLLM started and healthy"
}

# Background memory monitor (--monitor flag; see MONITORING_ARCHITECTURE.md and
# monitor_memory.sh). --max-kv-size caps admitted token counts, but not baseline memory
# pressure from everything else running on the machine, and it's only been verified against
# short single-turn prompts so far (PROJECT_STATUS.md "What's NOT done yet" #2/#5) — this is
# a real-time backstop, not a replacement for that still-open verification work.
start_memory_monitor() {
    local mlx_pid
    mlx_pid=$(pgrep -f "$MLX_SERVER_MODULE" | head -1)
    if [[ -z "$mlx_pid" ]]; then
        log_error "Could not find a running $MLX_SERVER_MODULE process to monitor — skipping monitor."
        return 1
    fi

    # No per-machine GB math needed here — monitor_memory.sh's thresholds are percent-free
    # (kern.memorystatus_level), which is already machine-size-normalized. An earlier version
    # computed a flat-GB headroom per machine here and it didn't hold up across machine sizes
    # (see monitor_memory.sh's own header comment for the real data behind this change).
    log_info "Starting background memory monitor for MLX server (PID $mlx_pid)..."
    "$SCRIPT_DIR/monitor_memory.sh" \
        --pid "$mlx_pid" \
        --interval 5 \
        --log-file "$MONITOR_LOG" \
        --litellm-pattern "litellm.*$LITELLM_CONFIG" \
        > /dev/null 2>&1 &
    MONITOR_PID=$!
    log_success "Monitor running (PID $MONITOR_PID) → $MONITOR_LOG"
}

stop_memory_monitor() {
    if [[ -n "$MONITOR_PID" ]] && kill -0 "$MONITOR_PID" 2>/dev/null; then
        kill "$MONITOR_PID" 2>/dev/null || true
    fi
}

# Proxy validation
validate_proxy_stack() {
    log_info "Validating proxy stack..."

    # Check all three components are healthy
    log_info "Checking MLX health..."
    if ! curl -fsS --max-time 5 "http://127.0.0.1:$MLX_PORT/health" > /dev/null 2>&1; then
        log_error "MLX server is not responding"
        exit 1
    fi
    log_success "MLX healthy"

    log_info "Checking PostgreSQL..."
    if ! "$PG_BIN/pg_isready" -h "$PG_HOST" -p "$PG_PORT" > /dev/null 2>&1; then
        log_error "PostgreSQL is not responding"
        exit 1
    fi
    log_success "PostgreSQL healthy"

    log_info "Checking LiteLLM..."
    if ! curl -fsS --max-time 5 "http://127.0.0.1:$LITELLM_PORT/health/readiness" > /dev/null 2>&1; then
        log_error "LiteLLM is not responding"
        exit 1
    fi
    log_success "LiteLLM healthy"

    log_success "Proxy stack validated and ready"
}

# OpenCode's own config (~/.config/opencode/opencode.jsonc) requires every model it can
# launch to be explicitly listed under its "litellm" provider's "models" object --
# there's no live-API fallback for an unrecognized model string (confirmed against
# opencode's own source: Provider.getModel does a plain object lookup and throws
# ModelNotFoundError otherwise). Registers ONLY the one model actually being launched,
# and only if it isn't already there -- deliberately not the whole models.json catalog,
# since most of those will never be used from opencode and the user doesn't want their
# hand-maintained config bloated with entries they don't need.
ensure_opencode_model_registered() {
    local opencode_config="$HOME/.config/opencode/opencode.jsonc"
    if [ ! -f "$opencode_config" ]; then
        log_info "No opencode.jsonc found at $opencode_config -- skipping model auto-registration."
        return 0
    fi

    log_info "Checking whether $LOCAL_MODEL is registered in opencode's litellm provider..."
    local result
    result=$(python3 - "$opencode_config" "$LOCAL_MODEL" "$CONTEXT_WINDOW" <<'PYEOF'
import json
import re
import sys

config_path, model_id, context_window = sys.argv[1], sys.argv[2], int(sys.argv[3])

with open(config_path) as f:
    text = f.read()

# Strip // and /* */ comments and trailing commas ONLY to check (read-only, in-memory)
# whether model_id is already registered -- this parsed copy is never written back, so
# it's fine that it would destroy comments/formatting. Must be string-literal-aware:
# confirmed live that a naive `//` regex mistakes the "https://..." in this file's own
# $schema value for a comment start and corrupts the parse.
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
    parsed = json.loads(strip_jsonc(text))
except Exception as e:
    print(f"PARSE_ERROR: {e}")
    sys.exit(2)

if model_id in parsed.get("provider", {}).get("litellm", {}).get("models", {}):
    print("ALREADY_REGISTERED")
    sys.exit(0)

# Not registered -- insert it as the first entry of the litellm provider's "models"
# object via brace-depth-aware scanning (not a naive regex, since model entries can
# have their own nested "limit": {...} objects), so every existing comment and every
# other provider's config is left completely untouched.
idx = text.find('"litellm"')
if idx == -1:
    print("NO_LITELLM_PROVIDER")
    sys.exit(3)

obj_start = text.find('{', idx)
depth = 1
i = obj_start + 1
while depth > 0 and i < len(text):
    if text[i] == '{':
        depth += 1
    elif text[i] == '}':
        depth -= 1
    i += 1
litellm_block = text[obj_start:i]

models_idx = litellm_block.find('"models"')
if models_idx == -1:
    print("NO_MODELS_KEY")
    sys.exit(4)
models_brace = litellm_block.find('{', models_idx)
insert_at = obj_start + models_brace + 1  # absolute offset, right after "models": {

new_entry = (
    f'\n        "{model_id}": {{ "name": "{model_id} (auto-registered by start_litellm_proxy.sh)", '
    f'"limit": {{ "context": {context_window}, "output": 4096 }} }},'
)
with open(config_path, "w") as f:
    f.write(text[:insert_at] + new_entry + text[insert_at:])
print("REGISTERED")
PYEOF
) || true
    case "$result" in
        ALREADY_REGISTERED)
            log_success "$LOCAL_MODEL already registered in opencode.jsonc"
            ;;
        REGISTERED)
            log_success "Registered $LOCAL_MODEL in opencode.jsonc's litellm provider"
            ;;
        *)
            log_error "Could not auto-register $LOCAL_MODEL in opencode.jsonc ($result)"
            log_error "Add it manually under provider.litellm.models, or opencode will fail with 'Model not found'."
            ;;
    esac
}

# Client launch
launch_client() {
    log_info "Configuring environment for $MODE..."

    case "$MODE" in
        claude)
            export ANTHROPIC_BASE_URL="http://127.0.0.1:$LITELLM_PORT"
            export ANTHROPIC_API_KEY="local-token"
            export ANTHROPIC_AUTH_TOKEN="ollama"
            export ANTHROPIC_MODEL="$LOCAL_MODEL"
            export ANTHROPIC_SMALL_FAST_MODEL="$LOCAL_MODEL"
            export CLAUDE_CODE_MAX_CONTEXT_TOKENS="$CONTEXT_WINDOW"

            log_success "Launching Claude Code with $LOCAL_MODEL (context: ${CONTEXT_WINDOW} tokens)"
            exec claude --model "$LOCAL_MODEL"
            ;;
        opencode)
            export OPENAI_BASE_URL="http://127.0.0.1:$LITELLM_PORT/v1"
            export OPENAI_API_KEY="sk-opencode-router-local"

            ensure_opencode_model_registered

            log_success "Launching OpenCode with litellm/$LOCAL_MODEL (context: ${CONTEXT_WINDOW} tokens)"
            exec opencode --model "litellm/$LOCAL_MODEL"
            ;;
    esac
}

# Main flow
main() {
    validate_args
    check_system_dependencies

    # Bootstrap environments (needed before interactive selection: model_advisor.py and
    # ensure_model_downloaded both run inside the MLX venv, via $MLX_PYTHON)
    bootstrap_mlx
    bootstrap_litellm

    # Tool is selected BEFORE the model, not after: model_advisor.py needs to know
    # which tool will launch so it can factor that tool's own context preload into
    # which models are actually usable (see run_interactive_selection's --tool flag).
    if [[ -z "$MODE" ]]; then
        run_interactive_tool_selection
    fi

    if [[ "$INTERACTIVE" == "true" ]]; then
        run_interactive_selection
    fi

    log_info "Starting LiteLLM proxy setup for $MODE with model $LOCAL_MODEL"
    log_info "MLX model: $MLX_MODEL"

    ensure_model_downloaded

    # Ensure services are running
    ensure_postgres_running
    ensure_mlx_running
    ensure_litellm_running

    if [[ "$MONITOR" == "true" ]]; then
        trap stop_memory_monitor EXIT INT TERM
        start_memory_monitor
    fi

    # Validate everything works together
    validate_proxy_stack

    # Launch client
    launch_client
}

main "$@"
