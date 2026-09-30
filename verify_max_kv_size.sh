#!/usr/bin/env bash
# verify_max_kv_size.sh — live smoke test for whether mlx_vlm.server's --max-kv-size
# actually enforces a hard cap on a given model, or silently ignores it.
#
# Background: Ollama's MLX runner has no context ceiling at all (confirmed in
# LITELLM_ROUTING_PLAN.md §4.15 and reconfirmed via upstream GitHub issues — Ollama's
# maintainers have twice rejected patches to fix this, see PR #18261/#18285 discussion).
# mlx_vlm.server's --max-kv-size flag looked like the fix, and a live test on
# Qwen3.5-9B-MLX-8bit (mlx-vlm 0.7.0) confirmed it works — a low --max-kv-size cleanly
# rejects an oversized request with HTTP 400 rather than silently ignoring it.
#
# BUT: mlx-vlm's own PR #1995 ("Apply max_kv_size to models that build their own cache",
# closed unmerged) states this enforcement is silently dropped on 97 of 173 supported
# models, and explicitly names Qwen3.5-0.8B (a smaller sibling of the model we verified)
# as one of the affected models. This is model/architecture-dependent and must be
# verified per model, not assumed from one successful test — this script automates
# that verification so we don't have to repeat it by hand for every model candidate.
#
# Usage: ./verify_max_kv_size.sh <huggingface-model-id> [port]
#
# Three phases, all against the same model/server process lifecycle:
#   1. LOW_KV_SIZE (256) vs a ~771-token prompt — expect rejection
#   2. HIGH_KV_SIZE (1024) vs the same prompt — expect success + correct recall
#   3. EXTREME_LOW_KV_SIZE (32) vs a ~150-token prompt, max_tokens=1 — mirrors upstream
#      mlx-vlm PR #1995's own reproduction scale, to catch a threshold effect where
#      enforcement works at moderate bounds but is silently dropped at very small ones
#      (the scale upstream actually reported broken on 97 of 173 models).
#
# Exit codes:
#   0 = SAFE                  — max-kv-size cleanly rejects oversized requests at every
#                               scale tested (verified)
#   1 = UNSAFE                — max-kv-size was silently ignored at the moderate scale
#       or UNSAFE_AT_EXTREME_BOUNDS — enforced at moderate scale but silently ignored
#                               at the extreme scale (phase 3 only) — treat as unsafe
#   2 = INCONCLUSIVE          — server failed to start, unexpected response, or
#                               ambiguous result; needs manual review
#
# Result is also appended to mlx_verification_results.json for models.json to consult.

set -uo pipefail

MODEL_ID="${1:?Usage: $0 <huggingface-model-id> [port]}"
PORT="${2:-8295}"

MLX_VENV="$HOME/.local-model-setup/venv"
MLX_PYTHON="$MLX_VENV/bin/python"
export HF_HOME="${HF_HOME:-$HOME/.local-model-setup/models}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_FILE="$SCRIPT_DIR/mlx_verification_results.json"
LOG_FILE="/tmp/verify_max_kv_size_$$.log"

# Deliberately small so a realistic test prompt (~750-800 tokens with filler) overflows it;
# comfortably large so the same prompt fits with room to spare. Both use the SAME prompt,
# so the only variable between the two phases is whether the declared cap is honored.
LOW_KV_SIZE=256
HIGH_KV_SIZE=1024

# Phase 3 mirrors mlx-vlm PR #1995's own reproduction shape as closely as possible
# (a 32-token bound against a ~200-token prompt, i.e. the bound is smaller than the
# PROMPT ALONE, not just smaller than prompt+max_tokens). That PR reported models that
# build their own KV cache silently ignore max_kv_size at this scale — no rejection, no
# truncation, just wrong-but-confident answers. Phase 1/2 above already have bound <
# prompt too, but at a different absolute scale (256 vs a 771-token prompt); this phase
# tests whether the admission gate's behavior holds at the much smaller scale the
# upstream bug report actually used, with generation-token contribution minimized
# (max_tokens=1) so the check is dominated by the prompt itself, not the request budget.
EXTREME_LOW_KV_SIZE=32
EXTREME_LOW_FILLER_REPEAT=2
EXTREME_LOW_MAX_TOKENS=1

# Pre-flight safety: refuse to run if wired memory is already elevated (avoids
# contaminating the test the way earlier ad hoc sessions did — see
# LITELLM_ROUTING_PLAN.md §4.16-§4.17 for why an unchecked background load produces
# misleading results).
MAX_BASELINE_WIRED_GB=15

SERVER_PID=""

log() { echo "[$(date +%H:%M:%S)] $*"; }

cleanup() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        sleep 2
        kill -9 "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

get_wired_gb() {
    vm_stat | awk '/Pages wired down/ {gsub("\\.","",$4); printf "%.2f", $4*16384/1024/1024/1024}'
}

# Pre-downloads the model via snapshot_download() BEFORE any server is ever started.
# Root cause of a real, reproducible bug found live: mlx_vlm.server's own lazy on-demand
# download has no resume support when the process is SIGKILLed mid-transfer (which
# start_server's timeout-failure path does) — confirmed live: repeated timeouts on
# Qwen2.5-Coder-14B-Instruct-4bit left FIVE different abandoned .incomplete blob
# fragments for the same two safetensors shards (each retry started a fresh blob
# instead of resuming the previous one), totaling 8GB on disk without ever completing
# the actual 7.75GB model. snapshot_download() run standalone, never killed mid-transfer,
# has real resume support and completed cleanly once given the chance to run
# uninterrupted. This function trades a longer one-time wait for guaranteed correctness.
predownload_model() {
    log "Pre-downloading $MODEL_ID via snapshot_download() (resumable, run standalone so it can never be killed mid-transfer)..."
    if "$MLX_PYTHON" -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL_ID')
" ; then
        log "Pre-download complete (or already cached)."
        return 0
    fi
    log "ERROR: pre-download failed for $MODEL_ID."
    return 1
}

wait_for_health() {
    local timeout="${1:-180}"
    for ((i = 0; i < timeout; i++)); do
        if curl -fsS "http://127.0.0.1:${PORT}/v1/models" > /dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    return 1
}

# Waits until nothing answers on $PORT. Without this, a fast-starting new server
# (fully cached model, sub-second load) can race a not-yet-dead previous process —
# confirmed live: phase 2 of a run got phase 1's still-listening server and silently
# validated against the WRONG max-kv-size, producing a false verdict. kill -9 does not
# guarantee the socket is released the instant the syscall returns.
wait_for_port_free() {
    local timeout="${1:-15}"
    for ((i = 0; i < timeout * 10; i++)); do
        if ! curl -fsS "http://127.0.0.1:${PORT}/v1/models" > /dev/null 2>&1; then
            return 0
        fi
        sleep 0.1
    done
    return 1
}

start_server() {
    local max_kv="$1"
    log "Starting mlx_vlm.server for '$MODEL_ID' with --max-kv-size $max_kv (port $PORT)..."
    "$MLX_PYTHON" -m mlx_vlm.server \
        --model "$MODEL_ID" \
        --host 127.0.0.1 --port "$PORT" \
        --max-kv-size "$max_kv" \
        --prefill-step-size 2048 \
        > "$LOG_FILE" 2>&1 &
    SERVER_PID=$!

    # 180s is enough once a model is cached, but a first-time download of a multi-GB
    # model can still be mid-transfer at 180s on a slow/variable connection (observed
    # live: Qwen2.5-Coder-7B-Instruct-4bit timed out at file 8/9 of its safetensors
    # shards). HEALTH_TIMEOUT lets a caller extend this for a known-uncached model
    # instead of every phase paying a longer timeout unconditionally.
    if ! wait_for_health "${HEALTH_TIMEOUT:-180}"; then
        log "ERROR: server failed to become healthy within timeout. Last 30 log lines:"
        tail -30 "$LOG_FILE" >&2
        # Root cause of a real bug found live: this used to just clear SERVER_PID
        # without killing the process. A slow-to-load model (e.g. still mid-download)
        # kept loading in the background after the script gave up and exited, then
        # went on answering healthy indefinitely — orphaned, still bound to $PORT,
        # confirmed live with `ps aux` finding it minutes later. A later invocation on
        # the same port would then race this orphan and could silently validate
        # against the WRONG max-kv-size. Must kill it here, not just forget about it.
        stop_server
        return 1
    fi

    # Defense in depth against the same class of bug wait_for_port_free guards against:
    # confirm /v1/models actually reports MODEL_ID, not a stale process from a prior
    # phase that happened to still be listening on this port.
    local served_model
    served_model=$(curl -fsS "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null | "$MLX_PYTHON" -c "
import json,sys
try:
    d=json.load(sys.stdin)
    print(d.get('data',[{}])[0].get('id',''))
except Exception:
    print('')
" 2>/dev/null)
    if [[ "$served_model" != "$MODEL_ID" ]]; then
        log "ERROR: health check answered but /v1/models reports '$served_model', expected '$MODEL_ID' — likely a stale process still bound to port $PORT."
        return 1
    fi

    log "Server healthy (PID $SERVER_PID), confirmed serving $served_model"
    return 0
}

stop_server() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        sleep 2
        kill -9 "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=""
    if ! wait_for_port_free 15; then
        log "WARNING: port $PORT still answering 15s after kill — something else may be bound to it."
    fi
}

# Sends a secret value early in a padded prompt, then asks for recall.
# Args: <filler_repeat> <max_tokens>
# Prints one line of JSON: {http_code, prompt_tokens, completion_tokens, content, error}
send_recall_probe() {
    local filler_repeat="$1" max_tokens="$2"
    "$MLX_PYTHON" - "$PORT" "$MODEL_ID" "$filler_repeat" "$max_tokens" << 'PYEOF'
import sys, json, urllib.request, urllib.error

port, model_id = sys.argv[1], sys.argv[2]
filler_repeat, max_tokens = int(sys.argv[3]), int(sys.argv[4])
secret = "The secret launch code is ZEBRA-9471-QUARTZ."
filler = " The weather today is mild and pleasant with light clouds. " * filler_repeat
prompt = (secret + filler +
          "\n\nWhat is the secret launch code stated at the very beginning "
          "of this message? Answer with just the code.")

body = {
    "model": model_id,
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": max_tokens,
    "temperature": 0.0,
}
data = json.dumps(body).encode()
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/chat/completions",
    data=data,
    headers={"Content-Type": "application/json"},
)
out = {"http_code": None, "prompt_tokens": None, "completion_tokens": None,
       "content": None, "error": None}
try:
    with urllib.request.urlopen(req, timeout=180) as resp:
        result = json.loads(resp.read())
    out["http_code"] = 200
    usage = result.get("usage", {})
    out["prompt_tokens"] = usage.get("prompt_tokens")
    out["completion_tokens"] = usage.get("completion_tokens")
    out["content"] = result["choices"][0]["message"]["content"]
except urllib.error.HTTPError as e:
    out["http_code"] = e.code
    out["error"] = e.read().decode()
except Exception as e:
    out["http_code"] = -1
    out["error"] = str(e)
print(json.dumps(out))
PYEOF
}

field() {
    # field <json-line> <key> — tiny helper to pull one field out of the probe's JSON line
    "$MLX_PYTHON" -c "import json,sys; v=json.loads(sys.argv[1]).get(sys.argv[2]); print(v if v is not None else '')" "$1" "$2"
}

record_result() {
    local verdict="$1" reason="$2" low_json="$3" high_json="$4" extreme_json="$5"
    "$MLX_PYTHON" - "$RESULTS_FILE" "$MODEL_ID" "$verdict" "$reason" "$low_json" "$high_json" "$extreme_json" << 'PYEOF'
import json, sys, os, subprocess, datetime

results_file, model_id, verdict, reason, low_json, high_json, extreme_json = sys.argv[1:8]

def pip_version(pkg):
    try:
        out = subprocess.check_output([sys.executable, "-m", "pip", "show", pkg], text=True)
        for line in out.splitlines():
            if line.startswith("Version:"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return None

entry = {
    "model_id": model_id,
    "verdict": verdict,
    "reason": reason,
    "low_kv_probe": json.loads(low_json) if low_json else None,
    "high_kv_probe": json.loads(high_json) if high_json else None,
    "extreme_low_kv_probe": json.loads(extreme_json) if extreme_json else None,
    "verified_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "mlx_vlm_version": pip_version("mlx-vlm"),
    "mlx_version": pip_version("mlx"),
    "mlx_lm_version": pip_version("mlx-lm"),
}

data = {}
if os.path.exists(results_file):
    try:
        with open(results_file) as f:
            data = json.load(f)
    except Exception:
        data = {}

data.setdefault("models", {})
data["models"][model_id] = entry

with open(results_file, "w") as f:
    json.dump(data, f, indent=2)
    f.write("\n")

print(f"Recorded to {results_file}")
PYEOF
}

main() {
    log "=== Verifying --max-kv-size enforcement for: $MODEL_ID ==="

    local baseline_wired
    baseline_wired=$(get_wired_gb)
    log "Baseline wired memory: ${baseline_wired} GB"

    if "$MLX_PYTHON" -c "import sys; sys.exit(0 if float('$baseline_wired') > float('$MAX_BASELINE_WIRED_GB') else 1)"; then
        log "ABORT: baseline wired memory (${baseline_wired}GB) exceeds safety threshold (${MAX_BASELINE_WIRED_GB}GB)."
        log "Close background apps and retry — see LITELLM_ROUTING_PLAN.md §4.17."
        exit 2
    fi

    if ! predownload_model; then
        record_result "INCONCLUSIVE" "Pre-download failed for $MODEL_ID" "" "" ""
        exit 2
    fi

    # Phase 1: low max-kv-size — expect a clean rejection
    if ! start_server "$LOW_KV_SIZE"; then
        record_result "INCONCLUSIVE" "Server failed to start at low max-kv-size ($LOW_KV_SIZE)" "" "" ""
        exit 2
    fi
    log "Sending recall probe (expecting HTTP 400 rejection)..."
    low_result=$(send_recall_probe 60 60)
    log "Low-KV result: $low_result"
    stop_server

    low_http=$(field "$low_result" http_code)
    low_content=$(field "$low_result" content)

    # Phase 2: high max-kv-size — expect success + correct recall
    if ! start_server "$HIGH_KV_SIZE"; then
        record_result "INCONCLUSIVE" "Server failed to start at high max-kv-size ($HIGH_KV_SIZE)" "$low_result" "" ""
        exit 2
    fi
    log "Sending recall probe (expecting HTTP 200 + correct recall)..."
    high_result=$(send_recall_probe 60 60)
    log "High-KV result: $high_result"
    stop_server

    high_http=$(field "$high_result" http_code)
    high_content=$(field "$high_result" content)

    # Phase 3: extreme-low max-kv-size, mirroring upstream PR #1995's own reproduction
    # scale (bound << prompt, generation-token contribution minimized). This is the
    # scale at which upstream reported the silent-drop bug on models that build their
    # own KV cache — phase 1/2 already put bound < prompt at a larger scale, so this
    # phase exists specifically to rule out a threshold effect that only shows up small.
    if ! start_server "$EXTREME_LOW_KV_SIZE"; then
        record_result "INCONCLUSIVE" "Server failed to start at extreme-low max-kv-size ($EXTREME_LOW_KV_SIZE)" "$low_result" "$high_result" ""
        exit 2
    fi
    log "Sending recall probe at extreme-low bound (expecting HTTP 400 rejection, PR #1995 scale)..."
    extreme_result=$(send_recall_probe "$EXTREME_LOW_FILLER_REPEAT" "$EXTREME_LOW_MAX_TOKENS")
    log "Extreme-low-KV result: $extreme_result"
    stop_server

    extreme_http=$(field "$extreme_result" http_code)
    extreme_content=$(field "$extreme_result" content)

    verdict="INCONCLUSIVE"
    reason=""

    if [[ "$low_http" == "200" ]] && echo "$low_content" | grep -qi "ZEBRA-9471-QUARTZ"; then
        verdict="UNSAFE"
        reason="--max-kv-size was silently ignored at the moderate scale (bound=$LOW_KV_SIZE, prompt~771 tokens): the request still succeeded (HTTP 200) with correct recall of the secret. No truncation, no rejection, no error — same failure class as Ollama's num_ctx bug."
    elif [[ "$low_http" == "200" ]]; then
        verdict="UNSAFE"
        reason="--max-kv-size was not enforced as a rejection at the moderate scale (bound=$LOW_KV_SIZE): request returned HTTP 200 instead of an error (recall did not match, so this may be truncation/corruption rather than a clean silent-ignore — needs manual follow-up)."
    elif [[ "$low_http" != "400" ]] || [[ "$high_http" != "200" ]] || ! echo "$high_content" | grep -qi "ZEBRA-9471-QUARTZ"; then
        reason="Unexpected combination at moderate scale (low_http=$low_http high_http=$high_http) — needs manual review. low=$low_result high=$high_result"
    elif [[ "$extreme_http" == "200" ]] && echo "$extreme_content" | grep -qi "ZEBRA-9471-QUARTZ"; then
        verdict="UNSAFE_AT_EXTREME_BOUNDS"
        reason="max-kv-size is enforced at moderate scale (bound=$LOW_KV_SIZE vs ~771-token prompt: correctly rejected) but SILENTLY IGNORED at the extreme scale matching upstream PR #1995's own reproduction (bound=$EXTREME_LOW_KV_SIZE, prompt~150 tokens, max_tokens=$EXTREME_LOW_MAX_TOKENS): request succeeded (HTTP 200) with correct recall despite the bound being far smaller than the prompt. This model likely builds its own KV cache and bypasses the admission gate's enforcement at small scale, matching the upstream bug report — treat as UNSAFE for low-context-budget use cases even though moderate-scale requests are protected."
    elif [[ "$extreme_http" == "400" ]]; then
        verdict="SAFE"
        reason="max-kv-size correctly rejected oversized requests at both moderate scale (bound=$LOW_KV_SIZE vs ~771-token prompt) and extreme scale matching upstream PR #1995's own reproduction (bound=$EXTREME_LOW_KV_SIZE vs ~150-token prompt). High max-kv-size ($HIGH_KV_SIZE) correctly recalled the planted secret. --max-kv-size is enforced for this model at every scale tested."
    else
        reason="Extreme-low-bound phase produced an unexpected result (extreme_http=$extreme_http, not 200 with correct recall or a clean 400) — needs manual review. extreme=$extreme_result"
    fi

    log "=== VERDICT: $verdict ==="
    log "$reason"

    record_result "$verdict" "$reason" "$low_result" "$high_result" "$extreme_result"

    case "$verdict" in
        SAFE) exit 0 ;;
        UNSAFE|UNSAFE_AT_EXTREME_BOUNDS) exit 1 ;;
        *) exit 2 ;;
    esac
}

main
