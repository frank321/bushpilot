#!/usr/bin/env bash
# STATUS: historical research artifact (PHASE_PLAN.md Phase 4). Its findings are
# archived in mlx_long_context_stress_results.json and folded into the real supervisor
# (see start_mlx_server.sh's check_memory_backstop). It shells out to monitor_memory.sh
# below, which Phase 4 removed as a standalone script (its logic now lives directly in
# start_mlx_server.sh's own poll loop) -- this script is kept for its methodology and
# is not expected to run as-is until that call is ported to the new backstop. Its
# MLX_VENV/HF_HOME values also predate the Phase 1 migration (completed 2026-09-30):
# the venv and model cache now live under ~/.bushpilot/ (venv/, models/) --
# update those two lines as well if porting this back to a runnable state.
#
# verify_long_context_memory.sh — real long-context memory stress test for a single
# mlx_vlm.server-backed model, addressing PROJECT_STATUS.md's open item #1: every "SAFE"
# verdict in models.json / mlx_verification_results.json so far only confirms the
# --max-kv-size ADMISSION GATE works against short (~150-800 token) single-turn prompts
# (verify_max_kv_size.sh). None of that proves the server stays memory-safe once it's
# actually asked to fill most of that window with a real prompt, the way a real coding
# session does.
#
# Methodology (adapted from devx-agent-eval/LITELLM_ROUTING_PLAN.md §4.14/§4.15's
# near-cap-fill stress methodology, itself built for a different backend — see the
# per-step comments below for what carries over and what's new here):
#   - One server instance, started with the SAME flags start_litellm_proxy.sh actually
#     uses in production (--max-kv-size, --kv-bits, --kv-group-size, --quantized-kv-start,
#     --kv-quant-scheme uniform) — not --max-kv-size alone, which is what
#     verify_max_kv_size.sh tested. This run also therefore contributes real evidence
#     toward PROJECT_STATUS.md open item #2 (kv-bits combination not separately verified),
#     though it isn't a substitute for a dedicated test of that combination alone.
#   - A background monitor_memory.sh watches the server's real PID the entire time and
#     will SIGTERM/SIGKILL it if wired memory crosses an abort threshold computed from
#     THIS machine's actual total RAM — the actual safety net for this test, not a
#     formality.
#   - Escalating steps, each a uniquely-nonced prompt built to an ABSOLUTE target token
#     count (not a percentage guessed from word count) — reports the server's own
#     `usage.prompt_tokens` as the real figure, same as §4.15's own correction of its
#     word-count-based estimate.
#   - Wired memory is sampled immediately before and after each step so a memory leak
#     across requests (not just within one) would show up as a rising floor between steps.
#   - Any step that fails, times out, or trips the monitor's abort stops the escalation —
#     later steps are not attempted once memory safety is already in question.
#
# Usage: ./verify_long_context_memory.sh <huggingface-model-id> [max_kv_size] [port]
#   max_kv_size defaults to 131072 — start_litellm_proxy.sh's own CONTEXT_WINDOW default,
#   chosen deliberately over this model's architectural max (262144) so this test mirrors
#   what a real launch would actually configure, not the most extreme case possible.
#
# Exit codes: 0 = completed all steps without a memory-safety event (see JSON results file
# for whether every step individually succeeded); 1 = a step failed AFTER at least one
# succeeded; 2 = aborted early (baseline memory too high, server wouldn't start, or the
# very first step failed) — needs manual review before retrying.

set -uo pipefail

MODEL_ID="${1:?Usage: $0 <huggingface-model-id> [max_kv_size] [port]}"
MAX_KV_SIZE="${2:-131072}"
PORT="${3:-8296}"

MLX_VENV="$HOME/pip_env/mlx_lm/venv"
MLX_PYTHON="$MLX_VENV/bin/python"
export HF_HOME="${HF_HOME:-$HOME/MLX_Models}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_FILE="$SCRIPT_DIR/mlx_long_context_stress_results.json"
SERVER_LOG="/tmp/verify_long_context_$$.log"
MONITOR_LOG="/tmp/verify_long_context_monitor_$$.log"

# Same production defaults as start_litellm_proxy.sh (KV_BITS/KV_GROUP_SIZE/etc.) — kept
# in sync deliberately, since testing anything other than the real production flags would
# defeat the point of this script.
KV_BITS=8
KV_GROUP_SIZE=64
QUANTIZED_KV_START=0
PREFILL_STEP_SIZE=2048

MAX_BASELINE_WIRED_GB=15

SERVER_PID=""
MONITOR_PID=""

log() { echo "[$(date +%H:%M:%S)] $*"; }

cleanup() {
    if [[ -n "$MONITOR_PID" ]] && kill -0 "$MONITOR_PID" 2>/dev/null; then
        kill -TERM "$MONITOR_PID" 2>/dev/null || true
    fi
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

predownload_model() {
    log "Pre-downloading $MODEL_ID via snapshot_download() (resumable — see verify_max_kv_size.sh's own comment for why this matters)..."
    if "$MLX_PYTHON" -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL_ID')
"; then
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
    log "Starting mlx_vlm.server for '$MODEL_ID' with PRODUCTION flags (--max-kv-size $MAX_KV_SIZE --kv-bits $KV_BITS --kv-group-size $KV_GROUP_SIZE --quantized-kv-start $QUANTIZED_KV_START --kv-quant-scheme uniform), port $PORT..."
    "$MLX_PYTHON" -m mlx_vlm.server \
        --model "$MODEL_ID" \
        --host 127.0.0.1 --port "$PORT" \
        --max-kv-size "$MAX_KV_SIZE" \
        --prefill-step-size "$PREFILL_STEP_SIZE" \
        --kv-bits "$KV_BITS" \
        --kv-group-size "$KV_GROUP_SIZE" \
        --quantized-kv-start "$QUANTIZED_KV_START" \
        --kv-quant-scheme uniform \
        > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!

    if ! wait_for_health 300; then
        log "ERROR: server failed to become healthy within timeout. Last 30 log lines:"
        tail -30 "$SERVER_LOG" >&2
        return 1
    fi

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
        log "ERROR: health check answered but /v1/models reports '$served_model', expected '$MODEL_ID'."
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
    wait_for_port_free 15 || log "WARNING: port $PORT still answering 15s after kill."
}

start_monitor() {
    # No per-machine GB math needed — monitor_memory.sh's thresholds are percent-free
    # (kern.memorystatus_level), already machine-size-normalized. See monitor_memory.sh's own
    # header comment for the real data (from this exact model) behind that design.
    log "Starting monitor_memory.sh on PID $SERVER_PID (percent-free-based thresholds — see monitor_memory.sh) — real safety net for this test, not a formality."
    "$SCRIPT_DIR/monitor_memory.sh" \
        --pid "$SERVER_PID" \
        --interval 5 \
        --log-file "$MONITOR_LOG" \
        --litellm-pattern "definitely-not-a-real-process-litellm-not-running-during-this-test" \
        > /dev/null 2>&1 &
    MONITOR_PID=$!
}

stop_monitor() {
    if [[ -n "$MONITOR_PID" ]] && kill -0 "$MONITOR_PID" 2>/dev/null; then
        kill -TERM "$MONITOR_PID" 2>/dev/null || true
    fi
    MONITOR_PID=""
}

# Sends a uniquely-nonced, secret-bearing prompt built to an ABSOLUTE target token count
# (via repeated filler text) and asks for recall. Prints one JSON line:
# {target_tokens, http_code, prompt_tokens, completion_tokens, elapsed_s, recalled, error}
send_long_context_probe() {
    local target_tokens="$1" max_tokens="$2" timeout_s="$3"
    "$MLX_PYTHON" - "$PORT" "$MODEL_ID" "$target_tokens" "$max_tokens" "$timeout_s" << 'PYEOF'
import sys, json, time, uuid, urllib.request, urllib.error

port, model_id = sys.argv[1], sys.argv[2]
target_tokens, max_tokens, timeout_s = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])

import random
# Digits-only nonce, not a random hex string — confirmed live via a controlled A/B (fixed
# secret, quantized vs unquantized KV, same ~28k-token size) that this model can recall a
# short alphanumeric secret perfectly at this context length either way, but a raw
# uuid4().hex nonce produced a near-miss (dropped/garbled characters) even at only 20% of
# the configured max-kv-size. That's a known LLM verbatim-copying failure mode for
# random hex strings at long context, NOT evidence of a memory-safety problem — using it
# as this test's recall probe was conflating "answer imprecise" with "server unsafe".
# Digits are tokenized far more predictably across models than mixed hex, so a numeric
# nonce is a fairer test of whether the SERVER is corrupting/losing context, independent
# of the model's raw string-copying accuracy.
nonce = "".join(random.choices("0123456789", k=8))
secret = f"The secret launch code is ZEBRA-{nonce}-QUARTZ."
filler_unit = "The weather today is mild and pleasant with light clouds over the hills. "
# ~13 tokens/unit is a rough heuristic used only to choose how many times to repeat the
# filler — the ACTUAL token count reported below comes from the server's own
# usage.prompt_tokens, per LITELLM_ROUTING_PLAN.md §4.15's correction of exactly this kind
# of word-count estimate.
repeat = max(1, target_tokens // 13)
filler = filler_unit * repeat
prompt = (secret + " " + filler +
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
out = {"target_tokens": target_tokens, "http_code": None, "prompt_tokens": None,
       "completion_tokens": None, "elapsed_s": None, "recalled": False, "error": None}
start = time.monotonic()
try:
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        result = json.loads(resp.read())
    out["elapsed_s"] = round(time.monotonic() - start, 1)
    out["http_code"] = 200
    usage = result.get("usage", {})
    out["prompt_tokens"] = usage.get("prompt_tokens")
    out["completion_tokens"] = usage.get("completion_tokens")
    content = result["choices"][0]["message"]["content"]
    out["recalled"] = nonce in content
    out["content_sample"] = content[:120]
except urllib.error.HTTPError as e:
    out["elapsed_s"] = round(time.monotonic() - start, 1)
    out["http_code"] = e.code
    out["error"] = e.read().decode()[:500]
except Exception as e:
    out["elapsed_s"] = round(time.monotonic() - start, 1)
    out["http_code"] = -1
    out["error"] = str(e)[:500]
print(json.dumps(out))
PYEOF
}

field() {
    "$MLX_PYTHON" -c "import json,sys; v=json.loads(sys.argv[1]).get(sys.argv[2]); print(v if v is not None else '')" "$1" "$2"
}

record_results() {
    local steps_json="$1" summary="$2"
    "$MLX_PYTHON" - "$RESULTS_FILE" "$MODEL_ID" "$MAX_KV_SIZE" "$KV_BITS" "$summary" "$steps_json" << 'PYEOF'
import json, sys, os, datetime

results_file, model_id, max_kv_size, kv_bits, summary, steps_json = sys.argv[1:7]

entry = {
    "model_id": model_id,
    "max_kv_size_tested": int(max_kv_size),
    "kv_bits_tested": int(kv_bits),
    "summary": summary,
    "steps": json.loads(steps_json),
    "tested_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
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
    log "=== Long-context memory stress test: $MODEL_ID (max-kv-size=$MAX_KV_SIZE) ==="

    local baseline_wired
    baseline_wired=$(get_wired_gb)
    log "Baseline wired memory: ${baseline_wired} GB"
    if "$MLX_PYTHON" -c "import sys; sys.exit(0 if float('$baseline_wired') > float('$MAX_BASELINE_WIRED_GB') else 1)"; then
        log "ABORT: baseline wired memory (${baseline_wired}GB) exceeds safety threshold (${MAX_BASELINE_WIRED_GB}GB). Close background apps and retry."
        exit 2
    fi

    if ! predownload_model; then
        record_results "[]" "Pre-download failed"
        exit 2
    fi

    if ! start_server; then
        record_results "[]" "Server failed to start at max-kv-size=$MAX_KV_SIZE"
        exit 2
    fi

    start_monitor

    # Escalating ABSOLUTE token targets: ~20/45/70/85/92% of max_kv_size, spanning from
    # "well under cap" to "close to the ceiling" — the near-cap-fill methodology
    # LITELLM_ROUTING_PLAN.md §4.14 used, adapted to this server/model instead of Ollama.
    local pcts=(20 45 70 85 92)
    local steps_json="["
    local first=true
    local any_success=false
    local aborted_early=false
    local any_recall_failure=false

    for pct in "${pcts[@]}"; do
        local target=$(( MAX_KV_SIZE * pct / 100 ))

        local pre_wired
        pre_wired=$(get_wired_gb)

        if ! curl -fsS "http://127.0.0.1:${PORT}/v1/models" > /dev/null 2>&1; then
            log "Server no longer responding before step ${pct}% (target ~${target} tokens) — monitor likely aborted it. Stopping escalation."
            aborted_early=true
            break
        fi

        log "Step ${pct}% of max-kv-size: target ~${target} tokens (pre-wired=${pre_wired}GB)..."
        # Timeout scales with target size — a 92%-of-131072 (~120k token) prompt on a
        # slower model can genuinely take minutes; §4.14's own 131072-token GGUF run took
        # 365s, and this is a different backend/model so budget generously rather than
        # guessing low and mistaking a slow-but-healthy request for a hang.
        local timeout_s=$(( 120 + target / 100 ))
        local result
        result=$(send_long_context_probe "$target" 80 "$timeout_s")
        log "Step ${pct}% result: $result"

        local post_wired
        post_wired=$(get_wired_gb)

        local http_code prompt_tokens elapsed_s recalled
        http_code=$(field "$result" http_code)
        prompt_tokens=$(field "$result" prompt_tokens)
        elapsed_s=$(field "$result" elapsed_s)
        recalled=$(field "$result" recalled)

        local step_record
        step_record=$("$MLX_PYTHON" -c "
import json,sys
r = json.loads(sys.argv[1])
r['target_pct'] = int(sys.argv[2])
r['pre_wired_gb'] = float(sys.argv[3])
r['post_wired_gb'] = float(sys.argv[4])
print(json.dumps(r))
" "$result" "$pct" "$pre_wired" "$post_wired")

        if [[ "$first" == "true" ]]; then first=false; else steps_json+=","; fi
        steps_json+="$step_record"

        log "Step ${pct}%: http=$http_code prompt_tokens=$prompt_tokens elapsed=${elapsed_s}s recalled=$recalled pre_wired=${pre_wired}GB post_wired=${post_wired}GB"

        # Gate escalation on MEMORY SAFETY (server responded, request completed, no
        # crash/timeout) — NOT on recall correctness. A wrong-but-served answer is a
        # model quality/tokenization question (confirmed via a controlled A/B: this exact
        # model recalled a fixed secret correctly at this same context size with BOTH
        # quantized and unquantized KV — see the code comment above the nonce generator),
        # not evidence the server is unsafe. Recall failures are still recorded per-step
        # and surfaced in the summary, just not treated as a safety-relevant abort signal.
        if [[ "$http_code" == "200" ]]; then
            any_success=true
            if [[ "$recalled" != "True" ]]; then
                any_recall_failure=true
                log "Step ${pct}%: recall did not match (informational only — not a memory-safety signal, see script header). Continuing escalation."
            fi
        else
            log "Step ${pct}% failed at the transport/server level (http=$http_code) — this IS a safety-relevant signal. Stopping escalation."
            aborted_early=true
            break
        fi
    done

    steps_json+="]"

    stop_monitor
    stop_server

    local final_wired
    final_wired=$(get_wired_gb)
    log "Final wired memory after cleanup: ${final_wired}GB (baseline was ${baseline_wired}GB)"

    local recall_note=""
    if [[ "$any_recall_failure" == "true" ]]; then
        recall_note=" NOTE: at least one step returned HTTP 200 (memory-safe) but with an incorrect secret recall — see per-step 'recalled' fields; this is a model-quality signal, not a memory-safety one."
    fi

    local summary
    if [[ "$aborted_early" == "true" && "$any_success" == "false" ]]; then
        summary="First step failed at the transport/server level — see steps[0] for detail. No conclusion possible about long-context memory safety."
    elif [[ "$aborted_early" == "true" ]]; then
        summary="Escalation stopped early after at least one successful step — see steps for where it broke and whether the monitor's abort fired (check $MONITOR_LOG).$recall_note"
    else
        summary="All escalation steps (20/45/70/85/92% of max-kv-size=$MAX_KV_SIZE) completed with HTTP 200, no monitor abort, no crash/timeout.$recall_note"
    fi
    log "=== SUMMARY: $summary ==="

    record_results "$steps_json" "$summary"

    if [[ "$any_success" == "false" ]]; then
        exit 2
    elif [[ "$aborted_early" == "true" ]]; then
        exit 1
    else
        exit 0
    fi
}

main
