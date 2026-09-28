#!/usr/bin/env bash
set -euo pipefail

# monitor_memory.sh — background safety monitor for a running mlx_vlm.server process.
#
# Design: MONITORING_ARCHITECTURE.md. --max-kv-size (see verify_max_kv_size.sh) caps how many
# *tokens* the server will admit, but it doesn't protect against baseline memory pressure from
# everything else running on the machine, and it was only verified against short single-turn
# prompts (see PROJECT_STATUS.md "What's NOT done yet" #2/#5). This script watches actual memory
# pressure live and kills the model before the OS starts swapping/compressing aggressively, which
# on a unified-memory Apple Silicon machine is the real failure mode (not an OOM crash — a slow,
# system-wide stall).
#
# THRESHOLD DESIGN — percent-free, not a fixed wired-GB number, and why:
# The first version of this script used a flat wired-memory GB threshold (e.g. 38GB), with
# start_litellm_proxy.sh computing that number per-machine as (total_ram_gb - 10). That formula
# was checked against MONITORING_ARCHITECTURE.md's own two worked examples (48GB machine -> 38GB
# abort, 24GB machine -> 20GB abort) and DIDN'T hold up: those two examples imply headroom that's
# roughly proportional to total RAM (~21% and ~17%), not a flat 10GB — and a flat 10GB heads
# straight for near-zero margin on a 16GB machine (abort=6GB, below typical OS baseline wired
# usage), which would make the monitor fire on startup before the model even finishes loading.
#
# Switched instead to `kern.memorystatus_level` — the kernel's own real-time "percent memory
# free" figure (0-100, higher=healthier), the same subsystem macOS's own memory-pressure/jetsam
# machinery uses. This is inherently machine-size-normalized: "10% free" means the same thing on
# a 16GB Mac as a 128GB one, with no per-machine math needed at the call site.
#
# Calibrated against a REAL failure, not guessed: a live stress test (verify_long_context_memory.sh
# against qwen3-coder-30b-a3b-instruct-4bit, 2026-09-27) that ran wired memory from 3.7GB to
# 38GB+ on this 48GB machine found kern.memorystatus_level tracking smoothly from 84 down to ~10
# right as wired crossed the old 38GB threshold — so the default abort-below of 10 lines up with
# real data, not intuition. In that SAME run, the discrete `kern.memorystatus_vm_pressure_level`
# (1=NORMAL/2=WARN/4=CRITICAL, the enum apps subscribe to via dispatch_source_memorypressure)
# stayed at NORMAL for the entire 104 seconds and only flipped to WARN in the very last sample —
# essentially concurrent with, not ahead of, the percent-free-based abort point. That single
# 3-state signal has no graduated warning of its own, and per Apple's own documentation WARN is
# routinely raised as an advisory hint (not evidence of real trouble), so it's used here only as
# an unconditional CRITICAL(4) override — a definite, unambiguous "abort now" backstop — layered
# on top of the percent-free zones, not as the primary/only signal. `vm.swapusage` was also
# checked live in that same run and stayed completely flat throughout (0 bytes of change) despite
# 34GB of wired growth — on Apple Silicon's unified memory, GPU/Metal-committed buffers apparently
# don't page to swap the way ordinary app memory would, so swap growth is NOT a usable signal for
# this failure mode and isn't tracked here.
#
# Usage:
#   ./monitor_memory.sh --pid 12345 [--interval 5] [--abort-below-pct 10] [--warn-below-pct 18] \
#       [--caution-below-pct 25] [--verbose] [--log-file /tmp/monitor.log] [--litellm-pattern litellm]
#
# Exit codes: 0 = clean exit (monitored process ended on its own), 1 = aborted due to memory
# pressure, 2 = usage/argument error.

LOG_FILE="/tmp/monitor.log"
INTERVAL=5
ABORT_BELOW_PCT=10
WARN_BELOW_PCT=18
CAUTION_BELOW_PCT=25
VERBOSE=false
TARGET_PID=""
LITELLM_PATTERN="litellm"

usage() {
    cat >&2 <<'EOF'
Usage: monitor_memory.sh --pid <pid> [options]

Options:
  --pid <pid>                 PID of the process to monitor (required; typically mlx_vlm.server)
  --interval <seconds>        Sample interval (default: 5)
  --abort-below-pct <pct>     Abort when kern.memorystatus_level (percent free) drops below this,
                               OR when kern.memorystatus_vm_pressure_level reports CRITICAL(4)
                               regardless of this threshold (default: 10)
  --warn-below-pct <pct>      Warning threshold (default: 18)
  --caution-below-pct <pct>   Caution threshold (default: 25)
  --verbose                   Also echo samples to stdout (default: log file only)
  --log-file <path>           Log file location (default: /tmp/monitor.log)
  --litellm-pattern <pat>     pkill -f pattern used to also stop the LiteLLM proxy on abort
                               (default: "litellm")
EOF
    exit 2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pid) TARGET_PID="$2"; shift 2 ;;
        --interval) INTERVAL="$2"; shift 2 ;;
        --abort-below-pct) ABORT_BELOW_PCT="$2"; shift 2 ;;
        --warn-below-pct) WARN_BELOW_PCT="$2"; shift 2 ;;
        --caution-below-pct) CAUTION_BELOW_PCT="$2"; shift 2 ;;
        --verbose) VERBOSE=true; shift ;;
        --log-file) LOG_FILE="$2"; shift 2 ;;
        --litellm-pattern) LITELLM_PATTERN="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "[ERROR] Unknown argument: $1" >&2; usage ;;
    esac
done

if [[ -z "$TARGET_PID" ]]; then
    echo "[ERROR] --pid is required" >&2
    usage
fi

if ! kill -0 "$TARGET_PID" 2>/dev/null; then
    echo "[ERROR] No such process: PID $TARGET_PID" >&2
    exit 2
fi

log_line() {
    local line="$1"
    echo "$line" >> "$LOG_FILE"
    if [[ "$VERBOSE" == "true" ]]; then
        echo "$line"
    fi
}

# vm_stat reports "Pages wired down" against its own page size header (16384 bytes on Apple
# Silicon, 4096 on Intel) — kept purely as informational/diagnostic context alongside the real
# percent-free-based decision below, not as the safety gate itself (see design note above).
get_wired_gb() {
    local vm_stat_output page_size wired_pages
    vm_stat_output=$(vm_stat)
    page_size=$(echo "$vm_stat_output" | head -1 | grep -o '[0-9]\+' | head -1)
    wired_pages=$(echo "$vm_stat_output" | grep "Pages wired down" | grep -o '[0-9]\+' | head -1)
    awk -v p="$wired_pages" -v s="$page_size" 'BEGIN { printf "%.2f", (p * s) / (1024*1024*1024) }'
}

get_proc_kb() {
    ps -p "$TARGET_PID" -o rss= 2>/dev/null | tr -d ' '
}

# kern.memorystatus_level: kernel's own real-time "percent memory free" (0-100, higher=healthier).
# Machine-size-normalized by construction — see design note above.
get_percent_free() {
    sysctl -n kern.memorystatus_level 2>/dev/null || echo ""
}

# kern.memorystatus_vm_pressure_level: 1=NORMAL, 2=WARN, 4=CRITICAL — the DISPATCH_MEMORYPRESSURE
# enum apps subscribe to. Used here only for the unconditional CRITICAL override, not as the
# primary signal — see design note above for why.
get_pressure_level() {
    sysctl -n kern.memorystatus_vm_pressure_level 2>/dev/null || echo ""
}

pressure_level_name() {
    case "$1" in
        1) echo "NORMAL" ;;
        2) echo "WARN" ;;
        4) echo "CRITICAL" ;;
        *) echo "UNKNOWN($1)" ;;
    esac
}

# Best-effort GPU/CPU thermal pressure read. Not every Mac exposes this via pmset (confirmed live
# on this M5 Pro: "No thermal warning level has been recorded" under normal load) — when pmset
# gives nothing parseable, report "Unknown" rather than fabricating a Nominal/Heavy verdict.
get_thermal_state() {
    local therm_output
    therm_output=$(pmset -g therm 2>/dev/null || true)
    if echo "$therm_output" | grep -qi "no thermal warning level"; then
        echo "Unknown"
        return
    fi
    if echo "$therm_output" | grep -qi "CPU_Scheduler_Limit"; then
        local limit
        limit=$(echo "$therm_output" | grep -i "CPU_Scheduler_Limit" | grep -o '[0-9]\+' | head -1)
        if [[ -n "$limit" ]]; then
            if [[ "$limit" -ge 100 ]]; then
                echo "Nominal"
            elif [[ "$limit" -ge 50 ]]; then
                echo "Moderate"
            else
                echo "Heavy"
            fi
            return
        fi
    fi
    echo "Unknown"
}

format_gb() {
    printf "%.1f" "$1"
}

# Action 1-5 from MONITORING_ARCHITECTURE.md's ABORT zone: SIGTERM the MLX server, give it a
# moment to shut down cleanly, SIGKILL if it's still alive, then best-effort stop the LiteLLM
# proxy so a runaway model doesn't keep the proxy serving requests it can no longer safely answer.
abort_and_cleanup() {
    local reason="$1" pct_free="$2" pressure_name="$3" wired_gb="$4"
    log_line "[ERROR] CRITICAL: $reason (percent_free=${pct_free}% pressure=${pressure_name} wired=${wired_gb}GB)"
    log_line "[INFO] Stopping MLX server (PID $TARGET_PID, graceful shutdown)..."

    kill -TERM "$TARGET_PID" 2>/dev/null || true
    sleep 2
    if kill -0 "$TARGET_PID" 2>/dev/null; then
        log_line "[INFO] MLX server still alive after SIGTERM — sending SIGKILL..."
        kill -KILL "$TARGET_PID" 2>/dev/null || true
    fi

    log_line "[INFO] Terminating LiteLLM proxy (pattern: $LITELLM_PATTERN)..."
    pkill -TERM -f "$LITELLM_PATTERN" 2>/dev/null || true

    sleep 1
    local final_wired
    final_wired=$(get_wired_gb)
    log_line "[✓] Cleanup complete, wired=$(format_gb "$final_wired")GB"
    log_line "[INFO] Process exited cleanly after abort"
}

log_line "[INFO] monitor_memory.sh started — watching PID $TARGET_PID (interval=${INTERVAL}s, caution<${CAUTION_BELOW_PCT}% warn<${WARN_BELOW_PCT}% abort<${ABORT_BELOW_PCT}% free, or pressure=CRITICAL)"

start_ts=$(date +%s)
last_zone=""

while true; do
    if ! kill -0 "$TARGET_PID" 2>/dev/null; then
        log_line "[INFO] Monitored process (PID $TARGET_PID) has exited — stopping monitor."
        exit 0
    fi

    wired_gb=$(get_wired_gb)
    proc_kb=$(get_proc_kb)
    thermal=$(get_thermal_state)
    pct_free=$(get_percent_free)
    pressure=$(get_pressure_level)
    pressure_name=$(pressure_level_name "${pressure:-0}")
    now_ts=$(date +%s)
    elapsed=$((now_ts - start_ts))
    elapsed_fmt=$(printf "%02d:%02d" $((elapsed / 60)) $((elapsed % 60)))
    timestamp=$(date +%H:%M:%S)

    zone="green"
    status="✓"
    critical_override=false
    if [[ -n "$pressure" && "$pressure" -eq 4 ]]; then
        zone="abort"; status="❌ ABORT (pressure=CRITICAL)"; critical_override=true
    elif [[ -n "$pct_free" ]] && awk -v p="$pct_free" -v a="$ABORT_BELOW_PCT" 'BEGIN{exit !(p<a)}'; then
        zone="abort"; status="❌ ABORT"
    elif [[ -n "$pct_free" ]] && awk -v p="$pct_free" -v a="$WARN_BELOW_PCT" 'BEGIN{exit !(p<a)}'; then
        zone="red"; status="⚠️  WARNING"
    elif [[ -n "$pct_free" ]] && awk -v p="$pct_free" -v a="$CAUTION_BELOW_PCT" 'BEGIN{exit !(p<a)}'; then
        zone="amber"; status="⚠️  CAUTION"
    fi

    log_line "[$timestamp] t=$elapsed_fmt  pct_free=${pct_free:-?}%  pressure=$pressure_name  wired=$(format_gb "$wired_gb")GB  proc=${proc_kb:-0}KB  thermal=$thermal  $status"

    if [[ "$zone" == "amber" && "$last_zone" != "amber" && "$last_zone" != "red" && "$last_zone" != "abort" ]]; then
        log_line "[SUGGEST] Context approaching ceiling — consider compacting context soon."
    elif [[ "$zone" == "red" && "$last_zone" != "red" && "$last_zone" != "abort" ]]; then
        log_line "[SUGGEST] Memory critical — compact context now or restart the session."
    fi

    if [[ "$thermal" == "Heavy" ]]; then
        log_line "[WARN] GPU/CPU thermal pressure: Heavy — throughput is likely degraded."
    fi

    if [[ "$zone" == "abort" ]]; then
        if [[ "$critical_override" == "true" ]]; then
            abort_and_cleanup "kern.memorystatus_vm_pressure_level reported CRITICAL" "$pct_free" "$pressure_name" "$wired_gb"
        else
            abort_and_cleanup "percent_free dropped below ${ABORT_BELOW_PCT}% safety threshold" "$pct_free" "$pressure_name" "$wired_gb"
        fi
        exit 1
    fi

    last_zone="$zone"
    sleep "$INTERVAL"
done
