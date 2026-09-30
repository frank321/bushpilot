# Implementation Notes: Interactive Memory-Aware LiteLLM Proxy

**Status: superseded (2026-09-29, PHASE_PLAN.md Phases 2-4).** This is the design
doc for the pre-redesign architecture — the LiteLLM-fronted proxy it describes was
dropped entirely (MLX serves both wire formats natively), the supervisor it planned
became `start_mlx_server.sh`, and its paths (`~/pip_env/*`, `~/MLX_Models`) predate
the Phase 1 migration of everything into `~/.local-model-setup/` (completed
2026-09-30). Kept for the memory-detection and tiering design research below, which
informed the current implementation — read `start_mlx_server.sh` and
`model_advisor.py` for where that logic actually lives now, not this doc.

## Summary of Changes Needed

### 1. Memory Detection Functions

```bash
get_wired_memory_bytes() {
    # Returns wired memory in bytes
    vm_stat | grep "Pages wired down" | awk '{print $4}' | tr -d '.' | awk '{print $1 * 4096}'
}

get_wired_memory_gb() {
    # Returns wired memory in GB (one decimal)
    local bytes=$(get_wired_memory_bytes)
    echo "scale=1; $bytes / 1024 / 1024 / 1024" | bc
}

get_total_memory_gb() {
    # macOS total physical RAM
    sysctl -n hw.memsize | awk '{printf "%.1f\n", $1 / 1024 / 1024 / 1024}'
}

get_memory_tier() {
    # Returns tier name based on available wired memory
    local wired=$(get_wired_memory_gb)
    
    if (( $(echo "$wired < 8" | bc -l) )); then
        echo "CRITICAL"  # < 8GB
    elif (( $(echo "$wired < 12" | bc -l) )); then
        echo "SMALL"     # 8-12GB
    elif (( $(echo "$wired < 16" | bc -l) )); then
        echo "MEDIUM"    # 12-16GB
    elif (( $(echo "$wired < 24" | bc -l) )); then
        echo "LARGE"     # 16-24GB
    else
        echo "XLARGE"    # 24GB+
    fi
}
```

### 2. Model Database (JSON)

Create `models.json` with metadata:

```json
{
  "models": [
    {
      "id": "qwen3.5-9b-mlx",
      "huggingface": "mlx-community/Qwen3.5-9B-MLX-8bit",
      "size_gb": 8.9,
      "params": "9.4B",
      "framework": "mlx",
      "tier": "RECOMMENDED",
      "min_wired_gb": 10,
      "recommended_context": {
        "SMALL": 32000,
        "MEDIUM": 65536,
        "LARGE": 131072,
        "XLARGE": 131072
      },
      "kv_bits": 8,
      "prefill_step": 2048,
      "notes": "Best throughput (1.7x faster than GGUF), safe memory behavior"
    },
    {
      "id": "qwen3.5-7b-mlx",
      "huggingface": "mlx-community/Qwen3.5-7B-MLX-8bit",
      "size_gb": 7.2,
      "params": "7B",
      "framework": "mlx",
      "tier": "SAFE",
      "min_wired_gb": 9,
      "recommended_context": {
        "SMALL": 32000,
        "MEDIUM": 65536,
        "LARGE": 131072,
        "XLARGE": 131072
      },
      "kv_bits": 8,
      "prefill_step": 2048,
      "notes": "More headroom than 9B, 1-2% slower"
    },
    {
      "id": "qwen3.5-1b-mlx",
      "huggingface": "mlx-community/Qwen3.5-1B-MLX-Instruct",
      "size_gb": 2.1,
      "params": "1.5B",
      "framework": "mlx",
      "tier": "LIGHT",
      "min_wired_gb": 5,
      "recommended_context": {
        "SMALL": 32000,
        "MEDIUM": 32000,
        "LARGE": 65536,
        "XLARGE": 131072
      },
      "kv_bits": 8,
      "prefill_step": 2048,
      "notes": "Ultra-light, good for testing or resource-constrained machines"
    }
  ]
}
```

### 3. Interactive Model Selection Menu

```bash
show_model_menu() {
    local tier="$1"
    local wired_gb="$2"
    
    clear
    cat << "EOF"
╔══════════════════════════════════════════════════════════════╗
║         LiteLLM Proxy - Select Model & Context               ║
╚══════════════════════════════════════════════════════════════╝
EOF

    echo ""
    echo "Available wired memory: ${wired_gb} GB (tier: $tier)"
    
    if [[ "$tier" == "CRITICAL" ]]; then
        echo "❌ CRITICAL: Less than 8GB wired memory"
        echo "   Please close apps (Chrome, Slack, Teams, etc.) and try again"
        return 1
    fi
    
    echo ""
    case "$tier" in
        SMALL)
            cat << 'EOF'
Models safe for 8-12 GB:

1) qwen3.5-9b-mlx    [RECOMMENDED] 65K context, 8-bit KV cache
   └─ Fast (1450 tok/s), efficient memory, proven safe
2) qwen3.5-7b-mlx    [SAFE]        65K context, 8-bit KV cache
   └─ More headroom, 1-2% slower than 9B
3) qwen3.5-1b-mlx    [LIGHT]       32K context, 8-bit KV cache
   └─ Ultra-small, only for testing

Select [1-3] or 'q' to quit: 
EOF
            ;;
        MEDIUM)
            cat << 'EOF'
Models safe for 12-16 GB:

1) qwen3.5-9b-mlx    [RECOMMENDED] 65K-131K context, 8-bit KV cache
   └─ Best throughput, proven at 131K in research
2) qwen3.5-7b-mlx    [SAFE]        65K context, 8-bit KV cache
   └─ More headroom, good for coding
3) qwen3.5-1b-mlx    [LIGHT]       32K context, 8-bit KV cache
   └─ Room for multiple tools/files

Select [1-3] or 'q' to quit:
EOF
            ;;
        LARGE)
            cat << 'EOF'
Models safe for 16-24 GB:

1) qwen3.5-9b-mlx    [BEST]        131K context, 8-bit KV cache ⭐
   └─ Full context window, ideal config
2) qwen3.5-14b-mlx   [CAPABLE]     65K context, 8-bit KV cache
   └─ More parameters, reduced context
3) gemma-2-27b (GGUF)[LARGE]       32K context, 4-bit quantization
   └─ Bigger model, be careful with context

Select [1-3] or 'q' to quit:
EOF
            ;;
        XLARGE)
            cat << 'EOF'
Models safe for 24+ GB:

1) qwen3.5-9b-mlx    [BEST]        131K context, 8-bit KV cache ⭐
   └─ Still the throughput winner
2) qwen3.5-14b-mlx   [CAPABLE]     131K context, 8-bit KV cache
   └─ More parameters, same context as 9B
3) gemma-4-26b (GGUF)[LARGE]       65K context, 4-bit quantization
   └─ Larger model, thermal onset later
4) mixtral-8x7b (GGUF)[EXPERT]     32K context, 4-bit quantization
   └─ MoE model, careful tuning needed

Select [1-4] or 'q' to quit:
EOF
            ;;
    esac
}
```

### 4. Model Download with Progress

```bash
download_model_if_missing() {
    local model_name="$1"
    local huggingface_id="$2"
    local size_gb="$3"
    
    local cache_dir="$HOME/MLX_Models"
    
    # Check if already cached
    if [ -d "$cache_dir/$huggingface_id" ]; then
        log_success "Model cached: $huggingface_id"
        return 0
    fi
    
    # Prompt user
    echo ""
    echo "Model '${huggingface_id}' not found"
    read -p "Download now? (${size_gb}GB) [y/N]: " -r response
    
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        log_error "Model download skipped"
        return 1
    fi
    
    # Download with progress
    log_info "Downloading ${huggingface_id}..."
    
    python3 << PYTHON
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

cache_dir = Path("$cache_dir")
cache_dir.mkdir(parents=True, exist_ok=True)

try:
    snapshot_download(
        repo_id="$huggingface_id",
        cache_dir=str(cache_dir),
        local_dir_use_symlinks=False,
        resume_download=True
    )
    print(f"✓ Download complete: $huggingface_id")
except Exception as e:
    print(f"✗ Download failed: {e}")
    sys.exit(1)
PYTHON
    
    return $?
}
```

### 5. Memory Monitoring Script

Create `monitor_memory.sh`:

```bash
#!/usr/bin/env bash
# monitor_memory.sh — background memory and thermal monitor
# Usage: ./monitor_memory.sh --pid <PID> --interval <seconds> --abort-at <GB>

set -euo pipefail

PID="${1:?Process ID required}"
INTERVAL="${2:-5}"
ABORT_GB="${3:-38}"

get_process_wired_memory() {
    local pid="$1"
    # Get process RSS (rough approximation of wired memory for this process)
    ps -p "$pid" -o rss= 2>/dev/null || echo "0"
}

get_system_wired_memory_gb() {
    vm_stat | grep "Pages wired down" | awk '{print $4}' | tr -d '.' | \
        awk '{printf "%.1f\n", $1 * 4 / 1048576}'
}

get_thermal_pressure() {
    # macOS thermal pressure (0=Nominal, 1=Moderate, 2=Heavy)
    pmset -g therm 2>/dev/null | grep "Thermal" | awk '{print $NF}' || echo "unknown"
}

get_gpu_power() {
    # Approximate GPU power from powermetrics (requires sudo)
    # Fallback to "unknown" if not available
    echo "unknown"
}

log_sample() {
    local elapsed="$1"
    local wired_gb="$2"
    local proc_kb="$3"
    local thermal="$4"
    
    local status="✓"
    if (( $(echo "$wired_gb > $((ABORT_GB - 5))" | bc -l) )); then
        status="⚠️ WARNING"
    elif (( $(echo "$wired_gb > $ABORT_GB" | bc -l) )); then
        status="❌ CRITICAL"
    fi
    
    printf "[%s] t=%s  wired=%5.1fGB  proc=%6dKB  thermal=%s %s\n" \
        "$(date +%H:%M:%S)" "$elapsed" "$wired_gb" "$proc_kb" "$thermal" "$status"
}

main() {
    log_info "Starting memory monitor for PID $PID"
    log_info "Abort threshold: ${ABORT_GB}GB wired"
    log_info "Sample interval: ${INTERVAL}s"
    log_info ""
    
    local start_time=$(date +%s)
    local prev_wired=0
    
    while kill -0 "$PID" 2>/dev/null; do
        local elapsed=$(($(date +%s) - start_time))
        local elapsed_min=$((elapsed / 60))
        local elapsed_sec=$((elapsed % 60))
        local elapsed_fmt=$(printf "%02d:%02d" $elapsed_min $elapsed_sec)
        
        local wired_gb=$(get_system_wired_memory_gb)
        local proc_kb=$(get_process_wired_memory "$PID")
        local thermal=$(get_thermal_pressure)
        
        log_sample "$elapsed_fmt" "$wired_gb" "$proc_kb" "$thermal"
        
        # Check abort threshold
        if (( $(echo "$wired_gb > $ABORT_GB" | bc -l) )); then
            log_error "CRITICAL: Wired memory exceeded ${ABORT_GB}GB (${wired_gb}GB used)"
            log_info "Aborting to prevent system instability..."
            pkill -TERM -P "$PID" > /dev/null 2>&1 || true
            kill -TERM "$PID" > /dev/null 2>&1 || true
            sleep 2
            break
        fi
        
        # Check thermal pressure
        if [[ "$thermal" == "2" ]]; then
            log_info "⚠️  Heavy thermal pressure detected"
            log_info "   Suggestion: Reduce prefill step size or context window"
        fi
        
        sleep "$INTERVAL"
    done
    
    local final_wired=$(get_system_wired_memory_gb)
    log_info "Process $PID exited"
    log_info "Final wired memory: ${final_wired}GB"
}

main
```

### 6. New Script Flags

```bash
# Current (unchanged):
./start_litellm_proxy.sh <mode> <model> [model_id]

# New interactive mode:
./start_litellm_proxy.sh --interactive <mode>
./start_litellm_proxy.sh --interactive --monitor <mode>

# Example:
./start_litellm_proxy.sh --interactive claude
./start_litellm_proxy.sh --interactive --monitor opencode
```

## Integration Checklist

- [ ] Add memory detection functions
- [ ] Create `models.json` with model metadata
- [ ] Implement `show_model_menu()` function
- [ ] Implement `download_model_if_missing()` function
- [ ] Add `--interactive` flag parsing
- [ ] Add `--monitor` flag for background monitoring
- [ ] Create `monitor_memory.sh` executable
- [ ] Update `validate_args()` to handle new flags
- [ ] Test on 16GB MacBook Pro (baseline)
- [ ] Test on 32GB+ machine
- [ ] Update README with new usage examples
- [ ] Test model download interruption/resume

## Testing Strategy

1. **16GB MacBook (baseline):**
   - Run menu with ~4GB baseline wired (apps closed)
   - Select qwen3.5-9b-mlx at 65K
   - Verify prefill completes successfully
   - Check memory doesn't exceed 12GB during request

2. **24GB MacBook:**
   - Select qwen3.5-9b-mlx at 131K
   - Verify sustained throughput
   - Monitor thermal pressure

3. **Download scenarios:**
   - First download of 8.9GB model
   - Interrupt and resume download
   - Model already cached (fast path)

4. **Memory pressure:**
   - Trigger abort by running background workload
   - Verify graceful shutdown, no system freeze

## Files to Create/Modify

```
~/scripts/
├── start_litellm_proxy.sh        (modify: add interactive + monitoring)
├── monitor_memory.sh             (NEW: background memory monitor)
├── models.json                   (NEW: model metadata database)
├── LITELLM_PROXY_USAGE.md        (updated ✓)
└── IMPLEMENTATION_NOTES.md       (this file)
```
