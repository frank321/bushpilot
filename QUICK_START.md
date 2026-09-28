# Quick Start: Memory-Aware LiteLLM Proxy

## For Users (How to Use)

### First Time Setup

```bash
cd ~/scripts
./start_litellm_proxy.sh --interactive --monitor claude
```

This will:
1. **Detect your Mac's wired memory**
2. **Show you safe model options** (based on your available memory)
3. **Let you choose context window size**
4. **Download the model** (if not cached) with progress indicator
5. **Start all services** (PostgreSQL, MLX server, LiteLLM proxy)
6. **Run memory monitor in background** (alerts if running out of memory)
7. **Launch Claude Code** with your selected configuration

### What Happens in the Menu

```
Your Mac has 12.4 GB free wired memory?
  ↓
MEDIUM tier → Shows 3 safe models for 12-16 GB Macs
  ↓
You select: qwen3.5-9b-mlx
  ↓
Choose context: 65K (safe), 131K (risky), or 32K (ultra-safe)
  ↓
Model not cached? 
  → "Download 8.9 GB? [y/N]: " 
  → [████████████░░░░░░░░] 65% (5.8 GB / 8.9 GB)
  ↓
Download complete
  ↓
Services start
  ↓
Claude Code launches with your model
```

### Common Scenarios

**Old 16GB MacBook:**
```bash
./start_litellm_proxy.sh --interactive --monitor claude
→ Shows: "SMALL tier" (8-12GB wired)
→ Recommends: qwen3.5-9b-mlx at 65K context
→ Safe choice for your machine
```

**24GB MacBook:**
```bash
./start_litellm_proxy.sh --interactive --monitor claude
→ Shows: "LARGE tier" (16-24GB wired)
→ Recommends: qwen3.5-9b-mlx at 131K context (full window!)
→ You get full 128K token context window
```

**32GB+ Mac Studio:**
```bash
./start_litellm_proxy.sh --interactive --monitor claude
→ Shows: "XLARGE tier" (24GB+ wired)
→ Recommends: Multiple options, qwen3.5-14b-mlx, larger models
→ More options available
```

### Monitoring: What It Does

While you work, the monitor quietly watches in background:

```
✓ Memory OK (9.2 GB wired)
✓ Memory OK (10.4 GB)
✓ Memory OK (12.8 GB) ← Context growing
⚠️ Memory WARNING (15.2 GB) ← Getting full
⚠️ Memory WARNING (16.9 GB) ← Close to limit
❌ CRITICAL (39.1 GB) ← ABORT! Kill model gracefully
```

If memory gets dangerous, it:
1. Stops the model (gracefully, preserves current state)
2. Frees memory immediately
3. Shows: "Memory exceeded safety threshold — please compact context or restart"

**You don't have to do anything.** It protects your machine automatically.

### View Monitoring Log (Optional)

While working:
```bash
tail -f /tmp/monitor.log
```

Shows real-time memory, thermal pressure, tokens/sec.

---

## For Developers (How to Build)

### Files to Implement

```
~/scripts/
├── start_litellm_proxy.sh          [MODIFY]  Add interactive + monitoring
├── monitor_memory.sh               [CREATE]  Background memory/thermal monitor
├── models.json                     [CREATE]  Model metadata database
├── LITELLM_PROXY_USAGE.md          [✓ DONE] Updated usage guide
├── IMPLEMENTATION_NOTES.md         [✓ DONE] Code snippets
├── MONITORING_ARCHITECTURE.md      [✓ DONE] How monitoring works
└── QUICK_START.md                  [✓ DONE] This file
```

### Implementation Steps

#### 1. Add Memory Detection to Main Script

```bash
# In start_litellm_proxy.sh, add:

get_wired_memory_gb() {
    vm_stat | grep "Pages wired down" | awk '{print $4}' | tr -d '.' | \
        awk '{printf "%.1f\n", $1 * 4 / 1048576}'
}

get_memory_tier() {
    local wired=$(get_wired_memory_gb)
    if (( $(echo "$wired < 8" | bc -l) )); then echo "CRITICAL"
    elif (( $(echo "$wired < 12" | bc -l) )); then echo "SMALL"
    elif (( $(echo "$wired < 16" | bc -l) )); then echo "MEDIUM"
    elif (( $(echo "$wired < 24" | bc -l) )); then echo "LARGE"
    else echo "XLARGE"
    fi
}
```

#### 2. Create models.json

```json
{
  "models": [
    {
      "id": "qwen3.5-9b-mlx",
      "huggingface": "mlx-community/Qwen3.5-9B-MLX-8bit",
      "size_gb": 8.9,
      "min_wired_gb": 10,
      "recommended_context": {
        "SMALL": 32000,
        "MEDIUM": 65536,
        "LARGE": 131072,
        "XLARGE": 131072
      },
      "tier": "RECOMMENDED",
      "notes": "Best throughput (1.7x faster than GGUF)"
    }
    // ... more models
  ]
}
```

#### 3. Add Interactive Menu

```bash
show_model_menu() {
    local tier="$1"
    # Display options based on memory tier
    # Prompt user to select
    # Return: model_id, context_size
}
```

#### 4. Create monitor_memory.sh

See MONITORING_ARCHITECTURE.md for full code.

Core logic:
```bash
while kill -0 "$PID" 2>/dev/null; do
    wired_gb=$(vm_stat | grep "Pages wired down" | awk '{print $4 * 4 / 1048576}')
    
    if (( $(echo "$wired_gb > $ABORT_GB" | bc -l) )); then
        kill -TERM "$PID"
        break
    fi
    
    log_sample $elapsed $wired_gb
    sleep 5
done
```

#### 5. Update start_litellm_proxy.sh

Add flags:
```bash
INTERACTIVE=false
MONITOR=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --interactive) INTERACTIVE=true; shift ;;
        --monitor) MONITOR=true; shift ;;
        *) break ;;
    esac
done

if [[ "$INTERACTIVE" == "true" ]]; then
    detect_memory_and_show_menu
    get_user_selection
    download_model_if_missing
fi

if [[ "$MONITOR" == "true" ]]; then
    start_memory_monitor &
    MONITOR_PID=$!
    trap "kill $MONITOR_PID 2>/dev/null" EXIT
fi
```

### Testing Checklist

- [ ] **Memory detection works on 16GB MacBook**
  ```bash
  ./test_memory_detection.sh
  ```

- [ ] **Interactive menu displays correctly**
  - Check menu renders for each tier (SMALL, MEDIUM, LARGE, XLARGE)
  - Verify model recommendations match tier

- [ ] **Model download works**
  - First download (full 8.9 GB)
  - Resume from interruption
  - Cache hit (fast path)

- [ ] **Monitor catches memory pressure**
  - Run synthetic workload that fills 80% of available memory
  - Verify monitor logs warnings
  - Verify abort triggers at configured threshold

- [ ] **Graceful abort on memory limit**
  - Trigger abort scenario (>38GB on test machine)
  - Verify model stops cleanly
  - Verify no system freeze
  - Verify memory releases within 5-10 seconds

---

## Key Design Decisions

### Why wired memory?

From LITELLM_ROUTING_PLAN.md §4.17:
- Model weights are pinned to GPU unified memory (cannot be paged to disk)
- macOS counts reclaimable cache pages as "used" (misleading)
- Only `wired` memory predicts true OOM/memory-pressure risk
- Alternative metrics (Activity Monitor "free", `top` "available") are unreliable

### Why interactive instead of fixed?

- **User's needs vary:** 16GB MacBook ≠ 32GB Mac Studio
- **Background state varies:** Close 1 browser = 2GB freed
- **Model choice is personal:** Prefer speed? Available memory? Thermal impact?
- **Interactive = safe:** Users see warning upfront, not crash later

### Why multiple models?

From benchmarking data:
- **qwen3.5-9b-mlx:** Best for most users (fast, efficient memory, proven safe to 131K)
- **qwen3.5-7b-mlx:** More headroom on tight memory (16GB Macs)
- **qwen3.5-1b-mlx:** Testing/ultra-light workloads
- **gemma models:** Trade-offs (slower thermal profile, larger size)
- **GGUF models:** Larger context, but less memory-efficient than qwen

### Why background monitoring instead of user-managed?

- **Prevents crashes:** Machine will not freeze (monitor aborts before critical)
- **Reduces user burden:** Doesn't require constant supervision
- **Detects unexpected patterns:** Catches KV cache leaks, thermal degradation
- **Safe abort:** Graceful shutdown preserves conversation state when possible

---

## Resource Requirements by Model

Based on real measurements from LITELLM_ROUTING_PLAN.md:

### qwen3.5-9b-mlx (RECOMMENDED)

| Config | Baseline Wired | Model Weight | Peak KV | Total | Thermal |
|--------|---|---|---|---|---|
| 32K context | 5GB | 8.9GB | 1GB | **15GB** | Nominal 100% |
| 65K context | 5GB | 8.9GB | 3GB | **17GB** | Nominal 61%, Heavy 38% |
| 131K context | 5GB | 8.9GB | 6GB | **20GB** | Nominal 29%, Heavy 71% |

**For 16GB Mac:** Use 65K (safe)
**For 24GB Mac:** Use 131K (full window)

### gemma4:12b-mlx (ALTERNATIVE)

| Config | Baseline Wired | Model Weight | Peak KV | Total | Thermal |
|--------|---|---|---|---|---|
| 32K context | 5GB | 7.7GB | 1GB | **14GB** | Nominal 100% |
| 65K context | 5GB | 7.7GB | 4GB | **17GB** | **Heavy 92%** ⚠️ |
| 131K context | — | — | — | **22GB** | Heavy 100%, risky |

**Caveat:** MLX backend is less efficient than GGUF for Gemma architecture. qwen3.5:9b-mlx is faster.

---

## Questions?

Refer to:
- **LITELLM_PROXY_USAGE.md** — Full user guide with all options
- **IMPLEMENTATION_NOTES.md** — Code snippets for each function
- **MONITORING_ARCHITECTURE.md** — Deep dive on how monitoring works
- **LITELLM_ROUTING_PLAN.md** — Original research & findings (in devx-agent-eval repo)
