# LiteLLM Proxy Setup & Launch Guide

## Overview

`start_litellm_proxy.sh` is an intelligent setup and launch script that bootstraps a complete local LLM inference stack. It:

- **Auto-detects** whether services are already running
- **Bootstraps from scratch** if needed (creates venvs, installs packages)
- **Validates** each component before proceeding
- **Launches** Claude Code or OpenCode with a local model backend

This eliminates manual setup steps and makes it easy to get up and running on a new machine.

---

## What It Sets Up

The script manages three key components:

1. **MLX Server** (port 11435)
   - Runs local language models using Apple's MLX framework
   - Executes inference directly on your machine
   - Configured to run Qwen3.5 9B by default

2. **PostgreSQL** (port 5433)
   - Embedded database for LiteLLM to store request logs and configuration
   - Installed as a Python package dependency
   - Data stored in `~/pip_env/litellm/pgdata/`

3. **LiteLLM Proxy** (port 4100)
   - Routes API requests from Claude Code/OpenCode to the local MLX model
   - Provides OpenAI-compatible API interface
   - Sits between your client and the local model

---

## Prerequisites

- **macOS** with Apple Silicon (M1/M2/M3 or later) recommended for MLX
- **Python 3.8+** installed and in your PATH
- **curl** installed (usually pre-installed on macOS)
- **~8-16 GB free disk space** for models and dependencies
- **~10 GB free RAM** minimum for running Qwen3.5 9B

---

## Installation & First Run

### Option 1: Fresh Setup (New Machine)

On a fresh machine, the script will automatically:
1. Create Python virtual environments in `~/pip_env/`
2. Install MLX, LiteLLM, and PostgreSQL packages
3. Initialize PostgreSQL database
4. Start all services
5. Launch your chosen client

**Run:**
```bash
./start_litellm_proxy.sh claude qwen3.5-9b-mlx-local mlx-community/Qwen3.5-9B-MLX-8bit
```

This will take 10-15 minutes on first run (downloading model, installing packages).

### Option 2: Existing Setup

If you already have the environments set up:
```bash
./start_litellm_proxy.sh claude qwen3.5-9b-mlx-local
```

The script skips installation and goes straight to health checks → service startup → client launch (takes 30-60 seconds).

---

## Usage

### Basic Syntax

```bash
./start_litellm_proxy.sh <MODE> <MODEL_NAME> [MODEL_ID]
```

### Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `MODE` | Client to launch | `claude` or `opencode` |
| `MODEL_NAME` | Local model identifier | `qwen3.5-9b-mlx-local` |
| `MODEL_ID` | Hugging Face model ID (optional) | `mlx-community/Qwen3.5-9B-MLX-8bit` |

### Examples

**Launch Claude Code with Qwen3.5:**
```bash
./start_litellm_proxy.sh claude qwen3.5-9b-mlx-local mlx-community/Qwen3.5-9B-MLX-8bit
```

**Launch OpenCode with Qwen3.5:**
```bash
./start_litellm_proxy.sh opencode qwen3.5-9b-mlx-local mlx-community/Qwen3.5-9B-MLX-8bit
```

**Reuse existing setup (faster startup):**
```bash
./start_litellm_proxy.sh claude qwen3.5-9b-mlx-local
```

---

## How It Works

### Startup Flow

```
1. Validate Arguments
   └─ Check mode (claude|opencode) and model name

2. Check System Dependencies
   └─ python3, curl available?

3. Bootstrap Environments (if needed)
   ├─ Create MLX venv at ~/pip_env/mlx_lm/
   ├─ Install: mlx-lm, mlx-vlm
   ├─ Create LiteLLM venv at ~/pip_env/litellm/
   └─ Install: litellm, pgserver

4. Start Services (in order)
   ├─ PostgreSQL (port 5433)
   │  └─ Health check: pg_isready
   ├─ MLX Server (port 11435)
   │  └─ Health check: /health endpoint
   └─ LiteLLM Proxy (port 4100)
      └─ Health check: /health/liveliness endpoint

5. Validate Proxy Stack
   └─ All three components responding?

6. Configure & Launch Client
   ├─ Set ANTHROPIC_BASE_URL or OPENAI_BASE_URL
   ├─ Set API keys
   └─ Launch claude or opencode
```

### Idempotent Design

The script is **safe to run repeatedly**:

- If services already running → skips startup, proceeds to health checks
- If venvs exist → skips creation, proceeds to package checks
- If packages installed → skips installation, proceeds to startup

### Health Checks

Each service has a health endpoint that's polled before proceeding:

| Service | Endpoint | Timeout |
|---------|----------|---------|
| MLX | `http://127.0.0.1:11435/health` | 600 seconds |
| PostgreSQL | `pg_isready -h 127.0.0.1 -p 5433` | Immediate |
| LiteLLM | `http://127.0.0.1:4100/health/liveliness` | 30 seconds |

---

## Configuration

### Environment Variables Set by Script

When launching Claude Code:
```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:4100
ANTHROPIC_API_KEY=local-token
ANTHROPIC_AUTH_TOKEN=ollama
ANTHROPIC_MODEL=qwen3.5-9b-mlx-local
ANTHROPIC_SMALL_FAST_MODEL=qwen3.5-9b-mlx-local
CLAUDE_CODE_MAX_CONTEXT_TOKENS=131072
```

When launching OpenCode:
```bash
OPENAI_BASE_URL=http://127.0.0.1:4100/v1
OPENAI_API_KEY=sk-opencode-router-local
```

### Customizing the Setup

**To use a different model**, modify the script:
- Edit the `MLX_MODEL` variable (line 23)
- Or pass it as the third argument

**To change port numbers**, edit:
- `MLX_PORT` (line 21)
- `LITELLM_PORT` (line 32)
- `PG_PORT` (line 35)

**To use different installation directory**, set:
- `MLX_DIR` (line 17)
- `LITELLM_DIR` (line 30)

---

## Logs & Troubleshooting

### Log Files

The script writes detailed logs to `/tmp/`:

| Log | Purpose |
|-----|---------|
| `/tmp/mlx-llm-proxy.log` | MLX server output and errors |
| `/tmp/litellm-llm-proxy.log` | LiteLLM proxy output and errors |
| `/tmp/pgserver-llm-proxy.log` | PostgreSQL startup logs |

**View logs:**
```bash
tail -50 /tmp/mlx-llm-proxy.log
tail -50 /tmp/litellm-llm-proxy.log
tail -50 /tmp/pgserver-llm-proxy.log
```

### Common Issues

#### "MLX server failed to start"
- Check `/tmp/mlx-llm-proxy.log` for the specific error
- Ensure you have ~4GB free RAM for the model
- Try waiting longer (first model load can be slow)

#### "PostgreSQL failed to start"
- Check `/tmp/pgserver-llm-proxy.log`
- Ensure `~/pip_env/litellm/pgdata/` is writable
- Try removing `pgdata` directory and re-running (will reinitialize)

#### "LiteLLM failed to start"
- Check `/tmp/litellm-llm-proxy.log`
- Ensure PostgreSQL is running: `ps aux | grep postgres`
- Verify `opencode-router-config.yaml` exists in `~/pip_env/litellm/`

#### "Cannot import mlx_vlm"
- The MLX packages may not have installed correctly
- Remove `~/pip_env/mlx_lm/venv/` and re-run the script

### Cleanup & Reset

**Stop all services:**
```bash
pkill -f "mlx_vlm.server"
pkill -f "litellm"
pkill -f "pg_ctl"
```

**Full reset (removes everything):**
```bash
rm -rf ~/pip_env/mlx_lm
rm -rf ~/pip_env/litellm
./start_litellm_proxy.sh claude qwen3.5-9b-mlx-local
```

---

## Performance Tips

1. **Free up RAM** before starting
   - Close unnecessary applications
   - Check available memory: `vm_stat`

2. **Use SSD** for model storage
   - Models stored in `~/MLX_Models/`
   - Ensure it's on your fast drive

3. **First run is slowest**
   - Model download: ~5-10 GB
   - First inference: model loads to memory (~8-15s)
   - Subsequent calls: much faster

---

## Smart Memory Detection & Interactive Model Selection (Part 2)

The script now detects your available wired memory and intelligently recommends safe model/context configurations.

### How Memory Detection Works

The script uses `vm_stat` to read **wired memory** (non-pageable RAM), which is the real bottleneck for local models:

```bash
wired_kb=$(vm_stat | grep "Pages wired down" | awk '{print $4}' | tr -d '.')
wired_gb=$((wired_kb * 4 / 1048576))
```

**Why wired memory matters (from §4.17 of LITELLM_ROUTING_PLAN.md):**
- Model weights are pinned to GPU unified memory (cannot be paged to disk)
- KV cache grows dynamically with actual context usage
- macOS's "free memory" is misleading (includes lazy-reclaimable cache pages)
- Only `wired` memory predicts true OOM risk

### Pre-Flight Safety Check

Before launching, the script will:

1. **Detect available wired memory**
   ```
   [INFO] Detected 8.2 GB free wired memory (total: 16 GB)
   ```

2. **Validate sufficient headroom**
   - If wired memory drops below 3 GB during a request, the machine enters memory pressure
   - The script requires a 4 GB safety buffer minimum
   - Refuses to launch if insufficient

3. **Display risk level**
   ```
   ⚠️  WARNING: Only 8.2 GB wired memory available
       Close unnecessary apps (especially browsers, Slack, Teams)
       Current apps using memory:
       - Chrome: 2.1 GB
       - Slack: 0.8 GB
   ```

### Interactive Model & Context Selection Menu

Instead of hard-coded parameters, the script now prompts you:

```
╔══════════════════════════════════════════════════════════════╗
║         LiteLLM Proxy - Select Model & Context               ║
║                                                               ║
║  Available wired memory: 10.5 GB (recommended minimum: 12GB)   ║
║                                                               ║
║  ⚠️  WARNING: Limited memory for local models                  ║
║  Close unnecessary apps (Chrome, Slack, Teams) for safety     ║
╚══════════════════════════════════════════════════════════════╝

Choose a configuration (wired memory → model → max context):

1) 8-12 GB wired
   ├─ qwen3.5-1b-mlx      [extra small]  → max 32K tokens
   ├─ qwen3.5-7b-mlx      [small]        → max 32K tokens
   └─ gemma-2-9b-mlx      [small]        → max 32K tokens

2) 12-16 GB wired  ✓ SAFE (Your tier: 10.5 GB - close apps!)
   ├─ qwen3.5-7b-mlx      → max 65K tokens   (FP16 KV cache)
   ├─ qwen3.5-9b-mlx      → max 65K tokens   (8-bit KV cache) ⭐ Recommended
   └─ gemma-2-9b-mlx      → max 32K tokens

3) 16-24 GB wired
   ├─ qwen3.5-9b-mlx      → max 131K tokens  (8-bit KV cache) ⭐ Best
   ├─ qwen3.5-14b-mlx     → max 65K tokens   (8-bit KV cache)
   └─ gemma-4-12b-mlx     → max 32K tokens   (thermal concern)

4) 24-32 GB wired
   ├─ qwen3.5-9b-mlx      → max 131K tokens  (8-bit KV cache)
   ├─ qwen3.5-14b-mlx     → max 131K tokens  (8-bit KV cache)
   ├─ gemma-4-26b (GGUF)  → max 65K tokens
   └─ qwen3.5-72b-mlx     → max 32K tokens

5) 32+ GB wired
   ├─ qwen3.5-9b-mlx      → max 131K tokens
   ├─ qwen3.5-14b-mlx     → max 131K tokens
   ├─ gemma-4-26b (GGUF)  → max 131K tokens
   └─ mixtral-8x7b (GGUF) → max 32K tokens

Select option [1-5] or 'q' to quit: _
```

### Model Recommendations by Memory Tier

Based on real benchmarking data from LITELLM_ROUTING_PLAN.md:

#### 16 GB MacBook Pro (Most Common Older Model)

**Baseline wired memory: ~4-5 GB** (system, background apps)
**Available for model: 11-12 GB** (if apps closed)

| Model | Size | Context | KV Bits | Prefill | Notes |
|-------|------|---------|---------|---------|-------|
| **qwen3.5-9b-mlx** | 8.9GB | 65K | 8-bit | 2048 | ⭐ Safe choice — 1.7x faster than GGUF |
| qwen3.5-7b-mlx | 7.2GB | 65K | 8-bit | 2048 | More headroom, 1-2% slower |
| qwen3.5-1b-mlx | 2.1GB | 32K | 8-bit | 2048 | Ultra-light for testing |
| gemma-2-9b-mlx | 9.2GB | 32K | 8-bit | 2048 | Thermal pressure onset at 65K |

**Thermal profile (§4.14):** qwen3.5:9b-mlx at 65K hits Nominal 61%, Heavy 38% — acceptable sustained load.

#### 24 GB MacBook Pro

**Baseline wired: ~5-6 GB**
**Available: 18-19 GB**

| Model | Size | Context | KV Bits | Prefill | Notes |
|-------|------|---------|---------|---------|-------|
| **qwen3.5-9b-mlx** | 8.9GB | 131K | 8-bit | 2048 | ⭐ Full context window possible |
| qwen3.5-14b-mlx | 14GB | 65K | 8-bit | 2048 | More capable, reduced context |
| gemma-2-27b (GGUF) | 19GB | 32K | 4-bit | 2048 | Large model, conservative context |

**Best config:** qwen3.5:9b-mlx at 131K with 8-bit KV cache (proven safe in §4.18).

#### 32+ GB MacBook Pro / Mac Studio

**Baseline wired: ~6-8 GB**
**Available: 24-26+ GB**

| Model | Size | Context | KV Bits | Prefill | Notes |
|-------|------|---------|---------|---------|-------|
| qwen3.5-9b-mlx | 8.9GB | 131K | 8-bit | 2048 | Still the throughput winner |
| **qwen3.5-14b-mlx** | 14GB | 131K | 8-bit | 2048 | More capable, same context |
| gemma-4-26b (GGUF) | 18GB | 65K | 4-bit | 2048 | Larger but limited context |
| mixtral-8x7b (GGUF) | 25GB | 32K | 4-bit | 2048 | Mixture-of-Experts, careful sizing |

### Model Download Behavior

When you select a model:

```
1) Script checks if model exists in ~/MLX_Models/

2a) If present → Skip to "Starting MLX server"

2b) If missing → Prompt user:

    Model 'mlx-community/Qwen3.5-9B-MLX-8bit' not found
    
    Download now? [y/N]: y
    
    [===============================>        ] 65% (5.8 GB / 8.9 GB)
    Downloading: mlx-community/Qwen3.5-9B-MLX-8bit
    Size: 8.9 GB  |  Elapsed: 3m 24s  |  Speed: 42.8 MB/s
    
3) Once downloaded, proceeds to start

4) Script remembers this choice for next run (no re-download)
```

**Implementation notes:**
- Uses `huggingface-hub` library's `snapshot_download()`
- Shows real-time progress with file size / ETA
- Validates checksum after download
- Cleanly handles network interruptions (resume-able)

### Memory Monitoring During Runtime

A separate monitoring script runs in the background to detect memory pressure:

```bash
# Background monitoring loop
./monitor_memory.sh --pid $MLX_PID --abort-at 38GB --interval 5s
```

**What it does:**

1. **Samples wired memory every 5 seconds**
   ```
   t=0s:   wired=10.2GB ✓ OK
   t=5s:   wired=15.8GB ✓ OK
   t=10s:  wired=18.1GB ⚠️ WARNING
   t=15s:  wired=22.4GB ⚠️ WARNING
   t=20s:  wired=30.5GB ❌ CRITICAL — aborting
   ```

2. **Tracks KV cache growth trajectory**
   - If growth rate accelerates unexpectedly, flags contamination
   - Example: "KV cache grew 8GB in last 10s — context might be stuck"

3. **Detects thermal throttling**
   ```
   GPU Power: 45W avg → 32W (throttling detected)
   Throughput: 450 tok/s → 310 tok/s (33% degradation)
   → Suggestion: Reduce prefill step size from 2048 to 1024
   ```

4. **Aborts automatically if unsafe**
   - When wired memory approaches machine's ceiling (configurable, default 38GB on 48GB)
   - Gracefully stops MLX server and LiteLLM proxy
   - Preserves logs and current context for debugging
   ```
   [CRITICAL] Wired memory exceeded 38GB safety threshold (39.2GB used)
   [INFO] Stopping MLX server...
   [INFO] Stopping LiteLLM proxy...
   [INFO] Memory released to 8.3GB
   [ERROR] Request aborted to prevent system instability
   ```

5. **On-demand reporting**
   ```bash
   # Show current memory state while session is running
   ./monitor_memory.sh --report
   
   Active MLX sessions: 1 (qwen3.5:9b-mlx)
   Wired memory: 19.4 GB / 48 GB (40%)
   KV cache estimated: 8.2 GB (growing)
   Thermal pressure: Nominal 80%, Moderate 20%
   Time on model: 14m 32s
   Requests processed: 23
   ```

### Usage: Interactive Model Selection

```bash
# Old way (deprecated):
./start_litellm_proxy.sh claude qwen3.5-9b-mlx-local mlx-community/Qwen3.5-9B-MLX-8bit

# New way (interactive):
./start_litellm_proxy.sh --interactive claude

# Or with monitoring (background safety):
./start_litellm_proxy.sh --interactive --monitor claude
```

### What Happens in Interactive Mode

```
$ ./start_litellm_proxy.sh --interactive claude

[INFO] Checking system dependencies...
[✓] System dependencies OK

[INFO] Detecting available memory...
[✓] Available wired memory: 12.4 GB

╔══════════════════════════════════════════════════════════════╗
║         LiteLLM Proxy - Select Model & Context               ║
║                                                               ║
║  Your tier: 12-16 GB wired (good for smaller models)          ║
╚══════════════════════════════════════════════════════════════╝

1) qwen3.5-9b-mlx    [RECOMMENDED] 65K tokens, 8-bit KV cache
2) qwen3.5-7b-mlx    [SAFE]        65K tokens, 8-bit KV cache
3) qwen3.5-1b-mlx    [LIGHT]       32K tokens, 8-bit KV cache
4) Custom (enter model ID manually)

Select [1-4]: 1

[INFO] Selected: qwen3.5-9b-mlx with 65K context window

Context window options:
1) 32K   (safe, standard)
2) 65K   [RECOMMENDED] (proven safe, good capacity)
3) 131K  (risky on 12GB — close apps first)

Select context [1-3]: 2

[INFO] Checking if model is cached...
[✓] mlx-community/Qwen3.5-9B-MLX-8bit found in ~/MLX_Models/

[INFO] Bootstrapping MLX environment...
[✓] MLX bootstrap complete

[INFO] Bootstrapping LiteLLM environment...
[✓] LiteLLM bootstrap complete

[INFO] Starting PostgreSQL...
[✓] PostgreSQL started and ready

[INFO] Starting MLX server with model qwen3.5:9b-mlx...
[█████████████████████████████░░░░░░░░░░░░] 64% Model loaded
[✓] MLX server started and healthy

[INFO] Starting LiteLLM...
[✓] LiteLLM started and healthy

[INFO] Validating proxy stack...
[✓] Proxy stack validated and ready

[✓] Launching Claude Code with qwen3.5-9b-mlx (context: 65K tokens)
```

### Monitoring Script Output Example

```
$ ./monitor_memory.sh --pid 12345 --interval 5s

[MONITOR] Starting memory monitor for MLX (PID 12345)
[MONITOR] Safety threshold: 38GB wired (abort if exceeded)
[MONITOR] Interval: 5 seconds

t=00:00  wired= 9.2GB (model loaded)    tok/s=1450 ✓
t=00:05  wired=10.4GB (context growing) tok/s=1440 ✓
t=00:10  wired=12.8GB (65% context cap) tok/s=1420 ✓
t=00:15  wired=15.2GB (85% context cap) tok/s=1380 ⚠️
t=00:20  wired=17.1GB (95% context cap) tok/s=1150 ⚠️ Thermal 38% Heavy
t=00:25  wired=17.9GB (near ceiling)    tok/s=890  ⚠️ Thermal 89% Heavy

→ Context window approaching capacity at 65K
→ Next request may hit memory pressure
→ Suggestion: Compact context (opencode auto-compaction) or reduce request size
```

---

## Future Enhancements (Part 3)

- **Model profiler** — benchmark new models to generate safety profiles
- **Thermal management** — auto-throttle prefill step size under sustained heavy pressure
- **Context compaction** — integrate with opencode's native compaction for long sessions
- **Performance replay** — replicate baseline measurements under different load conditions

---

## Related Files

- **Script:** `~/scripts/start_litellm_proxy.sh`
- **Original reference:** `~/scripts/start_llm_proxy.sh`
- **LiteLLM config:** `~/pip_env/litellm/opencode-router-config.yaml`
- **Environment:** `~/MLX_Models/` (model cache), `~/pip_env/` (venvs)

---

## Support & Debugging

For detailed debugging:
1. Run with verbose output: Add `set -x` near the top of the script
2. Check all logs in `/tmp/`
3. Verify each service manually:
   ```bash
   curl http://127.0.0.1:11435/health          # MLX
   curl http://127.0.0.1:4100/health/liveliness # LiteLLM
   pg_isready -h 127.0.0.1 -p 5433              # PostgreSQL
   ```

---

## Questions?

Refer to the script comments or check individual log files for specific errors.
