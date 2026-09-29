# Memory Monitoring Architecture

**Status: superseded (2026-09-29, PHASE_PLAN.md Phase 4).** `monitor_memory.sh` — the
standalone script this doc describes — has been removed. Its logic (the percent-free
thresholds, the consecutive-poll debounce, the CRITICAL-pressure-level override) is now
folded directly into `start_mlx_server.sh`'s own dashboard poll loop
(`check_memory_backstop`), triggering the exact same full-teardown path as Ctrl-C/`q`
instead of a separate background daemon. This doc is kept for the threshold-design
research below, which is still accurate and still the source of truth for those
numbers — just read `start_mlx_server.sh` for where the logic actually lives now, not
a separate script invoked with `--monitor`.

---

*(Original doc, describing the now-removed standalone script, preserved below for the
threshold-design research.)*

**Status when written: current.** This doc originally described `monitor_memory.sh` as a
design to be implemented. It has since been built (`monitor_memory.sh`), wired into
`start_litellm_proxy.sh` via `--monitor`, and used as the safety net for
`verify_long_context_memory.sh`'s real stress tests. This revision replaces the original
wired-memory-GB threshold design with the percent-free design actually implemented, after that
first design was found not to hold up across machine sizes and was recalibrated against real
failure data. See "Threshold design" below for why, and `PROJECT_STATUS.md` for how this fits
into the rest of the project.

---

## High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│  User runs: ./start_litellm_proxy.sh --interactive --monitor claude     │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │  Detect Wired Memory     │
                    │  └─ vm_stat              │
                    │  └─ Show memory tier     │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼──────────────────┐
                    │  Interactive Model Menu       │
                    │  └─ Load models.json          │
                    │  └─ Show safe options for tier
                    │  └─ Prompt user selection     │
                    └────────────┬──────────────────┘
                                 │
                ┌────────────────▼────────────────────┐
                │  Validate Selection                 │
                │  ├─ Check model exists locally      │
                │  ├─ If missing → Prompt download    │
                │  └─ Resolve context window size     │
                └────────────┬───────────────────────┘
                             │
        ┌────────────────────▼─────────────────────┐
        │  Start Services (PostgreSQL, MLX, etc)   │
        └────────────┬──────────────────────────────┘
                     │
     ┌───────────────▼──────────────────────┐
     │  Launch Background Monitor (if flag) │
     │  ├─ Fork: monitor_memory.sh          │
     │  ├─ Args: --pid $MLX_PID             │
     │  │         --interval 5              │
     │  │         (thresholds default to    │
     │  │          percent-free, no GB math │
     │  │          needed — see below)      │
     │  └─ Redirect to /tmp/monitor.log     │
     └───────────────┬──────────────────────┘
                     │
     ┌───────────────▼──────────────────────┐
     │  Launch Client (Claude/OpenCode)     │
     │  ├─ Sets up env vars                 │
     │  ├─ Routes to LiteLLM proxy          │
     │  └─ Proxy routes to MLX server       │
     └───────────────┬──────────────────────┘
                     │
     ┌───────────────▼───────────────────────┐
     │  User Session (Monitor in background) │
     └───────────────┬───────────────────────┘
                     │
                     └─→ [See diagram below]
```

## Monitoring Lifecycle (Parallel to User Session)

```
┌────────────────────────────────────────────────────────────────────┐
│  monitor_memory.sh (background process, runs in parallel)          │
└────────────────┬───────────────────────────────────────────────────┘
                 │
     ┌───────────▼────────────┐
     │  Main Loop (infinite)  │
     │  repeat every 5 seconds│
     └───────────┬────────────┘
                 │
    ┌────────────▼──────────────────────────┐
    │  Sample System Metrics                │
    ├─ sysctl kern.memorystatus_level       │
    │  └─ real-time percent-free (PRIMARY)  │
    ├─ sysctl kern.memorystatus_vm_pressure_level │
    │  └─ NORMAL/WARN/CRITICAL (override)   │
    ├─ vm_stat                              │
    │  └─ wired memory (informational only) │
    ├─ ps -p $PID -o rss=                 │
    │  └─ Get process memory (informational)│
    ├─ pmset -g therm                      │
    │  └─ Get thermal pressure (best-effort)│
    └────────────┬──────────────────────────┘
                 │
    ┌────────────▼──────────────────────────┐
    │  Log Sample                           │
    │  Format: timestamp | pct_free |       │
    │          pressure | wired | proc | T  │
    │  Write to /tmp/monitor.log            │
    │  (also echo to terminal if verbose)   │
    └────────────┬──────────────────────────┘
                 │
    ┌────────────▼──────────────────────────────────┐
    │  Evaluate Safety Thresholds                   │
    ├─ pressure == CRITICAL(4)?                     │
    │  ├─ YES → Kill MLX, stop LiteLLM, abort NOW   │
    │  │        (unconditional override, ignores    │
    │  │         percent-free entirely)              │
    │  └─ NO → Continue                              │
    ├─ pct_free < ABORT_BELOW_PCT (10)?             │
    │  ├─ YES → Kill MLX, stop LiteLLM, abort        │
    │  └─ NO → Continue                              │
    ├─ pct_free < WARN_BELOW_PCT (18)?              │
    │  ├─ YES → Log WARNING, suggest intervention    │
    │  └─ NO → Normal status                         │
    ├─ pct_free < CAUTION_BELOW_PCT (25)?           │
    │  ├─ YES → Log CAUTION                          │
    │  └─ NO → Green                                 │
    └────────────┬──────────────────────────────────┘
                 │
    ┌────────────▼──────────────────────────┐
    │  Check Process Still Alive            │
    ├─ kill -0 $PID 2>/dev/null             │
    ├─ Return code 0?                       │
    │  ├─ YES → Loop again (sleep 5s)       │
    │  └─ NO → Exit cleanly                 │
    └────────────┬──────────────────────────┘
                 │
                 └─→ [Back to top of loop]
```

---

## Threshold design — why percent-free, not wired-GB, and what the real data showed

### Attempt 1: flat wired-GB headroom (abandoned)

The original design (and the first working implementation) used a flat wired-memory GB
threshold: e.g. `abort-at 38` on a 48GB machine, with `start_litellm_proxy.sh` computing that
per-machine as `total_ram_gb - 10`. This didn't hold up:

- The **same** doc originally gave two worked examples — 48GB→38GB abort (10GB headroom) and
  24GB→20GB abort (4GB headroom) — that imply headroom scaling *with* total RAM, not a flat
  10GB. The flat-10GB formula was only checked against the first example.
- On a 16GB machine, `total_ram_gb - 10 = 6GB` abort, `caution_gb = 1GB` — below typical idle OS
  baseline wired memory (~3.5-4GB observed on the actual 48GB test machine; smaller machines
  don't shrink that baseline proportionally, since it's largely fixed OS/kernel overhead, not a
  percentage of RAM). That configuration would fire the monitor before a mid-size model even
  finished loading.

### Attempt 2: percent-free, calibrated against a real failure

Rather than guess a better GB-based formula, the redesign uses **`kern.memorystatus_level`** —
the kernel's own real-time "percent of memory free" figure (0-100, higher = healthier), the same
subsystem macOS's own memory-pressure/jetsam machinery already uses to decide when to reclaim or
kill processes. It's inherently machine-size-normalized: "10% free" means the same thing on a
16GB Mac as a 128GB one, with zero per-machine arithmetic needed at the call site (confirmed by
deleting that arithmetic from both `start_litellm_proxy.sh` and `verify_long_context_memory.sh`
with no loss of function).

**Calibrated against real data, not intuition.** `verify_long_context_memory.sh`'s real
stress test against `qwen3-coder-30b-a3b-instruct-4bit` (see `PROJECT_STATUS.md` and
`mlx_long_context_stress_results.json`) produces a genuine unbounded-memory-growth failure
during prefill. Two independent runs of that same failure, sampled every 5s, gave this real
trajectory (2026-09-27/28, this 48GB machine):

| t (s) | pct_free (`kern.memorystatus_level`) | pressure (`kern.memorystatus_vm_pressure_level`) | wired (GB) | zone |
|---|---|---|---|---|
| 0 | 84 | NORMAL | 3.6 | green |
| 26 | 32 | NORMAL | 28.4 | green |
| 47 | 25 | NORMAL | 31.4 | green |
| 52 | 23 | NORMAL | 32.4 | **caution** |
| 97 | 14 | NORMAL | 36.0 | **warning** |
| 141 | 9 | NORMAL | 38.1 | **abort** — monitor kills the server |

(A second, separate run at a slightly different absolute timeline — see below — showed
`pressure` flip from NORMAL to WARN in its very last sample, essentially concurrent with the
abort, not ahead of it.)

Two real findings from this table drove the final design:

1. **`kern.memorystatus_level` tracks smoothly and monotonically with real memory pressure** —
   84→9 over ~140s as wired grew from 3.6GB to 38.1GB. It gives genuine graduated lead time
   (caution at ~52s, warning at ~97s, abort at ~141s), which is what lets the monitor log
   escalating `[SUGGEST]` messages before the hard kill. **Default thresholds — `--caution-below-pct
   25`, `--warn-below-pct 18`, `--abort-below-pct 10`** — come directly from this table, not
   a guess: 10% free is where wired crossed the old (validated-by-two-real-failures) 38GB point
   on this exact machine.

2. **`kern.memorystatus_vm_pressure_level` (the discrete NORMAL/WARN/CRITICAL enum apps
   subscribe to via `dispatch_source_memorypressure`) is NOT a useful leading indicator.**
   Across two real failure runs, it stayed at NORMAL(1) for the *entire* climb — in one run it
   flipped to WARN(2) in the very last sample (essentially concurrent with, not ahead of, the
   percent-free-based abort); in the other run it never left NORMAL(1) even at the moment of
   abort (pct_free=9%, wired=38.1GB). It offers no graduated warning of its own (binary jump,
   not a smooth signal) and, per Apple's own documentation, WARN is described as an *advisory*
   level raised routinely as a hint for apps to release caches — not necessarily evidence of
   real trouble. Using `pressure > 1` (i.e. including WARN) as a standalone abort trigger risks
   the opposite failure mode from what motivated this investigation: killing the model process
   over a routine, benign pressure blip caused by some *other* app on the system, unrelated to
   what our own process is doing.

   **CRITICAL(4) is the one unambiguous value in that enum** — Apple's docs describe it as the
   level where the OS itself starts aggressively reclaiming/jetsam-killing. So it's kept, but
   only as an **unconditional override**: if `pressure == CRITICAL`, abort immediately
   regardless of what `pct_free` currently reads. This is a backstop for cases the percent-free
   sampling interval might miss (e.g. a sudden spike between samples), not the primary signal.

3. **`vm.swapusage` (real disk-backed swap) is not a usable signal for this specific failure
   mode.** Sampled every 2s throughout a full failing run, `used` stayed completely flat
   (6339.94MB → 6339.94MB, zero change) despite wired memory growing by 34GB. On Apple
   Silicon's unified memory, GPU/Metal-committed buffers apparently aren't the kind of memory
   that gets paged to swap the way ordinary CPU app memory would — waiting for swap growth as
   corroborating evidence would mean waiting for a signal that may simply never arrive before a
   real crash. Not tracked as a signal in `monitor_memory.sh`.

**Sources consulted for sysctl semantics** (used to decide what these values actually mean,
not just what they read on this one machine):
- [kern.memorystatus_level = percent free](https://github.com/anthony-chaudhary/fak/issues/13537)
- [kern.memorystatus_vm_pressure_level / DISPATCH_MEMORYPRESSURE levels (1/2/4, WARN is advisory)](https://developer.apple.com/forums/thread/802771)
- [vm.swapusage meaning](https://forums.macrumors.com/threads/swap-is-used-even-though-50-of-ram-is-still-free.2222254/)

**What this deliberately does NOT claim:** the 25/18/10 percent thresholds are calibrated
against real data from *one* machine (this 48GB M5 Pro) and *one* failure mode (unbounded
prefill growth on a specific MoE model). They haven't been cross-checked on a smaller/larger
Mac or against a different kind of memory event. Being percent-based removes the need to
*rescale* them per machine, but doesn't by itself prove they're the exactly-correct cutoff on
every machine — just a much better-grounded starting point than the flat-GB formula was.

---

## Safety Thresholds & Actions (current)

```
Zone          pct_free range   Status      Monitor Action
──────────────────────────────────────────────────────────────
Green         >= 25%           ✓ OK        Log normally
Amber         18-25%           ⚠️  Caution  "Approaching ceiling" (first entry only)
Red           10-18%           ⚠️  Warning  "Compact context now or restart" (first entry only)
Abort         < 10%,           ❌ Kill      Force stop + cleanup
              OR pressure=CRITICAL
              (unconditional, any pct_free)
```

**Actions per zone:**

- **GREEN (pct_free >= 25%):** log normally, no suggestion.
- **AMBER (18% <= pct_free < 25%):** log `⚠️ CAUTION`; on first entry into this zone, log
  `[SUGGEST] Context approaching ceiling — consider compacting context soon.`
- **RED (10% <= pct_free < 18%):** log `⚠️ WARNING`; on first entry into this zone, log
  `[SUGGEST] Memory critical — compact context now or restart the session.`
- **ABORT (pct_free < 10%, OR pressure == CRITICAL regardless of pct_free):**
  1. Log `[ERROR] CRITICAL: <reason> (percent_free=X% pressure=Y wired=ZGB)`
  2. SIGTERM the monitored PID (the MLX server)
  3. Wait 2s for graceful shutdown
  4. SIGKILL if still alive
  5. Best-effort `pkill -f <litellm-pattern>` to also stop the LiteLLM proxy
  6. Log final wired memory after cleanup, exit 1

Thermal pressure (`pmset -g therm`) is sampled and logged best-effort alongside these zones —
purely informational, not part of the abort decision (see `monitor_memory.sh`'s own comment:
`pmset -g therm` frequently reports no data at all on this machine, so a "Heavy" verdict is
only logged when the tool actually gives one, never fabricated).

---

## Example Monitoring Output

### Normal Session (Safe Progression)

```
$ tail -f /tmp/monitor.log

[INFO] monitor_memory.sh started — watching PID 45823 (interval=5s, caution<25% warn<18% abort<10% free, or pressure=CRITICAL)
[07:23:45] t=00:00  pct_free=84%  pressure=NORMAL  wired=3.6GB  proc=13147600KB  thermal=Unknown  ✓
[07:23:50] t=00:05  pct_free=79%  pressure=NORMAL  wired=6.1GB  proc=13147600KB  thermal=Unknown  ✓
[07:23:55] t=00:10  pct_free=61%  pressure=NORMAL  wired=14.2GB proc=13147600KB  thermal=Unknown  ✓
```

### Real Failing Session (from an actual `verify_long_context_memory.sh` run)

```
$ tail -f /tmp/monitor.log

[21:32:01] t=00:00  pct_free=84%  pressure=NORMAL  wired=3.6GB   thermal=Unknown  ✓
[21:32:48] t=00:47  pct_free=25%  pressure=NORMAL  wired=31.4GB  thermal=Unknown  ✓
[21:32:53] t=00:52  pct_free=23%  pressure=NORMAL  wired=32.4GB  thermal=Unknown  ⚠️  CAUTION
[SUGGEST] Context approaching ceiling — consider compacting context soon.
[21:34:21] t=02:20  pct_free=17%  pressure=NORMAL  wired=35.2GB  thermal=Unknown  ⚠️  WARNING
[SUGGEST] Memory critical — compact context now or restart the session.
[21:34:46] t=02:45  pct_free=9%   pressure=NORMAL  wired=38.1GB  thermal=Unknown  ❌ ABORT
[ERROR] CRITICAL: percent_free dropped below 10% safety threshold (percent_free=9% pressure=NORMAL wired=38.07GB)
[INFO] Stopping MLX server (PID 18054, graceful shutdown)...
[INFO] MLX server still alive after SIGTERM — sending SIGKILL...
[INFO] Terminating LiteLLM proxy (pattern: definitely-not-a-real-process-litellm-not-running-during-this-test)...
[✓] Cleanup complete, wired=3.7GB
[INFO] Process exited cleanly after abort
```

Note `pressure` stayed `NORMAL` through the entire failure, including at the abort moment
itself — direct live confirmation of why it isn't used as the primary/only signal (see
"Threshold design" above).

---

## Monitoring Tuning Parameters (current)

`monitor_memory.sh` accepts:

```bash
./monitor_memory.sh \
  --pid 12345                # Process to monitor (required)
  --interval 5                # Sample every N seconds (default: 5)
  --caution-below-pct 25      # Caution threshold, percent free (default: 25)
  --warn-below-pct 18         # Warning threshold, percent free (default: 18)
  --abort-below-pct 10        # Abort threshold, percent free — OR pressure=CRITICAL
                               # unconditionally (default: 10)
  --verbose                   # Show live output (default: silent, write log only)
  --log-file /tmp/m.log       # Output log location
  --litellm-pattern <pat>     # pkill -f pattern for the LiteLLM proxy on abort
```

**No machine-size tuning example is given here anymore** — the whole point of switching to
percent-free was that these defaults don't need to change between a 16GB MacBook and a 128GB
Mac Studio. If a different tolerance is wanted (e.g. more conservative on a shared/multi-user
machine), tune the percentages directly; there's no GB conversion step.

---

## Integration with LiteLLM Proxy Script (current)

`start_litellm_proxy.sh`:

1. **Parses `--monitor` flag:**
   ```bash
   if [[ "$MONITOR" == "true" ]]; then
       trap stop_memory_monitor EXIT INT TERM
       start_memory_monitor
   fi
   ```

2. **Starts the monitor after the MLX server is confirmed healthy** — no per-machine threshold
   math needed (see "Threshold design" above for why that's gone):
   ```bash
   start_memory_monitor() {
       local mlx_pid
       mlx_pid=$(pgrep -f "$MLX_SERVER_MODULE" | head -1)
       "$SCRIPT_DIR/monitor_memory.sh" \
           --pid "$mlx_pid" \
           --interval 5 \
           --log-file "$MONITOR_LOG" \
           --litellm-pattern "litellm.*$LITELLM_CONFIG" \
           > /dev/null 2>&1 &
       MONITOR_PID=$!
   }
   ```

3. **Cleans up the monitor subprocess on exit:**
   ```bash
   stop_memory_monitor() {
       if [[ -n "$MONITOR_PID" ]] && kill -0 "$MONITOR_PID" 2>/dev/null; then
           kill "$MONITOR_PID" 2>/dev/null || true
       fi
   }
   ```

`verify_long_context_memory.sh` follows the same pattern (no per-machine math, just
`--pid`/`--interval`/`--log-file`/`--litellm-pattern`) since it's the script that actually
produced the calibration data above.

**Known gap:** `--monitor` itself (the flag/wiring in `start_litellm_proxy.sh`) hasn't been
exercised end-to-end against a real interactive launch yet — only `monitor_memory.sh` directly,
and only via `verify_long_context_memory.sh`'s own direct invocation. See `PROJECT_STATUS.md`'s
open items.
