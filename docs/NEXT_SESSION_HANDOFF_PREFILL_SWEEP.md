# Handoff — Comprehensive per-model sweep (prefill step size + diagnostic metrics)

**Written:** 2026-10-02, from a planning-only conversation (no sweep code written yet —
see "What's already built and tested" for what WAS written). This file is
the spec + context for the work that builds the sweep harness. Read the
whole thing before writing code; it front-loads two real bash parser gotchas that would
otherwise cost real time rediscovering.

This is a separate, newer piece of work from `NEXT_SESSION_HANDOFF.md` (mlx-dspark/
speculative-decoding integration, already done) — that file's gotchas section is still
worth reading if you're touching `start_mlx_server.sh`'s backend-branching code, but it's
not required for this task.

---

## Why this exists

Context from the conversation that produced this doc (not written up anywhere else):

`PREFILL_STEP_SIZE=2048` in `start_mlx_server.sh` is a single hardcoded global constant,
derived from sweeping exactly 2 Gemma-4 variants on exactly 1 machine (this one, Apple M5
Pro / 48GB — see `HARDWARE_SPEED_ESTIMATION.md` §12). The project's own research doc
already flags this as too narrow (§17 item #8). Before building a bigger version of that
same manual sweep, we worked through why a single global number can't actually be
correct for every machine:

1. **The TTFT-optimal step size is compute-bound, not memory-bound** — prefill is a
   batched/parallel operation (unlike decode, which is bandwidth-bound), so the sweet
   spot depends on GPU architecture/core count. §12's own finding (2048 beats *both*
   512 and 4096 — non-monotonic) is itself evidence this is architecture-sensitive, not
   a universal constant. An M3 (10-core GPU) has no reason to share M5 Pro's optimum.
2. **The memory-safe ceiling is a live, per-launch quantity, not even a chip property**
   — §12 also measured step size's real memory cost (6.25→6.49→6.71GB as step size
   grows for one model). On 24GB free vs. 48GB free, the same step size has a very
   different safety margin, independent of chip identity.

Conclusion: don't hardcode a new global constant. Build a per-`(chip, model)`
self-calibrating sweep (reusing the calibration-file infrastructure already shipped
this session — see below), with the existing global constant kept as the fallback
default for any `(chip, model)` pair that hasn't been calibrated yet.

Separately: since loading a model and filling its context to various depths is the
expensive part of any sweep, we decided to maximize what gets measured per model-load
rather than running narrow, single-purpose passes repeatedly. The full metrics list is
below.

---

## What's already built and tested (don't redo this)

Three gaps closed, all shipped and passing 107 tests (`pytest -q` from repo root):

1. **`model_advisor.py:55` `CALIBRATION_DIR`** (`~/.bushpilot/calibration/`) +
   **`_calibration_path()` (line 501)**, **`load_calibration()` (line 510)**,
   **`save_calibration()` (line 534)** — per-`(chip, model)` JSON files, atomic
   write (tmp + `os.replace`), last-write-wins. `estimate_speed_tok_s()` (line 568)
   checks `load_calibration()` first and returns a `"measured on this machine"`
   reason string instead of `"estimated"` when one exists.
2. **`model_advisor.py` `main()`'s `--save-calibration MODEL_ID --decode-tok-s N
   [--context-tokens N] [--ttft-s N] [--peak-memory-gb N]`** CLI entry point —
   catalog-independent, just resolves chip name and writes the calibration file.
3. **`start_mlx_server.sh:1149` `save_calibration_snapshot()`**, called from
   **`full_teardown()` (line 1200)** before the server process is killed — curls
   `/metrics` one last time, parses either backend's shape (`mlx_vlm` vs.
   `mlx_dspark` have different JSON layouts, handled via a single Python heredoc
   branching on `$MLX_BACKEND`), calls `model_advisor.py --save-calibration`.
   Best-effort throughout (`|| true` at every step) — never blocks teardown.
4. **`models.json`'s `_schema_notes.speed_correction_factor`** (new optional
   per-model field) — `estimate_speed_tok_s()` reads
   `model_entry.get("speed_correction_factor", SPEED_CORRECTION_FACTOR)` instead
   of the global constant. Not yet populated for any real catalog entry — that
   needs a live measurement, which this sweep should produce as a side effect for
   Gemma-4 at minimum (§17 item #2 already has the two target numbers: 1.29 for
   `gemma-4-e4b-it-4bit`, 0.75 for `gemma-4-12b-it-4bit`).
5. **`recommend_kv_bits_and_context()` (line 660)** returns
   `kv_bits_apc_caveat` (bool) — True only when the quantized KV fallback is
   actually used *and* the model has sliding-window layers
   (`_has_sliding_window_layers()`, line 637). Flags the real upstream gap
   (mlx-vlm issue #1559) non-blockingly in the picker, the launch log, and the
   JSON selection contract. Not yet independently verified against a real
   long-context + `--kv-bits` + APC request on an actual Gemma-4 model — **this
   sweep's APC test (below) is the natural place to finally check that live.**

All of the above only touches `decode_tok_s`/`ttft_s`/`peak_memory_gb` as a single
flat point per `(chip, model)`. The sweep below needs a **richer, separate** file —
see schema design.

---

## The sweep spec

### What to measure, grouped by marginal cost

Everything in Tier 1 comes free from requests the sweep already has to make (filling
context to various depths at various step sizes). Each later tier costs one additional
request type per model.

**Tier 1 — same requests you're already making, just read more fields**

| Metric | Closes |
|---|---|
| TTFT + prompt tok/s at each step size × context depth | §17 #8 |
| Decode tok/s at each context depth (a real curve, not one point) | §17 #9 |
| Real peak memory at each depth, checked against `compute_kv_formula`'s prediction | §17 #4 |
| KV-bits decode-speed cost **at long context**, not just short | §17 #6 / #13 — the sharpest named gap |
| Load time (process start → first healthy response) | not tracked anywhere today |

**Tier 2 — one extra paired request per model**

| Metric | Closes |
|---|---|
| APC: same long prefix twice; hit rate, cached-vs-cold TTFT/decode, exact output match | §17 #11; also finally tests the `kv_bits_apc_caveat` gap live |
| Cold vs. cache-restored output exact-match | upstream mlx-vlm PR #2356 (restored-row regression) — turns a known risk into a per-model regression check |

**Tier 3 — one extra request type, model-dependent**

| Metric | Why |
|---|---|
| Re-run the live tool-call test; record pass/fail | keeps `tool_calling.verdict` fresh instead of a stale one-time check |
| If `multimodal_capable`: TTFT/decode with an image attached | currently undocumented — correctness was verified, speed never was |
| If a drafter's attached: acceptance rate across context depths, not one short-prompt number | `serving.measured_speedup` is currently a single point |

**Tier 4 — whole-session, not per-model**

| Metric | Why |
|---|---|
| Batch throughput at concurrency 1/2/4 | validates `compute_bonus_concurrency`'s recommended slot count against reality, never checked before |
| Decode speed over time across the whole sweep (thermal throttling) | free to log, Apple Silicon fanless/low-fan configs can throttle |
| Does every catalog entry's HF repo still resolve | side effect of touching every model; catches dead repos (§17 #14's NVFP4 case) early |

**Deliberately excluded:** tokenizer round-trip correctness, streaming-vs-non-streaming
parity — no documented gap anywhere motivates testing these; would just inflate runtime.

### Storage design

Two artifacts, kept **separate** so the sweep never has to touch the already-tested
`load_calibration`/`save_calibration`/`estimate_speed_tok_s` code:

1. **New: rich diagnostic file**, one per `(chip, model)`, in a **new**
   subdirectory so it never collides with the existing lightweight files:
   `~/.bushpilot/calibration/sweeps/<chip>__<model_id>.sweep.json`

   ```json
   {
     "schema_version": 1,
     "model_id": "gemma-4-e4b-it-4bit",
     "chip": "Apple M5 Pro",
     "total_ram_gb": 48.0,
     "swept_at": "2026-10-05T12:00:00+00:00",
     "mlx_vlm_version": "0.7.4",
     "mlx_dspark_version": null,

     "load": { "load_time_s": 4.2 },

     "prefill_step_size_sweep": [
       {"step_size": 512,  "context_tokens": 7200, "ttft_s": 1.716, "prompt_tok_s": 4208.3, "decode_tok_s": 72.29, "peak_memory_gb": 6.253},
       {"step_size": 2048, "context_tokens": 7200, "ttft_s": 1.647, "prompt_tok_s": 4385.6, "decode_tok_s": 71.60, "peak_memory_gb": 6.486},
       {"step_size": 4096, "context_tokens": 7200, "ttft_s": 1.823, "prompt_tok_s": 3962.0, "decode_tok_s": 72.08, "peak_memory_gb": 6.710}
     ],

     "decode_speed_vs_context": [
       {"context_tokens": 100,   "decode_tok_s": 82.23, "kv_bits": null, "peak_memory_gb": 2.1},
       {"context_tokens": 7200,  "decode_tok_s": 71.99, "kv_bits": null, "peak_memory_gb": 6.49},
       {"context_tokens": 16384, "decode_tok_s": 68.0,  "kv_bits": null, "peak_memory_gb": 9.1},
       {"context_tokens": 16384, "decode_tok_s": 63.2,  "kv_bits": 8,    "peak_memory_gb": 7.0}
     ],

     "apc_cache_test": {
       "tested": true, "prefix_tokens": 26690,
       "cold_ttft_s": 1.2, "cold_decode_tok_s": 71.0,
       "warm_ttft_s": 0.3, "warm_decode_tok_s": 70.8,
       "hit_rate": 0.98, "output_matches_cold_exactly": true,
       "tested_with_kv_bits_quantized": false
     },

     "tool_calling_reverified": {"verdict": "SUPPORTED", "matches_catalog": true},
     "multimodal_speed": {"tested": false, "reason": "model is text-only"},
     "speculative_decoding": {"drafter_attached": false},

     "concurrency_throughput": [
       {"concurrency": 1, "aggregate_tok_s": 71.6, "per_request_tok_s": 71.6},
       {"concurrency": 2, "aggregate_tok_s": 128.0, "per_request_tok_s": 64.0},
       {"concurrency": 4, "aggregate_tok_s": 210.0, "per_request_tok_s": 52.5}
     ],

     "thermal_check": {
       "requests_run": 40, "first_decode_tok_s": 72.3, "last_decode_tok_s": 70.1,
       "max_decode_tok_s": 73.0, "min_decode_tok_s": 69.8, "throttling_suspected": false
     },

     "catalog_health": {"hf_repo_resolves": true}
   }
   ```

2. **Existing: lightweight calibration file gets 1-2 new optional top-level keys**,
   derived from the sweep above as a condensed summary — this is what
   `start_mlx_server.sh` reads at launch time, so it never needs to parse the rich
   diagnostic file:

   ```json
   {
     "decode_tok_s": 71.6, "context_tokens": 7200, "ttft_s": 1.647,
     "peak_memory_gb": 6.486, "saved_at": "...",
     "prefill_step_size": 2048,
     "kv_bits_safe_at_long_context": true
   }
   ```

   This keeps `load_calibration()`/`estimate_speed_tok_s()` completely untouched —
   they already ignore unknown keys (`load_calibration` only validates
   `decode_tok_s`), so adding `prefill_step_size` is purely additive. A new
   `recommend_prefill_step_size(model_entry, chip_name, free_wired_gb)` function
   (not yet written) would read it, falling back to the global
   `PREFILL_STEP_SIZE` default the same way `recommend_kv_bits_and_context`
   falls back from unquantized to 8-bit — try the calibrated value, check it
   against current free memory using §12's measured memory-scaling relationship,
   fall back if it doesn't fit.

---

## Implementation gotchas (don't rediscover these)

Both bitten while writing `save_calibration_snapshot()`'s Python heredoc in
`start_mlx_server.sh` — the sweep harness will need many more heredocs like it, so
both are very likely to recur if not read first.

1. **An ODD number of literal apostrophes inside a heredoc nested in `$(...)` breaks
   bash's parser**, even though the heredoc delimiter is quoted
   (`<<'PYEOF'`). Minimal repro:
   ```bash
   foo() {
       parsed="$(cat <<'PYEOF'
   ttft_s = ""  # not reported by mlx-dspark's own /metrics
   PYEOF
   )"
   }
   ```
   fails with `unexpected EOF while looking for matching '''`. Removing the single
   apostrophe in "dspark's" fixes it. An EVEN count (e.g. a pair of empty-string
   Python literals `''`) does NOT trigger this — it's specifically an odd/even
   quote-balance miscount in bash's pre-scan of the `$(...)` substitution, not a
   real heredoc-content issue. The existing working heredoc at
   `start_mlx_server.sh:532` avoids this only by using *paired* single quotes
   (`shlex.quote(str(serving.get(key, '') or ''))`) — not by any special escaping.
   **Rule: never write an odd number of apostrophes inside a heredoc that's
   nested inside a command substitution. Rephrase the comment/string instead of
   trying to escape it.**

2. **Forgetting Python's `-` stdin placeholder when invoking it with a heredoc
   AND positional args.** `"$MLX_PYTHON" "$MLX_BACKEND" "$metrics_json" <<'PYEOF'`
   silently tries to execute `$MLX_BACKEND` *as a script file* (Python's first
   bare argument is a filename, not `sys.argv[0]`), producing `can't open file
   'mlx_vlm': No such file or directory` — not a bash syntax error, so `bash -n`
   won't catch it; only a live run will. Fix: `"$MLX_PYTHON" - "$MLX_BACKEND"
   "$metrics_json" <<'PYEOF'` (note the bare `-` right after the interpreter).
   The existing correct pattern is at `start_mlx_server.sh:532`
   (`"$MLX_PYTHON" - "$SCRIPT_DIR/models.json" "$LOCAL_MODEL" <<'PYEOF'`) —
   copy that shape exactly for any new heredoc-with-args.

   **Testing implication: `bash -n` only catches gotcha #1, never #2.** Verify
   any new heredoc with a live mock server (a trivial `http.server` on a free
   port is enough), not just a syntax check — this is how #2 was actually caught
   this session, after `bash -n` reported clean.

---

## Remaining work (not started — this is the actual TODO for the next session)

Roughly in build order:

1. **Decide the sweep's entry point.** Options discussed but not resolved: a new
   `--sweep` flag on `start_mlx_server.sh` (reuses existing bootstrap/download/launch
   plumbing, but that script is already large); or a standalone
   `sweep_model.py`/`sweep_model.sh` that drives the already-running server via
   plain HTTP (simpler, keeps the sweep's own complexity out of the main launcher,
   can be run against a server started normally). Leaning toward the standalone
   script — it can `curl`/`httpx` against an already-launched `mlx_vlm.server` or
   `mlx-dspark serve` instance exactly like `launch_tool.sh` does, without needing
   to know anything about model selection/bootstrapping itself.
2. **Pick the actual test prompts.** §12's methodology (repeated short filler
   sentences) is flagged in its own doc as possibly not representative of real
   code/tool-call content (§17 item #8's "genuinely different prompt content...
   could shift where the sweet spot sits"). Worth sourcing at least one prompt
   shape from this repo's own code (e.g. feed it a real source file) rather than
   only filler text, to actually close that caveat instead of repeating it.
3. **Write the schema writer** — a small Python module (`sweep_result.py`?) that
   builds the rich JSON above incrementally as the sweep runs (so a crash
   mid-sweep doesn't lose everything already measured), plus the derivation step
   that extracts `prefill_step_size`/`kv_bits_safe_at_long_context` into the
   existing lightweight calibration file via `model_advisor.save_calibration()`
   (already supports extra kwargs — just add the two new optional ones there
   first; it's an additive, low-risk change to a function that already has
   tests).
4. **Write `recommend_prefill_step_size()`** in `model_advisor.py`, mirroring
   `recommend_kv_bits_and_context`'s structure (try the calibrated choice, check
   current free memory using the §12 memory-scaling relationship, fall back to
   the global `PREFILL_STEP_SIZE` constant if either no calibration exists yet or
   it doesn't fit). Wire it into `start_mlx_server.sh` next to where `KV_BITS` is
   currently resolved from `advisor_output`.
5. **Decide concurrency-sweep safety.** Tier 4's batch-throughput test
   deliberately runs N requests at once — needs the same memory-guard awareness
   `start_mlx_server.sh`'s existing `check_memory_backstop` already has, so a
   sweep doesn't itself trigger the emergency teardown it's supposed to avoid.
   Probably: run the concurrency sweep at a SMALLER context depth than the
   step-size sweep's deepest point, not the max.
6. **Only after the above works on this M5 Pro machine:** populate
   `models.json`'s new `speed_correction_factor` field for the two Gemma-4
   entries using the real sweep output (targets already known: 1.29 and 0.75,
   per §17 item #2 — but let the sweep re-derive them rather than hardcoding
   those exact historical numbers, in case anything's shifted).

---

## Quick orientation

- Run `pytest -q` from repo root first — should show `107 passed` before you add
  anything. If it doesn't, something regressed since this doc was written.
- `tests/conftest.py` already has a `sliding_window_model` fixture (gemma-4-shaped
  kv_config) and a `dense_model` fixture — reuse both rather than hand-rolling new
  ones for sweep-related tests.
- The existing calibration functions are pure logic + filesystem, no network, no
  real MLX runtime required to test — `tests/test_recommendation_math.py`'s
  calibration tests (`monkeypatch.setattr(ma, "CALIBRATION_DIR", str(tmp_path))`)
  are the pattern to copy for any new schema-writer tests.
- `HARDWARE_SPEED_ESTIMATION.md` §12 has the exact existing step-size sweep
  methodology/table to extend, §13 has the APC measurement methodology to extend,
  §17 and §19's "Open items" lists are the authoritative source for which gaps
  this sweep is meant to close — re-read both before finalizing the schema above,
  in case anything shifts between now and when this is picked up.
- This machine already has real drafters/models downloaded (see
  `NEXT_SESSION_HANDOFF.md`'s "Quick orientation" section for the exact HF repo
  ids) — you should not need to re-download anything to start sweeping Gemma-4.
