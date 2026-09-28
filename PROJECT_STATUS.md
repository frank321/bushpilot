# Project Status — Memory-Aware Local MLX Model Launcher

**Last updated:** 2026-09-28 (end of this working session — added live tool-calling verification
for all 14 models, plus KV-cache quantization decode-speed cost data for 2 models)
**Location:** `~/scripts/` (this project is separate from the `devx-agent-eval` repo, though it
references research originally done there — `LITELLM_ROUTING_PLAN.md` in that repo is the
upstream source for several findings reused here)

---

## The goal

Make `start_litellm_proxy.sh` (a script that launches Claude Code / OpenCode against a local MLX
model via a LiteLLM proxy) **interactive and memory-aware**: detect the machine's hardware and
free memory, recommend a model + context size that's actually safe to run, and enforce that
recommendation as a real server-side cap — not just a client-side hint.

---

## Document map — what to read for what

| If you want... | Read this file |
|---|---|
| **The single best starting point to resume work** | This file (`PROJECT_STATUS.md`), then the "Resume prompt" section at the bottom |
| The current, real, verified model catalog (14 models) | `models.json` — the source of truth; every entry has a `verification` block AND (new this session) a `tool_calling` block |
| Which models actually work with OpenCode/Claude Code tool use (webfetch, bash, edit, etc.) | `models.json`'s `tool_calling.verdict` field, or the "What's actually done" #13 below for the summary + methodology |
| Raw verification data behind `models.json` (short-prompt admission gate) | `mlx_verification_results.json` |
| Raw data behind the real long-context stress test (only 1 of 14 models tested so far) | `mlx_long_context_stress_results.json` + `verify_long_context_memory.sh` (the tool itself) |
| **Why** `--max-kv-size` matters and how it was verified | `verify_max_kv_size.sh` (the tool itself, heavily commented) + its header comment explains the 3-phase methodology |
| The token-speed / context-size **formula research** (sourced, not guessed) | `HARDWARE_SPEED_ESTIMATION.md` — read this before touching `model_advisor.py`'s formulas |
| The actual hardware-aware recommender implementation | `model_advisor.py` |
| The main launcher script (now wired to the advisor) | `start_litellm_proxy.sh` |
| User-facing usage guide | `LITELLM_PROXY_USAGE.md` — **partially stale**, see caveat below |
| Original code-snippet design notes | `IMPLEMENTATION_NOTES.md` — **mostly superseded**, see caveat below |
| Background memory-monitor design + implementation, including the percent-free threshold redesign and its real calibration data | `MONITORING_ARCHITECTURE.md` (rewritten this session — current, not just historical) and `monitor_memory.sh` (see "What's actually done" #10 and #12) |
| Quick reference / TL;DR | `QUICK_START.md` — **partially stale**, see caveat below |

### ⚠️ Staleness caveats — read before trusting these files

- **`LITELLM_PROXY_USAGE.md`**'s "Part 2" section describes a fixed-tier model menu
  (8-12GB/12-16GB/etc.) designed *before* we discovered `--max-kv-size` actually works. That
  design is superseded by `model_advisor.py`'s live per-machine computation. The file hasn't been
  rewritten to match — treat its tier tables as historical, not current behavior.
- **`IMPLEMENTATION_NOTES.md`** has code snippets for the same pre-`max-kv-size` design (fixed
  `models.json` schema without verification data, a different menu-rendering approach). Mostly
  superseded by the real `model_advisor.py` + `models.json` now in place.
- **`QUICK_START.md`** references the old fixed-tier examples. Still roughly right in spirit, but
  specific numbers/examples predate the real implementation.
- **`MONITORING_ARCHITECTURE.md`** is CURRENT again, not just historical intent — it was rewritten
  this session (see "What's actually done" #12) to match the real implementation after the
  threshold design itself changed (wired-GB → percent-free + CRITICAL-pressure override), and now
  includes the real calibration data and sources behind that change, not just the original
  hypothetical design.

None of these are urgent to fix — the code (`models.json`, `model_advisor.py`,
`start_litellm_proxy.sh`, `verify_max_kv_size.sh`) is the real, current source of truth regardless
of what the prose docs say. Fix the docs opportunistically, not before the next feature.

---

## What's actually done (verified, not assumed)

1. **Root-caused why Ollama's MLX runner can't cap context** — confirmed via upstream GitHub
   issues that this is intentional (maintainers rejected the fix twice), not a bug being worked on.
2. **Found and verified `mlx_vlm.server --max-kv-size` as the real alternative** — live-tested,
   not just read about: it correctly rejects oversized requests with HTTP 400 rather than silently
   ignoring them.
3. **Built `verify_max_kv_size.sh`** — a repeatable 3-phase test (moderate bound, high bound,
   extreme-low bound matching upstream's own bug-report scale) that any candidate model must pass
   before being trusted. Found and fixed two real bugs in this script itself along the way:
   - A stale-process race between phases that could silently validate against the wrong bound
   - A resume-less download that left abandoned `.incomplete` blobs after a timeout-kill
4. **Verified 14 models SAFE** across the whole memory spectrum (0.6GB to 16GB), spanning distinct
   architectures (`qwen2`, `qwen3`, `qwen3_5`, `gemma4`/`gemma4_unified`, `phi3`, MoE) — see
   `models.json` for the full list with real sizes, architectural context ceilings, and per-model
   usage notes (tagline/good-for/2 usage examples/known issues, as requested).
5. **Researched and validated a token-speed estimation formula** — sourced published Apple
   Silicon memory-bandwidth figures, cross-checked against 3 live measurements on this exact
   machine (M5 Pro), confirmed the memory-bandwidth-bound physics, and confirmed MoE models need
   active-parameter scaling (not total size) with a real 5.4x discrepancy caught live. Full
   derivation in `HARDWARE_SPEED_ESTIMATION.md`.
6. **Built an exact (not estimated) context-size-recommendation formula** — computed directly from
   each model's own `config.json`, correctly handling hybrid architectures (Qwen3.5's
   linear/full-attention layers, Gemma-4's sliding/full-attention layers) rather than naively
   counting all layers.
7. **Built `model_advisor.py`** — implements both formulas, detects real hardware/memory live,
   renders an interactive menu, outputs a clean machine-readable selection.
8. **Wired it into `start_litellm_proxy.sh`** via `--interactive` — found and fixed a critical gap
   in the process: `CONTEXT_WINDOW` was previously only passed to the *client*, never to the MLX
   server itself, meaning the server ran with no real cap at all until this session's fix.
9. **End-to-end tested the front-end flow** (hardware detection → menu → selection → variable
   propagation → cached-model detection) — found and fixed a real bug where Python's `input()`
   prompt was leaking into stdout and corrupting the JSON handoff to bash.
10. **Built `monitor_memory.sh`** per `MONITORING_ARCHITECTURE.md`'s design — samples wired memory
    (parsed from `vm_stat`, page-size-aware) and monitored-process RSS on an interval, classifies
    into green/amber/red/abort zones, and on the abort zone sends SIGTERM→(2s)→SIGKILL to the MLX
    server plus a best-effort `pkill` of the LiteLLM proxy, then exits non-zero. GPU/CPU thermal
    reads via `pmset -g therm` are best-effort — confirmed live on this M5 Pro that `pmset` often
    reports no thermal data at all, so the script reports "Unknown" rather than fabricating a
    verdict when that happens. Live-tested (not just written): normal sampling against a real
    process, clean exit when the monitored process dies on its own, and the full abort+cleanup
    path with an artificially low threshold — all three verified working. Wired into
    `start_litellm_proxy.sh` via a new `--monitor` flag, which finds the running `mlx_vlm.server`
    PID, derives abort/warn/caution thresholds from this machine's total RAM (10GB headroom below
    total → abort, matching the 48GB→38GB example in the design doc), and cleans up the monitor
    subprocess on exit via `trap`. `--monitor` itself (the flag/wiring in
    `start_litellm_proxy.sh`) still hasn't been exercised end-to-end — but `monitor_memory.sh`
    the underlying tool HAS since been proven against the real MLX server, twice, in a genuine
    danger scenario, not a synthetic one — see item #11 below.
11. **Ran a real long-context memory stress test** (`verify_long_context_memory.sh`, new this
    session) against `qwen3-coder-30b-a3b-instruct-4bit` — the one model explicitly flagged with
    an unresolved historical memory-safety concern — using the SAME production flags
    `start_litellm_proxy.sh` actually launches with (`--max-kv-size 131072 --kv-bits 8
    --kv-group-size 64 --quantized-kv-start 0 --kv-quant-scheme uniform`), with
    `monitor_memory.sh` watching the real server PID as the safety net throughout.
    **Result: CONFIRMED UNSAFE above ~28K tokens, not just an unresolved old flag.** A
    28,276-token prompt completed safely (wired settled ~21GB). A 40,000-token-target probe and
    the escalation's next step (58,982-token-target, 45% of the cap) BOTH failed the same way:
    wired memory grew unboundedly through prefill — NOT proportional to token count, since both
    failures hit the ~38GB abort threshold at a similar ~113-117s elapsed mark despite the
    different target sizes, pointing at elapsed-prefill-time/chunk-count rather than KV-cache
    size as the real driver — and `monitor_memory.sh` had to forcibly kill the server both times
    before the request ever completed. Along the way, caught and fixed a real bug in the test
    itself: its first version used a random-hex-string secret as the recall probe and treated a
    wrong recall as a hard abort signal, conflating "model mis-copied a hard-to-tokenize random
    string" (confirmed via a controlled A/B to be unrelated to KV quantization — the model
    recalls a fixed secret correctly at the same context size either way) with "server memory
    is unsafe." Fixed by switching to a digits-only nonce and gating escalation on
    memory/transport safety only, with recall accuracy now tracked as a separate informational
    field. `models.json`'s entry for this model has been updated with the confirmed finding, a
    new `max_context_verified_safe: 28276` field, and a `long_context_verification` block. Full
    per-step data in `mlx_long_context_stress_results.json`. **Only this one model has been
    tested this way so far** — the other 13 "SAFE" verdicts in `models.json` still only reflect
    the short-prompt admission-gate test; see the new open item below.
12. **Redesigned `monitor_memory.sh`'s abort trigger from wired-memory GB to percent-free,
    calibrated against real failure data, after a user question exposed a real flaw in the
    first design.** The original flat `total_ram_gb - 10` headroom formula was checked only
    against `MONITORING_ARCHITECTURE.md`'s own 48GB example and quietly broke on smaller
    machines (16GB → abort at 6GB, below typical idle OS baseline, meaning the monitor would
    fire before a mid-size model even finished loading). Researched macOS's real memory-pressure
    sysctls online and confirmed live on this machine: `kern.memorystatus_level` (0-100 percent
    free, the same figure macOS's own jetsam/memorystatus subsystem uses) tracks smoothly and
    monotonically with real pressure and is inherently machine-size-normalized — no per-machine
    math needed. `kern.memorystatus_vm_pressure_level` (the discrete NORMAL/WARN/CRITICAL enum)
    was tested too and found NOT useful as a leading indicator — across two real failure runs it
    stayed NORMAL almost the entire time, only reaching WARN essentially concurrent with (not
    ahead of) the percent-free-based abort point, consistent with Apple's own docs describing
    WARN as a routine advisory, not a crisis signal. `vm.swapusage` was also checked live and
    stayed completely flat through a 34GB wired-memory climb — not usable for this failure mode
    on Apple Silicon's unified memory. Final design: percent-free drives graduated
    caution(<25%)/warn(<18%)/abort(<10%) zones — thresholds taken directly from where they
    landed on the real calibration data, not guessed — plus an unconditional
    `pressure == CRITICAL(4)` override as a backstop. Removed the now-unnecessary per-machine GB
    computation from both `start_litellm_proxy.sh` and `verify_long_context_memory.sh`. Re-ran
    the real qwen3-coder-30b failure end-to-end with the new monitor and confirmed it still
    catches it (aborted at percent_free=9%, wired=38.07GB — matching the old threshold's real
    danger point almost exactly, just without needing to know this machine's specific RAM size to
    get there). Full design writeup, calibration table, and sources now in
    `MONITORING_ARCHITECTURE.md` (rewritten this session — see its own header for what changed).
13. **Live-tested tool-calling support for all 14 models** (new this session) — not the same
    thing as `verification`/KV-size safety above, and not determined by static template
    inspection either. For each model: started a real `mlx_vlm.server` instance with production
    flags, sent a real `/v1/chat/completions` request with a `web_search` tool schema attached
    (`temperature=0`), and checked whether `finish_reason` came back `tool_calls` with a real
    parseable function call, vs. `stop` with the model just answering in plain text (meaning
    OpenCode/Claude Code's actual tools — webfetch, bash, edit, etc. — would silently no-op).
    **Result: 9 of 14 SUPPORTED, 5 of 14 NOT_SUPPORTED — and the 5 failures are not evenly spread,
    they're two specific, identifiable causes:**
    - **The entire `qwen2.5-coder-*-instruct` family fails** (3B, 7B-3bit, 7B-4bit, 14B — all
      four sizes tested, all four fail). `mlx_vlm` auto-assigns them the `json_tools` parser
      (their chat template's markers match it), but the models themselves don't reliably emit the
      `<tool_call></tool_call>` tags their own template instructs them to use — each size fails
      with a *different* wrong wrapper instead (`<tools>`/`<response>` for 3B, a fenced
      ` ```json ` block for 7B-3bit, a fenced ` ```xml `-wrapped `<tools>` block for 7B-4bit, bare
      `<tools>` for 14B). Confirmed this is a property of the model's own fine-tuning, not an
      `mlx_vlm`-specific bug, three ways: (a) `qwen3-14b-4bit` uses the exact same `json_tools`
      parser and works perfectly — rules out the parser itself; (b) an independent third party
      hit the identical symptom (`finish_reason=stop`, empty `tool_calls`) against **vLLM** with
      this same model family — see `github.com/QwenLM/Qwen3-Coder/issues/180`; (c) re-tested
      `qwen2.5-coder:3b-instruct` directly against **Ollama 0.33.2** (a completely different
      harness, llama.cpp-based, Q4_K_M GGUF instead of MLX) on this machine — same failure, same
      malformed-JSON-as-plain-text result, on both its OpenAI-compat and native `/api/chat`
      endpoints. Three independent serving stacks, same model family, same failure.
    - **`phi-4-mini-instruct-4bit` fails for the opposite reason** — this is a genuine
      serving-stack gap, not a model limitation. Microsoft's own model card documents real
      function-calling support for this exact model and benchmarks it on the Berkeley
      Function-Calling Leaderboard; its `added_tokens.json` has dedicated
      `<|tool|>`/`<|/tool|>`/`<|tool_call|>`/`<|/tool_call|>`/`<|tool_response|>` tokens. But
      `mlx_vlm`'s `tool_parsers` registry (`mlx_vlm/tool_parsers/__init__.py`'s
      `_TEMPLATE_MARKERS`) has no entry matching Phi's specific token format, so
      `_infer_tool_parser()` returns `None` and `mlx_vlm` never even tries to parse tool calls out
      of its output — confirmed live: `prompt_tokens` was suspiciously small (26), meaning the
      tool schema wasn't even rendered into the prompt. vLLM has a dedicated
      `--tool-call-parser phi4_mini_json` for this exact model (merged
      `vllm-project/vllm#14886`), but that has no `mlx_vlm` equivalent, and vLLM itself has no
      real Apple Silicon GPU acceleration (every Metal/MPS PR has been closed/abandoned as of
      2026-09) — so switching serving stacks just to fix this one model isn't practical here.
    - The 9 SUPPORTED models span every architecture family in this list except
      `qwen2`/`phi3` (which are exactly the 2 architectures that failed) — `qwen3_5` (both
      `qwen3_coder`-parser sizes), `qwen3` plain, and `gemma4`/`gemma4_unified` (all 4 Gemma
      variants) all passed on the first attempt, no retries needed.
    - `models.json` updated with a `tool_calling` block per model (verdict/method/parser/notes,
      same style as `verification`) plus a new `_schema_notes.tool_calling` entry explaining the
      full methodology and cross-stack evidence. The 5 NOT_SUPPORTED models also got an explicit
      `❌ TOOL CALLS DO NOT WORK` warning appended to their `display.known_issues` text — the field
      actually shown in the interactive picker, unlike the schema notes.
14. **Confirmed `--kv-bits 8` (the KV-cache quantization flag `start_litellm_proxy.sh` launches
    every model with unconditionally) has a real decode-speed cost at short context, and that
    cost is NOT the same across model sizes** — full methodology, data, and analysis now in
    `HARDWARE_SPEED_ESTIMATION.md` §10 (this is formula-research territory, not a one-off
    finding, since it's meant to feed a future dynamic KV-bit-selection feature). Same
    fixed-prompt benchmark as this doc's existing speed-calibration data (§6): tested
    `Qwen2.5-Coder-3B-Instruct-4bit` and `Qwen3.5-9B-MLX-8bit`, each with vs. without the
    production `--kv-bits 8 --kv-group-size 64 --quantized-kv-start 0` flags. Result: **3B model
    is 8.7% SLOWER decode with KV quantization on (125.8→114.8 tok/s); 9B model is only ~1.0%
    slower (30.3→30.0 tok/s, within run-to-run noise).** `peak_memory` was identical with/without
    quantization on both models — confirms this is pure per-step quantize/dequantize compute
    overhead, not a memory-savings trade-off, at this short (150-output-token) context length;
    quantization's actual memory benefit only shows up once the KV cache itself is large, which a
    short exchange never reaches. The mechanism (not yet independently confirmed, but the
    straightforward explanation): quantization overhead is roughly fixed-cost per decode step,
    but a bigger model's dense compute per token is proportionally much larger, so the same fixed
    overhead is a smaller fraction of total time for bigger models — the opposite of the naive
    assumption that a bigger KV cache means bigger quantization savings/cost. Only 2 models (both
    dense, one short context length) measured so far — see new open item below and
    `HARDWARE_SPEED_ESTIMATION.md` §11's item 6 for what's needed before this becomes an actual
    dynamic-selection feature.

---

## What's NOT done yet (explicitly open, not silently skipped)

1. **`model_advisor.py` does not know about the confirmed long-context memory-safety failure
   above, and will keep recommending unsafe context sizes.** It computed `recommended_context:
   262144` for `qwen3-coder-30b-a3b-instruct-4bit` on this 48GB machine (pure free-memory/KV-size
   math), but real testing found the model is unsafe above ~28K tokens regardless of free memory
   — this is a software behavior (unbounded growth during prefill), not a memory-availability
   problem, so no amount of free RAM fixes it. `models.json` now has a
   `max_context_verified_safe` field for this model, but `model_advisor.py`'s
   `recommend_max_context()` doesn't read it yet — needs to clamp its recommendation to that
   field when present, for any model, not just this one.
2. **The other 13 models in `models.json` still only have the short-prompt admission-gate
   verdict** — only `qwen3-coder-30b-a3b-instruct-4bit` has been through a real long-context
   stress test so far (see "What's actually done" #11). Given what was just found on that model,
   the other 13 "SAFE" verdicts should not be read as "verified safe at long context" — they
   haven't been tested that way at all yet. `verify_long_context_memory.sh` is now built and
   working; it just needs to be run against the rest of the catalog.
3. **`--max-kv-size` + `--kv-bits` combination not separately re-verified for the ADMISSION GATE
   specifically** — item #11's stress test only sent under-cap prompts (never tried to trigger a
   rejection while KV quantization was active), so whether `--max-kv-size` still correctly
   REJECTS an oversized request with quantization enabled remains untested. `verify_max_kv_size.sh`
   itself still only tests `--max-kv-size` alone.
4. **`gemma-4-26b-a4b-it-4bit`'s speed estimate is `null`** — its MoE config is missing
   `num_experts_per_tok`, so there's no way to compute its active-weight fraction without a
   dedicated live benchmark (unlike `qwen3-coder-30b-a3b-instruct-4bit`, which has one).
5. **Multi-turn / gradual context growth untested** — every verification so far is single-request.
   A real coding session's context grows turn-by-turn; whether the admission gate and the
   recommended-context math hold up under that pattern hasn't been checked.
6. **Full live end-to-end launch not yet run** — the front-end (menu → selection → download-check)
   is tested; actually running `postgres` → `mlx_vlm.server` → `litellm` → `claude`/`opencode` with
   an interactively-selected model has not been executed in this session (only a partial dry-run
   that deliberately stopped before touching those services).
7. **Minor, non-blocking:** `bootstrap_mlx`'s package-existence check for `mlx-lm` always reports
   "not installed" and reinstalls it every run, because `import mlx-lm` (with the literal hyphen)
   is invalid Python — the check should strip/translate the hyphen before the `import` test. Found
   during testing, not yet fixed. Harmless (just wastes ~10-20s per launch) but worth a one-line fix.
8. **`model_advisor.py` does not know about the tool-calling verdicts either** (same pattern as
   open item #1's `max_context_verified_safe` gap) — it will still recommend
   `qwen2.5-coder-*-instruct` or `phi-4-mini-instruct-4bit` for coding tasks with no warning that
   OpenCode/Claude Code's actual tools (webfetch, bash, edit, etc.) will silently no-op on them.
   `models.json` now has a `tool_calling.verdict` field per model (see "What's actually done"
   #13), but nothing in `model_advisor.py` reads it yet — it should at minimum warn, and arguably
   deprioritize/exclude `NOT_SUPPORTED` models, when the launch mode is `opencode`/`claude` (a
   non-tool-calling chat mode wouldn't need this filter at all, so this should be conditional on
   mode, not a blanket exclusion).
9. **Dynamic `--kv-bits` selection is a documented finding, not yet a feature.** Per item #14
   above, KV quantization costs real decode speed on smaller models at short context for no
   memory-safety benefit, but `start_litellm_proxy.sh` still passes `--kv-bits 8` unconditionally
   to every model regardless of size or requested context. Needs the additional calibration data
   `HARDWARE_SPEED_ESTIMATION.md` §11 item 6 describes (an MoE model, a 14B+ model, and a genuine
   long-context run to find the actual crossover point) before a real heuristic can be wired into
   `model_advisor.py`.

---

## Files in `~/scripts/` and their current role

```
models.json                      ← source of truth: 14 verified models + display notes +
                                    tool_calling verdicts (new this session)
mlx_verification_results.json    ← raw per-model verify_max_kv_size.sh results
verify_max_kv_size.sh            ← the verification tool (run this before trusting a new model)
model_advisor.py                 ← hardware detection + context/speed recommendation + menu
start_litellm_proxy.sh           ← main launcher, --interactive wired to model_advisor.py,
                                    --monitor wired to monitor_memory.sh
monitor_memory.sh                ← background memory-pressure safety monitor (percent-free +
                                    CRITICAL-pressure override, live-tested); see
                                    MONITORING_ARCHITECTURE.md for the design + real data
verify_long_context_memory.sh    ← real long-context memory stress test (new this session) —
                                    escalating near-cap-fill prompts + monitor_memory.sh as the
                                    safety net; only run against qwen3-coder-30b-a3b so far
mlx_long_context_stress_results.json ← raw per-model results from the script above

HARDWARE_SPEED_ESTIMATION.md     ← research backing model_advisor.py's formulas (current)
PROJECT_STATUS.md                ← this file
LITELLM_PROXY_USAGE.md           ← user guide (partially stale, see caveats above)
IMPLEMENTATION_NOTES.md          ← old design notes (mostly superseded)
MONITORING_ARCHITECTURE.md       ← monitor_memory.sh design (current — rewritten this session
                                    with the real percent-free calibration data)
QUICK_START.md                   ← quick reference (partially stale)
```

---

## Resume prompt

Paste this into a new session to pick up where this one left off:

```
I'm continuing work on a memory-aware local MLX model launcher in ~/scripts/. Please read
~/scripts/PROJECT_STATUS.md first — it has a document map, what's verified vs. still open, and
which other docs in that directory are stale vs. current. Don't just skim it, actually follow its
document map to the specific files it points to before making changes.

Short version: models.json has 14 real models verified SAFE via verify_max_kv_size.sh's 3-phase
test (short-prompt admission gate only). model_advisor.py does live hardware detection + an exact
context-size formula + an estimated speed formula, and start_litellm_proxy.sh --interactive is
wired to it, including passing --max-kv-size to the actual MLX server. monitor_memory.sh
(background memory-pressure safety monitor) is built and has now been proven live against the
real MLX server, three times, in genuine danger scenarios — it correctly killed a runaway server
each time before things got worse. It's wired into start_litellm_proxy.sh via a --monitor flag,
though that specific flag/wiring itself is still untested end-to-end.

monitor_memory.sh's abort trigger was redesigned mid-session from a flat wired-memory GB
threshold to kern.memorystatus_level (percent-free, machine-size-normalized) plus an
unconditional kern.memorystatus_vm_pressure_level==CRITICAL override, after the original flat-GB
formula was found to break down on smaller machines (guessed 10GB headroom didn't match
MONITORING_ARCHITECTURE.md's own second worked example, and would leave almost no margin on a
16GB Mac). The new thresholds are calibrated against real failure data, not guessed — full
writeup, calibration table, and sources in MONITORING_ARCHITECTURE.md (rewritten this session,
now current rather than just historical design intent).

A real long-context memory stress test (verify_long_context_memory.sh, new this session) found a
CONFIRMED memory-safety failure in qwen3-coder-30b-a3b-instruct-4bit: safe up to 28,276 prompt
tokens, but unbounded wired-memory growth during prefill above that (not proportional to token
count — driven by elapsed prefill time/chunk count instead), monitor-killed before ~38GB both
times it was pushed further. models.json has been updated with the confirmed finding and a new
max_context_verified_safe field for this model, but model_advisor.py does NOT yet read that field
and will still recommend unsafe context sizes for it based on free memory alone. Only this one
model (of 14) has been through this real stress test so far.

Also this session: live-tested tool-calling (OpenCode/Claude Code's actual tools — webfetch, bash,
edit, etc.) for all 14 models, by sending a real /v1/chat/completions request with a web_search
tool schema to each and checking whether it came back as a proper tool_calls response. Result: 9
of 14 SUPPORTED, 5 NOT_SUPPORTED. All 4 qwen2.5-coder-*-instruct sizes fail identically — the
model doesn't reliably emit the <tool_call> tags its own chat template instructs, confirmed as a
model-level (not mlx_vlm-specific) issue by reproducing the same failure on vLLM (a third party
independently hit it, see github.com/QwenLM/Qwen3-Coder/issues/180) and on Ollama/llama.cpp
(tested directly on this machine). phi-4-mini-instruct-4bit fails for the opposite reason — it's a
genuine mlx_vlm gap: Microsoft's own docs confirm this model supports function calling, but
mlx_vlm's tool_parsers registry has no entry matching its <|tool|>/<|tool_call|> token format, so
it never even tries to parse tool calls out of its output. models.json now has a tool_calling
block per model (same style as verification), but model_advisor.py does NOT read it either — same
gap pattern as max_context_verified_safe above.

Also this session: confirmed --kv-bits 8 (the KV quantization flag start_litellm_proxy.sh passes
to every model unconditionally) has a real decode-speed cost at short context, and that cost is
NOT uniform across model sizes. Same fixed-prompt benchmark as HARDWARE_SPEED_ESTIMATION.md's
existing speed-calibration data: Qwen2.5-Coder-3B-Instruct-4bit is 8.7% slower with KV
quantization on (125.8→114.8 tok/s decode); Qwen3.5-9B-MLX-8bit is only ~1.0% slower (30.3→30.0
tok/s, within noise). peak_memory was identical with/without quantization on both — confirms this
is pure per-step compute overhead, not a memory trade-off, at this short context length. Full
data, mechanism discussion, and open items (need an MoE model, a 14B+ model, and a genuine
long-context run before this becomes a real dynamic-selection feature) are in
HARDWARE_SPEED_ESTIMATION.md §10-11.

Next priorities, in the order PROJECT_STATUS.md's "What's NOT done yet" section lists them:
1. model_advisor.py needs to clamp recommendations to a model's max_context_verified_safe field
   when present, instead of only computing from free memory/KV-cache size.
2. The other 13 models in models.json haven't been through the real long-context stress test yet
   (verify_long_context_memory.sh is built and working — just needs to be run against the rest).
3. --max-kv-size's ADMISSION GATE (rejecting an oversized request) hasn't been separately
   re-verified with --kv-bits quantization active — this session's stress test only sent
   under-cap prompts.
4. gemma-4-26b-a4b-it-4bit's speed estimate is null (missing num_experts_per_tok).
5. Multi-turn / gradual context growth untested.
6. The full live launch (postgres → mlx_vlm.server → litellm → claude/opencode) with an
   interactively-selected model hasn't actually been run end-to-end yet — only tested up through
   model selection + download-check. This would also be the first real test of --monitor (the
   flag/wiring, as opposed to monitor_memory.sh itself) against the actual MLX server.
7. Minor, non-blocking: bootstrap_mlx's mlx-lm package-existence check always reports "not
   installed" due to the hyphen-vs-import-name mismatch.
8. model_advisor.py should warn on (and arguably deprioritize/exclude, when launch mode is
   opencode/claude specifically) models with tool_calling.verdict == NOT_SUPPORTED
   (qwen2.5-coder-3b/7b-3bit/7b-4bit/14b-instruct and phi-4-mini-instruct-4bit) — it doesn't
   consult that field at all right now.
9. Dynamic --kv-bits selection needs more calibration data (an MoE model, a 14B+ model, a genuine
   long-context run to find the real crossover point) before it can become an actual
   model_advisor.py feature rather than just a documented finding.
8. model_advisor.py should warn on (and arguably deprioritize/exclude, when launch mode is
   opencode/claude specifically) models with tool_calling.verdict == NOT_SUPPORTED
   (qwen2.5-coder-3b/7b-3bit/7b-4bit/14b-instruct and phi-4-mini-instruct-4bit) — it doesn't
   consult that field at all right now.

Ask me which of these (or something else) to work on next rather than assuming.
```
