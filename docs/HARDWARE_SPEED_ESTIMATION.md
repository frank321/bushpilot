# Hardware Detection & Token-Speed Estimation — Research Notes

**Status:** Research complete, validated against live measurements on this machine.
Implemented in `model_advisor.py`, called by `start_mlx_server.sh` (renamed from
`start_mlx_server.sh` — see PHASE_PLAN.md Phase 4). This document exists so the
formula below is traceable back to its sources and isn't reinvented or drifted from
silently later.

**Owner:** this repo (`bushpilot`)
**Related:** `MLX_SERVER_USAGE.md` (renamed from `LITELLM_PROXY_USAGE.md` — interactive
menu design), `models.json` (per-model metadata this feeds into),
`verify_max_kv_size.sh` / `mlx_verification_results.json` (separate verification —
confirms `--max-kv-size` is *enforced*, not how *fast* a model runs).

---

## 1. What problem this solves

Two related but distinct estimation needs for the interactive model-picker:

1. **Token generation speed** — so the menu can tell a user "~65 tok/s on your machine" instead
   of nothing, without running a live benchmark for every model on every machine (too slow for an
   interactive launch).
2. **Recommended context size** — so the menu can compute a safe `--max-kv-size` from *this
   machine's* current free wired memory and *this model's* actual KV-cache cost per token, instead
   of a fixed tier table that's wrong the moment the user opens another app.

Both need to be honest about being **estimates**, sourced from real data where possible, and never
presented as measured fact for a combination that wasn't actually measured.

---

## 2. Hardware detection — confirmed live, macOS only

```bash
system_profiler SPHardwareDataType SPDisplaysDataType 2>/dev/null
```

Confirmed live on this machine (a MacBook Pro, chip reported as "Apple M5 Pro"):

```
Chip: Apple M5 Pro
Total Number of Cores: 18 (6 Super and 12 Performance)
Memory: 48 GB
...
Total Number of Cores: 20        ← GPU core count, under SPDisplaysDataType
```

**Timing confirmed live:** `system_profiler SPHardwareDataType SPDisplaysDataType` completes in
**0.45s** — fast enough to run on every script launch without a noticeable delay, no caching
needed.

`sysctl` gives a faster subset (chip name, CPU core split, total RAM) if GPU core count isn't
needed for a given code path:
```bash
sysctl -n machdep.cpu.brand_string      # "Apple M5 Pro"
sysctl -n hw.perflevel0.physicalcpu     # 6  (performance cores)
sysctl -n hw.perflevel1.physicalcpu     # 12 (efficiency cores)
sysctl -n hw.memsize                    # bytes, total RAM
```

Wired memory detection (already used by `verify_max_kv_size.sh` and the earlier
`LITELLM_ROUTING_PLAN.md` §4.17 finding that `wired`, not "free"/"used", is the metric that
predicts real memory pressure):
```bash
vm_stat | awk '/Pages wired down/ {gsub("\\.","",$4); printf "%.2f", $4*16384/1024/1024/1024}'
```
(16384 = this machine's page size, confirmed via `vm_stat`'s own header line and `pagesize`; not
universal — Intel Macs use 4096-byte pages, so a real implementation must read the page size from
`vm_stat`'s header rather than hardcoding 16384.)

---

## 3. Token speed is memory-bandwidth-bound — established methodology, not invented here

Confirmed via multiple independent sources during research (not assumed):

> "speed scales with memory bandwidth, not GPU cores" — M5 Max's 2x bandwidth advantage over M5
> Pro translates directly to ~2x generation speed on identical models.
> — [PromptQuorum: M5 Pro vs M5 Max 2026 Benchmark Comparison](https://www.promptquorum.com/local-llms/m5-pro-max-llm-benchmarks-2026)

> Since LLM inference is memory-bandwidth bound (not compute bound), this bandwidth gap directly
> affects tokens/second — performance scales approximately proportionally with memory bandwidth
> across different Apple Silicon chips.
> — cross-referenced summary from general Apple Silicon LLM benchmark search, 2026-09-27

This matches the standard explanation for single-batch (batch-size-1) autoregressive decode on any
memory-bandwidth-limited accelerator: each generated token requires reading essentially all active
model weights (plus the KV cache) once, so throughput is bounded by
`memory_bandwidth / bytes_read_per_token`, not by raw compute (FLOPs) — the GPU is normally waiting
on memory during decode, not compute-bound, for consumer-scale local inference.

**Corollary, confirmed important below:** for a Mixture-of-Experts (MoE) model, only the *active*
experts for a given token get read from memory — not the full on-disk parameter count. Using total
on-disk size for a MoE model's speed estimate is not a minor rounding error, it's off by a large,
predictable factor (see §5).

---

## 4. Real published memory bandwidth table (Apple Silicon, M1–M6)

Cross-referenced from two independent sources during research (2026-09-27); the M5 Pro/Max figures
were confirmed consistent between both:

| Chip | Bandwidth (GB/s) |
|---|---|
| M1 | 68.25 |
| M1 Pro | 200 |
| M1 Max | 400 |
| M1 Ultra | 800 |
| M2 | 100 |
| M2 Pro | 200 |
| M2 Max | 400 |
| M2 Ultra | 800 |
| M3 | 100 |
| M3 Pro | 150 |
| M3 Max | 300–400 (14-core CPU/30-GPU vs 16-core/40-GPU variant) |
| M3 Ultra | 819 |
| M4 | 120 |
| M4 Pro | 273 |
| M4 Max | 546 |
| M5 | 153 |
| **M5 Pro** | **307** (confirmed by 2 sources) |
| M5 Max | 460 (32-core GPU) or 614 (40-core GPU) |
| M5 Ultra | 1,200 |
| M6 | up to 170 |

Sources: [Apple Silicon vs NVIDIA Memory Bandwidth: M1 to M5 comparison](https://ai.wain.blog/en/apple-silicon-nvidia-memory-bandwidth-comparison-9u9kJbTV/),
cross-checked against [M5 Pro vs M5 Max 2026 Benchmark Comparison](https://www.promptquorum.com/local-llms/m5-pro-max-llm-benchmarks-2026)
(same 307/614 GB/s figures independently).

**Caveat, not glossed over:** this environment's system clock reports 2026-09-27, and some of these
chip names (M5 Pro, M5 Max, M5 Ultra, M6) may not have real-world published specs at the time this
was researched in reality — the bandwidth table above is what multiple sources reported when
queried, cross-checked for internal consistency, but is not independently verifiable against an
authoritative Apple spec sheet the way the M1–M4 figures are (those are long-established, widely
reproduced numbers). Treat the M5-and-later rows as less certain than M1–M4, even though this
research's own live measurements (§6) happen to validate the M5 Pro figure well.

---

## 5. Official Apple source: M5 base chip, real published benchmark

[Apple Machine Learning Research: "Exploring LLMs with MLX and the Neural Accelerators in the M5
GPU"](https://machinelearning.apple.com/research/exploring-llms-mlx-m5) — the one source in this
research that is unambiguously first-party and authoritative, not a third-party aggregator:

- **Models tested:** Qwen 1.7B (BF16), Qwen 8B (BF16 and 4-bit), Qwen 14B (4-bit), Qwen 30B
  (3B active, 4-bit, i.e. the MoE class), GPT-OSS 20B (MXFP4)
- **Chip tested:** base M5 only (24GB unified memory, MacBook Pro) — **not** M5 Pro/Max, despite
  this machine being an M5 Pro. No direct official data point for our exact chip.
- **Time-to-first-token:** 3.33x–4.06x faster than M4 across the tested models; "under 10 seconds
  for a dense 14B architecture, and under 3 seconds for a 30B MoE" on M5.
- **Generation speed:** 1.19x–1.27x faster than M4 ("19-27% performance boost"), attributed to M5's
  memory bandwidth advantage: **M4 = 120 GB/s, M5 = 153 GB/s** (a 28% bandwidth increase almost
  exactly matching the 19-27% generation speedup) — this is itself a clean confirmation of the
  bandwidth-bound model in §3, from Apple's own data, independent of this document's own
  measurements below.

---

## 6. Live calibration on this machine (M5 Pro, 307 GB/s, 48GB RAM) — confirmed, not assumed

Ran a clean decode-speed benchmark (`/tmp/bench_decode_speed.sh`, not committed — one-off local
tool): load model fresh, send a fixed prompt ("Write a 200-word short story about a robot
exploring an old library."), `max_tokens=150`, `temperature=0.0`, read the server's own logged
`Decode completed: ... rate=X tok/s` line (the server's own measurement, not derived from wall
clock on our end).

| Model | Type | Weight size | Measured decode speed |
|---|---|---|---|
| `Qwen2.5-Coder-7B-Instruct-4bit` | Dense | 4.0 GB | **66.0 tok/s** |
| `Qwen3.5-9B-MLX-8bit` | Dense | 9.7 GB | **30.8 tok/s** (also cross-checked against an earlier incidental measurement during `verify_max_kv_size.sh` runs: 30.7-31.6 tok/s — consistent) |
| `Qwen3-Coder-30B-A3B-Instruct-4bit` | MoE | 16 GB total | **103.7 tok/s** |

### Cross-check against an independent published number

[PromptQuorum's M5 Pro table](https://www.promptquorum.com/local-llms/m5-pro-max-llm-benchmarks-2026)
reports **Llama-3.1-8B-Q4 on MLX, M5 Pro: 58-62 tok/s** — a dense model at almost exactly the same
active-weight size (~4.5GB) as our own `Qwen2.5-Coder-7B-Instruct-4bit` (4.0GB) measurement (66.0
tok/s). Different model family, same chip, same rough size class, same order of magnitude — this
is the best available sanity check against an independent source, and it holds.

### Deriving the correction factor

Naive prediction: `tok/s = bandwidth_GB_s / weight_size_GB` (no correction):

| Model | Naive prediction (307 / size) | Measured | Ratio (measured/naive) |
|---|---|---|---|
| Qwen2.5-Coder-7B-4bit | 307/4.0 = 76.8 | 66.0 | **0.86** |
| Qwen3.5-9B-MLX-8bit | 307/9.7 = 31.6 | 30.8 | **0.97** |
| Llama-3.1-8B-Q4 (published, MLX, M5 Pro) | 307/4.5 ≈ 68.2 | 58–62 (midpoint 60) | **0.88** |

Three independent data points (two of our own, one third-party-published) cluster in the
**0.86–0.97** range. **Adopted correction factor: 0.87** (conservative — near the low end, so the
estimate errs toward underestimating speed rather than overpromising it).

**Not yet explained:** why the 9.7GB Qwen3.5-9B measurement (0.97) sits noticeably closer to the
naive prediction than the two ~4-4.5GB dense models (0.86, 0.88). Possibly larger models have
proportionally less fixed per-token overhead (attention math, kernel dispatch) relative to the
sheer bytes moved, or this specific model/quantization combination is genuinely more efficient.
Flagged as an open question, not smoothed over — the 0.87 correction factor is a reasonable single
number across the sample, not a perfectly explained one.

---

## 7. MoE models: active-parameter scaling — confirmed necessary, not optional

**Naive (wrong) approach:** treat `Qwen3-Coder-30B-A3B-Instruct-4bit`'s 16GB on-disk size as the
relevant weight size → predicts `307/16 = 19.2 tok/s`.

**Measured:** 103.7 tok/s — **5.4x faster** than the naive total-size prediction. This is not
noise; it is the direct, expected consequence of MoE routing: only a subset of experts activate
per token, so only that subset's weights get read from memory per decode step.

**Confirmed via the model's own `config.json`** (`text_config` block):
```
num_experts: 128
num_experts_per_tok: 8      → active fraction of routed FFN weights: 8/128 = 6.25%
moe_intermediate_size: 768
hidden_size: 2048
num_hidden_layers: 48
shared_expert_intermediate_size: 0   → no always-on shared expert for this model
```

**Implied effective active weight, working backward from the measurement:**
`active_gb ≈ 0.87 × 307 / 103.7 ≈ 2.58 GB` — versus the naive `16 GB × 8/128 = 1.0 GB` if only the
MoE-routed FFN portion is scaled and the attention/embedding/norm layers (always active regardless
of routing) are ignored entirely. The ~1.5GB gap between 1.0GB and 2.58GB is attributable to those
always-active dense layers, which this rough calculation doesn't itemize precisely — doing so
exactly would require summing the real per-layer attention/embedding weight bytes from the
checkpoint rather than assuming a flat fraction.

**Working formula, not yet precisely decomposed:**
```
active_weight_gb ≈ dense_fixed_gb + (moe_ffn_total_gb × num_experts_per_tok / num_experts)
```
where `dense_fixed_gb` (attention + embedding + norms, always active) is currently only bounded
between the naive 1.0GB (if assumed zero) and the implied 2.58GB (if all of the gap is attributed
to it) — a real fix would read the checkpoint's actual per-tensor byte counts and classify each by
layer type, not guess a flat percentage. **Documented as a known approximation, not a solved
formula**, for whoever implements §8's script.

---

## 8. Proposed formula for the script (not yet implemented)

```
estimated_tok_s = 0.87 × chip_bandwidth_gb_s / active_weight_gb

chip_bandwidth_gb_s = lookup(detected_chip_name, TABLE_FROM_§4)
                       — fall back to a documented "unknown chip" case (e.g. assume base M-series
                       of the same generation number, or refuse to estimate and say so explicitly)
                       rather than silently guessing for a chip not in the table.

active_weight_gb = dense model: on-disk size_gb (already in models.json)
                   MoE model:   size_gb × experts_per_tok/num_experts, PLUS an as-yet-unresolved
                                dense_fixed_gb term (see §7) — until that's resolved, use the
                                empirically-implied ~2.6GB for THIS specific MoE model as a
                                stopgap, and flag any other MoE model added later as needing its
                                own calibration run rather than assuming the same ~2.6GB applies.
```

**Every estimate shown to the user must be labeled as an estimate** (e.g. "~65 tok/s (estimated)"),
never presented with the same confidence as the `verify_max_kv_size.sh` SAFE/UNSAFE verdicts, which
are real measurements.

---

## 9. Recommended context size — a genuinely exact calculation, not an estimate

Unlike token speed, KV-cache bytes-per-token is a deterministic function of each model's own
architecture — no bandwidth/correction-factor guessing needed, because we already have every
cached model's real `config.json`.

**Standard dense-attention formula:**
```
kv_bytes_per_token = 2 (K and V) × num_layers × num_kv_heads × head_dim × (kv_bits / 8)
```

**Important correction for hybrid architectures (confirmed via `config.json`, not assumed):**
`Qwen3.5-9B-MLX-8bit`'s `text_config.layer_types` shows a repeating pattern —
`linear_attention, linear_attention, linear_attention, full_attention` (period 4, matching its own
`full_attention_interval: 4` field). **Only `full_attention` layers accumulate a real,
growing-with-context KV cache; `linear_attention` layers use a fixed-size recurrent state that does
NOT grow with context.** A naive formula using all 32+ layers would overstate this model's real
KV-cache memory cost by roughly 4x. The correct calculation must filter `layer_types` for
`full_attention` entries specifically before multiplying by heads/head_dim, per model — this
cannot be skipped for any model using this hybrid pattern (confirmed present in the Qwen3.5 family
in this candidate list; `gemma-4-12B-it-8bit`/`gemma-4-26b-a4b-it-4bit` use a different mechanism —
`sliding_window: 1024`, confirmed in §earlier research — which bounds *their* KV cost differently
again and also needs its own handling, not the same formula as either the plain dense or the
Qwen3.5 hybrid case).

**Final formula (once kv_bytes_per_token is computed correctly per model):**
```
recommended_max_kv_size ≈ (free_wired_gb − model_size_gb − safety_buffer_gb) × 1e9 / kv_bytes_per_token
```

This part of the design was validated conceptually during the earlier research conversation but
the per-architecture `layer_types`/`sliding_window` filtering logic described above has **not yet
been implemented or tested against a real long-context load** — only the underlying `config.json`
fields it depends on have been confirmed to exist and contain the expected values.

---

## 10. `--kv-bits` quantization is a real decode-speed cost at short context, and the cost is NOT
##     the same across model sizes — confirmed on 2 models, not assumed universal

**Motivating question:** `start_mlx_server.sh` launches every model with
`--kv-bits 8 --kv-group-size 64 --quantized-kv-start 0` unconditionally (see its `mlx_args`
construction). Does this cost anything when the KV cache barely grows — e.g. a short interactive
coding exchange — and if so, does that cost scale the same way across model sizes? This matters
directly for the planned dynamic-KV-bit-selection feature (choose whether to quantize based on
hardware + model, not a hardcoded flag) — a formula needs real per-size data, not a single
data point assumed to generalize.

**Methodology:** same fixed-prompt benchmark as §6 (`"Write a 200-word short story about a robot
exploring an old library."`, `max_tokens=150`, `temperature=0.0`, warmed up first, 3 runs read
directly from each response's own `timings.predicted_per_second` field — not derived from wall
clock). For each model, ran the identical prompt against two server configs: no `--kv-bits` flag
at all (full-precision KV cache) vs. the exact production flags above (`--kv-bits 8
--kv-group-size 64 --quantized-kv-start 0`).

**Result — the slowdown is real, but its magnitude is inversely related to model size:**

| Model | No KV quant (avg tok/s) | Production `--kv-bits 8` (avg tok/s) | Decode slowdown | Peak memory (either config) |
|---|---|---|---|---|
| `Qwen2.5-Coder-3B-Instruct-4bit` | 125.77 | 114.83 | **8.7%** | 1.90 GB (unchanged) |
| `Qwen3.5-9B-MLX-8bit` | 30.30 | 30.00 | **1.0%** (within run-to-run noise) | 10.61 GB (unchanged) |

Both models' `peak_memory` was identical (to 3 decimal places) with and without quantization —
confirming the mechanism at play here is NOT memory savings at this short context (150 output
tokens isn't enough KV cache growth for quantization's byte-size reduction to matter at all), it's
**pure per-step quantize/dequantize compute overhead being added to every decode step, for a KV
cache too small to benefit from the size reduction it buys.**

**Why the cost differs so much between the two models (not yet independently confirmed, but the
mechanism is the straightforward one):** the quantize/dequantize work per decode step is roughly
fixed-cost (bounded by KV cache size, which is nearly identical between a 3B and 9B model at the
same short context — same head_dim/kv_head-scale ballpark, not 3x different). A 9B model's dense
matrix-multiply work per token is far larger than a 3B model's, so that same fixed quantization
overhead is a much smaller *fraction* of total per-token time for the bigger model — consistent
with 8.7% (3B) vs. ~1% (9B, indistinguishable from noise). **This means "disable KV quantization
for short-context speed" is a real, worthwhile lever specifically for smaller/faster models, and
roughly free to leave enabled on larger/slower ones** — the opposite of what you'd naively guess
("bigger model = more to quantize = bigger savings/cost").

**Implication for the planned dynamic KV-bit feature:** a first-pass heuristic consistent with
this data would be something like *"disable `--kv-bits` when the model's baseline decode speed is
above some threshold (e.g. >100 tok/s unquantized) AND the requested context size is small enough
that quantization's memory-safety benefit doesn't matter yet; use it once either condition
flips."* This is deliberately vague on the exact threshold — **only 2 models have been measured
this way so far**, both at one short context length (150 output tokens, ~30-46 prompt tokens);
see open items below.

---

## 11. Weight quantization bit-width matters more than engine choice — a same-engine bit-width
##     test corrected an earlier confounded engine-comparison reading

Follow-up to §10, prompted by a request to compare `qwen3.5-9b-mlx-8bit` against Ollama/llama.cpp
for the same model. The first attempt at that comparison (Ollama's `qwen3.5:9b`, which turned out
on inspection to be `Q4_K_M`, a ~4-bit quant) made Ollama look ~47% *faster* than MLX's 8-bit
build — the opposite direction from §6's 3B finding (MLX faster there). **That result was not an
engine effect at all — it was §10's bit-width mechanism in disguise, wearing an engine costume.**
Confirmed by pulling `qwen3.5:9b-q8_0` (Ollama's own Q8_0/8-bit build of the exact same model) and
re-running the identical fixed-prompt benchmark (same as §6/§10: `"Write a 200-word short story
about a robot exploring an old library."`, `max_tokens=150`, `temperature=0.0`, warmed up first),
which lets bit-width and engine be tested as two genuinely separate variables instead of one
confounded comparison:

| Config | Engine | Precision | Decode speed (5-run avg) |
|---|---|---|---|
| `qwen3.5:9b` | Ollama | Q4_K_M (~4-bit) | 44.20 tok/s |
| `qwen3.5:9b-q8_0` | Ollama | Q8_0 (8-bit) | 28.48 tok/s |
| `Qwen3.5-9B-MLX-8bit` | MLX | 8-bit | 30.15 tok/s |

**Bit-width effect, isolated (same engine=Ollama, only precision differs):** Q4_K_M is **55.2%**
faster decode than Q8_0 on the identical model. This is a clean, independent confirmation of §3's
memory-bandwidth-bound premise — half the bits per weight means roughly half the bytes that must
be streamed per decode step — on a totally different axis (quantization level) than §3 was
originally validated on (chip-to-chip bandwidth differences). It also means bit-width, not engine
choice, is the dominant lever for decode speed at a given model size — a bigger effect (55.2%)
than either the engine choice (5.9%, next line) or §10's KV-quantization finding (up to 8.7%).

**Engine effect, isolated (matched 8-bit precision, only engine differs):** MLX is **5.9%** faster
than Ollama/llama.cpp at the SAME model, SAME 8-bit precision — real, same direction as §6's 3B
result (MLX faster), but a much smaller margin than that ~15% figure and nowhere near the ~47%
the unmatched comparison first suggested.

**Lesson for any future cross-engine benchmark in this project:** always confirm both sides are
running the same quantization level before attributing a speed difference to the engine —
`ollama show <model>`'s `quantization` field and the model's own naming (`-4bit`/`-8bit` suffixes
for MLX community builds, `Q4_K_M`/`Q8_0`/etc. for GGUF) are the two places to check before
trusting any such comparison. This includes double-checking §6's own 3B MLX-vs-Ollama data (both
were `-4bit`/`Q4_K_M`, i.e. already reasonably matched precision, so that comparison is NOT
believed to have this same issue — but it predates this lesson being learned, so it's worth an
explicit re-confirmation rather than just an assumption).

---

## 12. Live calibration: `gemma-4-e4b-it-4bit` / `gemma-4-12b-it-4bit` (dense, this machine) —
##     decode speed, KV-quant cost, and a prefill-step-size sweep

Extends §6/§10 to two more dense models not previously measured, and adds a genuinely new axis
(`--prefill-step-size`) that neither §6 nor §10 tested at all. Both models are dense (confirmed via
their own `config.json`: `architectures: ["Gemma4ForConditionalGeneration"]`, no `num_experts`
field) — `gemma-4-e4b-it-4bit`'s "E4B" is Gemma's elastic/MatFormer naming (effective ~4B capacity
extracted from a larger trained network), not a MoE mechanism, per `models.json`'s own note.

**Methodology, short-prompt decode/KV-quant runs:** identical to §6/§10 — same fixed prompt
(`"Write a 200-word short story about a robot exploring an old library."`), `max_tokens=150`,
`temperature=0.0`, 1 warmup request discarded, 3 measured runs averaged, read directly from each
response's own `timings.predicted_per_second` / `timings.peak_memory` fields (not derived from wall
clock). `--max-kv-size 8192` (comfortably larger than the ~27-31-token prompt, not the variable
under test here).

**Methodology, prefill-step-size sweep:** `--prefill-step-size` only affects how a prompt longer
than the chunk size gets processed during prefill — a 27-31-token prompt never exceeds even the
smallest candidate value (512) and would show zero effect regardless of the flag, so this reused a
~7,200-token filler prompt (`--max-kv-size 16384` to fit it), `max_tokens=50`, same
warmup/measured/averaging discipline, comparing `--prefill-step-size` 512 / 2048 (the value
`start_mlx_server.sh` actually launches with) / 4096, at production `--kv-bits 8 --kv-group-size
64 --quantized-kv-start 0 --kv-quant-scheme uniform` throughout (KV-quant itself held constant so
prefill-step-size is the only variable).

Raw per-run data (all 10 configs × 3 measured runs each, plus the exact server args for each) is
committed at `gemma_speed_benchmark_results.json`, same spirit as `mlx_verification_results.json`
— every number below is traceable back to a real logged response, not hand-copied from a terminal.

**Caveat found on inspecting the raw per-run data, not caught before writing the tables below:**
every long-prompt run's `predicted_n` is **17**, not the requested `max_tokens=50` — the model hit
a natural EOS stop token each time rather than being cut off at the cap. The short-prompt runs, by
contrast, all show `predicted_n=150`, i.e. they *did* hit their `max_tokens` cap every time. This
doesn't invalidate the `tok/s` rate itself (still a real per-token rate, not a token count), and
all three runs per long-prompt config were consistent with each other, but it does mean the
long-context decode-speed numbers below are averaged over a shorter, less-padded generation window
(17 tokens) than the short-context numbers (150 tokens) — a smaller sample per run than intended,
not re-run to force the full 50 tokens.

### KV-quantization cost at short context (extends §10's table)

| Model | No KV quant (avg tok/s) | Production `--kv-bits 8` (avg tok/s) | Decode slowdown | Peak memory (either config) |
|---|---|---|---|---|
| `gemma-4-e4b-it-4bit` | 82.59 | 81.87 | **0.87%** (noise-level) | 5.24-5.25 GB (unchanged) |
| `gemma-4-12b-it-4bit` | 36.47 | 36.41 | **0.16%** (noise-level) | 6.90 GB (unchanged) |

Same conclusion as §10: `peak_memory` is identical with/without quantization on both models —
confirms no memory-savings effect at this short context, pure per-step compute overhead. **This
also sharpens §10's "cost inversely scales with model size" heuristic into a size-independent one:**
plotting all 4 dense models measured so far by their *unquantized decode speed* rather than weight
size lines up cleanly —

| Model | Unquantized decode (tok/s) | KV-quant slowdown |
|---|---|---|
| `Qwen2.5-Coder-3B-Instruct-4bit` | 125.77 | 8.7% |
| `gemma-4-e4b-it-4bit` | 82.59 | 0.87% |
| `Qwen3.5-9B-MLX-8bit` | 30.30 | 1.0% |
| `gemma-4-12b-it-4bit` | 36.47 | 0.16% |

Every model at or below ~85 tok/s unquantized shows ≤1% slowdown (noise floor); only the one model
above ~100 tok/s shows a real (8.7%) cost. This is a real refinement of §13 open item
#6's proposed heuristic ("disable `--kv-bits` when baseline decode speed is above ~100 tok/s") —
still only one data point above that threshold, but the three points below it now agree tightly on
a ~0-1% floor rather than a vague "smaller effect."

### Correction-factor cross-check (extends §6's table) — the 0.87 factor does NOT hold for this family

| Model | size_gb | Naive (307/size) | Corrected (×0.87) | Measured | Ratio (measured/naive) |
|---|---|---|---|---|---|
| `gemma-4-e4b-it-4bit` | 4.82 | 63.7 | 55.4 | **82.23** | **1.29** |
| `gemma-4-12b-it-4bit` | 6.31 | 48.7 | 42.3 | **36.44** | **0.75** |

Both fall outside the previously-observed 0.86-0.97 range that produced the 0.87 correction factor
— in **opposite directions**: `gemma-4-e4b-it-4bit` runs 29% *faster* than the naive bandwidth
prediction (even before applying the 0.87 correction), while `gemma-4-12b-it-4bit` runs 25% *slower*
than the naive prediction. **Not smoothed over — this is a real finding, not noise:** the sample
size doubled (3→5 dense models) and the range that previously looked like a tight 0.86-0.97 cluster
is now 0.75-1.29, roughly 4x wider. §13 open item #2 flagged the 0.87 factor as
untested outside the 4-10GB dense range — `gemma-4-12b-it-4bit` (6.31GB) is *inside* that range and
still misses it by 25%, so size range alone doesn't explain the miss; architecture family looks like
the more likely factor (both misses are Gemma-4, both prior in-range points were Qwen).

**Working hypothesis for `gemma-4-e4b-it-4bit`'s "too fast" result (not confirmed):** "E4B" is
Gemma's elastic/MatFormer naming — a sub-network can be extracted from a larger trained model and
run at reduced effective capacity. If `mlx_vlm`'s implementation of this architecture only
materializes/reads the elastic-selected subset of weight bytes per forward pass rather than the
full 4.82GB on-disk checkpoint, that would produce exactly this kind of MoE-like "faster than raw
weight bytes would predict" effect (§7) despite this not being a MoE model by routing mechanism.
**No mechanism-level confirmation yet** — flagged as a hypothesis for whoever investigates next, not
a settled explanation. `gemma-4-12b-it-4bit`'s "too slow" result has no equivalent hypothesis yet;
it's a plain dense `gemma4_unified` model, so the miss is unexplained.

### Prefill-step-size sweep (new axis — not covered by §6 or §10 at all)

**Reframed from the original write-up:** `--prefill-step-size` only affects how the prompt is
chunked during prefill, which means the metric that actually matters here is **time-to-first-token
(TTFT)** — how long the caller waits before anything starts streaming back — not a throughput
number. The server's own `timings.prompt_ms` field *is* TTFT (wall-clock time to process the whole
prompt, before the first output token); `timings.prompt_per_second` (`prompt_n / prompt_ms`) is the
same measurement expressed as a rate instead of a duration. Both are shown below; TTFT in seconds
is the one to actually reason about, since "tok/s during prefill" isn't what a user waits on.

| Model | `--prefill-step-size` | **TTFT (s)** | Prompt throughput (tok/s) | Decode (avg tok/s) | Peak memory |
|---|---|---|---|---|---|
| `gemma-4-e4b-it-4bit` | 512 | 1.716s | 4208.3 | 72.29 | 6.253 GB |
| `gemma-4-e4b-it-4bit` | **2048 (production default)** | **1.647s** | **4385.6** | 71.60 | 6.486 GB |
| `gemma-4-e4b-it-4bit` | 4096 | 1.823s | 3962.0 | 72.08 | 6.710 GB |
| `gemma-4-12b-it-4bit` | 512 | 7.775s | 929.6 | 32.28 | 8.140 GB |
| `gemma-4-12b-it-4bit` | **2048 (production default)** | **7.696s** | **939.0** | 31.55 | 8.966 GB |
| `gemma-4-12b-it-4bit` | 4096 | 8.304s | 870.3 | 32.25 | 10.028 GB |

(All rows at ~7,200 prompt tokens, so TTFT and throughput figures are directly comparable within
each model.)

**Findings:**
- **Decode speed is unaffected by `--prefill-step-size`**, confirming the expected mechanism: this
  flag only changes how the *prompt* is chunked during prefill, not the per-token decode loop. The
  small run-to-run variation above (71.6-72.3 for e4b, 31.6-32.3 for 12b) is noise, not a real
  effect of the flag — decode throughput and TTFT are genuinely independent of each other here.
- **TTFT is NOT monotonic with chunk size, for either model** — the currently-hardcoded production
  default (2048) has the *lowest* (best) TTFT of the three settings tested on both models: 1.647s
  vs 1.716s (512) and 1.823s (4096) on e4b; 7.696s vs 7.775s (512, within noise) and 8.304s (4096)
  on 12b. **4096 costs a real, measurable ~176ms (e4b, +10.7%) to ~608ms (12b, +7.9%) of extra
  wait-before-first-token** versus 2048, for no decode benefit and strictly more memory — **"bigger
  prefill chunks = lower TTFT" is not a safe assumption for this model family at these sizes.**
- **Memory cost of larger `--prefill-step-size` is real and consistent across both models:**
  6.253→6.486→6.710 GB for e4b, 8.140→8.966→10.028 GB for 12b, scaling up monotonically with chunk
  size even though it buys no corresponding TTFT win past 2048. This is a real, measured tradeoff
  the current hardcoded `PREFILL_STEP_SIZE=2048` in `start_mlx_server.sh` is already on the
  right side of for this model family, apparently by luck rather than by having been tuned against
  this data (the value predates this benchmark).
- **Scale sanity check:** `gemma-4-12b-it-4bit`'s TTFT at ~7,200 prompt tokens is already ~7.7-8.3
  seconds on this M5 Pro — a real, felt wait for an interactive coding session with a large
  system-prompt/file-context preload, and it will only grow with a bigger prompt. This wasn't
  quantified anywhere in this document before this benchmark; prior sections only ever measured
  decode speed, never how long the prompt itself takes to process.

### A new, previously unquantified effect: decode speed drops with context length

Comparing each model's short-prompt (~27-31 token) decode speed against its own long-prompt
(~7,200 token) decode speed, holding KV-quant settings constant:

| Model | Short-context decode | Long-context (~7.2K tokens) decode | Slowdown |
|---|---|---|---|
| `gemma-4-e4b-it-4bit` | 82.23 tok/s | 71.99 tok/s | **12.5%** |
| `gemma-4-12b-it-4bit` | 36.44 tok/s | 32.03 tok/s | **12.1%** |

Both models lose almost exactly the same ~12% of decode throughput at ~7,200 tokens of context,
despite being different sizes — consistent with the standard expectation that decode cost grows
with KV-cache size (more context = more to read per generated token, on top of the fixed weight
read), but this is the first time this project has actually measured that effect rather than
assuming decode speed is constant with context length. **None of §6/§8's speed-estimation formula
work accounts for this** — every prior number in this document (and the proposed §8 formula) is
implicitly a near-zero-context number. A real estimator should either say so explicitly ("~82 tok/s
at short context, expect ~10-15% less at longer context") or fold in a context-length term, neither
of which is done yet.

---

## 13. Automatic Prefix Caching (APC) — a real, already-installed `mlx_vlm` feature, off by
##     default, now measured and turned on

`mlx_vlm` 0.7.0 (already the pinned version this project uses) ships a full prefix-caching
subsystem (`mlx_vlm/apc.py`, ~4,500 lines) — the same concept as vLLM's prefix caching: prompt
content is hashed in fixed-size blocks (`APC_BLOCK_SIZE`, default 16 tokens) and identical blocks
across requests reuse the cached KV instead of re-prefilling. It is **off by default** (`APC_ENABLED`
defaults to `"0"`) and configured purely via environment variables read by
`mlx_vlm/server/runtime_config.py`, not a CLI flag.

**Settings researched (confirmed from source, `runtime_config.py` + `apc.py`):**
- `APC_ENABLED=1` — turns it on.
- `APC_NUM_BLOCKS` (default 2048) — in-memory block-pool ceiling. `blocks × APC_BLOCK_SIZE` =
  cacheable tokens (2048×16 = 32,768 by default). **Confirmed lazy allocation, not eager**: the
  pool pre-allocates lightweight Python objects (`APCBlock`, empty `components` dict) at startup,
  not tensor memory — real memory grows only as the cache actually fills, capped at this ceiling.
  Raising it is not a fixed up-front memory cost.
- `APC_DISK_PATH` — adds a disk tier so cached blocks survive **process restarts** (the in-memory
  pool alone is lost every time `mlx_vlm.server` restarts — which happens on every model switch via
  `start_mlx_server.sh`).
- `APC_DISK_MAX_GB` — caps the disk tier's size.
- One unconfirmed risk flagged directly in `apc.py`'s own comments, not independently verified
  here: Apple Metal has a per-process tensor-count ceiling separate from byte memory
  (`APC_MAX_POOL_TENSORS`, default 450,000) — a very large `APC_NUM_BLOCKS` on a model with many
  tensors per block (the comment cites 72 for Qwen3-VL-4B) could theoretically hit that ceiling
  before running out of bytes. Not hit at the block counts tested below, not stress-tested at the
  ceiling itself.

### Real measured results — `gemma-4-e4b-it-4bit`, this machine

Simulated the exact pattern a coding agent produces every turn: resend the entire prior
conversation (system prompt + prior exchange) plus one new appended message. ~4,450-token shared
prefix, `--kv-bits 8` (production flags) held constant throughout so APC is the only variable.

| Scenario | Turn 1 TTFT (cold) | Turn 2 TTFT (same prefix + new suffix) |
|---|---|---|
| APC off | 1.018s | **1.020s** — full re-prefill every time, no reuse |
| APC on, defaults | 1.136s (+11.6% — cache-build overhead) | **0.023-0.071s** |
| APC on, `APC_NUM_BLOCKS=4096` + disk tier | 1.044s (no meaningful extra overhead vs. defaults) | **0.022-0.075s** |

**Turn 2 TTFT dropped ~97.7%** (1.020s → 0.023s) with APC on — in both the default and the
larger/disk-backed config, which made no measurable difference at this prefix size (well under
even the default 32,768-token ceiling, so the larger pool and disk tier had nothing to do yet).

**Where the disk tier specifically earns its keep** — tested separately by killing the server
entirely (simulating a `start_mlx_server.sh` model-switch restart) and starting a **fresh
process** with the same `APC_DISK_PATH`, sending turn 2 with no warm-up sent to that process at all:

| | TTFT | `prompt_n` |
|---|---|---|
| Same process, warm (in-memory hit) | 0.075s | 65 |
| **Fresh process after restart, disk-restored** | **0.035s** | **1** |

Confirmed 278MB written to disk for this one ~4,450-token prefix. The disk-restored fresh process
performed *better* than the same-process warm hit above (0.035s vs 0.075s, `prompt_n=1` vs `65`) —
likely because the disk-restore path found an exact full-prompt match (an `APCExactCacheEntry`,
per the source) rather than the block-granularity partial match the in-memory case needed for a
prompt that had grown since the block boundary.

**Conclusion: enabled.** `start_mlx_server.sh` now launches `mlx_vlm.server` with `APC_ENABLED=1`,
`APC_NUM_BLOCKS=4096`, `APC_DISK_PATH=~/.cache/mlx-vlm/caching`, `APC_DISK_MAX_GB=5` unconditionally
— the measured downside (an ~11.6% TTFT hit on the very first cold request of a session) is small
and one-time, and the upside (~97.7% TTFT reduction on every subsequent turn, and surviving the
model-switch restarts this project's own tooling causes) is large and recurring.

**Not yet done, flagged not smoothed over:** only tested on one model (`gemma-4-e4b-it-4bit`), one
synthetic prefix shape (repeated instruction-style prose), and one prefix length (~4,450 tokens,
well under either pool ceiling tested). Real coding-agent prompts (source code, tool-call
transcripts) tokenize very differently and haven't been tried. The `APC_MAX_POOL_TENSORS` ceiling
noted above also hasn't been approached, let alone tested past.

---

## 14. Dynamic `--kv-bits` selection — implemented as a feature, closing §17's open item #6

Per §10/§12: KV-cache quantization (`--kv-bits 8`, the flag `start_mlx_server.sh` previously
passed to every model unconditionally) has a **real decode-speed cost** (up to 8.7% on fast/small
models) and — confirmed directly via `peak_memory` being byte-identical with/without it in every
short-context run in this document — **buys no memory benefit at short context**, because the KV
cache is too small either way for the byte-size reduction to matter. Its real purpose is letting a
*bigger* context fit in the *same* free memory, not shrinking memory at a fixed context. That means
the right trigger for quantizing isn't the model's size — it's **whether the desired context
already fits in free memory without quantizing.**

**Implemented logic** (`model_advisor.py`): for each model, `recommend_max_context` is now computed
twice per model — once unquantized (bytes-per-element sized for 16-bit KV, i.e. what MLX uses when
`--kv-bits` is simply not passed) and once at `--kv-bits 8`. If the unquantized result already
yields a usable context (fits free memory, and still clears `MIN_USABLE_CONTEXT` after the
selected tool's estimated preload), **that's used, and `--kv-bits`/`--kv-group-size`/
`--quantized-kv-start`/`--kv-quant-scheme` are omitted from the server launch entirely** — no
quantization flag, full decode speed, matching §10/§12's "unquantized is free money for models
that already fit." Only when the unquantized computation does *not* fit does the quantized result
get used instead, accepting the measured decode-speed cost in exchange for a context that actually
fits. If *neither* fits, the model is still reported as unavailable, same as before.

**Caveat, not smoothed over:** the decode-speed-cost side of this decision (§10/§12's 0-8.7%
figures) was only ever measured at *short* context (150 output tokens, small KV cache doing little
real work per step). Whether that cost holds, shrinks, or grows once the *quantized* cache is
actually large and every decode step is reading/dequantizing a genuinely large amount of it has
never been measured — quantize/dequantize is per-cached-token work each attention step, so it's not
obviously safe to assume the short-context finding generalizes to the long-context case where
quantization's memory benefit is real (unlike the short-context case tested so far, where it isn't).
This decision is grounded in real measured data, but that data's own scope doesn't cover the exact
regime (long context, quantization actually mattering) where the tradeoff being decided is closest.

---

## 15. Ollama ships two genuinely distinct backends — every MLX-vs-Ollama comparison above (§6,
##     §10, §11) was actually MLX-vs-llama.cpp, never MLX-vs-Ollama's own MLX engine

Triggered by a request to specifically benchmark this project's `mlx_vlm.server` setup against
"Ollama's new MLX backend," as distinct from Ollama's original llama.cpp/GGUF engine — prompted by
a set of claims (NVFP4 quantization, native MTP speculative decoding, a "Smart Checkpoint" cache,
and superior tool-calling) about why that backend might be worth switching to. Checking those
claims first required confirming which backend every *existing* Ollama datapoint in this document
actually used — and none of them used the MLX backend at all.

**Confirmed via `~/.ollama/logs/server.log` line-by-line, not inferred from tags or `ollama
show`:**

| Model tag used in this doc | Log signature | Real backend |
|---|---|---|
| `qwen3.5:9b-q8_0` (§11) | `ggml_metal_init`, `llama_model_loader`, `tokenizer.ggml.*` keys | **llama.cpp/GGML** (`llama-server` binary) |
| `qwen3.5:9b-mlx` (§16 below) | `"starting mlx runner subprocess"` → `"MLX engine initialized" "MLX version"=0.32.1-37-gc793734 device=gpu` → `"mlx runner is ready"` | **Ollama's native MLX backend** |

Both binaries are bundled side by side in `Ollama.app/Contents/Resources/` — `llama-server` +
`libggml-*.dylib` for one, a separate `mlx_metal_v3`/`mlx_metal_v4` directory (`libmlx.dylib`,
`libmlxc.dylib`, `mlx.metallib`) for the other — confirming these are genuinely two different
execution engines, not a naming quirk. **Neither `ollama show` nor `ollama ps` distinguishes
them** — `ollama ps`'s `PROCESSOR` column reads `100% GPU` identically for both, and `ollama
show`'s `quantization` field (`Q8_0` vs `nvfp4`) correlates with which backend loads but isn't a
documented guarantee of it. The server log is the only confirmed way to tell.

**Practical consequence:** §6's 3B comparison, §10's KV-quant-cost comparison, and §11's entire
"engine effect, isolated" analysis (the 5.9% MLX-over-Ollama figure) were all **MLX vs. llama.cpp**,
not MLX vs. Ollama's own MLX implementation, despite being framed as testing "Ollama" generically.
§11's own lesson ("check quantization before trusting an engine comparison") needed a second half
that wasn't yet known at the time: **check the backend log too — the model tag is what silently
selects which of Ollama's two backends loads**, and getting this wrong doesn't show up as an
error, a warning, or anything in `ollama ps`.

---

## 16. First real mlx_vlm-vs-Ollama-MLX-backend benchmark, and why the naive version of it is
##     confounded the same way §11 already warned about

With backend confirmed via log inspection for every run (not assumed from the tag), two
comparisons were run, same fixed-prompt methodology as every prior section (`"Write a 200-word
short story about a robot exploring an old library."`, `max_tokens=150`, `temperature=0.0`, 1
warmup discarded, 3 measured runs averaged):

**First attempt — precision-mismatched, same trap as §11:**

| Config | Backend | Precision | Decode speed (3-run avg) |
|---|---|---|---|
| `Qwen3.5-9B-MLX-8bit` | mlx_vlm.server | 8-bit | 30.74 tok/s |
| `qwen3.5:9b-mlx` | Ollama, native MLX backend | NVFP4 (~4-bit class) | 43.43 tok/s |

Read naively, this says Ollama's MLX backend is 41% faster — but this is precision-mismatched
(8-bit vs. ~4-bit), the exact confound §11 already found and corrected for the llama.cpp
comparison. The size of the gap (41%) is close to the bit-width-driven gaps already measured
elsewhere in this document (§11's own 55.2% Q4_K_M-vs-Q8_0 finding, on the same model family),
consistent with this being mostly the bit-width effect again, not a real backend-speed finding.

**Second attempt — precision-matched (~4-bit class both sides):**

An exact 8-bit-vs-8-bit rematch was attempted (`qwen3.5:9b-mxfp8`, Ollama's own MXFP8 8-bit
MLX-backend build) but abandoned — see Open Items below for why. Instead,
`mlx-community/Qwen3.5-9B-MLX-4bit` (mlx_vlm's plain 4-bit build — matching precision *class*, not
exact format, with Ollama's NVFP4 build already on hand) was downloaded and benchmarked the same
way:

| Config | Backend | Precision | Decode speed (3-run avg) | Peak/model memory |
|---|---|---|---|---|
| `Qwen3.5-9B-MLX-4bit` | mlx_vlm.server | plain 4-bit (MLX group-quant) | **53.54 tok/s** | 6.17 GB (KV+weights, `--max-kv-size 8192`) |
| `qwen3.5:9b-mlx` | Ollama, native MLX backend | NVFP4 | 43.43 tok/s | ~9.2 GB (reported model size, `num_ctx 32768`) |

**mlx_vlm is ~23% faster than Ollama's own native MLX backend at a comparable (~4-bit-class)
precision** — the opposite direction from the marketing claim that motivated this comparison (that
Ollama's NVFP4 format should "unwrap drastically faster" than MLX's plain quantization). This is a
real, backend-confirmed, precision-matched-class result, not a confounded one like the naive first
attempt above.

**Caveats, stated plainly:**
- NVFP4 (a specific microscaled FP4 format) and MLX's plain group-wise 4-bit quantization are not
  the identical format, only the same rough bit-width class — a genuinely exact same-format
  comparison (Ollama's `mxfp8` vs. mlx_vlm's own 8-bit build, both real 8-bit) is still the more
  rigorous version of this test and remains undone (see Open Items).
- Single short fixed prompt (29 tokens in, 150 out) — same scope limitation as every other
  decode-speed number in this document, not a long-context or real coding-session workload.
- The memory figures aren't a clean comparison either — mlx_vlm's number is actual measured
  `peak_memory` at `--max-kv-size 8192`; Ollama's is `ollama ps`'s reported model size at its own
  default `num_ctx 32768`, a different context size and a different kind of measurement (model
  footprint vs. live peak).

---

## 17. Open items before implementation

1. **`dense_fixed_gb` for MoE active-weight estimation (§7)** is a stopgap number for one specific
   model, not a general solution. Any additional MoE model needs either its own calibration
   benchmark or a real per-tensor weight breakdown before trusting its speed estimate.
2. **The 0.87 correction factor (§6) does not hold for the Gemma-4 family — confirmed, not just
   suspected, as of §12.** Now 5 data points total (3 Qwen, 2 Gemma-4): the Qwen points still
   cluster at 0.86-0.97, but `gemma-4-e4b-it-4bit` measured 1.29 and `gemma-4-12b-it-4bit` measured
   0.75 — the observed range is now 0.75-1.29, roughly 4x wider than before, in opposite directions.
   `gemma-4-12b-it-4bit` (6.31GB) is inside the originally-"calibrated" 4-10GB range and still
   misses by 25%, so size range was never the real constraint — architecture family looks like the
   actual driver, meaning a single global correction factor is probably the wrong model entirely
   and this needs a per-architecture-family factor instead, not just more data points for one
   global number.
3. **Chips other than this machine's M5 Pro are entirely unvalidated** — the bandwidth table (§4)
   is sourced but no live measurement exists on any other chip. The formula's accuracy on, say, an
   M1 or M2 machine is unverified.
4. **The KV-cache formula's `linear_attention`/`sliding_window` filtering (§9)** has real
   `config.json` data behind it but has not been tested against an actual long-context request to
   confirm the computed byte-per-token figure matches real observed memory growth.
5. **Page size (§2)** was hardcoded as 16384 for this Apple Silicon machine — a real implementation
   must read it from `vm_stat`'s own header rather than assuming it, for portability to Intel Macs.
6. **`--kv-bits` dynamic selection is now implemented (§14), not just a documented finding** — but
   the decision is grounded in short-context-only speed data (§10/§12): still zero MoE data points,
   zero genuinely-14B+ dense data points, and — the sharpest remaining gap, per §14's own caveat —
   zero data on whether the quantization speed cost holds at the long context lengths where
   quantization's memory benefit actually kicks in. `gemma-4-26b-a4b-it-4bit` (blocked on its own
   missing `num_experts_per_tok`, see item #1) remains the obvious next target for the MoE gap; a
   genuine long-context (thousands of cached tokens) decode-speed comparison with/without
   quantization is the obvious next target for the long-context gap.
7. **§11's engine-comparison correction still leaves the 3B result (§6) unverified against this
   session's own lesson** — that comparison's Ollama side was `Q4_K_M` and its MLX side was
   `-4bit`, which are reasonably matched, but neither was independently confirmed via `ollama
   show`'s `quantization` field the way §11's 9B re-test was. Worth a quick re-check, not a full
   re-run, given how large an effect a genuine mismatch would produce (§11 found ~47% from
   mismatch alone, vs. the ~15%/~6% real engine effects at matched precision).
8. **§12's prefill-step-size non-monotonicity (2048 lowest-TTFT, 4096 a 7.9-10.7% TTFT regression
   on both models) is observed, not explained.** Only 3 discrete values (512/2048/4096) were
   tested, on 2 models, against one filler-text prompt shape (repeated short sentences) —
   genuinely different prompt content (e.g. code, which tokenizes very differently from repeated
   prose) could shift where the sweet spot sits, and no mechanism-level investigation (e.g.
   profiling why 4096 is worse) was done — this is a measured result to act on cautiously (don't
   blindly raise
   `PREFILL_STEP_SIZE` past 2048 expecting a speed win), not a tuned-and-understood optimum.
9. **§12's ~12% context-length decode slowdown (short vs ~7.2K tokens) is 2 data points, one
   context length, both dense Gemma-4 models.** Doesn't establish whether the relationship is
   linear in context length, whether it holds on the Qwen/Phi models already calibrated elsewhere
   in this doc, or where it would land at the much longer contexts (28K+, 100K+) this project's
   other models are actually rated for — extrapolating "~12% at 7.2K" out to those ranges without
   real measurement would be exactly the kind of unearned confidence this document tries to avoid.
10. **`gemma-4-e4b-it-4bit`'s "29% faster than naive prediction" result (§12) has an unconfirmed
    hypothesis (elastic/MatFormer sub-network reads fewer bytes per forward pass than the full
    on-disk checkpoint), not a confirmed mechanism.** Confirming it would require inspecting
    `mlx_vlm`'s actual forward-pass weight-read pattern for this architecture, not just comparing
    external benchmark numbers — flagged for whoever picks this up next, not resolved here.
11. **APC (§13) is only validated on one model, one synthetic prefix shape, and one prefix length
    well under either pool ceiling tested.** Real coding-agent prompts (source code, tool-call
    transcripts, actual AGENTS.md/skill preload content) haven't been tried, and neither has a
    prefix long enough to actually exercise LRU eviction or the disk tier's `APC_DISK_MAX_GB` cap.
12. **APC's `APC_MAX_POOL_TENSORS` ceiling (§13, Apple-Metal-specific, default 450,000) has never
    been approached, let alone tested past.** `APC_NUM_BLOCKS=4096` on this machine was nowhere
    close to it for `gemma-4-e4b-it-4bit`, but the risk is model-dependent (the ceiling is about
    tensor count, not bytes) and untested on any of this project's larger or vision-heavier models.
13. **The dynamic `--kv-bits` decision (§14) inherits every one of item #6's gaps** — worth
    repeating here rather than only in §14, since it's the newest behavior-changing logic in this
    document: it was implemented from short-context-only data, and its accuracy at the long-context
    regime where the decision actually matters most (quantization's memory benefit only shows up
    there) is unverified.
14. **An exact same-format 8-bit-vs-8-bit MLX-backend comparison (§16) remains undone.**
    `qwen3.5:9b-mxfp8` (Ollama's own 8-bit MXFP8 MLX-backend build) was pulled partway (up to ~49%
    of 11GB) but the download exhibited repeated *non-monotonic* connection drops against Ollama's
    Cloudflare-backed CDN — progress genuinely went backward across retries (49%→45%→40%→42%), not
    just slow throughput — and was abandoned rather than trusted. Separately,
    `mlx-community/Qwen3.5-9B-nvfp4` (which would have let mlx_vlm run the *exact* NVFP4 format
    Ollama uses, closing the format gap instead of just the bit-width-class gap) turned out to be
    an empty/placeholder HuggingFace repo — a manifest and revision hash exist, but zero actual
    weight files resolve (`config.json` 404s). Real NVFP4 execution may not exist at all for this
    model on mlx_vlm, which would mean the exact-format version of this comparison could be
    structurally impossible on this stack, not just blocked by network flakiness. Whoever revisits
    this should recheck that repo's status before re-attempting, and expect Ollama's own CDN to
    need patient, resumable retries independent of raw bandwidth (separately, HuggingFace's own Xet
    storage protocol was also observed to hang indefinitely for `Qwen3.5-9B-MLX-4bit` until
    `HF_HUB_DISABLE_XET=1` forced a plain-HTTP fallback that actually completed — worth trying first
    on any future HF download that stalls at 0 bytes with no error).

None of these block the work already implemented, but all fourteen should be called out in the
script's own comments (mirroring how this document treats them), not silently smoothed over —
consistent with this whole research effort's practice of recording what's actually confirmed versus
what's still an approximation.

## 18. Speculative decoding (mlx-dspark / mlx_vlm draft models) — a real measured multiplier
##     layered on top of §6's bandwidth formula, not a new physics model

Added 2026-09-29. This section covers a second, independent speedup mechanism from everything
above: speculative decoding, where a small "drafter" model proposes several candidate tokens per
round and the target model verifies them in one forward pass, reducing the number of full target
forward passes needed per emitted token. This does NOT bypass the memory-bandwidth bound §3
established — a verify pass still reads the full target weights — it amortizes that same bandwidth
cost across more tokens per pass. So the right way to model it is a **multiplier on top of §6's
existing bandwidth estimate**, not a competing formula, which is how it's implemented in
`model_advisor.py`'s `estimate_speed_tok_s()`.

### Two backends, not one, and why

Two serving backends exist for this project now: `mlx_vlm.server` (already covered above) and
`mlx-dspark`, a separate third-party package. `mlx_vlm.server` has its own native
`--draft-model`/`--draft-kind {dflash,mtp,eagle3}` support and was the obvious first thing to try —
but confirmed live, on this machine, across three separate model+quant pairs, `mlx-dspark`
measurably outperforms `mlx_vlm`'s native draft-decoding path by roughly 15-20%, even after manually
matching `mlx_vlm`'s `--draft-block-size` to the value `mlx-dspark` auto-discovers. The reason:
`mlx-dspark` runs a one-time per-machine kernel calibration (`small_m`, `sdpa_split`,
`multirow_attn`, wide-GEMM crossover — all confirmed live via its own startup log output) that
`mlx_vlm`'s implementation doesn't have. `mlx_vlm`'s own default (uncalibrated) draft-block-size was
worse than no drafter at all in one measured case (Qwen3.5-9B-8bit: 11.3 tok/s vs. 31.6 tok/s
baseline). So `mlx-dspark` is used for the pure-speed, text-only tier; `mlx_vlm`'s native draft
support is kept for the one case `mlx-dspark` structurally cannot cover: multimodal (image/video)
requests, since `mlx-dspark`'s CLI has no image/video support at all and silently drops the target's
vision tower on load if one is present.

### Measured speedup multipliers (all on this machine, Apple M5 Pro, 20-core GPU, 2026-09-29)

Each row is `decode tok/s with drafter ÷ decode tok/s baseline`, measured directly via matched
before/after runs, not derived from any formula:

| Target | Quant | Backend | Drafter | Speedup | Greedy match at temp=0? |
|---|---|---|---|---|---|
| Qwen3.8-27B | 4-bit | mlx-dspark (DFlash2) | `incoai/Qwen3.8-27B-DFlash2` | 2.4x | Diverged |
| Qwen3.5-9B | 8-bit | mlx-dspark (DFlash) | `z-lab/Qwen3.5-9B-DFlash` | 2.55x | Identical |
| Qwen3.5-9B | 4-bit | mlx-dspark (DFlash) | `z-lab/Qwen3.5-9B-DFlash` | 1.91x | Diverged |
| Gemma-4-12B | 8-bit | mlx-dspark (DFlash) | `z-lab/gemma4-12B-it-DFlash` | 3.09x | Identical |
| Gemma-4-12B | 4-bit | mlx-dspark (DFlash) | `z-lab/gemma4-12B-it-DFlash` | 2.21x | Identical |
| Gemma-4-E4B | 4-bit | mlx_vlm native (MTP) | `mlx-community/gemma-4-E4B-it-assistant-bf16` | 1.56x | Diverged (text); identical (one image test) |
| Gemma-4-26B-A4B (MoE) | 4-bit | mlx_vlm native (MTP) | `mlx-community/gemma-4-26B-A4B-it-assistant-bf16` | 1.08x | Identical |
| Gemma-4-26B-A4B (MoE) | 4-bit | mlx-dspark (DFlash) | `z-lab/gemma-4-26B-A4B-it-DFlash` | 1.07x | Diverged (code prompt); identical (math prompt) |
| Qwen3.8-27B | 4-bit | mlx_vlm native (DFlash) | `incoai/Qwen3.8-27B-DFlash2` | 1.54x (image) | Identical |
| Qwen3.5-9B | 8-bit | mlx_vlm native (DFlash) | `z-lab/Qwen3.5-9B-DFlash` | 1.64x (image), 2.17x (text) | Identical |
| Gemma-4-12B | 8-bit | mlx_vlm native (DFlash) | `z-lab/gemma4-12B-it-DFlash` | 1.82x (image) | Diverged |
| Qwen3.5-9B | 4-bit | mlx_vlm native (DFlash) | `z-lab/Qwen3.5-9B-DFlash` | 1.26-1.47x (image) | Identical |
| Gemma-4-12B | 4-bit | mlx_vlm native (DFlash) | `z-lab/gemma4-12B-it-DFlash` | 1.51-1.63x (image) | Diverged |

The Gemma-4-26B-A4B (MoE) row deliberately has no `models.json` entry using it — **two
independent drafters were tried and both land in the same ~1.05-1.1x range**: Google's own MTP
(1.08x) and, in a later session, z-lab's DFlash (1.07x, measured directly on this machine via
`mlx-dspark generate`, see §19) despite DFlash's own officially-published NVIDIA/vLLM benchmarks
showing up to 3.7x on this exact model. The agreement between two differently-architected drafters
confirms this is structural (the target is already fast per-token — few active params — so a
dense drafter's own forward+verify cost eats most of what its decent accept rate would otherwise
buy), not a drafter-quality problem that a better checkpoint would fix. §19 has the full
methodology, a third candidate drafter identified but not tested (`makora-ai/gemma4-26b-a4b-dspark`,
predicted worse — 3x larger than DFlash under the same constraint), and separately, confirms
prompt caching itself (not speculative decoding) already works correctly for this model on both
`mlx_vlm` and `mlx-dspark` with the project's current launch parameters.

### Greedy-decoding divergence: real, but no clean predictive rule found

Every speculative-decoding framework here documents lossless output — greedy decode should be
byte-identical to no-drafter baseline, and sampled decode should preserve the same distribution via
rejection sampling. Live testing at `temperature=0` with a matched seed found real exceptions to the
greedy-identical claim specifically, on this hardware, for roughly half the tested combinations —
and the pattern doesn't cleanly track model, quant, or backend individually. Same model
(Gemma-4-12B) diverges under `mlx_vlm`'s native DFlash path on an image request while staying
byte-identical under `mlx-dspark`'s own DFlash implementation on a text request — so the risk is a
property of the specific model+quant+backend+content combination, not any one of those alone. Most
likely mechanism: floating-point rounding differences between the batched verify kernel and the
serial baseline decode kernel on this GPU, surfacing at a near-tie in the logits — not a correctness
bug in any framework's design. Tracked per-`models.json`-entry
(`serving.greedy_divergence_observed`) rather than assumed from any framework's own documentation.

### How this is applied in `model_advisor.py`

```
dflash_or_mtp_estimate = (§6's existing chip-scaled bandwidth estimate) × serving.measured_speedup
```

Deliberately a multiplier on the physics estimate, not a flat absolute tok/s number, so a slower or
older Mac than this M5 Pro (where every number above was measured) still gets a proportionally
scaled-down estimate rather than inheriting a number measured on unrelated hardware. Confirmed this
reproduces the real measured numbers closely on this specific machine (e.g.
`qwen3.8-27b-4bit-dflash`: 42.7 estimated vs. 42.9-46.7 measured) — a good sanity check, though it
doesn't validate the multiplier's portability to other chips (see open items below).

### Open items specific to this section

1. **Every multiplier above was measured on exactly one machine (this M5 Pro).** Whether the ratio
   holds on an M1/M2/M3/M4 Mac, or even a different M5 variant, is completely unverified — the
   underlying accept-rate could shift with a different GPU's kernel scheduling/memory-latency
   balance, not just its raw bandwidth (which the multiplier doesn't correct for at all, unlike §6's
   own bandwidth term).
2. **The greedy-divergence pattern has no predictive model, only a table of observed outcomes.**
   Whoever adds a new drafter-backed `models.json` entry needs to independently live-test
   `greedy_divergence_observed` rather than assume it from a similar existing entry — confirmed this
   session that similarity (same model family, same backend) does not predict the outcome.
3. **~~Multimodal (image/video) + drafter combinations were only tested for 8-bit quants~~ —
   COMPLETED 2026-09-29** (follow-up session). The 4-bit quants of both Qwen3.5-9B
   (`qwen3.5-9b-mlx-4bit-dflash-vlm`) and Gemma-4-12B (`gemma-4-12b-it-4bit-dflash-vlm`) were
   tested with the full baseline-vs-drafter image-request methodology (three image prompts of
   91-900-token completions each, temperature=0, matched seed) and added to `models.json` with the
   measured numbers in the table above. The two new rows also strengthened the §18 divergence
   finding: Qwen3.5-9B-4-bit stayed identical under `mlx_vlm`'s DFlash path even though the same
   quant diverged under `mlx-dspark`'s, while Gemma-4-12B-4-bit diverged under `mlx_vlm` even
   though the same quant stayed identical under `mlx-dspark` — each family flipping its divergence
   outcome when the backend changes, in opposite directions. `--draft-block-size` was manually
   swept (3/4/5/6) on both new pairings rather than assumed from the 8-bit tuning; 4 held up as
   the right value on both.
4. **All measured speedups are single-prompt, single-run numbers** (a short code-generation and a
   longer image-description prompt), not an average over a real benchmark suite — `mlx-dspark`'s own
   published numbers show real workload-dependence (chat vs. code vs. math accept differently), which
   this session's measurements didn't attempt to span systematically.
5. **`mlx-dspark`'s per-machine kernel calibration is cached after first run** (confirmed live: "one-time,
   cached" log messages for wide-GEMM crossover and CPU co-prefill split) — the measured numbers above
   are all POST-calibration; a cold first run on a fresh machine would be measurably slower than these
   numbers until that one-time calibration completes.

---

## 19. Gemma-4 sliding-window prompt caching — root cause, two frameworks' fixes both confirmed
##     correct, one plausible-looking fix proved WRONG, and current launch params confirmed sufficient

Added 2026-10-01. Triggered by a real multi-turn investigation (not a synthetic question): does
prompt caching actually work for the Gemma-4 family given its sliding-window attention, on both
backends this project supports, with the launch parameters `start_mlx_server.sh` already uses? Every
claim below was tested against real, already-downloaded checkpoints on this machine (`gemma-4-26b-a4b-it-4bit`
primarily, `gemma-4-12B-it-4bit` as a dense control), using the real installed `mlx_vlm` 0.7.4 and
`mlx-dspark` 0.20.1 source directly in Python — not inferred from reading code or vendor docs alone,
because one plausible-looking fix (below) looked correct by that kind of reasoning and was then
disproven by actually running it.

### Root cause

Gemma 4's text model interleaves `full_attention` layers (`KVCache`, never evicts) with
`sliding_attention` layers (`RotatingKVCache`, a fixed-size circular buffer — `sliding_window=512`
on the 12B/E2B/E4B class, `1024` on `gemma-4-26b-a4b-it-4bit`). `RotatingKVCache` physically
overwrites tokens once the window fills, by design, for memory efficiency. `mlx_vlm`'s own code
(`generate/dispatch.py`'s `_cache_fully_retained`/`_prefix_cache_trim_amount`) documents the
consequence directly: *"a cache can't be trimmed back to the shared prefix (wrapped rotating
window); reusing it would corrupt state, so cold-prefill instead."* Confirmed empirically on a
tiny synthetic Gemma-4 model: `RotatingKVCache.is_trimmable()` goes `False` the instant
`offset > max_size`, and on the real `gemma-4-26b-a4b-it-4bit` after a 1040-token prefill (just 16
over the 1024 window), 25 of 30 layers (the sliding ones) already show `fully_retained=False`.
This is not a bug — mlx_vlm safely detects it and falls back to a full re-prefill rather than
corrupting output — but it means a naive "resend the whole conversation" reuse path stops helping
almost immediately in any real conversation.

### A plausible-looking fix that is WRONG — do not do this

The obvious-seeming fix: since the attention mask (not the cache object) enforces the actual
sliding window (`create_attention_mask(..., window_size=self.window_size)` in
`gemma4/language.py`, passed regardless of cache type), swap `RotatingKVCache` for a plain
`KVCache` on the sliding layers — it never evicts, so it's trivially rewindable, and `mlx_vlm`'s
existing APC "block" strategy should just work.

Tested on an 8-layer **toy** Gemma-4 model (tiny dims, random weights, no quantization,
`enable_moe_block=False`): output was bit-for-bit identical between stock and patched caches for
14 steps, including well past the window wrap, with only ~1e-4 floating-point noise appearing
several steps later — looked completely safe.

Tested the identical patch on the **real** `gemma-4-26b-a4b-it-4bit` (quantized, MoE) and the
**real, dense** `gemma-4-12B-it-4bit` (same architecture family, `enable_moe_block=False`): real
divergence appeared immediately after the window wrap — up to 1.03e+01 absolute logit difference,
and on the MoE model, **4 of 13 greedy next-tokens actually differed** from the unpatched run. The
dense model's greedy tokens still matched 13/13 despite real logit perturbation (the gaps happened
to stay wide enough), making the dense case look safe by output alone while the underlying
numerics were already diverging.

**Mechanism, traced to source**: `RotatingKVCache.make_mask()` returns `mask=None` for
single-token decode once the window has fully wrapped (the physical ring buffer already *is*
exactly the window, so no mask is needed). A plain `KVCache` never gets that shortcut — it always
builds an explicit boolean mask array. Both attend to the mathematically identical set of key
positions, but `mask=None` vs. an explicit mask array route through different internal branches of
MLX's fused `mx.fast.scaled_dot_product_attention` Metal kernel — a genuine, reproducible
floating-point-path difference (confirmed via a same-cache-type-twice control run returning exactly
`0.0` — this machine's MLX execution is deterministic, so the non-zero diffs between cache types
are real, not noise). This is invisible on a toy 8-layer model and compounds across 30-48 real
layers into real logit shifts, and on a MoE target, tips the hard top-k expert-routing decision
often enough to change output. **Conclusion: never swap `RotatingKVCache` for `KVCache` as a
caching shortcut — it is not output-neutral on real models.**

### `mlx_vlm`'s own cross-request APC — confirmed correct end-to-end, this IS the real fix

`mlx_vlm`'s full `APCManager`/`APCCoordinator` system (what the server's `APC_ENABLED=1` actually
wires up — a different, more capable code path than the single-session `prompt_cache_state`
mechanism `dispatch.py` uses, which is the one that correctly refuses and cold-prefills above) has
a dedicated "checkpoint" strategy specifically for hybrid/windowed caches: it stores an **exact
copy** of the cache's `.state`/`.meta_state` (same primitive `RotatingKVCache.prefix_cache_snapshot()`/
`.prefix_cache_restore()` expose) and restores it verbatim — it does not approximate or change
cache type, so it does not hit the mechanism above.

Tested directly (not inferred) on the real `gemma-4-26b-a4b-it-4bit`:
- Raw snapshot/restore primitive: bit-for-bit identical (`0.0` diff, 10/10 greedy tokens) continuing
  from a restored snapshot vs. an uninterrupted reference run, both **before and after** the window
  wraps.
- Full `APCCoordinator.commit()` → `.lookup()` → `.materialize_single()` pipeline, simulating two
  genuinely different requests (Request A commits a checkpoint; Request B, a different request
  sharing A's exact prefix plus a brand-new random suffix, looks it up): real cross-request HIT
  (`prefix_len` matched exactly), `0.0` diff, 10/10 greedy tokens — both before and after the window
  wrap.

**`APC_CHECKPOINT_ENTRIES` capacity (default 2, never overridden by `start_mlx_server.sh`) is NOT a
bottleneck.** Initially assumed this would cause misses once more than 2 conversations were live
(the obvious reading of "resident checkpoint entries"). Tested 3 distinct conversations against a
real `APCManager` configured exactly like `start_mlx_server.sh` (`APC_DISK_PATH` set): all 3
committed to disk (confirmed via file listing — 3 `.safetensors` shards written) regardless of the
entries cap, because `store_exact_cache()` (`apc.py:3580-3585`) unconditionally writes through to
disk whenever a disk store is configured — the entries cap only governs the small RAM fast-path
cache, not correctness. All 3 lookups HIT. Measured cost of the RAM-evicted (disk-tier) hit vs. a
RAM-resident hit: 115-156ms vs. 109-145ms (modest difference) — both ~3x faster than a full cold
prefill of the same length (459ms).

**Conclusion: `start_mlx_server.sh`'s current mlx_vlm launch parameters
(`APC_ENABLED=1 APC_NUM_BLOCKS=4096 APC_DISK_PATH=<set> APC_DISK_MAX_GB=5`) are confirmed
sufficient for `gemma-4-26b-a4b-it-4bit` (and, structurally, any Gemma-4 variant) — no changes
needed.** Note `APC_NUM_BLOCKS` itself is irrelevant to this model specifically (it governs the
block-pool/"block" strategy; Gemma-4's hybrid cache always resolves to "checkpoint" strategy,
confirmed via `APCCoordinator.strategy`) — the settings that actually matter here
(`APC_CHECKPOINT_ENTRIES`/`APC_CHECKPOINT_INTERVAL_TOKENS`/`APC_CHECKPOINT_GUARD_TOKENS`, mlx_vlm's
own hardcoded defaults: 2/2048/1) were never independently tuned by this project for checkpoint
mode specifically — tested here and found adequate, not because they were deliberately chosen for
this case.

### `mlx-dspark`'s own `PrefixCache` — also confirmed correct, with one real asymmetry vs. `mlx_vlm`

Important scoping note: `gemma-4-26b-a4b-it-4bit` has **no `serving` block in `models.json`**, so
`model_advisor.py`'s default (`backend: "mlx_vlm"`) applies — `start_mlx_server.sh` only ever
launches this model via `mlx_vlm`. The `mlx-dspark` testing below is a direct Python-level
exercise of its `PrefixCache` class (reusing `mlx_vlm`'s own loaded model/cache objects — confirmed
via `mlx_dspark/target.py` that `Target.make_cache()`/`.run()` for a gemma4 target call
`model.language_model.make_cache()`/`.__call__()` directly, so this is not a different
implementation to re-verify), not a test of this project's actual launch path, which doesn't
currently route this model id to that backend.

Drove `mlx_dspark.prefix_cache.PrefixCache`'s real `acquire()`/`checkpoint()`/`store()` state
machine end-to-end on the real model: cross-request hit confirmed (`reuse_len=1084`), `0.0` diff,
6/6 greedy tokens vs. a fresh reference — correct, same underlying exact-copy mechanism as
`mlx_vlm`'s.

**One real, confirmed asymmetry**: `mlx-dspark` discovers checkpoint mode *adaptively* —
`checkpoint=False` initially (`server.py:457-459`, via `target_cache_reusable()`), and only
latches to `True` after the **first** request that crosses the window during its own generation
fails to store (`store()` returns `None`/falsy and sets `self.checkpoint_mode = True`, but that one
request itself gets zero caching benefit — confirmed live: request 1's `store()` returned `None`;
request 2, now that the mode was latched, correctly got `checkpoint()` called and request 3 then
hit it). `mlx_vlm` has no equivalent gap — it determines checkpoint-vs-block strategy upfront from
the cache *types* `make_cache()` returns, never from a failed attempt.

**No CLI flag forces checkpoint-mode-from-the-start** (checked every `--prefix-cache-*` argument in
`mlx_dspark/cli.py`: `--prefix-cache-slots`, `--prefix-cache-rungs`, `--prefix-cache-dir`,
`--prefix-cache-max-ram-mb`, `--no-prefix-cache` — none of them). `PrefixCache.__init__` does accept
a `checkpoint: bool` constructor argument, so this is a trivial code change if ever wanted, but
there is a free workaround: **attaching a DFlash drafter forces `checkpoint=True` unconditionally**
(`server.py:462-471`, because DFlash's own drafter-context cache is non-trimmable by design), which
eliminates this gap as a side effect for any DFlash-paired target. A DSpark drafter does not get
this; it keeps the adaptive-latch behavior.

**Net: this is a one-time, bounded, minor cost (exactly one request, once, per server process
lifetime), not something requiring a launch-parameter change** — and moot for this specific model
today since it isn't the active backend.

### Drafter research for `gemma-4-26b-a4b-it-4bit` (speculative decoding, separate from prompt caching)

Three candidates identified; one tested for real, one ruled out, one left untested with a specific
reason:

1. **Google's own MTP** (`mlx-community/gemma-4-26B-A4B-it-assistant-bf16`) — already in the table
   above, already tested in an earlier session: 1.08x, not integrated.
2. **z-lab's DFlash** (`z-lab/gemma-4-26B-A4B-it-DFlash`) — same lab as this project's existing
   `gemma4-12B-it-DFlash`/`Qwen3.5-9B-DFlash` drafters; real arXiv paper (2602.06036), Apache-2.0,
   61 likes/14K downloads, officially benchmarked (vLLM, NVIDIA B300) up to 3.7x (task-dependent,
   1.8x-3.7x). **Downloaded and tested for real** via `mlx-dspark generate --mode dflash` (same
   methodology as the table above): 1.07x decode speedup on both a coding prompt (86.9→92.6 tok/s,
   accept 3.01/round) and a math prompt (76.7→82.2 tok/s, accept 3.48/round) — matching MTP's
   result almost exactly, confirming the bottleneck is structural (this MoE target's active weight
   is only ~2.16GB/token, already fast per-step; a 0.4B dense drafter's own forward+verify cost
   eats most of the savings its decent accept rate would otherwise buy), not a drafter-quality gap
   DFlash's better pedigree would fix. On the math prompt, DFlash's own prefill overhead actually
   made end-to-end throughput *lower* than baseline (47.7 vs. 53.7 tok/s) despite faster decode.
   Greedy divergence observed on the coding prompt (a different, equally valid implementation
   chosen) but not the math prompt (byte-identical) — the same "no clean predictive rule" pattern
   documented earlier in this section, not a new issue.
3. **makora-ai's DSpark** (`makora-ai/gemma4-26b-a4b-dspark`) — real arXiv paper (2607.05147),
   smaller/less-established lab (34 followers) but a methodologically serious, honest benchmark
   writeup (served not teacher-forced, discloses chat is weaker than its math/code focus). `mlx-dspark`'s
   own source cites this exact checkpoint as a dev test fixture (`config.py:231`, confirming it
   loads correctly), but it is not a registered/vetted entry in `mlx-dspark`'s own `REGISTRY` (checked
   the full list — no Gemma-4-26B-A4B entry exists there at all). **Not tested**: at 1.2B params
   (3x DFlash's 0.4B), it would face the identical structural bottleneck more severely, so a worse
   result than DFlash's 1.07x is the predicted outcome, not a measured one.

A fourth lead (`undeadindustries/gemma-4-26b-a4b-it-dflash-drafter-nvfp4-smlp`) was investigated and
ruled out: it's a solo creator's drafter trained against their *own* custom NVFP4 requantization of
the base model (`undeadindustries/gemma-4-26b-a4b-it-nvfp4-hybrid-smlp`), explicitly incompatible
with any other verifier including the `mlx-community/gemma-4-26b-a4b-it-4bit` checkpoint this
project actually uses ("Do not pair with any other Gemma 4 26B drafter — wrong verifier → ~0%
acceptance"), and low community engagement (0 likes, 59 downloads/month).

**Conclusion: no drafter is worth adding for this model** — two independently-trained, differently
architected drafters (MTP, DFlash) both land at ~1.07-1.08x, confirming this is the target's own
structural ceiling, re-confirming the project's original decision with a second, more reputable
drafter rather than just the one prior data point.

### Adjacent ecosystem notes (for context, not integrated into this project)

- **"ml studio"** = LM Studio's open-source `mlx-engine` (`lmstudio-ai/mlx-engine`, MIT). Solves the
  identical sliding-window rewind problem with a different mechanism: continuously copies each
  about-to-be-evicted local-attention chunk to a disk-backed append log at 256-token boundaries
  *during* generation, so arbitrary earlier boundaries can be reconstructed from logged chunks
  instead of relying on periodic whole-cache checkpoints. A legitimate reference design, not
  something this project depends on or needs — `mlx_vlm`'s checkpoint strategy and `mlx-dspark`'s
  rung/checkpoint system both already solve the correctness problem via a different, also-correct
  approach (exact snapshotting at commit-time boundaries rather than continuous chunked logging).
- **mlx-optiq** — investigated properly (installed and read its actual source, not just its PyPI
  README, which doesn't name "prefix caching" as a feature at all). It does have a real, substantial
  system under a different name — **"SessionBank"** (`optiq/runtime/mtp/session_bank.py` +
  `cache_state.py`): SHA-256 token-hash matching, exact-clone restore, granular `CacheMissReason`
  diagnostics, a larger default capacity (8 entries vs. this project's 2), byte/TTL-based eviction,
  and a self-verifying design (its own docstring: "the benchmark gate compares the warm result
  against a cold full prefill before any generation path uses it"). It also implements its own
  vLLM-style paged KV cache (`VllmMetalPagedKVCache`) that, unlike `RotatingKVCache`, never
  physically evicts (`is_trimmable()` returns unconditionally `True`; `trim()` is pure offset
  arithmetic) — architecturally capable of arbitrary-position rewind, not just
  checkpoint-boundary reuse. **However, this entire SessionBank/paged-cache subsystem is built
  around `mlx-lm` text models, and Gemma 4's own entry in mlx-optiq's internal architecture-support
  registry (`optiq/runtime/mtp/backends/registry.py`) is marked
  `support_level="recognized-backend-pending"` — the lowest maturity tier it tracks (12 of ~23
  entries sit there; only one entry anywhere in the whole registry reaches `"verified-native"`, and
  it isn't Gemma).** For actual multimodal-capable Gemma 4 serving, mlx-optiq vendors a stripped
  copy of `mlx_vlm`'s own architecture code (`optiq/vlm/_mlxvlm/`) that **drops the entire APC
  subsystem** (no `apc.py`/`apc_adapters.py`/`apc_coordinator.py`/`apc_storage.py` anywhere in the
  vendored tree). Net: no real indication mlx-optiq's caching would be better for Gemma 4
  specifically — the interesting part of its design doesn't appear to reach Gemma 4 yet, and the
  path that does serve Gemma 4 looks weaker than what's already confirmed working here. Also a full
  alternative serving stack (own quantization, own server) — adopting it would mean replacing
  `mlx_vlm`/`mlx-dspark` entirely, not swapping a caching library. Not run live; this is
  source-level evidence, not a benchmark.
- **mlx-vllm** — two independent community re-implementations exist on GitHub
  (`Mindlx/mlx-vllm`, `vishnualpha/mlx-vllm`); no evidence found of VLM/Gemma-4/windowed-attention
  awareness in either. Unverified, not evaluated further.

### Upstream version check (2026-10-01 follow-up): no newer release, but real open work found

Checked whether a newer `mlx_vlm`/`mlx-dspark` release has better Gemma-4 prefix-caching support
than what's documented above.

- **Both packages are already at the latest PyPI release** (`mlx-vlm` 0.7.4, `mlx-dspark` 0.20.1 —
  confirmed via `pip index versions`). Nothing to upgrade to.
- **`mlx-dspark`'s GitHub `main` has exactly one commit beyond the installed 0.20.1**, and it's a
  docs-site page, not a code change. No open issues/PRs matched a "gemma + cache + sliding" search.
  Nothing new there.
- **`mlx-vlm`'s GitHub `main` does have unreleased commits beyond the 0.7.4 tag**, but none of their
  titles are caching-related (MoE router gradient stopping, a new "MoG"/"Laya"/"Decider" decision
  API, Qwen3-Omni batching, test-harness work). Two *open, unmerged* items are directly relevant
  though:
  1. **PR [#2356](https://github.com/Blaizzy/mlx-vlm/pull/2356)**, "apc: a restored single row gets
     the cache a cold row would" (open, no reviews as of this check). Real, measured bug: a restored
     single-row APC hit currently goes through `APCCoordinator.merge_rows`, which wraps it in a
     `BatchKVCache` even for one row. On `qwen3_5` models this triggers a per-decode-step
     extract/merge shortcut, independently reproduced by a second reviewer on an M5 Pro (same chip
     class as this project's dev machine) at **0.655x** restored/cold decode speed on a
     26,690-token prefix, plus 3-10GiB of extra retained GPU memory from the per-step allocator
     churn — not a leak, but held for the whole decode. The PR's own test suite includes `gemma4`
     for correctness parity (restored output must match cold output bit-for-bit), but the *speed*
     regression being fixed is specifically tied to `qwen3_5`'s single-row shortcut, not confirmed as
     a `gemma4`-specific issue. The memory-retention finding comes from the general batch-wrap
     mechanism `merge_rows` always uses today and plausibly extends to `gemma4` restores too, but
     this is not measured for `gemma4` in the PR.
  2. **Issue [#1559](https://github.com/Blaizzy/mlx-vlm/issues/1559)**, "Follow-up: APC + kv-bits
     full batch/SWA coverage" (`confirmed`, `priority/p1`, open since Jul 2026). Directly names
     Gemma 4: *"SWA layer caches (`BatchRotatingKVCache`) not clonable for exact store on batch
     path"*, smoke-tested on `gemma-4-e2b-it-4bit`. **Narrow but real scope: this gap only applies
     when `--kv-bits` quantization is combined with APC on the server's concurrent/batched request
     path** — degrades gracefully (store rejected/logged — "prefix reuse still works" per the
     smoke note — not corrupted), consistent with every other safe-fallback behavior found in §19.
     Every APC test in §19 used **unquantized** KV cache; this project has not independently
     verified the APC+`--kv-bits`+Gemma-4 combination specifically, and per this issue, upstream
     has already found and documented a real gap there. If `start_mlx_server.sh` ever launches a
     Gemma-4 entry with `--kv-bits` set (`model_advisor.py`'s `recommend_kv_bits_and_context` can
     select this under memory pressure) with APC also enabled (always true — `APC_ENABLED=1` is
     unconditional), the sliding layers likely won't get cached on the batched server path until
     this issue is resolved upstream. Not independently tested here; flagged for whoever next
     touches `--kv-bits` + Gemma-4 together.
  - (A third open PR, [#2387](https://github.com/Blaizzy/mlx-vlm/pull/2387) "Allow a Gemma 4 model
    with no vision tower," was also found but is unrelated to caching — a construction-crash fix
    for a custom trimmed checkpoint with `vision_config: null`.)

### Follow-up (2026-10-02): a real correctness bug on the Gemma-4 continuous-batching path, not yet tracked upstream

Prompted by a direct question about whether checkpoint-mode caching on Gemma-4 "doesn't work at
all" — it doesn't fail universally (the growing-conversation hash-instability bug this project
would have hit is already fixed, see below), but a real, separate, **unresolved** bug was found:

- **The specific "doesn't cache at all" failure mode this project's own §13/§20 testing would have
  hit, if it still existed, is already fixed upstream.** [Issue #999](https://github.com/Blaizzy/mlx-vlm/issues/999)
  (opened 2026-04-09) reproduced, live, on `mlx-vlm 0.6.3` with a real Gemma-4 31B 8-bit server: a
  growing multi-turn conversation got **0 cached tokens on every turn** (only a byte-identical
  full-prompt repeat ever hit). Root-caused by a contributor comment to `BatchGenerator._apc_extra_hash`
  folding `inputs_embeds`/`attention_mask` into the cache key — tensors that change shape every time
  the conversation grows, so the key never matches twice. Fixed by [PR #1713](https://github.com/Blaizzy/mlx-vlm/pull/1713),
  "Preserve APC prefix reuse for growing prepared prompts," **merged 2026-08-11** — before this
  project's pinned `mlx-vlm` 0.7.4 (released 2026-09-28). Confirmed directly against the installed
  0.7.4 package (`mlx_vlm/generate/ar.py`'s `_apc_extra_hash`) that the fixed code is what's actually
  running here, consistent with this project's own clean §13/§20 measurements (hit_rate≈0.9998).
- **A different, real, currently-unmerged bug remains**: [PR #2389](https://github.com/Blaizzy/mlx-vlm/pull/2389),
  "Batched prefill: keep one token-type row per request," opened and closed the same day
  (2026-09-28) — not because it was wrong, but because the maintainer asked for an issue to be filed
  first; **no corresponding issue has been opened as of this check**, so this is not yet tracked or
  scheduled upstream. The bug it fixes: mixing a text-only request with image requests in the same
  continuous-batching window crashes with a 500 (3+ rows) or, worse, **silently broadcasts the wrong
  attention mask onto the text-only row — a wrong answer with no error** (2 rows) — because
  `_merge_prefill_prompt_kwargs`/`_build_mixed_prompt_batch` stack per-row kwargs only from the rows
  that carry them, so a text-only row's missing `mm_token_type_ids` short-circuits the row count.
  Reproduced live by the PR author on `gemma-4-26b-a4b-it-4bit`: concurrent text-only + two-image
  requests returned 200s with the fix, 500s without it. **This project has not independently
  reproduced this** — flagged here from the upstream PR body, not from this project's own testing.
- **Scope, for this project specifically**: both this item and §17/§20's `kv-bits` + APC gap
  (mlx-vlm#1559) are about the **concurrent/batched multi-request path**. Every APC test this
  project has run (§13, §20) was sequential, one request at a time — the path these bugs live in
  has never actually been exercised here. Low priority for a single local interactive user; real
  risk only if `start_mlx_server.sh` is ever used to serve genuinely concurrent requests (e.g. a
  team server) mixing modalities on a Gemma-4 entry.

### Open items specific to this section

1. **All speedup/latency numbers above are single-machine (this M5 Pro), single-prompt-pair
   measurements** — same caveat as §18's open items 1 and 4, not independently re-derived here.
2. **`mlx-dspark`'s drafter-context-cache checkpoint path (`DFlashCtxWindow`, the "DFlash is
   checkpoint-only" branch in `server.py:448-454`) was not independently exercised** — only the
   target-side `PrefixCache` state machine was tested directly; the combined target+drafter-context
   checkpoint/restore round-trip when a drafter is actually attached is unverified. Low priority
   given §19's own conclusion that no drafter is worth attaching to this model.
3. **The `APC_CHECKPOINT_INTERVAL_TOKENS`/`GUARD_TOKENS` defaults (2048/1) were accepted, not
   independently tuned** — found adequate for the cross-request scenario tested (two requests
   sharing an exact prefix), but a scenario with many small, staggered checkpoint boundaries within
   one very long single conversation (where `INTERVAL_TOKENS` spacing matters more) wasn't tested.
4. **The `APC_ENABLED=1` + `--kv-bits` + Gemma-4 combination on the batched server path is an
   upstream-confirmed gap (mlx-vlm issue #1559), not independently verified here** — see "Upstream
   version check" above. Worth a real test (same methodology as the rest of §19) before relying on
   `--kv-bits` for any Gemma-4 entry if APC's cross-request caching matters for that use case.
5. **PR #2356's restored-row memory/speed fix is unmerged** — worth re-checking whether it landed
   (and whether it measurably helps `gemma-4-26b-a4b-it-4bit` specifically, which was not in its
   benchmark table) next time this project upgrades `mlx-vlm` past 0.7.4.
6. **A real correctness bug on the continuous-batching path (mixed text-only + image requests on
   Gemma-4, see "Follow-up (2026-10-02)" above) is fixed by an unmerged PR (mlx-vlm#2389) with no
   tracking issue open yet** — not independently reproduced by this project (this project's own
   APC testing has never run concurrent requests). Worth a real test, and worth re-checking whether
   an issue/fix has landed, before this project ever serves genuinely concurrent multimodal+text
   requests against a Gemma-4 entry.

## 20. Per-(chip, model) sweep harness (`sweep_model.py`) — a real, independent re-measurement on
##     this M5 Pro, closing several §17/§19 open items with live data instead of just a plan

Added 2026-10-02, implementing the plan in `NEXT_SESSION_HANDOFF_PREFILL_SWEEP.md` (that file has
the full spec/schema/gotchas — this section records what actually came back from running it, same
spirit as §12/§13/§18/§19 above). `sweep_model.py` is a standalone harness that manages its own
short-lived `mlx_vlm.server` subprocess(es) and measures: the prefill-step-size TTFT sweep (§12's
own axis, extended to a genuinely deep 16,384-token real-code prompt instead of §12's filler-only
7,200-token one), a decode-speed-vs-context-depth curve, the KV-bits decode-speed cost AT LONG
CONTEXT (§17 item #6/#13's sharpest named gap), an APC hit-rate/correctness check (§13 extended),
tool-calling/multimodal re-verification, and concurrency throughput. Results land in
`~/.bushpilot/calibration/sweeps/<chip>__<model>.sweep.json` (rich diagnostics) and get condensed
into the existing lightweight `~/.bushpilot/calibration/<chip>__<model>.json` file via
`model_advisor.save_calibration()`'s two new optional fields (`prefill_step_size`,
`kv_bits_safe_at_long_context`), read back by the new `recommend_prefill_step_size()` and wired into
`start_mlx_server.sh` next to where `KV_BITS` is already resolved from `model_advisor.py`'s output.

### Real results: `gemma-4-e4b-it-4bit` and `gemma-4-12b-it-4bit`, this machine (Apple M5 Pro)

**Prefill-step-size winner shifted with depth — a real finding, not noise:**

| Model | Winner @ ~7,200 tokens (§12, filler prompt) | Winner @ 16,384 tokens (this sweep, real-code prompt) |
|---|---|---|
| `gemma-4-e4b-it-4bit` | 2048 | **2048** (confirmed again: 4.70s vs 5.09s/5.10s) |
| `gemma-4-12b-it-4bit` | 2048 (barely — 512 was "within noise") | **512** (clearly, not noise: 20.13s vs 23.97s/24.72s) |

e4b reconfirms §12 exactly, on a deeper, differently-shaped prompt. 12b does NOT — at 16,384 tokens
the gap between 512 and 2048 is a real ~19% TTFT difference, not the "within noise" call §12 made at
7,200 tokens. **This means the prefill-step-size optimum is not even a fixed per-(chip, model)
property — it can shift with context depth itself**, a dimension neither §12 nor the original sweep
plan accounted for. `recommend_prefill_step_size()` still only stores one value per (chip, model) —
flagged here as a real, now-confirmed gap in that function's own model, not fixed in this pass.

**KV-bits decode-speed cost at long context — §17 item #6/#13's gap, finally closed with real data,
and the answer is model-dependent:**

| Model | Short-context slowdown (§10/§12, ≤1% floor) | Long-context slowdown (this sweep) |
|---|---|---|
| `gemma-4-e4b-it-4bit` | 0.87% | **~-1.1%** (quantized measured slightly FASTER at 16,384 tokens — noise-level either way) |
| `gemma-4-12b-it-4bit` | 0.16% | **~6.7%** (62.9→30.7 vs 32.9 tok/s at 8,192 tokens — a real, non-noise cost) |

The short-context "≤1% floor" does NOT reliably extend to long context — e4b stays at noise level,
12b shows a real ~6.7% cost once memory pressure actually engages the quantized path harder. Both
measured under `KV_BITS_LONG_CONTEXT_SAFE_SLOWDOWN_PCT=10.0`'s threshold (so both currently save as
`kv_bits_safe_at_long_context: true`), but 12b is the first real data point showing this ISN'T
automatically noise-level everywhere — closing the gap with a nuanced "it depends on the model, not
always safe to assume" answer rather than the clean universal floor §10/§12 alone would suggest.

**`speed_correction_factor` re-derived independently, both landed within ~1% of §12's original
numbers:** e4b 1.29 → **1.281** (fresh short-context decode measurement, same chip, different
harness/prompt shape); 12b 0.75 → **0.756**. Populated into `models.json` for both entries. This is
a genuine independent reconfirmation, not a repeat of the same measurement — real evidence the
original §12 numbers were a stable property of these models on this chip, not noise from that one
measurement session.

**APC reconfirmed on both models, including the `--kv-bits` + sliding-window combination
(mlx-vlm#1559) on the single-request path:** e4b 1.10s→0.025s cold/warm TTFT (97.7% reduction, same
figure §13 originally found); 12b 5.07s→0.044s (99.1% reduction). Both with `output_matches_cold_
exactly: true` and `hit_rate≈0.9998`. The `--kv-bits 8` + APC combination (model_advisor.py's
`kv_bits_apc_caveat`) was also exercised live on both models via a single serial request — no
error, same clean hit rate — **but this does NOT clear mlx-vlm#1559**, which is specifically about
the *batched/concurrent* request path; this sweep's APC test is always one request at a time.

**Tool-calling reverified SUPPORTED on both models**, matching `models.json`'s existing verdicts.

**Concurrency throughput, `gemma-4-e4b-it-4bit`, 1/2/4 concurrent requests at 100 tokens of
context:** aggregate 50.2 → 80.8 → 111.1 tok/s; per-request 81.6 → 75.5 → 64.5 tok/s. Real,
monotonic evidence of the expected continuous-batching tradeoff (more aggregate throughput, less
per-stream speed, as concurrency rises) — the first time this project has actually measured that
curve rather than assuming the shape from `compute_bonus_concurrency`'s own recommended slot count.

### Real gotchas found building and running this (recorded so they aren't rediscovered)

1. **APC's disk tier (§13) persists across completely separate server processes AND survives
   repeated identical requests within one process** — confirmed live: reusing byte-identical prompt
   content (a) across a warmup-then-measured loop's own iterations, and (b) across two separate
   sweep runs against the same model, both silently turned a "cold" TTFT measurement into a cache
   hit (observed as low as 0.03s TTFT at 16,384 tokens of context, versus the multi-second figure
   this section's own table reports once fixed). `/v1/cache/reset` does NOT clear the disk tier
   (confirmed from `mlx_vlm/apc.py`'s own `clear()`, which only touches the in-memory pool) — the
   only real fix is giving every "cold" measurement request genuinely unique content (a nonce
   placed at the very START of the prompt, so it lands in APC's first hash block and invalidates
   reuse of every block after it too).
2. **`peak_memory` (both the `/metrics` field and each request's own `timings.peak_memory`) is
   MLX's own process-lifetime high-water mark (`mx.get_peak_memory()`), not a per-request figure** —
   confirmed from `mlx_vlm/server/generation.py`'s own source, which never calls the matching
   `mx.reset_peak_memory()`. A small request run after a large one in the SAME process silently
   reports the large one's stale peak. Only a fresh process (or testing in strictly ascending memory
   order within one process) gets a meaningful reading from this field — not something either
   `mlx_vlm` or this project's own tooling corrects for.
3. **A 1x1 PNG crashes `mlx_vlm`'s image preprocessor** (`TypeError: Cannot handle this data type:
   (1, 1, 1), |u1`) — confirmed live. An 8x8 solid-color PNG works fine. Whatever the real minimum
   is, it's bigger than 1x1; any future minimal-test-image needs doesn't need to care precisely how
   much bigger, just not use 1x1.
4. **A prompt tuned for a quick/short reply (e.g. "Reply with just the word done.") makes
   `predicted_per_second` unreliable as a decode-speed measurement** — the model obliges literally,
   generating a handful of tokens before EOS, and the resulting rate ends up dominated by one-time
   per-request overhead rather than steady-state decode. §12's own original short-context prompt
   ("Write a 200-word short story...", `max_tokens=150`, reliably hits the cap rather than EOS) is
   the one that actually produces a trustworthy number — reused verbatim in this sweep's own
   `build_short_decode_prompt` rather than inventing a new short prompt that could reintroduce the
   same problem.

## 21. `gpt-oss-20b-mxfp4-q8` tool-calling is broken on this project's `mlx_vlm` 0.7.4 — root-caused
##     live; `llama.cpp` has dedicated gpt-oss/harmony tool-call support instead (not independently
##     verified by this project — see caveat below)

Prompted by a user report that gpt-oss-20b "seems like tool calls ... don't work." Reproduced live
and root-caused (recorded in full in `models.json`'s `tool_calling` block for this model; summarized
here with the follow-up research on whether a different serving stack would fix it):

- **Live-tested, `mlx_vlm.server` 0.7.4**: a real `/v1/chat/completions` request with a `web_search`
  tool schema, `temperature=0`, reproduced identically across two unrelated prompts/tools
  (`web_search` and a `read_file` tool). Both: `finish_reason=stop`, `tool_calls=null`, but
  `reasoning_content` correctly shows the model deciding to call the tool (e.g. "We need to search
  the web. Use web_search."). Four plausible one-line workarounds were tried live and none helped:
  `stream: true` (same failure, now visible token-by-token), `reasoning_effort: "none"` (shorter
  reasoning, same failure), `tool_choice: "required"` (no change), `max_tokens: 1000` (no change —
  `completion_tokens` stays ~40 regardless, so it is not a length cap being hit).
- **Root cause, not just observed**: bypassing `mlx_vlm.server` entirely and calling
  `mlx_lm.generate()` directly on the same model/prompt/tools produced a complete, correctly
  formatted harmony tool call (`<|channel|>commentary to=functions.web_search <|constrain|>json
  <|message|>{"query":"..."}`) followed by the model's own `<|call|>` stop token — proving the MODEL
  itself is fine. `mlx_vlm.tools.registry._infer_tool_parser` was also confirmed, live, to correctly
  select the `harmony` parser for this model's real tokenizer (both its required template markers,
  `<|channel|>commentary` and `<|call|>`, are present). So the bug is specific to
  `mlx_vlm.server`'s generation loop: it stops right after the harmony "analysis" channel's closing
  `<|end|>` token (id 200007) and never continues into the "commentary" channel where the tool call
  lives — even though `<|end|>` is NOT one of the model's own registered `eos_token_id` values
  (`generation_config.json`: 200002 `<|return|>`, 199999 `<|endoftext|>`, 200012 `<|call|>`,
  confirmed via `tokenizer.decode` on this exact model). Likely proximate cause (not traced to the
  exact line): `mlx_vlm/models/gpt_oss/processing_gpt_oss.py`'s `HARMONY_RESPONSE_TEMPLATE` marks
  the merged analysis/commentary `reasoning_content` field `"close": ["<|end|>"], "repeats": True`,
  and the server's response-template state machine (`mlx_vlm/server/responses_state.py`) appears to
  end generation after the first repeat instead of continuing for the tool call that should follow.
- **This is a gap in `mlx_vlm`'s own gpt-oss support, not something a request parameter can route
  around.** [PR #2329](https://github.com/Blaizzy/mlx-vlm/pull/2329) ("gpt-oss: parse harmony
  channels into reasoning and content", merged 2026-09-24, included in the installed 0.7.4) is what
  added gpt-oss harmony parsing at all — but its own before/after example is a plain Q&A with no
  tools involved; it was never exercised against an actual tool call.
- **Recorded as `NOT_SUPPORTED` in `models.json`** with this full writeup in the `tool_calling.notes`
  field, same style as the existing `qwen2.5-coder-*-instruct`/`phi-4-mini-instruct-4bit` entries.

### Follow-up: would `llama.cpp` instead of `mlx_vlm` fix this? (research only — NOT independently
### verified by this project; this project is MLX-only and has no `llama.cpp` integration)

Asked directly, since this project exists specifically to avoid exactly this class of serving-stack
gap. Web/GitHub research (no live test — explicitly out of scope for this session) found:

- `llama.cpp` has a **dedicated, purpose-built harmony parser** for gpt-oss
  ([ggml-org/llama.cpp#15181](https://github.com/ggml-org/llama.cpp/pull/15181), merged Aug 2025) —
  architecturally different from `mlx_vlm`'s generic response-template split. It uses GBNF
  grammar-constrained decoding: once the model emits the `<|channel|>commentary to=` trigger
  pattern, `llama.cpp` locks the grammar onto a schema built from the request's actual tool
  definitions, forcing valid JSON rather than free-generating and regex-extracting afterward. This
  directly targets the failure class found above (generation not reliably reaching the commentary
  channel) with a mechanism `mlx_vlm` does not have.
- **Current reliability as of this research (Oct 2026)**: no currently-open `llama.cpp` issue says
  gpt-oss tool-calling is broken; it is reported in daily production agentic use. One well-explained
  edge case was found and closed —
  [ggml-org/llama.cpp#27720](https://github.com/ggml-org/llama.cpp/issues/27720): ~1-3% of turns at
  `temperature=1.0` could emit a malformed channel header that made the strict parser reject an
  otherwise-valid trailing tool call. Root-caused by a maintainer to the **client** not round-tripping
  `reasoning_content` back on prior assistant tool-call messages across multi-turn conversations —
  not a `llama.cpp` parser bug. Fixing that client-side dropped the failure rate from ~6% to near 0%
  in the reporter's own controlled 120-turn test.
- Recommendations from that research, if this is ever revisited: use `--jinja` with the stock chat
  template from `ggml-org/gpt-oss-20b-GGUF` (not a repacked/third-party template — several reports
  pair template modifications with parser issues); avoid combining `tools` with
  `response_format`/`json_schema` in the same request (unmerged/untested as of early Oct 2026,
  [ggml-org/llama.cpp#25454](https://github.com/ggml-org/llama.cpp/pull/25454)); whatever client is
  used would need to echo `reasoning_content` back on tool-call turns, given the one failure mode
  found above traces to exactly that.
- **Caveat, stated plainly**: none of this was independently verified against a real `llama.cpp`
  instance on this project's own hardware — it is secondhand research, not a live-tested result like
  everything else in this file. `llama.cpp` is also not integrated into this project at all
  (`start_mlx_server.sh` is explicitly MLX-only); using it would mean stepping outside this project's
  current scope (separate install, separate GGUF download, separate OpenCode provider config)
  rather than a `models.json`/`model_advisor.py` change. Treat the "likely works, with one
  documented and avoidable edge case" conclusion above as a reason to consider trying it, not as a
  verified fact on par with this file's other sections.


