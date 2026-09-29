# Hardware Detection & Token-Speed Estimation — Research Notes

**Status:** Research complete, validated against live measurements on this machine.
Implemented in `model_advisor.py`, called by `start_mlx_server.sh` (renamed from
`start_mlx_server.sh` — see PHASE_PLAN.md Phase 4). This document exists so the
formula below is traceable back to its sources and isn't reinvented or drifted from
silently later.

**Owner:** this repo (`local-model-setup`)
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
