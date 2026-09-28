# Hardware Detection & Token-Speed Estimation — Research Notes

**Status:** Research complete, validated against live measurements on this machine. Not yet
implemented in `start_litellm_proxy.sh`. This document exists so the formula below is traceable
back to its sources and isn't reinvented or drifted from silently later.

**Owner:** local scripts (`~/scripts/`)
**Related:** `LITELLM_PROXY_USAGE.md` (interactive menu design), `models.json` (per-model metadata
this feeds into), `verify_max_kv_size.sh` / `mlx_verification_results.json` (separate
verification — confirms `--max-kv-size` is *enforced*, not how *fast* a model runs).

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

**Motivating question:** `start_litellm_proxy.sh` launches every model with
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

## 12. Open items before implementation

1. **`dense_fixed_gb` for MoE active-weight estimation (§7)** is a stopgap number for one specific
   model, not a general solution. Any additional MoE model needs either its own calibration
   benchmark or a real per-tensor weight breakdown before trusting its speed estimate.
2. **The 0.87 correction factor (§6)** is based on 3 data points, all dense models in the 4-10GB
   range. Untested at very small (<2GB) or very large (15GB+) dense model sizes — the correction
   factor could plausibly differ outside the calibrated range.
3. **Chips other than this machine's M5 Pro are entirely unvalidated** — the bandwidth table (§4)
   is sourced but no live measurement exists on any other chip. The formula's accuracy on, say, an
   M1 or M2 machine is unverified.
4. **The KV-cache formula's `linear_attention`/`sliding_window` filtering (§9)** has real
   `config.json` data behind it but has not been tested against an actual long-context request to
   confirm the computed byte-per-token figure matches real observed memory growth.
5. **Page size (§2)** was hardcoded as 16384 for this Apple Silicon machine — a real implementation
   must read it from `vm_stat`'s own header rather than assuming it, for portability to Intel Macs.
6. **`--kv-bits` dynamic selection (§10)** only has 2 data points (3B and 9B, both dense, both at
   one short context length ~150 output tokens on this M5 Pro). Before wiring an actual
   quantize/don't-quantize decision into `model_advisor.py` or `start_litellm_proxy.sh`, need: (a)
   at least one MoE model and one larger dense model (14B+) measured the same way, to confirm the
   "cost inversely scales with model size" pattern holds rather than being a 2-point coincidence;
   (b) the SAME comparison repeated at a genuinely long context (thousands of tokens generated,
   not 150) to find the crossover point where quantization's memory-bandwidth savings start
   outweighing its per-step compute cost — §10's data only proves the cost exists at short
   context, it says nothing about where quantization starts being a net win rather than a net
   loss; (c) a decision on what "hardware spec" should feed into this — e.g. total RAM/available
   bandwidth likely changes where that crossover point sits, not just the model's own size.
7. **§11's engine-comparison correction still leaves the 3B result (§6) unverified against this
   session's own lesson** — that comparison's Ollama side was `Q4_K_M` and its MLX side was
   `-4bit`, which are reasonably matched, but neither was independently confirmed via `ollama
   show`'s `quantization` field the way §11's 9B re-test was. Worth a quick re-check, not a full
   re-run, given how large an effect a genuine mismatch would produce (§11 found ~47% from
   mismatch alone, vs. the ~15%/~6% real engine effects at matched precision).

None of these block writing the detection/estimation script, but all seven should be called out in
the script's own comments (mirroring how this document treats them), not silently smoothed over —
consistent with this whole research effort's practice of recording what's actually confirmed versus
what's still an approximation.
