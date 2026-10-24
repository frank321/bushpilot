# Docs

Research, design rationale, and verification data behind Bushpilot. None of this is
required to *use* the tool — for that, start at the [root README](../README.md) and
[QUICK_START.md](../QUICK_START.md).

These docs are kept in the repo because the code cites them: `start_mlx_server.sh`,
`bushpilot/*.py`, and `sweep_model.py` all reference specific sections here for the
"why" behind non-obvious decisions.

## Reference — read these first

| Doc | What it covers |
|---|---|
| [HARDWARE_SPEED_ESTIMATION.md](HARDWARE_SPEED_ESTIMATION.md) | The chip/memory detection and tok/s estimation research — every formula in `bushpilot/hardware.py`, `kv.py`, and `speed.py`, traceable to its sources and live calibration runs. The most useful doc for understanding *why* a recommendation is what it is. |
| [PHASE_PLAN.md](PHASE_PLAN.md) | The current project plan and the decision record for the two-script architecture. Where most design "why"s live. |

## Verification data

Raw results behind `models.json`'s `verification` / `tool_calling` fields:

| File | Produced by |
|---|---|
| [mlx_verification_results.json](mlx_verification_results.json) | `verify_max_kv_size.sh` — the short-prompt memory-admission gate (per-model `SAFE` / unsafe verdicts). |
| [mlx_long_context_stress_results.json](mlx_long_context_stress_results.json) | `verify_long_context_memory.sh` — the long-context prefill memory-stress test. |
| [gemma_speed_benchmark_results.json](gemma_speed_benchmark_results.json) | Raw decode/prompt tok/s timings for the Gemma entries, feeding the speed-correction factors. |

## Implementation history

Historical records of how the project got here. Several are explicitly marked
**superseded** at the top and are kept for their reasoning, not as current state —
read the status banner before trusting any path or script name in them.

| Doc | Status |
|---|---|
| [PROJECT_STATUS.md](PROJECT_STATUS.md) | Superseded tracker. Kept for the model-verification and tool-calling findings, which are still accurate. |
| [NEXT_SESSION_HANDOFF.md](NEXT_SESSION_HANDOFF.md) | The `mlx-dspark` / speculative-decoding integration record, including the gotchas that cost real time. |
| [NEXT_SESSION_HANDOFF_PREFILL_SWEEP.md](NEXT_SESSION_HANDOFF_PREFILL_SWEEP.md) | Spec + rationale for the prefill-step-size sweep harness (`sweep_model.py`). |
| [MONITORING_ARCHITECTURE.md](MONITORING_ARCHITECTURE.md) | Superseded. The memory-backstop threshold *design* research, still the source of truth for those numbers even though the logic now lives in `start_mlx_server.sh`. |
| [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md) | Superseded. Design notes for the pre-redesign LiteLLM-fronted architecture. |

## Build & site

| File | Purpose |
|---|---|
| [build_release.sh](build_release.sh) | Builds the end-user `bushpilot.zip` bundle (runtime files only, no internal docs). Run automatically by `.github/workflows/release.yml` on a `v*` tag. |
| [index.html](index.html) | The GitHub Pages landing site. |
| [release-README.md](release-README.md) | The standalone `README.md` that ships *inside* the zip. |
