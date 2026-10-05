# Bushpilot — agent notes

Local LLM cockpit for Apple Silicon Macs: a curated, tested model catalog
(`models.json`), a hardware-aware advisor, an MLX model server with a memory
backstop, and attachers for Claude Code / OpenCode / Aider / Cline / Pi.

## Run / test

```bash
pip install -r requirements-dev.txt   # pytest, ruff
pytest                                # pure-logic suite, no Apple Silicon needed
bats tests/shell                      # shell backstop safety logic (needs `bats`)
shellcheck *.sh                       # if installed
python3 model_advisor.py --list-json  # non-interactive smoke test of the advisor
```

`./ci.sh` runs everything that can run without Apple Silicon or a downloaded
model. The two `verify_*.sh` scripts are live-hardware checks — they cannot run
in CI or on any non-Mac machine.

## File map

| Path | Purpose |
|---|---|
| `start_mlx_server.sh` | Server lifecycle: env, venv, download, MLX launch, dashboard loop, memory + backend-liveness backstops, teardown calibration |
| `launch_tool.sh` | Attaches a coding tool to the running server (reads `~/.bushpilot/config.json`) |
| `model_advisor.py` | Thin shim over the `bushpilot/` package (kept at root so all bash call sites and the release bundle stay unchanged) |
| `bushpilot/` | The advisor logic, by concern: `hardware.py` (chip/memory), `kv.py` (context/KV math), `speed.py` (tok/s estimate), `prefill.py` (prefill step size), `calibration.py` (per-(chip,model) measurements), `preload.py` (tool context overhead), `recommend.py` (row builder), `picker/` (`cells.py` row model, `sort.py`, `display.py`), `cli.py` (`main`) |
| `sweep_model.py` | Per-(chip, model) prefill-step-size sweep harness |
| `add_model.py` | Generates a catalog entry from a HF repo id (never marks it tested) |
| `uninstall.sh` | Full removal: server, `~/.bushpilot`, provider registration, PATH symlinks, project folder |
| `models.json` | The curated catalog — source of truth for what the advisor computes |
| `verify_max_kv_size.sh`, `verify_long_context_memory.sh` | Live-hardware verification behind the catalog's tested data |
| `tests/` | `pytest` suite (advisor math, picker layout/sort, sweep harness) + `tests/shell/` bats suite (bash backstops) |
| `docs/` | GitHub/GitLab Pages site, release-bundle build (`build_release.sh`), and the project's research/history docs |
| `ci.sh`, `.github/workflows/ci.yml` | pytest + ruff + shellcheck + bats, run locally or on push/PR |
| `pyproject.toml`, `.shellcheckrc` | ruff and shellcheck config, each with inline reasoning for every disabled check |

## Invariants — do not break

1. **`model_advisor.py` stdout discipline.** All interactive UI goes to stderr;
   the ONLY thing on stdout is one machine-readable JSON line (or the
   `--list-json` / `--list-compact` formats the bash side parses). The bash
   scripts capture stdout with `$(...)` and `json.loads` it — anything else on
   stdout corrupts the selection. This holds for the shim AND every module in
   `bushpilot/` (use `log()` for human output, never bare `print` to stdout
   outside the documented emit paths).
2. **`models.json` schema.** Every entry carries `kv_config` (extracted once
   from the HF `config.json`, baked in — the advisor must work on a machine
   with no models downloaded), `size_gb`, `active_weight_gb` (dense models),
   `status` (`tested`/`untested`), `serving`, `verification`. `add_model.py`
   must keep producing entries that match what `bushpilot/` reads; `status`
   only becomes `tested` after `verify_*.sh` passes.
3. **Deliberate duplicated constants.** `DEFAULT_PREFILL_STEP_SIZE` in
   `bushpilot/prefill.py` and `PREFILL_STEP_SIZE` in `start_mlx_server.sh` are
   the same number on purpose (bash owns the launch flag; they are NOT shared
   across the bash/python boundary). Same for the memory "free" semantics:
   both sides read `kern.memorystatus_level` so the fit check and the backstop
   can never disagree. If you change one, change the other and the comment
   that documents the pairing.
4. **Calibration files are a cache, never a requirement.** Missing or corrupt
   `~/.bushpilot/calibration/*.json` must degrade to the formula/default path,
   never error. `save_calibration` writes atomically (tmp + `os.replace`).
5. **Release bundle.** `docs/build_release.sh` ships the runtime files (the
   `bushpilot/` package is part of it) — never the research docs in `docs/`.
   `docs/*.zip` are build artifacts, gitignored, and must not be committed.
6. **Safety-critical bash.** The memory backstop and backend-liveness check in
   `start_mlx_server.sh` use consecutive-poll debouncing on purpose — a single
   blip must never tear down a healthy session. Covered by `tests/shell/`.

## Conventions

- Shell scripts: bash, `set -euo pipefail`, `SCRIPT_PATH`/`BUSHPILOT_HOME`
  bootstrap near the top, everything under `~/.bushpilot/`.
- Python: stdlib only. No dependencies in the shipped path (pytest is dev-only).
- Docs: current-state guides live at the root (`README.md`, `QUICK_START.md`,
  `MLX_SERVER_USAGE.md`); research/history lives in `docs/`.
