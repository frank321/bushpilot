# Bushpilot

**Charted routes through the chaotic LLM landscape.**

*From "where do I start?" to running locally. Bushpilot has done the research, tested the models, and watches your memory so you can just fly.*

## Why

New models, frameworks, and inference tricks appear every week, and the tutorials you read six months ago are already out of date. Getting a local LLM running on a Mac usually means digging through instructions, learning a new framework, and guessing which model and context size won't crash your machine.

Bushpilot skips that. It ships a curated catalog of models that have been tested on Apple Silicon (`models.json`), serves them with Apple's MLX (`mlx_vlm`, plus mlx-dspark for speculative decoding), and attaches Claude Code, OpenCode, Aider, Cline, or Pi to the result.

## What it does

- **Curated, tested models:** pick from a catalog where each entry has verified memory and long-context behavior, instead of researching models yourself.
- **Hardware-aware picker:** detects your chip and free memory, greys out models that won't fit, and computes a safe context size and estimated tok/s for each.
- **Memory backstop:** a live dashboard watches memory, and the server is torn down automatically if free memory drops below 10% or pressure hits CRITICAL.
- **One shared server + live model hot-swap:** attach Claude Code, OpenCode, Aider, Cline, and/or Pi (from as many terminals as you like) to the same running server, and swap models live without restarting your coding tool (`m` on the dashboard, or `Tab`-select multiple models in the picker and switch `[BushPilot]` models directly inside your tool).
- **Single-terminal or two-terminal launch:** run `./launch_tool.sh` on its own and it will automatically split a pane/window (iTerm2, `tmux`, or macOS Terminal.app) for the server dashboard, wait for it to come up, and attach your coding tool in the original pane.
- **Speculative decoding:** optional speedup variants for supported models.
- **Self-contained:** everything (venv, weights, logs, state) lives under `~/.bushpilot/`. Delete that directory for a full reset, or run `./uninstall.sh` to remove it along with the OpenCode/Pi provider registrations, PATH symlinks, and this project folder.

## Requirements

- Mac with Apple Silicon (M1 or later), macOS
- [Homebrew](https://brew.sh) and Python 3.10+
- Claude Code, OpenCode, Aider, Cline, and/or Pi
- `fzf` (optional, for nicer pickers): `brew install fzf`

## Quick start

Get the latest release bundle (`bushpilot.zip`) — no build step needed:

**→ [Download the latest release](https://github.com/frank321/bushpilot/releases/latest)** — unzip, `cd` in, then run `./launch_tool.sh`.

Or clone the repo and run from source:

```bash
git clone https://github.com/frank321/bushpilot.git
cd bushpilot
```

Then, **single-terminal launch** (auto-splits iTerm2, `tmux`, or Terminal.app if the server isn't running yet):

```bash
./launch_tool.sh
```

Or run the server and tool in **two separate terminals**:

**Terminal 1:** start the model server and leave it running (`Tab` in the picker to pre-register multiple models; press `m` on the dashboard anytime to hot-swap models).

```bash
./start_mlx_server.sh
```

**Terminal 2:** attach a coding tool.

```bash
./launch_tool.sh
```

Press **Ctrl-C** or **`q`** in the server terminal to stop everything, or **`m`** to swap the active model live.

Skip the pickers for scripting:

```bash
./start_mlx_server.sh qwen3.5-9b-mlx-8bit mlx-community/Qwen3.5-9B-MLX-8bit 131072
./launch_tool.sh claude
```

## Files

| File | Purpose |
|---|---|
| `start_mlx_server.sh` | Brings up the model server, shows the dashboard, enforces the memory + backend-liveness backstops |
| `launch_tool.sh` | Attaches Claude Code, OpenCode, Aider, Cline, or Pi to the running server |
| `model_advisor.py` | Thin shim over `bushpilot/` (hardware detection and per-model context/speed recommendations) |
| `bushpilot/` | The advisor's logic, one concern per module -- see `AGENTS.md` for the breakdown |
| `models.json` | The curated model catalog |
| `uninstall.sh` | Removes the running server, `~/.bushpilot`, the OpenCode provider registration, PATH symlinks, and this project folder |
| `add_model.py` | Generates an untested `models.json` entry from a Hugging Face repo id, URL, or model name (`--write` to append; `--sync-verified` to ingest `verify_max_kv_size.sh` and `sweep_model.py` results) |
| `verify_max_kv_size.sh`, `verify_long_context_memory.sh` | Verification tools behind the catalog's tested data |
| `docs/` | GitHub/GitLab Pages site, release bundle build, and this project's research/history docs |
| `tests/` | `pytest` suite (`bushpilot`'s recommendation math and picker layout) + `tests/shell/` bats suite (shell backstops) -- see Tests below |
| `ci.sh` | Runs the full pytest + ruff + shellcheck + bats suite locally (same checks as CI) |

## Tests

`tests/` covers `bushpilot`'s pure logic -- the memory/context/speed recommendation math, the picker's terminal-width-aware column layout, and its sort state machine -- with synthetic chip/memory inputs, no real Apple Silicon required:

```bash
pip install -r requirements-dev.txt
pytest
```

`tests/shell/` covers `start_mlx_server.sh`'s memory and backend-liveness backstops (the consecutive-polls debounce that must never tear down a session over one transient blip) by extracting their real function bodies into an isolated, hermetic bats fixture -- no real server, no real memory pressure:

```bash
bats tests/shell/
```

`./ci.sh` runs both of the above plus `ruff` and `shellcheck`, skipping whichever of those two isn't installed locally (CI always has all four).

This is deliberately separate from `verify_max_kv_size.sh` / `verify_long_context_memory.sh`, which check things no unit test can (whether `mlx_vlm.server` itself enforces its own flags on real hardware) and need a real Mac + a downloaded model to run.

## More documentation

- [QUICK_START.md](QUICK_START.md): the fast path
- [MLX_SERVER_USAGE.md](MLX_SERVER_USAGE.md): full usage guide
- [docs/HARDWARE_SPEED_ESTIMATION.md](docs/HARDWARE_SPEED_ESTIMATION.md): research behind the recommendations
- [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md) and [docs/PHASE_PLAN.md](docs/PHASE_PLAN.md): project history and plan

## License

[MIT](LICENSE)
