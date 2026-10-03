# Bushpilot

**Charted routes through the chaotic LLM landscape.**

*From "where do I start?" to running locally. Bushpilot has done the research, tested the models, and watches your memory so you can just fly.*

## Why

New models, frameworks, and inference tricks appear every week, and the tutorials you read six months ago are already out of date. Getting a local LLM running on a Mac usually means digging through instructions, learning a new framework, and guessing which model and context size won't crash your machine.

Bushpilot skips that. It ships a curated catalog of models that have been tested on Apple Silicon (`models.json`), serves them with Apple's MLX (`mlx_vlm`, plus mlx-dspark for speculative decoding), and attaches Claude Code, OpenCode, Aider, or Cline to the result.

## What it does

- **Curated, tested models:** pick from a catalog where each entry has verified memory and long-context behavior, instead of researching models yourself.
- **Hardware-aware picker:** detects your chip and free memory, greys out models that won't fit, and computes a safe context size and estimated tok/s for each.
- **Memory backstop:** a live dashboard watches memory, and the server is torn down automatically if free memory drops below 10% or pressure hits CRITICAL.
- **One shared server:** attach Claude Code, OpenCode, Aider, and/or Cline (from as many terminals as you like) to the same running model.
- **Speculative decoding:** optional speedup variants for supported models.
- **Self-contained:** everything (venv, weights, logs, state) lives under `~/.bushpilot/`. Delete that directory for a full reset, or run `./uninstall.sh` to remove it along with the OpenCode provider registration, PATH symlinks, and this project folder.

## Requirements

- Mac with Apple Silicon (M1 or later), macOS
- [Homebrew](https://brew.sh) and Python 3.10+
- Claude Code, OpenCode, Aider, and/or Cline
- `fzf` (optional, for nicer pickers): `brew install fzf`

## Quick start

**Terminal 1:** start the model server and leave it running.

```bash
./start_mlx_server.sh
```

**Terminal 2:** attach a coding tool.

```bash
./launch_tool.sh
```

Press **Ctrl-C** or **`q`** in Terminal 1 to stop everything.

Skip the pickers for scripting:

```bash
./start_mlx_server.sh qwen3.5-9b-mlx-8bit mlx-community/Qwen3.5-9B-MLX-8bit 131072
./launch_tool.sh claude
```

## Files

| File | Purpose |
|---|---|
| `start_mlx_server.sh` | Brings up the model server, shows the dashboard, enforces the memory backstop |
| `launch_tool.sh` | Attaches Claude Code, OpenCode, Aider, or Cline to the running server |
| `model_advisor.py` | Hardware detection and per-model context/speed recommendations |
| `models.json` | The curated model catalog |
| `uninstall.sh` | Removes the running server, `~/.bushpilot`, the OpenCode provider registration, PATH symlinks, and this project folder |
| `add_model.py` | Generates an untested `models.json` entry from a Hugging Face repo id, URL, or model name (`--write` to append) |
| `verify_max_kv_size.sh`, `verify_long_context_memory.sh` | Verification tools behind the catalog's tested data |
| `docs/` | GitHub Pages site and release bundle build |
| `tests/` | Unit tests for `model_advisor.py`'s recommendation math and picker layout (see Tests below) |

## Tests

`tests/` covers `model_advisor.py`'s pure logic -- the memory/context/speed recommendation math, the picker's terminal-width-aware column layout, and its sort state machine -- with synthetic chip/memory inputs, no real Apple Silicon required:

```bash
pip install -r requirements-dev.txt
pytest
```

This is deliberately separate from `verify_max_kv_size.sh` / `verify_long_context_memory.sh`, which check things no unit test can (whether `mlx_vlm.server` itself enforces its own flags on real hardware) and need a real Mac + a downloaded model to run.

## More documentation

- [QUICK_START.md](QUICK_START.md): the fast path
- [MLX_SERVER_USAGE.md](MLX_SERVER_USAGE.md): full usage guide
- [HARDWARE_SPEED_ESTIMATION.md](HARDWARE_SPEED_ESTIMATION.md): research behind the recommendations
- [PROJECT_STATUS.md](PROJECT_STATUS.md) and [PHASE_PLAN.md](PHASE_PLAN.md): project history and plan
