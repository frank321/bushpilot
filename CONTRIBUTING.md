# Contributing

Thanks for your interest in Bushpilot! This is a local-first tool for running
LLMs on Apple Silicon Macs, so most changes can be developed and tested on any
machine — only the live-hardware scripts need a real Mac.

## Getting set up

```bash
git clone https://github.com/frank321/bushpilot.git
cd bushpilot
pip install -r requirements-dev.txt   # pytest, ruff
```

## Running the checks

Everything that can run without Apple Silicon or a downloaded model:

```bash
./ci.sh          # pytest + ruff + shellcheck + bats (skips what isn't installed)
pytest           # pure-logic suite only
bats tests/shell # shell backstop safety logic
```

`verify_max_kv_size.sh` and `verify_long_context_memory.sh` are **live-hardware
checks** — they need a real Mac with a downloaded model and cannot run in CI.
Please note in your PR if a change touches them but you couldn't execute them.

## Guidelines

- **Python is stdlib-only** in the shipped path. No new runtime dependencies.
- **Keep the stdout contract.** `model_advisor.py` and every module under
  `bushpilot/` write human output to stderr via `log()`. The bash scripts capture
  stdout with `$(...)` and `json.loads` it — anything extra on stdout corrupts
  model selection. See `AGENTS.md` for the full invariants list.
- **Shell scripts:** bash, `set -euo pipefail`, state under `~/.bushpilot/`.
- **Matching the surrounding style** matters more than personal preference.
- **Add tests** for logic changes (`tests/` for Python, `tests/shell/` for bash
  backstops).

## Pull requests

1. Keep changes focused; one concern per PR.
2. Run `./ci.sh` and make sure it's green.
3. Describe *what* changed and *why*, and how you verified it.

By contributing, you agree that your contributions are licensed under the
project's [MIT License](LICENSE).
