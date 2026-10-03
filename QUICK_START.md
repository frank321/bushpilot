# Quick Start: Bushpilot

Run Claude Code, OpenCode, Aider, Cline, or Pi against a model served locally on-device by Apple's MLX,
instead of a cloud API. Hardware-aware: it detects your Mac's chip and free memory
and recommends a model + context size that's actually safe to run.

**Rewritten for the two-script architecture (PHASE_PLAN.md Phase 4).** There is no
LiteLLM, no PostgreSQL, and no `--monitor`/`--interactive`-flag-per-run design
anymore — see `MLX_SERVER_USAGE.md` for the full guide. This doc is the fast path.

## First Time Setup

**Terminal 1** — start the model server and leave it running:

```bash
./start_mlx_server.sh
```

This detects your free memory, shows a picker of models that actually fit (with a
computed context size and estimated tok/s for each), downloads the model if needed,
starts `mlx_vlm.server`, and takes over the terminal with a live dashboard.

**Terminal 2** — attach a coding tool:

```bash
./launch_tool.sh
```

This shows a picker (Claude Code / OpenCode / Aider / Cline / Pi), checks the server from Terminal 1 is
healthy and has room for that tool's context preload, then launches straight into it.

Open a **third** terminal and run `./launch_tool.sh` again with another tool to
have multiple attached to the same running model at once.

## Common Scenarios

**Skip both pickers (scripting / restarts):**
```bash
./start_mlx_server.sh qwen3.5-9b-mlx-8bit mlx-community/Qwen3.5-9B-MLX-8bit 131072
./launch_tool.sh claude
```

**16GB Mac:** the model picker greys out anything that won't fit and shows why —
follow its recommendation rather than forcing a bigger model/context than it suggests.

**24GB+ Mac:** the picker will typically offer the full 131K context window for the
default model (`qwen3.5-9b-mlx-8bit`) unquantized.

## Stopping Everything

In the `start_mlx_server.sh` terminal: **Ctrl-C** or **`q`**. Both do a full teardown.
Closing that terminal (or a dead SSH session) does the same thing automatically. A
memory-pressure backstop will also trigger the same teardown on its own if free
memory gets dangerously low.

## Live Dashboard

```
┌─ bushpilot ── mlx-community/Qwen3.5-9B-MLX-8bit ── up 00:04:12 ─┐
│ port 11435   context 131072/262144   APC: on                                   │
├───────────────────────────────────────────────────────────────────────────────┤
│ REQUESTS        completed 14   failed 0   in-flight 0                          │
│ LIFETIME        prompt 612,304 tok   generated 3,208 tok   avg decode 231 tok/s │
├───────────────────────────────────────────────────────────────────────────────┤
│ LATEST REQUEST  prefill 10,180 tok/s   decode 231 tok/s   ttft 3.06s           │
│                 elapsed 3.53s   finish_reason: tool_use                        │
├───────────────────────────────────────────────────────────────────────────────┤
│ APC CACHE       hit rate 78%   hits 412 / misses 116   evictions 0             │
│                 mem pool 812/4096 blocks   disk 1.2/5.0 GB                     │
├───────────────────────────────────────────────────────────────────────────────┤
│ SYSTEM          free mem 34%   pressure: normal   MLX RSS 6.1 GB   peak 8.2 GB │
├───────────────────────────────────────────────────────────────────────────────┤
│ last error: none                                                                │
└───────────────────────────────────────────────────────────────────────────────┘
  Ctrl-C / q : stop everything
```

All numbers come free from `mlx_vlm.server`'s own `/metrics` and `/v1/cache/stats`
endpoints — no separate instrumentation needed.

## Full Reset

```bash
rm -rf ~/.bushpilot
```

Removes the venv, downloaded model weights, APC cache, state, and logs.

## Questions?

- **`MLX_SERVER_USAGE.md`** — the full usage guide, both scripts' env vars, troubleshooting
- **`HARDWARE_SPEED_ESTIMATION.md`** — the research behind the context-size math and speed estimates
- **`PHASE_PLAN.md`** — the redesign this project went through, and why
