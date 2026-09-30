# MLX Server & Tool Launcher — Usage Guide

**Renamed from `LITELLM_PROXY_USAGE.md` (PHASE_PLAN.md Phase 4).** LiteLLM and
PostgreSQL are gone from this project entirely — see PHASE_PLAN.md's "drop LiteLLM +
Postgres entirely" decision. `mlx_vlm.server` natively serves both the OpenAI
(`/v1/chat/completions`) and Anthropic (`/v1/messages`) wire formats, so there is no
translation-layer proxy process anymore, and this doc's old three-service
description (MLX + PostgreSQL + LiteLLM) is stale — it now describes one process.

## Overview

There are two scripts, each with one job:

1. **`start_mlx_server.sh`** — the supervisor. Brings up `mlx_vlm.server` safely
   (memory-aware, won't crash the machine), shows a live status dashboard in the
   foreground, and tears everything down on Ctrl-C / `q` / a closed terminal / a
   memory emergency. Run it in its own terminal and leave it running — it's the
   thing keeping the model server alive for every tool/terminal attached to it.
2. **`launch_tool.sh`** — the launcher. Attaches Claude Code or OpenCode to whatever
   `start_mlx_server.sh` already has running. Run it (in a *different* terminal) each
   time you want to open a coding session.

This split means multiple terminals/tools can share one running model — you start the
server once, then run `launch_tool.sh claude` in one terminal and
`launch_tool.sh opencode` in another, both talking to the same `mlx_vlm.server`
process, no matter which wire format each client speaks.

---

## Prerequisites

- **macOS on Apple Silicon** (MLX is Apple-Silicon-only)
- **Python 3.10+** available somewhere on the machine (`start_mlx_server.sh` scans
  `/opt/homebrew/bin`, `/usr/local/bin`, and `$PATH` for the newest qualifying
  `python3.X` and builds its own venv with it — your system `python3` doesn't need to
  meet this bar itself)
- **curl**
- Claude Code and/or OpenCode installed, for whichever tool(s) you plan to use
- **`fzf`** (optional) — both scripts' interactive pickers use it if present, and fall
  back to a plain numbered menu if it's missing

Everything else (venv, downloaded model weights, APC disk cache, state, logs) lives
under `~/.bushpilot/` — the whole install is removable by deleting that one
directory.

---

## First Run

```bash
./start_mlx_server.sh
```

With no arguments this runs `model_advisor.py`'s interactive picker: it detects your
Mac's chip and free wired memory, shows every model in `models.json` (greying out ones
that don't currently fit, with the reason why), and lets you pick one. It then:

1. Bootstraps `~/.bushpilot/venv` (creates it if needed, installs/upgrades
   `mlx-lm`/`mlx-vlm>=0.7.0`)
2. Pre-downloads the model if it isn't cached yet (so a killed download can resume,
   unlike `mlx_vlm.server`'s own lazy on-demand fetch)
3. Starts `mlx_vlm.server` on port 11435 with Automatic Prefix Caching always on
4. Runs a startup smoke check against `/v1/messages`, `/metrics`, and
   `/v1/cache/stats` before declaring itself ready
5. Takes over the terminal with a live dashboard (requests, token throughput, APC hit
   rate, free memory) refreshed every ~2 seconds

Leave that terminal open. In a **different** terminal:

```bash
./launch_tool.sh
```

This runs `launch_tool.sh`'s own tool picker (Claude Code / OpenCode), checks that the
server from step above is actually healthy, checks that the picked tool's own context
preload still leaves enough room in whatever context size you chose, and then `exec`s
straight into `claude` or `opencode` with all the right environment variables set.

### Skipping the pickers

```bash
./start_mlx_server.sh qwen3.5-9b-mlx-8bit mlx-community/Qwen3.5-9B-MLX-8bit 131072
./launch_tool.sh claude
./launch_tool.sh opencode
```

---

## Stopping the server

There's no separate stop script anymore. In the `start_mlx_server.sh` terminal, press
**Ctrl-C** or **`q`** — both do a full teardown (kills `mlx_vlm.server`, cleans up the
state file). A closed terminal or dead SSH session (SIGHUP) and a plain SIGTERM both
trigger the same teardown path. There's no "detach without stopping" mode: the
dashboard process *is* what's keeping the server up, so closing it always stops
everything, for every tool/terminal attached to it. (An automatic memory-pressure
backstop also triggers the same teardown if free memory gets dangerously low while
you're not looking — see `MONITORING_ARCHITECTURE.md`.)

---

## What each script sets up / does

### `start_mlx_server.sh`

- **MLX server** (port 11435, `127.0.0.1` only) — the only background process this
  project runs. No PostgreSQL, no LiteLLM.
- **Single home**: `~/.bushpilot/{venv,models,state,logs,apc-cache}/`
- **State file**: `~/.bushpilot/state/mlx-loaded.txt` —
  `<huggingface_id>|<context_window>|<kv_bits>`, read by `launch_tool.sh` to know
  what's actually running.
- **Reliability checks carried over from the old script**: a real (not just `/health`)
  one-token inference probe before declaring the server ready, detection of a foreign
  process squatting on port 11435, and the "different model already running, replace
  it?" confirmation prompt.

### `launch_tool.sh`

- Checks the server is up (offers to `exec` `start_mlx_server.sh` if not — see below)
- Checks the picked tool's estimated context preload still leaves a usable amount of
  the server's loaded context (errors out with a clear message instead of letting you
  hit a `PromptTooLongError` mid-session)
- **Claude Code**: sets `ANTHROPIC_BASE_URL` to the MLX server directly (no `/v1`
  suffix — `mlx_vlm.server` registers `/v1/messages` itself), `ANTHROPIC_MODEL`, all
  three `ANTHROPIC_DEFAULT_{HAIKU,SONNET,OPUS}_MODEL` vars (works around Claude Code
  sending a hardcoded `claude-haiku-4-5-...` for an internal call regardless of
  `ANTHROPIC_MODEL`), dummy credentials, and `CLAUDE_CODE_MAX_CONTEXT_TOKENS` (the
  loaded context minus a ~32K output-token reserve)
- **OpenCode**: registers the model under a `mlx-direct` provider
  (`@ai-sdk/openai-compatible`) in `~/.config/opencode/opencode.jsonc`, keyed by the
  literal Hugging Face repo id (what actually goes on the wire) with a hardcoded
  `baseURL` pointed at the MLX server's `/v1` — no LiteLLM provider entry is written
  or read anymore

If the server isn't running when `launch_tool.sh` starts, it asks to confirm, then
`exec`s `start_mlx_server.sh` directly in that same terminal (replacing itself, not
backgrounding) — because the supervisor's own dashboard is what keeps the server
alive. Once it's up, run `launch_tool.sh` again in a **different** terminal.

---

## Logs & Troubleshooting

| Log | Purpose |
|-----|---------|
| `~/.bushpilot/logs/mlx-server.log` | `mlx_vlm.server` stdout/stderr |

### "No healthy local model server detected"

`launch_tool.sh` couldn't reach `http://127.0.0.1:11435/health` or found no state
file. Start (or check) `start_mlx_server.sh` in its own terminal first.

### "Cannot import mlx_vlm" / package errors

Remove `~/.bushpilot/venv` and re-run `start_mlx_server.sh` — it rebuilds the
venv from scratch.

### Preload-fit error from `launch_tool.sh`

The tool's estimated system-prompt + tool-schema preload doesn't leave enough of the
currently-loaded context window for an actual conversation. Restart
`start_mlx_server.sh` with a larger context argument, or pick the other tool.

### Full reset

```bash
rm -rf ~/.bushpilot
```

Removes the venv, downloaded model weights, APC cache, state, and logs — everything
this project manages. Nothing else on the machine is touched.

---

## Related Files

- **`model_advisor.py`** — hardware detection + per-model context recommendation +
  interactive/`fzf` model picker, called by `start_mlx_server.sh`
- **`models.json`** — the curated, live-verified model catalog
- **`HARDWARE_SPEED_ESTIMATION.md`** — the research behind the context-size math and
  speed estimates
- **`MONITORING_ARCHITECTURE.md`** — the memory-pressure backstop now folded into
  `start_mlx_server.sh`'s own poll loop
- **`PHASE_PLAN.md`** — the redesign this doc reflects, and the decisions behind it
