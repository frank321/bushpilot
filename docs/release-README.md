# Bushpilot

Run Claude Code, OpenCode, Aider, Cline, or Pi against a model served locally on your Mac by Apple's
MLX, instead of a cloud API. Hardware-aware: it detects your Mac's chip and free
memory and recommends a model + context size that's actually safe to run, rather
than crashing your machine or silently truncating context.

## Requirements

- A Mac with Apple Silicon (M1 or later)
- macOS
- [Homebrew](https://brew.sh)
- Python 3.10+ somewhere on the machine (the setup script will find or help you
  install one if needed)
- Claude Code, OpenCode, Aider, Cline, and/or Pi, for whichever tool you plan to use
- `fzf` (optional) — both scripts' interactive pickers use it if present, and fall
  back to a plain numbered menu if it's missing:
  ```bash
  brew install fzf
  ```

## Setup

1. Unzip this archive and open a terminal in the resulting folder.
2. Make the scripts executable if they aren't already:
   ```bash
   chmod +x start_mlx_server.sh launch_tool.sh
   ```

## Running it

**Terminal 1** — start the model server and leave it running:

```bash
./start_mlx_server.sh
```

This detects your free memory, shows you a picker of models that actually fit
(with a computed context size and estimated tok/s for each), downloads the model
if needed, and starts serving it with a live status dashboard.

**Terminal 2** — attach a coding tool:

```bash
./launch_tool.sh
```

This shows a picker (Claude Code / OpenCode / Aider / Cline / Pi), checks the server from Terminal 1 is
healthy, and launches straight into it. You can open additional terminals and run
`./launch_tool.sh` again to attach another tool to the same running model.

To stop everything, go back to the Terminal 1 window and press **Ctrl-C** or
**`q`**.

Everything this project manages (downloaded model weights, its Python
environment, logs, and state) lives under `~/.bushpilot/` — remove that
one directory for a full reset.

## Uninstalling

```bash
./uninstall.sh
```

This stops the model server if it's running, removes the `mlx-direct` provider
OpenCode registered, removes any PATH symlinks `start_mlx_server.sh`/
`launch_tool.sh` created, deletes `~/.bushpilot` (all downloaded models, logs,
and state), and finally deletes this folder. Pass `-y`/`--yes` to skip the
confirmation prompt.

## More

This is the packaged runtime bundle. For the full source, development notes, and
the research behind the hardware-aware recommendations, see the project's
repository (link on the page you downloaded this from).
