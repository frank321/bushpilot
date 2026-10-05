#!/usr/bin/env python3
"""
model_advisor.py — thin compatibility shim over the bushpilot/ package.

This used to be one 1,743-line script; the logic now lives in bushpilot/ (see
that package's __init__.py docstring for the module breakdown, and AGENTS.md's
file map). This file stays at the repo root, under this name, re-exporting
bushpilot's full public surface, so that:
  - every start_mlx_server.sh / launch_tool.sh call site (`python3
    "$SCRIPT_DIR/model_advisor.py" ...`) keeps working unchanged,
  - docs/build_release.sh's release bundle keeps shipping a model_advisor.py
    (it now also ships the bushpilot/ package alongside it),
  - `import model_advisor as ma` in tests/ keeps seeing every name at the top
    level, matching the pre-split flat shape those tests were written against.

Design notes (unchanged by the split — see bushpilot/kv.py, bushpilot/speed.py
for the full reasoning behind each):
- Context-size recommendation is an EXACT calculation from each model's
  "kv_config" field in models.json, not an estimate.
- Token-speed is a genuine ESTIMATE, always labeled as such, overridden by a
  real per-(chip, model) calibration measurement when one exists.
- Interactive UI goes to stderr. The final machine-readable selection is the
  ONLY thing printed to stdout, as one JSON line, so the calling bash script
  can capture it cleanly with `SELECTION=$(python3 model_advisor.py)` without
  scraping prompts out of it. This holds for every bushpilot/ module too.

Usage:
  python3 model_advisor.py                  → interactive menu, prints selection JSON
  python3 model_advisor.py --list-json       → non-interactive: print every model's
                                                computed recommendation as JSON, no prompt
  python3 model_advisor.py --list-compact    → fzf-facing compact table (see
                                                bushpilot.picker.display.print_list_compact)
  python3 model_advisor.py --detail MODEL_ID → prints that model's description block
  python3 model_advisor.py --resolve MODEL_ID → prints that model's selection JSON directly
  python3 model_advisor.py --tool claude     → also estimate & factor in Claude Code's/
                                                OpenCode's context preload overhead
                                                — see bushpilot.preload.estimate_preload_overhead
  python3 model_advisor.py --save-calibration MODEL_ID --decode-tok-s N
                                              [--context-tokens N] [--ttft-s N]
                                              [--peak-memory-gb N]
                                              [--prefill-step-size N]
                                              [--kv-bits-safe-at-long-context 1|0]
                                              → records a real measurement from a
                                                session that already ran (called by
                                                start_mlx_server.sh at teardown, and
                                                by sweep_model.py's richer per-model
                                                sweep) — see bushpilot.calibration
"""
import sys  # noqa: F401 -- re-exported for `model_advisor.sys` call sites (tests)

from bushpilot import *  # noqa: F401,F403

if __name__ == "__main__":
    main()  # noqa: F405 -- provided by the star import above, via bushpilot.__all__
