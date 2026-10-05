"""Command-line entry point: flag parsing, the --save-calibration write path, and
the three non-interactive presentation modes (--list-json, --list-compact,
--detail/--resolve), falling back to the interactive plain-menu picker."""
import json
import os
import sys

from .calibration import save_calibration
from .hardware import get_chip_info, get_current_wired_gb, log
from .kv import MIN_USABLE_CONTEXT
from .picker.display import format_model_body_lines, print_list_compact, print_menu
from .picker.sort import SORT_KEYS
from .preload import estimate_preload_overhead
from .recommend import build_recommendations

# The catalog lives at the repo root next to the model_advisor.py shim (and inside
# docs/build_release.sh's release bundle), one directory ABOVE this package --
# unlike the pre-split layout, this is not relative to this file's own directory.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_JSON = os.path.join(REPO_ROOT, "models.json")


def _flag_value(argv, flag):
    if flag in argv:
        idx = argv.index(flag)
        if idx + 1 < len(argv):
            return argv[idx + 1]
    return None


def emit_selection_json(m, selected):
    # Machine-readable result — the ONLY thing on stdout.
    # kv_bits is None when this model's recommended context fits free memory
    # unquantized (the common case now — see docs/HARDWARE_SPEED_ESTIMATION.md §14); the
    # caller (start_mlx_server.sh) must omit --kv-bits entirely in that case rather
    # than treating None as a literal value.
    print(json.dumps({
        "id": m["id"],
        "huggingface": m["huggingface"],
        "size_gb": m["size_gb"],
        "recommended_context": selected["recommended_context"],
        "kv_bits": selected["kv_bits"],
        "kv_bits_apc_caveat": selected["kv_bits_apc_caveat"],
        "estimated_tok_s": selected["estimated_tok_s"],
        "concurrency": selected["concurrency"],
        "prefill_step_size": selected["prefill_step_size"],
    }))


def main():
    argv = sys.argv[1:]
    list_json = "--list-json" in argv
    list_compact = "--list-compact" in argv
    detail_id = _flag_value(argv, "--detail")
    resolve_id = _flag_value(argv, "--resolve")
    tool = _flag_value(argv, "--tool")
    concurrency_flag = _flag_value(argv, "--concurrency")
    # Explicit request wins; anything absent/invalid falls back to auto mode (None)
    # rather than erroring, since this is a convenience override, not a required arg.
    requested_concurrency = None
    if concurrency_flag:
        try:
            requested_concurrency = max(1, int(concurrency_flag))
        except ValueError:
            requested_concurrency = None

    # Same convenience-override philosophy as --concurrency: an optional upper bound
    # on every model's recommended context (config.json's "max_context_tokens",
    # forwarded by start_mlx_server.sh's ADVISOR_FLAGS). A cap, never a bypass --
    # the memory-fit math still runs underneath it. Invalid values fall back to no
    # cap rather than erroring.
    max_context_flag = _flag_value(argv, "--max-context-tokens")
    max_context_tokens = None
    if max_context_flag:
        try:
            max_context_tokens = int(max_context_flag)
            if max_context_tokens < 1:
                max_context_tokens = None
        except ValueError:
            max_context_tokens = None

    with open(MODELS_JSON) as f:
        catalog = json.load(f)
    models = catalog["models"]

    current_wired_gb = get_current_wired_gb()
    chip_name, gpu_cores, total_ram_gb = get_chip_info()

    # Standalone, catalog-fit-independent write path: start_mlx_server.sh's
    # full_teardown calls this with whatever that session's own /metrics last
    # reported, so a FUTURE launch of the identical (chip, model) pair can report a
    # measured decode speed instead of estimate_speed_tok_s's physics formula — see
    # save_calibration's docstring. Doesn't need total_ram_gb/free_wired_gb or any of
    # the fit-check machinery below; a session that already ran doesn't need its own
    # launch-time fit decision repeated just to record what actually happened.
    save_calibration_id = _flag_value(argv, "--save-calibration")
    if save_calibration_id is not None:
        decode_flag = _flag_value(argv, "--decode-tok-s")
        if decode_flag is None:
            log("ERROR: --save-calibration requires --decode-tok-s")
            sys.exit(2)
        try:
            decode_tok_s = float(decode_flag)
        except ValueError:
            log(f"ERROR: --decode-tok-s value '{decode_flag}' is not a number")
            sys.exit(2)
        if decode_tok_s <= 0:
            log(f"Skipping calibration save: non-positive decode_tok_s ({decode_tok_s}) — nothing real to record.")
            return
        context_flag = _flag_value(argv, "--context-tokens")
        ttft_flag = _flag_value(argv, "--ttft-s")
        peak_mem_flag = _flag_value(argv, "--peak-memory-gb")
        try:
            context_tokens = int(context_flag) if context_flag else None
            ttft_s = float(ttft_flag) if ttft_flag else None
            peak_memory_gb = float(peak_mem_flag) if peak_mem_flag else None
        except ValueError:
            context_tokens = ttft_s = peak_memory_gb = None
        # Both optional, additive (see save_calibration's own docstring) -- written by
        # the sweep harness (sweep_model.py), never by the plain teardown-time
        # save_calibration_snapshot() path, which has no sweep data to offer.
        prefill_step_size_flag = _flag_value(argv, "--prefill-step-size")
        kv_bits_safe_flag = _flag_value(argv, "--kv-bits-safe-at-long-context")
        try:
            prefill_step_size = int(prefill_step_size_flag) if prefill_step_size_flag else None
        except ValueError:
            prefill_step_size = None
        kv_bits_safe_at_long_context = None
        if kv_bits_safe_flag is not None:
            kv_bits_safe_at_long_context = kv_bits_safe_flag.strip().lower() in ("1", "true", "yes")
        save_calibration(chip_name, save_calibration_id, decode_tok_s,
                          context_tokens=context_tokens, ttft_s=ttft_s,
                          peak_memory_gb=peak_memory_gb,
                          prefill_step_size=prefill_step_size,
                          kv_bits_safe_at_long_context=kv_bits_safe_at_long_context)
        log(f"Saved calibration for '{save_calibration_id}' on '{chip_name}': {decode_tok_s} tok/s")
        return

    if total_ram_gb is None:
        log("ERROR: could not detect total RAM via system_profiler.")
        sys.exit(2)
    free_wired_gb = total_ram_gb - current_wired_gb

    preload = estimate_preload_overhead(tool, os.getcwd()) if tool else None
    rows = build_recommendations(
        models, free_wired_gb, chip_name, gpu_cores,
        preload_overhead=(preload["total"] if preload else 0),
        requested_concurrency=requested_concurrency,
        max_context_tokens=max_context_tokens,
    )

    # Phase 6: three non-interactive presentation modes for the fzf-based picker.
    # None of these touch recommend_kv_bits_and_context/build_recommendations --
    # purely how the same computed rows get rendered or looked up by id.
    if detail_id is not None:
        row = next((r for r in rows if r["model"]["id"] == detail_id), None)
        if row is None:
            log(f"ERROR: unknown model id '{detail_id}'")
            sys.exit(2)
        print(row["model"]["id"])
        print(row["model"]["display"]["tagline"])
        print("")
        for line in format_model_body_lines(row["model"], row):
            print(line)
        return

    if resolve_id is not None:
        row = next((r for r in rows if r["model"]["id"] == resolve_id), None)
        if row is None:
            log(f"ERROR: unknown model id '{resolve_id}'")
            sys.exit(2)
        if not row["fits"]:
            log(f"ERROR: '{resolve_id}' does not currently fit free memory.")
            sys.exit(3)
        emit_selection_json(row["model"], row)
        return

    if list_compact:
        # fzf's click-header passes whatever word was clicked ("Notes", "✓", ...), so an
        # unrecognized value falls back to the default instead of failing the reload.
        # Omitting --sort entirely (as the resize/start reloads below do) is NOT the same
        # as asking for "released" -- it means "whatever's already active", so a window
        # resize or the initial start-up render never resets the user's chosen sort (see
        # next_sort: None preserves the prior state, falling back to "released" only when
        # there isn't one yet).
        sort_arg = _flag_value(argv, "--sort")
        sort_key = sort_arg.lower() if sort_arg else None
        if sort_key is not None and sort_key not in SORT_KEYS:
            sort_key = "released"
        sort_col = _flag_value(argv, "--sort-col")
        # Caller's real terminal width (start_mlx_server.sh passes `tput cols`), used to
        # drop/truncate columns so the row never runs off the right edge of the window --
        # see visible_columns. Absent or garbage falls back to unlimited width (today's
        # pre-existing behavior) rather than guessing.
        term_width_flag = _flag_value(argv, "--term-width")
        term_width = None
        if term_width_flag:
            try:
                term_width = int(term_width_flag)
                if term_width < 20:
                    term_width = None
            except ValueError:
                term_width = None
        print_list_compact(rows, chip_name, gpu_cores, total_ram_gb, free_wired_gb, preload,
                           sort_key=sort_key, banner="--no-banner" not in argv,
                           sort_col=int(sort_col) if sort_col and sort_col.isdigit() else None,
                           sort_state=_flag_value(argv, "--sort-state"),
                           term_width=term_width, force_color="--color" in argv)
        return

    if list_json:
        print(json.dumps({
            "current_wired_gb": current_wired_gb, "free_wired_gb": free_wired_gb,
            "chip": chip_name, "gpu_cores": gpu_cores,
            "total_ram_gb": total_ram_gb, "preload_overhead": preload,
            # Exposed so callers (launch_tool.sh's attach-time preload-fit check) don't
            # have to hardcode a second copy of this constant.
            "min_usable_context": MIN_USABLE_CONTEXT,
            "recommendations": rows,
        }, indent=2))
        return

    fitting = print_menu(rows, free_wired_gb, chip_name, gpu_cores, total_ram_gb, preload)
    if not fitting:
        log("q) Quit")
        # input()'s own prompt argument writes to stdout, which must stay clean for the
        # final JSON line — confirmed live as a real bug during testing (the prompt
        # text ended up concatenated with the JSON, breaking the caller's json.loads).
        # Printing the prompt via log() (stderr) first and calling input() with no
        # argument avoids this.
        print("Select [q]: ", end="", file=sys.stderr, flush=True)
        input()
        sys.exit(1)

    log("q) Quit")
    while True:
        print(f"Select [1-{len(fitting)}] or 'q': ", end="", file=sys.stderr, flush=True)
        choice = input().strip().lower()
        if choice == "q":
            sys.exit(1)
        if choice.isdigit() and 1 <= int(choice) <= len(fitting):
            break
        log("Invalid choice, try again.")

    selected = fitting[int(choice) - 1]
    emit_selection_json(selected["model"], selected)
