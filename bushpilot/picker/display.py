"""Rendering: ANSI color, the plain numbered menu (print_menu), and the
--list-compact table -- header + rows (format_compact_header_row/
format_model_compact_row via print_list_compact) -- plus the preload banner and
unavailable-reason text shared by both presentation modes."""
import math
import os
import sys

from ..hardware import log
from ..kv import MIN_USABLE_CONTEXT, SAFETY_BUFFER_GB, UNQUANTIZED_KV_BITS, effective_size_gb
from ..preload import MCP_SERVER_TOKENS_ESTIMATE
from ..recommend import MIN_SPEED_TOK_S
from .cells import (
    COL_CTX,
    COL_ENGINE,
    COL_KV,
    COL_LAST,
    COL_NAME,
    COL_SPEED,
    COL_TOOLS,
    COMPACT_HEADERS,
    COMPACT_RIGHT_ALIGNED,
    _truncate_cell,
    cap_name_width,
    compact_row_cells,
    compact_row_widths,
    is_model_cached,
    model_drafter_label,
    model_engine,
    model_input_label,
    model_tools_label,
    visible_columns,
)
from .sort import COLUMN_SORT_KEYS, next_sort, sort_key_for_click, sort_rows

DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"


def use_color(force=False):
    """Color only on a real terminal and when NO_COLOR isn't set. stdout is a pipe into
    fzf here, so the TTY check is on stderr (where the banner is painted).

    `force=True` (--color) skips the stderr-isatty check entirely -- for the
    --list-compact picker path specifically, stderr isn't a reliable signal: every
    fzf reload/bind command in start_mlx_server.sh redirects this process's stderr to
    /dev/null (to swallow system_profiler's unrelated hw.cpufamily noise), which makes
    sys.stderr.isatty() False on every single reload after the picker's first instant
    -- not just on some other machine, on EVERY machine, every launch. The caller
    already knows this output is headed into fzf's --ansi-enabled screen, so it passes
    --color to assert that rather than relying on a TTY check that --color's own
    plumbing defeats. NO_COLOR is still honored even when forced -- that's an explicit
    user accessibility opt-out, not a detection heuristic, so it should never be
    overridden."""
    if "NO_COLOR" in os.environ:
        return False
    return True if force else sys.stderr.isatty()


def print_preload_banner(preload):
    tool_label = {"claude": "Claude Code", "opencode": "OpenCode"}.get(preload["tool"], preload["tool"])
    log(f"  Estimated preload overhead for {tool_label} in {preload['cwd']}:")
    log(f"    baseline (system prompt + built-in tools): ~{preload['baseline_tokens']:,} tokens")
    log("      (measured live on this machine 2026-09-29, see TOOL_BASELINE_PRELOAD_TOKENS — still varies by tool version)")
    if preload["doc_files"]:
        for path, tokens in preload["doc_files"]:
            log(f"    {os.path.basename(path)}: {tokens:,} tokens (measured: {path})")
    else:
        log("    AGENTS.md/CLAUDE.md: none found")
    if preload["skill_count"]:
        log(f"    skills ({preload['skill_count']} found): ~{preload['skill_tokens']:,} tokens (measured)")
    else:
        log("    skills: none found in .claude/skills")
    if preload["mcp_count"]:
        log(f"    MCP servers ({preload['mcp_count']} found): ~{preload['mcp_tokens']:,} tokens "
            f"(count measured, ~{MCP_SERVER_TOKENS_ESTIMATE:,}/server is a rough estimate)")
    else:
        log("    MCP servers: none found")
    log(f"    {'-' * 60}")
    log(f"    total estimated preload: ~{preload['total']:,} tokens")
    log(f"  A model needs recommended context - preload >= {MIN_USABLE_CONTEXT:,} tokens of real")
    log("  conversation headroom to be usable, i.e. at least "
        f"~{preload['total'] + MIN_USABLE_CONTEXT:,} tokens total.")
    log("=" * 68)
    log("")


def describe_unavailable_reason(r, preload):
    """Human-readable reason a model is greyed out -- shared by the plain numbered
    menu (print_menu) and --list-compact's hidden-model summary line."""
    context_reason = r["context_reason"]
    if context_reason == "insufficient_memory_for_model_weights":
        return "not enough free memory to load this model"
    elif context_reason == "insufficient_memory_for_fixed_kv_cost":
        return "not enough free memory left for this model's context cache"
    elif context_reason == "computed_context_too_small_to_be_useful":
        return f"would only fit under {MIN_USABLE_CONTEXT:,} tokens of context — not usable"
    elif context_reason == "architecture_fields_missing_from_config":
        return "can't compute (model not downloaded yet / config unavailable)"
    elif context_reason == "insufficient_context_after_tool_preload":
        return (
            f"{r['recommended_context']:,} tokens of context, but tool preload leaves only "
            f"~{r['recommended_context'] - (preload['total'] if preload else 0):,} for the conversation — below the "
            f"{MIN_USABLE_CONTEXT:,}-token floor"
        )
    elif context_reason == "estimated_speed_too_slow":
        return f"estimated ~{r['estimated_tok_s']} tok/s — below the {MIN_SPEED_TOK_S} tok/s usability floor"
    else:
        return context_reason


def format_model_body_lines(m, r=None):
    """The descriptive block shared between the plain numbered menu (print_menu) and
    --detail's fzf preview pane (Phase 6): good_for, usage examples, known_issues, and
    the KV-cache note. The text itself doesn't change between the two renderers, only
    where/how it's displayed -- indentation and surrounding chrome are each caller's
    own job."""
    disp = m.get("display") or {}
    lines = []
    if disp.get("good_for"):
        lines.extend([disp["good_for"], ""])
    for ex in disp.get("usage_examples") or []:
        lines.append(f"e.g. {ex}")
    if disp.get("known_issues"):
        lines.append(f"⚠ {disp['known_issues']}")
    if r is not None:
        if lines:
            lines.append("")
        if r["kv_bits"]:
            lines.append(f"[KV cache quantized to {r['kv_bits']}-bit — needed to fit this context in free memory]")
            if r.get("kv_bits_apc_caveat"):
                lines.append(
                    "[⚠ this model's sliding-window layers + quantized KV cache + prefix caching is a "
                    "known upstream gap (mlx-vlm #1559) — auto-concurrency is capped at 1 to keep "
                    "prefix reuse on the verified single-request path]"
                )
        else:
            lines.append("[KV cache unquantized — full decode speed, plenty of free memory for this context]")
        if r.get("concurrency", 1) > 1:
            lines.append(f"[{r['concurrency']} concurrent requests supported at this context size, without exceeding safe memory]")
        curve = r.get("decode_tok_s_by_context")
        if isinstance(curve, dict) and len(curve) >= 2:
            pts = []
            for k, v in curve.items():
                try:
                    d = int(k)
                    if d > 0 and isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
                        pts.append((d, float(v)))
                except (TypeError, ValueError):
                    continue
            if len(pts) >= 2:
                pts.sort(key=lambda item: item[0])
                curve_str = " → ".join(f"{rate:.1f} tok/s @ {depth:,} tok" for depth, rate in pts)
                lines.append(f"[Measured decode speed by context depth: {curve_str}]")
    return lines


def format_model_detail_lines(row, free_wired_gb, preload=None, force_color=False):
    """Full lines for `model_advisor.py --detail MODEL_ID` (fzf's preview pane).
    Includes the status/cache badges, the full metadata spec bar (so columns dropped on
    narrow terminals are always visible here), the live hardware fit summary or
    unavailable reason, and the descriptive body lines."""
    m = row["model"]
    disp = m.get("display") or {}
    color = use_color(force_color)
    b_on = BOLD if color else ""
    d_on = DIM if color else ""
    g_on = GREEN if color else ""
    y_on = YELLOW if color else ""
    c_on = CYAN if color else ""
    rst = RESET if color else ""

    status_badge = f"{d_on}[○ untested]{rst}" if m.get("status") == "untested" else f"{g_on}[✓ tested]{rst}"
    cached = is_model_cached(m)
    cache_badge = f"{g_on}[cached on disk]{rst}" if cached else f"{d_on}[download ~{m['size_gb']:g} GB]{rst}"
    hf_str = f"  {d_on}({m.get('huggingface', '')}){rst}" if m.get("huggingface") else ""

    header = f"{b_on}{m['id']}{rst}  {status_badge}  {cache_badge}{hf_str}"
    tagline = f"{c_on}{disp.get('tagline', '')}{rst}" if disp.get("tagline") else ""

    mtype = "MoE" if m.get("type") == "moe" else "Dense"
    engine = model_engine(m)
    drafter = model_drafter_label(m)
    inp = model_input_label(m)
    tools = model_tools_label(m)
    released = m.get("released") or "—"
    specs = (
        f"{d_on}Type:{rst} {mtype}  ·  "
        f"{d_on}Engine:{rst} {engine}  ·  "
        f"{d_on}Drafter:{rst} {drafter}  ·  "
        f"{d_on}Input:{rst} {inp}  ·  "
        f"{d_on}Tools:{rst} {y_on + tools + rst if tools == 'no tools' else tools}  ·  "
        f"{d_on}Released:{rst} {released}  ·  "
        f"{d_on}Size:{rst} {m['size_gb']:g} GB"
    )

    if row["fits"]:
        kv_bits = row["kv_bits"] or UNQUANTIZED_KV_BITS
        kv_hue = y_on if row["kv_bits"] else g_on
        if row["estimated_tok_s"] is not None:
            if row.get("speed_source") == "calibrated":
                spd_str = f"{b_on}{row['estimated_tok_s']} tok/s{rst} (measured on this Mac)"
            else:
                spd_str = f"~{row['estimated_tok_s']} tok/s (estimated)"
        else:
            spd_str = "speed unknown"
        fit_line = (
            f"{kv_hue}● Fits this Mac:{rst} {kv_hue}{row['recommended_context']:,} tok context "
            f"({kv_bits}-bit KV){rst}  ·  {spd_str}  ·  "
            f"{row.get('concurrency', 1)} concurrent"
        )
    else:
        reason = short_unavailable_reason(row, free_wired_gb, preload)
        fit_line = f"{y_on}✗ Does not fit current free memory:{rst} {reason}"

    out = [header, tagline, specs, fit_line, ""]
    for line in format_model_body_lines(m, row if row["fits"] else None):
        if color and line.startswith("⚠"):
            out.append(f"{y_on}{line}{rst}")
        elif color and line.startswith("[⚠"):
            out.append(f"{y_on}{line}{rst}")
        elif color and line.startswith("["):
            out.append(f"{d_on}{line}{rst}")
        elif color and line.startswith("e.g. "):
            out.append(f"{d_on}e.g.{rst} {line[5:]}")
        else:
            out.append(line)
    return out


def short_unavailable_reason(r, free_wired_gb, preload=None):
    """One-line version of describe_unavailable_reason for the greyed-out picker rows,
    with the actual numbers where we have them."""
    if r["context_reason"] == "insufficient_memory_for_model_weights":
        need = effective_size_gb(r["model"]) + SAFETY_BUFFER_GB
        return f"needs ~{math.ceil(need)} GB free, you have {free_wired_gb:.0f} GB"
    return describe_unavailable_reason(r, preload)


def format_compact_header_row(widths, cols, sort_key=None, descending=False, force_color=False):
    """First line of --list-compact, consumed by fzf's --header-lines=1: same
    `id<TAB>display<TAB>reason` shape as a data row (so --with-nth applies to it
    identically and the columns line up), id left as a placeholder. The column the list
    is currently sorted by gets a ▼ (high to low / newest first) or ▲ (low to high / A-Z).
    `cols` is the ordered subset of columns visible_columns() decided fits the terminal;
    a column dropped for width still keeps its sort-arrow label logic correct for the
    columns that remain. `force_color` is --list-compact's --color flag, forwarded to
    use_color() -- see its docstring for why stderr's own isatty status can't be trusted
    here."""
    labels = list(COMPACT_HEADERS)
    if sort_key in COLUMN_SORT_KEYS:
        k = COLUMN_SORT_KEYS.index(sort_key)
        labels[k] += " ▼" if descending else " ▲"
    padded = [labels[k].rjust(widths[k]) if k in COMPACT_RIGHT_ALIGNED else labels[k].ljust(widths[k])
              for k in cols]
    line = "  ".join(padded + [labels[COL_LAST]])
    return f"-\t{BOLD + line + RESET if use_color(force_color) else line}\t"


def format_model_compact_row(r, widths=None, cols=None, reason=None, term_width=None, force_color=False):
    """One row for --list-compact: `id<TAB>display<TAB>reason`. Field 1 (id) is used
    internally (fzf's {1} for the preview command, `cut -f1` for the final --resolve
    call); field 2 is what fzf shows and matches against (the caller hides the others
    with --with-nth=2); field 3 is empty for models that fit and the why-not text for
    the ones that don't, so the caller can refuse them. Columns are padded to `widths`
    (computed across all rows by print_list_compact) so they line up -- tab stops alone
    don't. `cols` restricts which padded columns actually get joined into the line (see
    visible_columns); columns left out here are still shown in full in --detail.

    Notes (the last cell) is never padded, but on a narrow terminal a long why-not reason
    can still run the line past term_width on its own -- so unlike the other columns it's
    truncated with an ellipsis to whatever room is left, rather than dropped outright."""
    # No known_issues marker here -- every model in this catalog carries some caveat
    # note (severity varies widely), so it wouldn't discriminate anything in this
    # compact view. Full known_issues text is always in the --detail preview pane.
    cells = compact_row_cells(r, reason)
    w = widths or (0,) * COL_LAST
    cols = cols if cols is not None else list(range(COL_LAST))
    cells[COL_NAME] = _truncate_cell(cells[COL_NAME], w[COL_NAME])
    padded = [cells[k].rjust(w[k]) if k in COMPACT_RIGHT_ALIGNED else cells[k].ljust(w[k])
              for k in cols]
    notes = cells[COL_LAST]
    if term_width is not None:
        fixed_width = sum(len(p) for p in padded) + 2 * len(padded)  # the "  " before Notes too
        room = term_width - fixed_width
        if room < 1:
            notes = ""
        elif len(notes) > room:
            notes = (notes[: room - 1] + "…") if room > 1 else "…"
    line = "  ".join(padded + [notes])
    if use_color(force_color):
        if not r["fits"]:
            line = f"{DIM}{line}{RESET}"
        else:
            # Named 16-color ANSI only (never 256/RGB), so the user's terminal decides
            # the actual hues. Green = fits unquantized; yellow = needs KV quantization
            # to fit. Meaning is also carried by the KV column text. Keyed by original
            # column index (not position in `cols`) since COL_CTX/KV/ENGINE may have
            # been dropped by visible_columns on a narrow terminal.
            hue = YELLOW if r["kv_bits"] else GREEN
            colored = dict(zip(cols, padded))
            if COL_CTX in colored:
                colored[COL_CTX] = f"{hue}{colored[COL_CTX]}{RESET}"
            if COL_KV in colored:
                colored[COL_KV] = f"{hue}{colored[COL_KV]}{RESET}"
            if COL_ENGINE in colored:
                colored[COL_ENGINE] = f"{DIM}{colored[COL_ENGINE]}{RESET}"
            if COL_TOOLS in colored:
                if cells[COL_TOOLS] == "no tools":
                    colored[COL_TOOLS] = f"{YELLOW}{colored[COL_TOOLS]}{RESET}"
                elif cells[COL_TOOLS] == "tools?":
                    colored[COL_TOOLS] = f"{DIM}{colored[COL_TOOLS]}{RESET}"
            if COL_SPEED in colored and r.get("speed_source") == "calibrated":
                colored[COL_SPEED] = f"{BOLD}{colored[COL_SPEED]}{RESET}"
            line = "  ".join([colored[k] for k in cols] + [f"{DIM}{notes}{RESET}"])
    return f"{r['model']['id']}\t{line}\t{reason or ''}"


def print_list_compact(rows, chip_name, gpu_cores, total_ram_gb, free_wired_gb, preload=None,
                       sort_key=None, banner=True, sort_col=None, sort_state=None,
                       term_width=None, force_color=False):
    """--list-compact: the fzf-facing presentation mode. Every catalog model is listed;
    the ones that don't fit your free memory are dimmed and carry the reason, so you can
    see what a bigger machine would unlock. Rows go to stdout (fzf's list input); the
    header banner goes to stderr, so it's visible in the terminal right above where fzf
    paints (suppressed with banner=False when fzf re-runs this to re-sort). `sort_key`
    of None means "keep whatever's active" (see next_sort) rather than resetting to the
    released-first default -- callers that actually want a specific sort (ctrl-r/s/t/o)
    pass one explicitly. `sort_col` is a header click (see sort_key_for_click);
    `sort_state` is the file remembering the current sort so a second request for the
    same key reverses it. `term_width` (the caller's actual terminal width --
    start_mlx_server.sh passes fzf's own $FZF_COLUMNS) is what keeps the line from
    running off the right edge of a narrower window or terminal; None means render every
    column at full width, as before. `force_color` is --color -- see use_color()'s
    docstring: start_mlx_server.sh's fzf reload/bind commands all redirect this
    process's stderr to /dev/null to swallow system_profiler's noise, which defeats the
    stderr-isatty auto-detection on every single reload, on every machine -- not just
    the picker's first frame. The caller passes --color because it already knows this
    output is going into fzf's --ansi-enabled screen."""
    fitting_count = sum(1 for r in rows if r["fits"])
    if banner:
        log(f"  Local Model Advisor — {chip_name or 'unknown chip'}   "
            f"RAM: {total_ram_gb or '?'}GB   Free: {free_wired_gb:.1f}GB   "
            f"{fitting_count}/{len(rows)} models fit")
        if preload:
            print_preload_banner(preload)

    widths = compact_row_widths(rows)
    cols = visible_columns(widths, term_width)
    widths = cap_name_width(widths, cols, term_width)
    requested = sort_key_for_click(sort_col, widths, cols) if sort_col is not None else sort_key
    sort_key, reverse = next_sort(requested, sort_state)
    rows, descending = sort_rows(rows, sort_key, reverse)
    print(format_compact_header_row(widths, cols, sort_key, descending, force_color=force_color))
    for r in rows:
        reason = None if r["fits"] else short_unavailable_reason(r, free_wired_gb, preload)
        print(format_model_compact_row(r, widths, cols, reason, term_width=term_width,
                                        force_color=force_color))
    return [r for r in rows if r["fits"]]


def print_menu(rows, free_wired_gb, chip_name, gpu_cores, total_ram_gb, preload=None):
    current_wired_gb = (total_ram_gb - free_wired_gb) if total_ram_gb else None
    log("=" * 68)
    log("  Local Model Advisor")
    log("=" * 68)
    log(f"  Chip: {chip_name or 'unknown'}   Total RAM: {total_ram_gb or '?'} GB")
    log(f"  Currently in use (wired): {current_wired_gb:.1f} GB   Free: {free_wired_gb:.1f} GB")
    log("=" * 68)
    log("  Model availability and recommended context size below are calculated from")
    log("  your system's CURRENT free memory, not a fixed spec. Close background apps")
    log("  or free up memory and re-run this script to refresh what's possible.")
    log("=" * 68)
    log("")
    if preload:
        print_preload_banner(preload)

    fitting = [r for r in rows if r["fits"]]
    not_fitting = [r for r in rows if not r["fits"]]

    if not fitting:
        log("  No model in the list fits your current free memory.")
        log("  Close some background apps and try again.")
        log("")

    i = 0
    for r in rows:
        m = r["model"]
        if r["fits"]:
            i += 1
            if r["estimated_tok_s"]:
                if r.get("speed_source") == "calibrated":
                    speed_str = f"{r['estimated_tok_s']} tok/s (measured on this machine)"
                else:
                    speed_str = f"~{r['estimated_tok_s']} tok/s (estimated)"
            else:
                speed_str = "speed unknown"
            headroom_str = ""
            if r["context_after_preload"] is not None:
                headroom_str = f"  (~{r['context_after_preload']:,} left for conversation after preload)"
            conc_str = f"  concurrency={r['concurrency']}" if r["concurrency"] > 1 else ""
            log(f"  {i}) {m['id']}  [{m['display']['tagline']}]")
            log(f"       size={m['size_gb']}GB  max context={m['max_context_architectural']:,} tokens  "
                f"recommended context={r['recommended_context']:,} tokens  {speed_str}{headroom_str}{conc_str}")
            for line in format_model_body_lines(m, r):
                log(f"       {line}" if line else "")
        else:
            reason = describe_unavailable_reason(r, preload)
            log(f"{DIM}  —) {m['id']}  [{m['display']['tagline']}]")
            log(f"       {m['display']['good_for']}")
            log(f"       size={m['size_gb']}GB  max context={m['max_context_architectural']:,} tokens  "
                f"✗ unavailable now: {reason}{RESET}")
        log("")

    if not_fitting:
        log(f"  ({len(not_fitting)} model(s) above are greyed out — they don't fit your current free memory)")
        log("")

    return fitting
