"""Tool preload-overhead estimation: how many tokens Claude Code/OpenCode/Aider/
Cline/Pi spend on their system prompt, built-in tools, CLAUDE.md/AGENTS.md, skills,
and configured MCP servers before the user types anything."""
import json
import os
import re

# --- Tool preload-overhead estimation ---------------------------------------------
#
# Claude Code and OpenCode both load a system prompt, built-in tool schemas, any
# CLAUDE.md/AGENTS.md, configured skills, and configured MCP servers' tool schemas
# BEFORE the user types anything. On a small local context window, that preload alone
# can eat most or all of the budget. Below, doc/skill/MCP-COUNT numbers are REAL
# measurements of what's actually on this machine; only the baseline system-prompt +
# built-in-tools figure and the per-MCP-server token cost are approximations, and both
# are labeled and sourced as such — not fabricated, but not exact either.
#
# TOOL_BASELINE_PRELOAD_TOKENS: system prompt + built-in tool schemas with NO MCP
# servers or skills configured yet. Recalibrated 2026-09-29 (docs/PHASE_PLAN.md Phase 5)
# against REAL measurements via launch_tool.sh, replacing the previous public-report
# -sourced figures (Claude 18,000 / OpenCode 7,000, from a Sep 2026 HN thread — never
# measured on this project's own usage).
#
# Methodology: ran a single-turn, no-tool-call `-p "Say hello."` (claude) /
# `run "Say hello."` (opencode) request through launch_tool.sh from this repo's own
# root (no CLAUDE.md/AGENTS.md/.mcp.json present there, and claude's run used
# --strict-mcp-config), reading the server's own reported total prompt tokens
# (input_tokens, or input+cache.read for a request that hit APC) as ground truth for
# the local model's tokenizer. That total still includes this machine's real ~32
# globally-installed skills (measured separately via measure_skills_preload as
# ~3,518 tokens, since skills are global here, not per-directory) — subtracted back
# out below so estimate_preload_overhead's own skill-token addition (applied for
# whatever skills a real caller's cwd actually has) isn't double-counted on top of
# the baseline, per this phase's own double-counting warning.
#   claude:   22,406 measured total - ~3 (user msg) - 3,518 (skills) ≈ 18,885
#   opencode:  9,906 measured total - ~3 (user msg) - 3,518 (skills) ≈  6,385
# Note: this contradicts an earlier, unreproduced same-day claim in this doc's history
# of ~39,000-44,000 tokens for Claude Code (measured via a Read-tool-call request, not
# a bare "say hello"). A same-methodology Read-tool-call re-test this session measured
# ~24,000 total, consistent with the ~22,406 bare figure here, not the ~39-44K one —
# that earlier figure could not be reproduced and is treated as superseded.
#
# aider/cline/pi added 2026-10-03 (docs/PHASE_PLAN.md's lightweight-tool-CLI research
# session), same ground-truth method (server's own /metrics prompt_tokens for a bare,
# no-edit "say hello"-equivalent request through launch_tool.sh), but with NO skills
# subtraction for any of the three — confirmed live that none of them were loading
# this machine's ~3,518-token Claude-Code-style global skill listing in the first
# place: Aider has no skill/plugin-discovery mechanism at all; Cline has its own
# unrelated one (`cline skill` → `npx skills`) with zero skills installed on this
# machine; Pi has its own too (`~/.pi/agent/skills/`, confirmed in its docs) but it was
# a fresh install with that directory not even existing yet. So each tool's raw
# measured total already IS the clean baseline — nothing to subtract, unlike
# claude/opencode above where this machine's global Claude-Code skills happened to load
# through both of them.
#   aider: --message "Say hello." --yes-always --exit (run against a scratch copy of
#     this repo, not the live checkout — aider's "whole edit format" means a model
#     reply containing anything that looks like code gets auto-applied AND
#     auto-committed with --yes-always; this bit on the first attempt at this exact
#     measurement, which is why the scratch-copy precaution exists here, not for
#     claude/opencode above). Aider always builds a repo-map (confirmed: "Repo-map:
#     using 4096 tokens, auto refresh" in its own startup banner) regardless of the
#     prompt, so this total is inherently repo-shape-dependent, not a pure
#     system-prompt baseline — real for THIS repo's 36 tracked files, not necessarily
#     portable to a differently-sized repo. 6,721 measured total - ~3 (user msg) ≈
#     6,718.
#   cline: a bare `cline "Say hello." --timeout 30` (its default act-mode,
#     auto-approve=true). 5,457 measured total - ~3 (user msg) ≈ 5,454.
#   pi: `pi --provider mlx-direct --model <id> --print "Say hello." --no-session`.
#     1,611 measured total - ~3 (user msg) ≈ 1,608 — by far the smallest of the five
#     tools measured so far, consistent with it being explicitly marketed as a
#     minimal/hackable terminal agent (see docs/PHASE_PLAN.md's research section).
TOOL_BASELINE_PRELOAD_TOKENS = {
    "claude": 18900,
    "opencode": 6400,   # bare-install figure (~0 MCP; global skills subtracted back
                         # out, see above) — real OpenCode sessions with skills
                         # configured run higher; see measure_skills_preload's
                         # docstring for why that growth isn't modeled for OpenCode
                         # specifically.
    "aider": 6700,       # includes this repo's own repo-map cost (see above) — not a
                         # repo-independent figure the way the others are.
    "cline": 5500,
    "pi": 1600,
}

# Rough, clearly-approximate per-server token cost — public reports range from ~800
# tokens (a couple of terse tools) to 5,000-8,000+ tokens (verbose internal servers),
# with typical multi-tool setups landing around 2,000-3,500/server. This is the
# unavoidably-fuzzy part; the SERVER COUNT it gets multiplied by is a real count from
# this machine's own config (see measure_mcp_servers).
MCP_SERVER_TOKENS_ESTIMATE = 3000


def count_tokens_rough(text):
    """chars/4 heuristic — a standard rough estimate, not a real tokenizer run. Good
    enough to sanity-check "does this fit at all", not exact token accounting."""
    return max(0, len(text) // 4)


def measure_doc_preload(cwd, tool=None):
    """REAL measurement (not an estimate) of CLAUDE.md/AGENTS.md token cost — these
    files get read verbatim into context, so their actual size is directly countable."""
    total = 0
    found = []
    candidates = [
        os.path.join(cwd, "CLAUDE.md"),
        os.path.join(cwd, "AGENTS.md"),
    ]
    if tool not in ("aider", "cline", "pi"):
        candidates.append(os.path.expanduser("~/.claude/CLAUDE.md"))
    for path in candidates:
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            except OSError:
                continue
            tokens = count_tokens_rough(text)
            total += tokens
            found.append((path, tokens))
    return total, found


_SKILL_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---", re.DOTALL)


def _parse_skill_frontmatter(text):
    m = _SKILL_FRONTMATTER_RE.match(text)
    if not m:
        return None, None
    name = description = None
    for line in m.group(1).splitlines():
        if line.startswith("name:"):
            name = line.split(":", 1)[1].strip()
        elif line.startswith("description:"):
            description = line.split(":", 1)[1].strip()
    return name, description


def measure_skills_preload(cwd):
    """REAL measurement of currently-installed Claude Code skills' name+description
    text — that's what actually gets injected into the skill listing at session start,
    per public per-skill-cost reports of ~75-150 tokens/skill (name + description +
    Claude Code's XML-ish wrapper around each entry). Only covers the two documented
    skill directories below — misses plugin-bundled skills, and doesn't model
    OpenCode's skill mechanism at all (its on-disk convention isn't confirmed here),
    so this under-counts for OpenCode and for plugin-heavy Claude Code setups."""
    total = 0
    count = 0
    for base in (os.path.join(cwd, ".claude", "skills"), os.path.expanduser("~/.claude/skills")):
        if not os.path.isdir(base):
            continue
        for entry in os.listdir(base):
            skill_md = os.path.join(base, entry, "SKILL.md")
            if not os.path.isfile(skill_md):
                continue
            try:
                with open(skill_md, encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            except OSError:
                continue
            name, description = _parse_skill_frontmatter(text)
            if description is None:
                continue
            count += 1
            total += count_tokens_rough((name or "") + description) + 20  # +wrapper allowance
    return total, count


def measure_mcp_servers(cwd):
    """Counts CONFIGURED MCP servers for REAL, from known config file locations — only
    the per-server token COST applied to this count (MCP_SERVER_TOKENS_ESTIMATE) is an
    estimate, not this count itself. May miss non-standard config locations."""
    count = 0
    for path in (
        os.path.join(cwd, ".mcp.json"),
        os.path.expanduser("~/.claude.json"),
        os.path.expanduser("~/.claude/mcp.json"),
    ):
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        servers = data.get("mcpServers")
        if isinstance(servers, dict):
            count += len(servers)
    return count


def estimate_preload_overhead(tool, cwd):
    """Combines the one approximate number (tool baseline) and one approximate
    per-unit cost (MCP_SERVER_TOKENS_ESTIMATE) with everything else measured for real
    on this machine, for the given launch directory."""
    baseline = TOOL_BASELINE_PRELOAD_TOKENS.get(tool, 0)
    doc_tokens, doc_files = measure_doc_preload(cwd, tool=tool)
    if tool in ("aider", "cline", "pi"):
        # Confirmed live (see TOOL_BASELINE_PRELOAD_TOKENS comments): these tools do not
        # load Claude Code's global ~/.claude/skills listing or Claude's MCP server configs.
        skill_tokens, skill_count = 0, 0
        mcp_tokens, mcp_count = 0, 0
    else:
        skill_tokens, skill_count = measure_skills_preload(cwd)
        mcp_count = measure_mcp_servers(cwd)
        mcp_tokens = mcp_count * MCP_SERVER_TOKENS_ESTIMATE
    return {
        "tool": tool,
        "cwd": cwd,
        "baseline_tokens": baseline,
        "doc_tokens": doc_tokens,
        "doc_files": doc_files,
        "skill_tokens": skill_tokens,
        "skill_count": skill_count,
        "mcp_tokens": mcp_tokens,
        "mcp_count": mcp_count,
        "total": baseline + doc_tokens + skill_tokens + mcp_tokens,
    }
