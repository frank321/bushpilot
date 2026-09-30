# Phase Plan: bushpilot redesign

## What this project is

Lets you run Claude Code or OpenCode against a **local** LLM served on-device by
Apple's MLX (Apple Silicon only) instead of a cloud API — for cost, privacy, or
offline use. It's **hardware-aware**: it detects your Mac's chip and free memory,
and recommends a model + context size that's actually safe to run rather than
crashing the machine or silently truncating context. `models.json` is the curated
model catalog (each entry live-verified for correctness and real tool-calling
support, not just copied from a spec sheet); `model_advisor.py` turns that catalog
plus live hardware detection into a concrete recommendation.

## What this doc tracks

This doc tracks the agreed redesign of this project — from "one script that does
everything" to a supervisor script (owns the model stack) + a separate launcher
script (attaches a coding tool to it) — plus the live end-to-end testing that
validated dropping LiteLLM/Postgres entirely (MLX now serves both wire formats
natively, so no translation-layer process is needed), and follow-on refinement
phases for the hardware-recommendation logic and the model picker's UI. This doc is
the working plan; update it as decisions change before/during implementation.

## Decisions locked in so far

*(Note: `start_litellm_proxy`/LiteLLM/Postgres mentions in this section predate the
later "drop LiteLLM + Postgres entirely" decision below — the supervisor script
described here is superseded by Phase 2's MLX-only version once implemented.)*

- Everything (venvs, downloaded models, Postgres data, logs, state) moves under a
  single `~/.bushpilot/` directory instead of scattering
  `~/pip_env/*`, `~/mlx_models`, `/tmp/*`. Fully removable by deleting one directory.
- The proxy launcher script is symlinked into a writable directory already on the
  user's `$PATH` (e.g. `~/.local/bin`), so it can be run from anywhere without typing
  a full path.
- `start_litellm_proxy` no longer knows about Claude Code / OpenCode at all — its
  only job is bringing up the model-serving stack (MLX server + LiteLLM + Postgres)
  safely (memory-aware, won't crash the machine).
- `start_litellm_proxy` runs in the **foreground** and shows a live status
  dashboard. No more separate `stop_litellm_proxy.sh`.
- Both **Ctrl-C** and **`q`** do the same thing: full teardown of the whole stack
  (MLX server, LiteLLM, Postgres). No "detach without stopping" mode — the dashboard
  process is the thing keeping the stack alive, full stop. Tradeoff accepted: an
  accidental Ctrl-C in that terminal kills the server for every other terminal/tool
  attached to it.
- A new `launch_tool` script replaces the tool-selection logic removed from
  `start_litellm_proxy`:
  - Detects Claude Code and OpenCode; greys out whichever isn't installed (same
    pattern as today's model advisor greying out models that don't fit memory).
  - **Antigravity is dropped entirely** — already researched and confirmed there's no
    way to point it at a custom local endpoint; not worth showing a permanently-dead
    option.
  - Takes the tool name as an optional CLI arg to skip the interactive picker.
  - Checks whether the proxy/model is already up first. If not, asks the user to
    confirm, then `exec`s `start_litellm_proxy` directly (replacing itself in that
    terminal) — no separate daemon/background mode needed, since
    `start_litellm_proxy`'s own dashboard already tells the user to open another
    terminal and re-run `launch_tool` there.
  - This is what enables multiple terminals/tool sessions pointed at the same shared
    proxy.

## Decision: drop LiteLLM + Postgres entirely (2026-09-29)

Superseded plan: earlier phases assumed LiteLLM stayed in place as the
protocol-translation layer between Claude Code (Anthropic wire format) / OpenCode
(OpenAI wire format) and the MLX backend. That assumption turned out to be wrong for
the currently-installed `mlx_vlm` version:

- **`mlx_vlm.server` already natively serves both wire formats** — confirmed by
  reading the installed package source, not by guessing. `mlx_vlm/server/app.py`
  registers `/v1/chat/completions` (OpenAI) via `openai.py` **and** `/v1/messages`
  (Anthropic) via `anthropic.py` (`register_anthropic_routes`, `app.post("/messages")`
  / `app.post("/v1/messages")`). Live-tested both directly against the server with no
  proxy in front — both work.
- LiteLLM's original job here (per this project's own `LITELLM_PROXY_USAGE.md`:
  *"Provides OpenAI-compatible API interface... Sits between your client and the
  local model"*) assumed MLX was OpenAI-only. Checked `frank-research`,
  `devx-agent-eval`, and this repo's own docs for prior research that might justify
  keeping LiteLLM — found none that re-examined that assumption. (`devx-agent-eval`'s
  `LITELLM_ROUTING_PLAN.md` is an unrelated use case — evaluating LiteLLM as a
  cloud-routing layer between OpenCode and Bedrock — but it does independently
  document a real correctness bug in LiteLLM's own `ollama/`→OpenAI-compatible
  streaming translation layer, which is evidence *for* removing a translation hop we
  don't need, not against it.)
- Model-name aliasing (the other real thing LiteLLM was doing — mapping a friendly
  `models.json` id to the actual HF repo id): confirmed live that `mlx_vlm.server`
  requires the wire-level `"model"` field to match a real, loadable HF repo id (a
  fake alias 401s trying to fetch it from Hugging Face). Not a blocker — our own
  scripts already control the env var value passed to each tool, so we just pass the
  literal HF id under the hood while still showing the friendly name in the picker UI.
  No proxy needed for this.
- Removing LiteLLM + Postgres also removes: the whole Prisma/Node codegen bootstrap
  (documented in `bootstrap_litellm()` as three separate hard-to-diagnose gaps),
  `brew install postgresql@15` as a hard dependency, role/db bootstrap, the
  generated YAML config, and master-key management. Net effect: one background
  process (MLX) instead of three, faster startup, lower bar for a teammate to get
  running — directly serves the "don't crash the user's machine" mission by cutting
  background daemons.

**This changes Phase 2 (see below): `start_litellm_proxy` becomes an MLX-only
supervisor. There is no LiteLLM/Postgres bootstrap step anymore.**

## Decision: dashboard stats source is `mlx_vlm.server` itself (2026-09-29)

Confirmed live (started the cached `mlx-community/Qwen3.5-0.8B-4bit` model directly,
polled both endpoints, sent real requests) that almost everything on the "needs new
instrumentation" list from the original stats discussion is **already computed by
the installed `mlx_vlm.server`, free, via two GETs — no new instrumentation
required**:

- **`GET /metrics`** (`mlx_vlm/server/generation.py`'s `ServerMetricsStore`) returns:
  requests started/completed/failed/in-flight, lifetime prompt/completion/generated
  token totals, lifetime avg prefill & decode tok/s, and the *latest* request's
  `prefill_tok_s`, `decode_tok_s`, `ttft_s`, `prompt_tokens`, `generated_tokens`,
  `request_elapsed_s`, `finish_reason`, `peak_memory_gb`, plus `last_error` and
  server uptime. Live example captured: cold request `prefill_tok_s=303`, same
  prompt replayed with APC hit `prefill_tok_s=2643`.
- **`GET /v1/cache/stats`** (`mlx_vlm/apc.py`'s `APCStats.snapshot`) returns the APC
  prefix-cache stats directly: `token_hit_rate`, `lookups_hit`/`lookups_miss`,
  memory-tier `pool_used`/`num_blocks`, disk-tier `disk_bytes`/`disk_max_bytes`,
  `evictions`, `rejects`/`rejects_by_reason`.
- Both are cheap local HTTP GETs (sub-ms, in-process dicts) — fine to poll every 1-2s
  for the live dashboard.
- Plain OS calls (already used elsewhere in this codebase, e.g. `monitor_memory.sh`,
  `model_advisor.py`) cover the rest: free memory % / pressure level
  (`kern.memorystatus_level`), MLX process RSS, currently loaded model/context.
- Genuinely dropped from scope (per explicit decision): per-tool/per-terminal
  attribution of load, GPU/Neural-Engine utilization (`powermetrics` needs sudo),
  and LiteLLM-side stats (not useful now that LiteLLM itself is being removed — see
  above).

So the "easy" vs. "needs work" split from the original discussion is obsolete: it's
now essentially all easy, sourced from `mlx_vlm.server`'s own `/metrics` and
`/v1/cache/stats`.

### Decision: dashboard layout (2026-09-29)

Six fixed, in-place-refreshed sections (poll `/metrics` + `/v1/cache/stats` + OS
calls every 1-2s; no scrolling log, just overwritten values — confirmed above that
mid-request live decode-rate would require log-tailing instead of stats polling, and
that's explicitly out of scope, so every number here only updates once per
completed request):

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

- Header: model id, server uptime, port, loaded/architectural context, APC on/off.
- REQUESTS / LIFETIME: from `/metrics`'s `summary` (`requests_completed`,
  `requests_failed`, `in_flight`, `prompt_tokens_total`, `generated_tokens_total`,
  `avg_decode_tok_s`).
- LATEST REQUEST: from `/metrics`'s `latest` (`prefill_tok_s`, `decode_tok_s`,
  `ttft_s`, `request_elapsed_s`, `finish_reason`) — this is what would have caught
  the 0.8B model's tool-call loop immediately (`finish_reason: tool_use` never
  changing across polls, `prompt_tokens` in `latest` climbing every poll).
- APC CACHE: from `/v1/cache/stats` (`token_hit_rate`, `lookups_hit`/`lookups_miss`,
  memory-tier `pool_used`/`num_blocks`, disk-tier `disk_bytes`/`disk_max_bytes`,
  `evictions`).
- SYSTEM: OS-level free memory %, `kern.memorystatus_level` pressure tier, MLX
  process RSS, `peak_memory_gb` from `latest`.
- Last error row: `summary.last_error`, hidden/blank when null.

## Open / still deciding

- **Resolved (2026-09-29): manual, one-time, this-machine-only migration — not
  scripted.** Chose not to auto-migrate on first run; the mapping from old scattered
  paths (`~/pip_env/mlx_lm`, `~/mlx_models`/`~/MLX_Models`, `/tmp/mlx-llm-proxy-*`)
  into the new `~/.bushpilot/` layout will be done by hand on this
  machine only, as the last step of implementing Phase 1/2 — not before, since the
  *current* `start_litellm_proxy.sh` still hardcodes the old paths (`MLX_DIR`,
  `HF_HOME` default) and would break if those directories moved before the rewrite
  lands. `~/pip_env/litellm` gets deleted outright at that point too (dead weight,
  nothing to migrate). No code needed for this — it's a one-off `mv`/`rm` on frank's
  machine, deliberately not written into any script.
  **EXECUTED (2026-09-30):** removed the interim `~/.bushpilot/models`
  symlink (which pointed back out at the still-visible `~/MLX_Models`) and
  `mv`'d the 147 GB cache to `~/.bushpilot/models` for real (same volume —
  instant rename); `~/.zshrc`'s `HF_HOME`/`SSL_CERT_FILE` exports updated to the
  hidden path; `~/pip_env/litellm` and `~/pip_env/mlx_lm` deleted (1.5 GB freed;
  `headroom_enabled_env` deliberately kept — unknown purpose, not in plan);
  `/tmp/mlx-llm-proxy-*` was already absent. Verified end-to-end after the move:
  TLS to huggingface.co via the relocated CA bundle, both cache checks, full
  launcher run, smoke check, and a real inference request all pass. Home directory
  now shows no MLX-related folders.

## Live end-to-end test (2026-09-29) — complete, both tools confirmed

Goal: confirm Claude Code and OpenCode both work talking **directly** to
`mlx_vlm.server` (no LiteLLM in front), including real tool-calling through the
client's own agentic loop — not just a raw curl with a tool schema attached (that
part was already verified per-model in `models.json`'s `tool_calling` field, but
never through the actual `claude`/`opencode` CLIs).

**Model used:** `mlx-community/Qwen3.5-0.8B-4bit` — already tagged in `models.json`
as "Ultra-light sanity-check model... smoke-test that opencode/Claude Code → ... →
MLX is wired up correctly", already cached, already has a `tool_calling.verdict:
SUPPORTED` entry. Good choice for this test, keep using it.

**Claude Code — findings so far:**
1. First attempt used `--max-kv-size 32768` and failed with a clean `400` —
   *correctly*, this is the safety mechanism working as designed:
   `PromptTooLongError: Request needs 71307 context tokens (39307 prompt + 32000 max
   generation), but MAX_KV_SIZE is 32768`. **Real data point: Claude Code's own
   system prompt + tool schemas cost ~39.3K tokens of preload** — matches the
   ballpark `model_advisor.py`'s preload estimator already assumes, good
   corroboration.
2. Restarted with `--max-kv-size 131072`. Retried — this time it worked: MLX server
   log shows a real completed request with `finish_reason=tool_use`
   (`endpoint=/v1/messages model=mlx-community/Qwen3.5-0.8B-4bit stream=True
   backend=continuous_batching prompt_tokens=40019 generated_tokens=79
   elapsed=4.896s prefill=9083.7 tok/s decode=209.2 tok/s`). **This is the real
   proof point: the model correctly issued a tool call over the native `/v1/messages`
   endpoint, through Claude Code's actual agentic loop, with zero LiteLLM in the
   path.** `/metrics` also confirmed 10 requests completed with no failures.
3. Two unrelated snags hit along the way, worth carrying into the real script:
   - Claude Code sent a hardcoded `claude-haiku-4-5-20251001` for what looks like an
     internal background call, ignoring `ANTHROPIC_MODEL`/`ANTHROPIC_SMALL_FAST_MODEL`
     — caused a 401 (MLX tried to fetch that as a real HF repo id) on the first run.
     Worked around by also setting `ANTHROPIC_DEFAULT_HAIKU_MODEL` /
     `ANTHROPIC_DEFAULT_SONNET_MODEL` / `ANTHROPIC_DEFAULT_OPUS_MODEL` to the local
     model id and passing `--model` explicitly on the CLI. **This is a pre-existing
     gap in the current (LiteLLM-fronted) design too** — nothing in the old
     `models.json`-derived LiteLLM config registers `claude-haiku-4-5-20251001` as an
     alias either, so this would 400 through the old architecture just the same. Not
     a regression from dropping LiteLLM; just never noticed until this was tested
     end-to-end for the first time. **Needs fixing in the real script regardless of
     the LiteLLM decision** — export all four env vars (or whatever the current
     Claude Code version actually reads) to the same local model id.
   - A background/non-interactive `claude -p` run inherited this dev session's own
     `cmux-cua` MCP server config and hung trying to connect it — an artifact of how
     I was invoking the nested test process, not a finding about the MLX/Claude Code
     integration itself. Use `--strict-mcp-config --mcp-config '{}'` to isolate
     future test runs (note: `--mcp-config '{}'` itself errored — "Invalid MCP
     configuration: mcpServers: Invalid input" — needs the right empty-config shape,
     e.g. `{"mcpServers":{}}`, or just omit `--mcp-config` and rely on
     `--strict-mcp-config` alone. Untested which works — next session should check.)
   - `-p` (print mode) exited 0 but printed only two diagnostic lines
     (`claude.ai connectors are disabled...`, `[claude-code:unrecognized_model]`)
     with **no final assistant text** on one run, even though the MLX server log
     shows the request completed successfully server-side. Not yet root-caused —
     could be `--output-format text` needing to be explicit, could be the "unrecognized
     model" warning suppressing final output for a model Claude Code doesn't
     recognize, could be an artifact of the MCP-hang issue above. **Next step: retry
     with `--output-format json` (attempted, but the run was interrupted before
     results came back) to get a structured result with any error/stop-reason
     field, and confirm the actual answer text (should be the literal string
     `BANANA-7742` from `/tmp/mlx-test/scratch/marker.txt`) comes through.**

**OpenCode — not started yet.** Need to figure out (and this session didn't get to
it): does OpenCode support a generic "openai-compatible custom endpoint" provider
pointed straight at `mlx_vlm.server`'s `/v1` without needing anything registered in
`opencode.jsonc` the way the current `ensure_opencode_model_registered()` function
does for the LiteLLM-fronted setup? Check `opencode --help` / its config schema for
a direct-base-URL provider option; if it still needs a models registry entry, decide
whether that registration step moves into `start_litellm_proxy`'s replacement or into
`launch_tool`.

**Cleanup already done:** all test artifacts stopped/removed (`/tmp/mlx-test/`, no
stray `mlx_vlm.server` or `claude -p` processes left running) as of end of this
session.

**Resumed and completed (2026-09-29):**

1. **MCP isolation fix confirmed:** `--strict-mcp-config --mcp-config /path/to/{"mcpServers":{}}.json`
   works (the earlier `--mcp-config '{}'` shape was the bug — needs the full
   `{"mcpServers":{}}` object, and it must be a real file path, not an inline string,
   to avoid shell-quoting issues).
2. **First re-run, same `mlx-community/Qwen3.5-0.8B-4bit` model used in the original
   test session, was NOT clean — new finding, worse than previously known:** instead
   of the earlier session's clean single-tool-call success, this run got stuck in an
   **infinite tool-call loop**. `/metrics` showed `finish_reason: tool_use` on every
   single request (32+ recorded), prompt tokens growing ~100/request every ~3.5s
   (26,532 → 30,612+) with no convergence to final text. Killed after 6.5 minutes
   (would have taken ~50+ more minutes to hit the 131,072 context cap). Root cause
   not isolated — plausibly a `qwen3_coder` tool-parser interaction specific to this
   0.8B model + `Read` tool + `--allowedTools` restriction, but not confirmed by
   further debugging (deprioritized once the 9B model resolved the immediate
   question). **Practical implication: the 0.8B model is not reliable enough for
   real agentic tool-calling loops and should not be the default recommendation for
   actual coding-tool usage** — its `models.json` role should stay "ultra-light
   sanity-check model" only, not a general-purpose pick.
3. **Same test with `mlx-community/Qwen3.5-9B-MLX-8bit` (already cached) worked
   cleanly** for both tools, confirming the hypothesis that model size/capability,
   not the LiteLLM-removal architecture, caused the 0.8B loop:
   - **Claude Code:** `claude -p ... --output-format json` completed in ~34s (real
     wall time dominated by a ~15s time-to-first-token on this bigger model), 2
     requests, 0 failed, `stop_reason: end_turn`, `is_error: false`. Final answer
     text confirmed correct: `"The file marker.txt contains:\n\n\`\`\`\nBANANA-7742..."`.
   - **OpenCode:** required registering the model directly in
     `~/.config/opencode/opencode.jsonc` under a new provider (OpenCode's
     `Provider.getModel` does a plain object lookup and throws on an unregistered
     model — same constraint the old `ensure_opencode_model_registered()` worked
     around for the LiteLLM-fronted setup, still applies with LiteLLM removed).
     Added a `mlx-direct` provider using `@ai-sdk/openai-compatible` with
     `baseURL: "{env:MLX_DIRECT_BASE_URL}"` = `http://127.0.0.1:11435/v1` (backup at
     `/tmp/mlx-test/opencode.jsonc.bak`; left in place, not reverted). Ran via
     `opencode run "..." --model mlx-direct/mlx-community/Qwen3.5-9B-MLX-8bit
     --format json --dir ...` (had to drop `--auto`; blocked by the safety
     classifier as an unsafe blanket permission-approval flag — turned out to be
     unnecessary anyway, the `read` tool call went through with no approval prompt
     in this non-interactive JSON mode). Completed in ~10s, `step-finish reason:
     stop`, final text confirmed correct.
   - **Confirms the core architectural claim from the LiteLLM-removal decision
     above:** the exact same `mlx_vlm.server` process, same port (11435), serves
     both wire protocols concurrently — Claude Code hits `/v1/messages` (via
     `ANTHROPIC_BASE_URL=http://127.0.0.1:11435`), OpenCode hits
     `/v1/chat/completions` (via the `@ai-sdk/openai-compatible` provider's
     `baseURL=http://127.0.0.1:11435/v1`). No proxy, no protocol-translation layer,
     no separate registration step beyond each client's own model lookup.

**Decision (2026-09-29): drop LiteLLM + Postgres, confirmed via live test, not just
static analysis.** The architectural decision recorded earlier in this doc is now
validated end-to-end for both target tools. Remaining follow-ups, not blockers:
- **Resolved (2026-09-29): not root-causing the 0.8B model's tool-call loop.**
  Documented as a known limitation directly in `models.json`'s `qwen3.5-0.8b-4bit`
  entry (`display.known_issues`) instead — small models getting stuck in infinite
  tool-call loops during real multi-turn agentic use (as opposed to a single
  isolated tool-call request, which it still passes structurally) is treated as a
  common, expected failure mode for this size class, not a bug worth debugging
  further. The model stays scoped to "sanity-check only."
- OpenCode's model-registration step (the `opencode.jsonc` edit) needs to move into
  whichever script replaces `ensure_opencode_model_registered()` — see Phase 3 below.
- `--mcp-config` needs a real file path with `{"mcpServers":{}}`, not an inline `{}`
  string — worth a one-line comment in the real script/docs if it ever shells out to
  `claude` non-interactively (e.g. for a health-check probe), so this isn't
  re-discovered from scratch.

## Phases

### Phase 1 — Directory & path migration
- Introduce `~/.bushpilot/` as the single home for venvs, model
  weights, Postgres data dir, logs, and state files.
- Decide + implement the migration story for existing installs (see Open items).
- Symlink the proxy script into a writable `$PATH` directory; detect one that
  already exists (`~/.local/bin`, `/opt/homebrew/bin`) rather than assuming.
- **Move APC's disk cache under the new single home** (panel finding, GLM, low —
  confirmed `APC_DISK_PATH="$HOME/.cache/mlx-vlm/caching"` in the current script,
  outside `~/.bushpilot/`): relocate to
  `~/.bushpilot/apc-cache/` so "fully removable by deleting one directory"
  is actually true, and keep exporting `APC_ENABLED`/`APC_NUM_BLOCKS`/
  `APC_DISK_PATH`/`APC_DISK_MAX_GB` plus the `mkdir -p` it needs (mlx_vlm doesn't
  create the dir itself).

### Phase 2 — `start_litellm_proxy` rewrite (MLX-only supervisor, no LiteLLM/Postgres)

**Panel-reviewed 2026-09-29** (Kimi K3 High + GLM-5.3 High via `devx-consult-panel`,
against the actual current `start_litellm_proxy.sh`/`model_advisor.py` — see
findings folded in below). Both consultants independently flagged that "strip and
rewrite" framing risked silently regressing several already-battle-tested pieces of
the current script; those are now explicit preserve/fix items, not just removals.

- Strip out all Claude Code / OpenCode / tool-launching logic.
- **Remove the LiteLLM/Postgres bootstrap entirely** — confirmed unnecessary by the
  live end-to-end test above (`mlx_vlm.server` natively serves both wire formats).
  Drop `bootstrap_litellm()`, the Prisma/Node codegen step, the
  `brew install postgresql@15` dependency, role/db bootstrap, the generated LiteLLM
  YAML config, and master-key management.
- **Explicitly PORT, don't drop, these existing reliability mechanisms** (confirmed
  present and load-bearing in the current script — panel finding, both consultants,
  high severity): `mlx_port_owner_is_expected` (foreign-process/squatter detection —
  a real incident already happened: a stray `mlx_lm.server` squatted on the port for
  17h, answering `/health` while hung), `mlx_inference_responds` (a real one-token
  inference probe — `/health` alone was already proven insufficient once),
  `wait_for_port_free`/`wait_for_process_gone`, the loaded-model/context/kv-bits
  state file (relocate from `/tmp/mlx-llm-proxy-loaded.txt` to
  `~/.bushpilot/state/`, matching the single-home decision), the
  "different model already running, confirm?" interactive flow, and
  `ensure_model_downloaded`'s pre-download step (no resume support if a download is
  killed mid-transfer, so pre-checking matters).
- **Fix the `install_packages_if_needed` "already installed" check**, a pre-existing
  latent bug confirmed in the current code (panel finding, Kimi): it runs
  `"$python_bin" -c "import $(echo $pkg | cut -d= -f1)"`, which for `pkg="mlx-lm"`
  literally executes `import mlx-lm` — a Python `SyntaxError`, since hyphens aren't
  valid in a bare import identifier. The fast path therefore **never succeeds**,
  and every invocation silently re-runs `pip install --upgrade pip setuptools
  wheel` plus a full reinstall of both packages, even when already installed. Fix
  by checking `pip show <dist>` or mapping dist name → import name explicitly
  (`mlx-lm`→`mlx_lm`, `mlx-vlm`→`mlx_vlm`).
- **Pin `mlx-vlm`/`mlx-lm` to a minimum verified version** (`mlx-vlm>=0.7.0`, not
  unbounded) — panel finding (GLM, high): the entire LiteLLM-removal decision rests
  on this specific installed version's behavior (native `/v1/messages` routes,
  `/metrics`'s `ServerMetricsStore` fields, `/v1/cache/stats`'s `APCStats` fields),
  and the current script installs both packages with no version constraint at all.
  **Also add a startup smoke check**: after the server reports healthy, actually
  call `/v1/messages`, `/metrics`, and `/v1/cache/stats` once and verify the
  expected shape before declaring the supervisor ready — catches behavior drift
  even within the allowed version range, not just "too old."
- Convert to foreground-only with a live status dashboard.
- Implement Ctrl-C and `q` both triggering full teardown of just the MLX server
  (no LiteLLM/Postgres processes left to tear down). **Also trap `HUP`/`TERM`, not
  just `INT`** — panel finding (GLM, medium): a closed terminal or dead SSH session
  currently only sends SIGHUP, which the plan's Ctrl-C/`q`-only framing doesn't
  cover, risking exactly the kind of orphaned-process squatter the port-owner check
  above exists to catch.
- **Fold `monitor_memory.sh`'s memory-pressure backstop directly into the
  supervisor's own poll loop, applying the exact same full-teardown path as
  Ctrl-C/`q`** (decided 2026-09-29, resolving a panel finding from Kimi, high
  severity: the current script's `--monitor` mode is described in its own comments
  as *"a real-time backstop"* against ambient memory pressure during long
  sessions — something admission checks alone don't cover, and the original phase
  text only covered *displaying* memory, not *acting* on it). The poll loop already
  reads `kern.memorystatus_level` for the dashboard's SYSTEM row; when pressure
  crosses the critical threshold for N consecutive polls, trigger the same teardown
  Ctrl-C/`q` use, with a visible dashboard banner explaining why. This removes
  `monitor_memory.sh` as a separate background daemon entirely — one fewer process,
  consistent with this project's own "don't crash the machine, minimize background
  daemons" mission.
- Ship the "easy" stats tier (`/metrics` + `/v1/cache/stats`, already confirmed
  sufficient — see decision above); no "needs work" tier remains to fast-follow.

### Phase 3 — New `launch_tool` script
- **Tool selection uses `fzf` too** (decided 2026-09-29), consistent with Phase 6's
  model picker. Unlike the model picker, an unavailable tool is **not** hidden from
  the list — selecting an uninstalled tool shows a clear message ("Claude Code is
  not installed — install it first: <link/command>") rather than silently greying
  it out of the picker. Supersedes the original "detects... greys out whichever
  isn't installed" wording.
- Support `launch_tool <toolname>` to skip the interactive picker.
- Check proxy/model health first; if down, confirm with the user and `exec
  start_litellm_proxy`.
- **Also validate preload fit at attach-time, not just liveness** (decided
  2026-09-29, resolving a panel finding — GLM raised it as "supervisor/launcher
  ordering breaks tool-before-model," Kimi as "`--resolve` drops the `--tool`
  dimension"; same root cause, different framing). The current script selects the
  tool *before* the model specifically so `model_advisor.py`'s `--tool` flag can
  factor that tool's preload into context selection (see the code comment: *"Tool
  is selected BEFORE the model, not after"*). In the redesign this ordering is
  gone by design — the supervisor is deliberately tool-agnostic, since its whole
  point is serving multiple tools/terminals off one shared process, so there's no
  longer a single "the tool" to optimize context for at select-time. That's fine,
  but the underlying resource question doesn't disappear — it just moves later:
  if the already-running proxy's loaded context doesn't leave enough room for
  *this* tool's real preload (Phase 5 measured Claude Code's at ~39-44K tokens),
  the user needs to find out at `launch_tool` attach-time, not mid-session via a
  cryptic `PromptTooLongError`. So `launch_tool`'s existing health check gains one
  more condition: `loaded_context - this_tool's_preload >= MIN_USABLE_CONTEXT`; if
  it fails, warn clearly and suggest restarting the proxy with a bigger context (or
  picking the other tool) instead of proceeding into a broken session.
- **Reserve output tokens when setting the client-side context cap** (decided
  2026-09-29, resolving a panel finding — GLM, medium, though the underlying data
  was already in this doc's own live-test notes: *"PromptTooLongError: Request
  needs 71307 context tokens (39307 prompt + 32000 max generation), but
  MAX_KV_SIZE is 32768"*). `recommended_context` is a memory ceiling, not a usable
  prompt budget — setting `CLAUDE_CODE_MAX_CONTEXT_TOKENS` directly to it will keep
  reproducing this exact error near the ceiling. Reserve the client's expected
  max-output-tokens (32K observed for Claude Code) when setting the cap.
- For Claude Code: export `ANTHROPIC_BASE_URL` (pointed straight at the MLX server,
  no `/v1` suffix) plus all four `ANTHROPIC_MODEL` /
  `ANTHROPIC_DEFAULT_{HAIKU,SONNET,OPUS}_MODEL` env vars to the real HF repo id (see
  the Claude Code hardcoded-haiku-model finding above), and pass `--model`
  explicitly. **Also carry over** (panel finding, both consultants, confirmed
  present in the current script and omitted from this phase until now):
  `ANTHROPIC_API_KEY`/`ANTHROPIC_AUTH_TOKEN` dummy credentials (Claude Code's
  behavior with no credential configured at all against a keyless server is
  unconfirmed — keep the dummies unless a startup test proves they're
  unnecessary), `ANTHROPIC_SMALL_FAST_MODEL`, and `CLAUDE_CODE_MAX_CONTEXT_TOKENS`
  (set per the output-token-reserve rule above).
- For OpenCode: port `ensure_opencode_model_registered()` to write directly into a
  new `mlx-direct`-style `@ai-sdk/openai-compatible` provider block (`baseURL`
  pointed at `http://<mlx-host>:<port>/v1`) instead of the old `litellm` provider —
  confirmed live that OpenCode still requires the model registered in
  `opencode.jsonc` even with LiteLLM removed (`Provider.getModel` does a plain
  object lookup and throws otherwise). Prototype version of this provider block is
  already sitting in `~/.config/opencode/opencode.jsonc` (backup at
  `/tmp/mlx-test/opencode.jsonc.bak`) — use it as the template, don't re-derive from
  scratch. **Panel findings to close before treating the prototype as done**: the
  prototype's `baseURL: "{env:MLX_DIRECT_BASE_URL}"` has no confirmed setter —
  either `launch_tool` must export that env var, or the provider block should just
  hardcode the URL (the port is already a fixed constant in the supervisor); the
  current `opencode.jsonc` still has the old `litellm` provider pointing at the
  now-dead port 4100 (confirmed: `LITELLM_PORT=4100`) and Phase 3 must clean that
  up, not just add alongside it; confirm the registered model block includes a
  required `apiKey` field; and register models keyed by the literal HF repo id
  (what actually goes on the wire), not the friendly `models.json` id the old
  `litellm` provider used.

### Phase 4 — Cleanup & docs
- Update `QUICK_START.md` / `LITELLM_PROXY_USAGE.md` for the new layout and two-script
  model.
- Remove now-stale `stop_litellm_proxy.sh`.
- **Remove `monitor_memory.sh`** — its logic is now folded directly into the
  supervisor's poll loop (see Phase 2), so the standalone script and its
  `--litellm-pattern` argument (panel finding, GLM: already stale once LiteLLM is
  gone) become dead weight, not just renamed.
- **Rename `start_litellm_proxy` itself** (panel finding, GLM, low) — it no longer
  proxies anything LiteLLM-related; keeping the old name after Phase 2's rewrite is
  a maintainability/confusion cost worth fixing while the Phase 1 symlink work is
  already being touched.
- Re-verify `HARDWARE_SPEED_ESTIMATION.md` / `PROJECT_STATUS.md` references to old
  paths.
- **Test the headline "multiple terminals/tool sessions share one proxy" scenario
  concurrently, not just sequentially** (panel finding, GLM, medium): both live
  tests so far ran Claude Code and OpenCode one at a time. Run them
  simultaneously against the same server, and check `/metrics`'s in-flight count
  and APC hit-rate behavior under interleaved requests from both wire formats at
  once — the actual new capability this redesign enables was never verified under
  real concurrency.

**Implemented and live-validated (2026-09-29):**

- `start_litellm_proxy.sh` renamed to `start_mlx_server.sh`; `launch_tool.sh` (Phase 3)
  and all doc cross-references updated to the new name.
- `stop_litellm_proxy.sh` and `monitor_memory.sh` deleted.
  `MONITORING_ARCHITECTURE.md` and `verify_long_context_memory.sh` (which shelled out
  to `monitor_memory.sh` directly for its stress-test safety net) both got a
  superseded/historical-status banner rather than a rewrite, pointing at
  `start_mlx_server.sh`'s `check_memory_backstop` as where that logic actually lives
  now.
- `LITELLM_PROXY_USAGE.md` renamed to `MLX_SERVER_USAGE.md` and rewritten for the
  two-script architecture (no LiteLLM/Postgres, `~/.bushpilot/` paths, both
  scripts' real env vars/behavior). `QUICK_START.md` rewritten the same way.
  `PROJECT_STATUS.md` and `HARDWARE_SPEED_ESTIMATION.md` got stale-path fixes (old
  `~/scripts/`/`~/pip_env` locations, old script name) — `PROJECT_STATUS.md` also got
  a banner pointing at this doc as the current tracker, since it predates the whole
  Phase 1-4 redesign and a full rewrite of its 500+ lines of historical
  model-verification notes was out of scope here.
- **Live concurrency test, not simulated:** started `start_mlx_server.sh` with
  `mlx-community/Qwen3.5-9B-MLX-8bit` (context 131072), then ran
  `launch_tool.sh claude -p ...` and `launch_tool.sh opencode run ...` against the
  same running server at the same time (both reading a marker file to confirm real,
  distinct tool-call results, not cached/mocked output). `/metrics` showed
  `in_flight` at 2-3 while both were mid-request, both completed successfully with
  the correct answer (`MANGO-1234`), `requests_failed: 0` throughout, and
  `/v1/cache/stats` showed real APC hits/disk writes across the interleaved requests.
  Confirms the core claim: one `mlx_vlm.server` process really does serve both wire
  formats to two independent clients concurrently, no proxy in the middle.
- **Teardown validated via SIGTERM** (full `full_teardown` path: stops
  `mlx_vlm.server`, removes the state file, frees the port) — confirmed clean via a
  live kill. Ctrl-C's SIGINT path uses the identical `full_teardown` function (same
  trap pattern, just a different signal) but could not be exercised in this same test
  because the server had to be started as a backgrounded job for automation, and
  POSIX/bash's "SIGINT ignored on entry to an async job cannot be re-trapped" rule
  means that specific test artifact — not the script — silently defeats `kill -INT`
  in that setup. Real interactive usage (running the script directly in a foreground
  terminal, as it's designed for) does not hit this at all.

### Phase 5 — Recalibrate `model_advisor.py`'s hardware-based recommendation logic

Added 2026-09-29 after cross-checking `model_advisor.py`'s two ESTIMATE-based
components against real numbers gathered during the live end-to-end test above.
The context-size math itself (`recommend_max_context`/`compute_kv_formula`) is an
*exact* calculation from each model's real `kv_config`, not an estimate — not in
scope here.

**Correction (2026-09-29, same day):** this section originally claimed the speed
estimate was off by 6-8x on this machine, based on a "~150-231 tok/s" figure. That
figure was a mistake — it was actually the 0.8B model's tool-call-loop metrics
(captured earlier while debugging that loop), mislabeled as the 9B-8bit model's
speed when a dashboard-mockup placeholder number got treated as real data. Six
separate, deliberate live benchmarks against the actual 9B-8bit model on this M5
Pro — varying APC hit/miss, streaming, generation length, prompt size (up to 32K
tokens), real tool-calling, and both wire endpoints (`/v1/chat/completions` and
`/v1/messages`) — all landed at **26.5-32.2 tok/s decode**, matching the physics
formula's **~27.5 tok/s** prediction to within ~15%. **The chip bandwidth table and
`SPEED_CORRECTION_FACTOR` are fine; there is no speed-estimate bug to fix.** (Also
independently confirmed while investigating this: this machine's M5 Pro variant —
20-core GPU — really is 307 GB/s per Apple's own published spec, and Apple's own
MLX/M5 research explicitly states decode stays memory-bandwidth-bound on M5, with
Neural Accelerators only speeding up prefill — so nothing about the M5 generation
changes the formula's underlying premise either.)

The one still-real, still-unfixed finding from the live test:

- **Claude Code's preload baseline (`TOOL_BASELINE_PRELOAD_TOKENS["claude"]` = 18,000)
  is likely more than 2x too low.** Both this session's and the earlier session's
  live tests measured Claude Code's real prompt-token preload at **~39,000-44,000
  tokens** for a trivial single-file-read request — not 18,000. This constant was
  sourced from public HN/issue-tracker reports, never measured on this project's
  actual usage.

**Why this is cheap to fix, and why it waits for Phase 2-4 first:** real Claude
Code/OpenCode preload numbers are cheap to keep collecting from real `launch_tool`
sessions (Phase 3) rather than one-off measurements, so recalibrating after Phase 3
exists means it's grounded in a real, growing dataset rather than the two
measurements this correction is based on.

- Replace `TOOL_BASELINE_PRELOAD_TOKENS["claude"]` (and re-verify `["opencode"]`,
  which was closer to its measured value in this session's test but only checked
  once) with values grounded in this project's own real measurements, not public
  reports. **Watch for double-counting when doing this** (panel finding, GLM,
  medium): `estimate_preload_overhead` already adds real measured doc/skill/MCP
  tokens *on top of* the baseline constant. The live-tested ~39,000-44,000-token
  figure was measured on frank's actual config, which already includes those
  additive components — if the new baseline constant is set to that full figure
  without subtracting them back out, preload gets counted twice and the advisor
  will grey out models that would actually fit. Measure the new baseline in a
  clean directory (`--strict-mcp-config`, no skills, no CLAUDE.md/AGENTS.md) so it
  stays a true no-extras baseline, or explicitly decompose the measured figure
  before using it.

**Decision (2026-09-29): MacBook Air bandwidth — documented, no code change.**
Checked directly: Apple's own specs show the Air ships the same base (non-Pro/Max)
chip as the base MacBook Pro/Mac mini, at *identical* peak bandwidth per generation
(e.g. M5 MacBook Air and M5 MacBook Pro are both 153 GB/s) — `CHIP_BANDWIDTH_GB_S`'s
existing base-tier rows (`"Apple M1"` through `"Apple M5"`) already cover the Air
correctly; no separate `"MacBook Air"` row is needed, and `get_chip_info()` reading
`Chip:` couldn't distinguish chassis anyway (that string is identical regardless of
chassis; `system_profiler`'s separate `Model Name:` field is what actually says
"MacBook Air" vs "MacBook Pro"/"Mac mini", and nothing currently reads it). What
*is* real but explicitly deferred: the Air's fanless/passive cooling means it can
thermal-throttle under a sustained decode run (potentially minutes of continuous
compute) and fail to hold the same real-world throughput the identical spec-sheet
bandwidth promises on an actively-cooled chassis — a genuine caveat, but not worth
adding `Model Name:`-based chassis detection for right now.

**Decision (2026-09-29): minimum speed floor for model availability = 15 tok/s.**
Confirmed live: `build_recommendations()` currently gates a model's
availability/greyed-out status purely on memory fit
(`recommend_kv_bits_and_context`'s `fits`) — `estimated_tok_s` is computed and
shown but never used to exclude anything, so a model estimated at even 1 tok/s
would still appear fully selectable today. Add a floor: a model whose
`estimated_tok_s` is below **15 tok/s** should be greyed out the same way an
insufficient-memory model already is, with its own `context_reason` (e.g.
`"estimated_speed_too_slow"`) surfaced in `print_menu`'s existing greyed-out
explanation branch. Applies only to models with a known estimate — a model with
`estimated_tok_s is None` (the MoE-without-active-weight-data case) keeps today's
behavior (shown, speed labeled "unknown") rather than being excluded by a floor it
can't be measured against.

**Implemented and validated (2026-09-29):**

- Added `MIN_SPEED_TOK_S = 15` and wired it into `build_recommendations()`: when a
  model fits memory-wise but its `estimated_tok_s` is a real number below 15,
  `fits` flips to `False` and `context_reason` becomes `"estimated_speed_too_slow"`,
  surfaced in `print_menu` as `"estimated ~N tok/s — below the 15 tok/s usability
  floor"`. `estimated_tok_s is None` (MoE case) is left untouched, per the decision
  above. Verified against the real catalog on this M5 Pro (nothing currently falls
  below the floor here — fastest chip in the table) and against a simulated `Apple
  M1`/8-core run, where 6 of 8 models correctly flip to greyed-out with the new
  reason text and the MoE model with `estimated_tok_s: None` correctly stays shown.
- **Floor lowered twice after this decision, from real-world use — current value is
  7, not 15.** 15 → 10 (`3e7cd93`: gemma-4-12B-it-8bit at ~17 tok/s felt pretty
  fast, so 15 was excluding models that are actually fine to use), then 10 → 7
  (`2954789`: Qwen3.8-27B-4bit at ~8 tok/s still works OK in practice; mainly meant
  to let 16GB-RAM users reach usably-good models previously filtered out). The
  grey-out reason string renders the constant's current value, so no other code
  carries a stale floor number.
- **Recalibrated `TOOL_BASELINE_PRELOAD_TOKENS` from real measurements**, not public
  reports: ran a single-turn, no-tool-call request through `launch_tool.sh` for both
  tools from this repo's own root (no local `CLAUDE.md`/`AGENTS.md`/`.mcp.json`;
  claude's run used `--strict-mcp-config`), read the server's real total prompt
  tokens back, and subtracted this machine's measured global-skills contribution
  (~3,518 tokens, measured via `measure_skills_preload` directly — skills are global
  here, not per-directory) to avoid double-counting against what
  `estimate_preload_overhead` adds separately for any given real caller. Result:
  `claude` 18,000 → **18,900**, `opencode` 7,000 → **6,400**.
  - **Note on a discrepancy surfaced by this recalibration:** the ~39,000-44,000-token
    Claude Code figure recorded earlier in this same document (under "Live
    end-to-end test") could not be reproduced. A same-methodology Read-tool-call
    request measured today came in at ~24,000 total, consistent with today's bare
    "say hello" baseline (~22,406) plus a bit of tool-result overhead — not with the
    earlier ~39-44K claim. Since today's numbers are internally consistent across
    two independently-run live tests (this session's Phase 4 concurrency test and
    this Phase 5 measurement) and the earlier figure isn't, today's measurement is
    what's now encoded in the constant; the earlier figure is superseded, not
    reconciled — worth re-measuring again in a future session if this drifts.

### Phase 6 — Arrow-key model picker with a live detail pane (`fzf`)

Added 2026-09-29. Today's picker (`model_advisor.py`'s `print_menu` + `input()`
loop) prints every model's full tagline/good_for/usage_examples/known_issues block
inline, for every model, before the user even picks one — a wall of text once the
catalog has more than 2-3 models. Replacing it with a compact list + arrow-key
navigation + a live-updating detail pane, using `fzf` rather than hand-rolled
Python TUI code (considered stdlib `curses` as the no-new-dependency alternative;
`fzf` was chosen because it's the tool actually built for this exact UX — arrow
keys, fuzzy filter, live preview pane — and it's a small, single, well-known static
binary, not a heavy dependency like the Postgres one already removed elsewhere in
this plan).

**Locked design:**

```
  Local Model Advisor — Apple M5 Pro   RAM: 48GB   Free: 22.4GB
> qwen3.5-9b-mlx-8bit    9.7GB  131K tok  ~30 tok/s
  qwen3.5-9b-4bit        5.2GB  131K tok  ~55 tok/s
  qwen3.5-0.8b-4bit      0.6GB  131K tok  ~200 tok/s ⚠
  2/3 models shown (1 hidden: too little free memory)
────────────────────────────────────────────────────────────────
qwen3.5-9b-mlx-8bit
Best all-around choice on 24GB+ Macs

The strongest general recommendation once you have 24GB+ of RAM — a
real 128K-class context window with the fastest measured MLX
throughput of any model in this list (1.7-2.5x faster than its own
GGUF counterpart at every context size tested).

e.g. "Work through this repo-wide refactor" — tasks that need the
model to hold a lot of file context at once
e.g. "Let's pair-program through implementing this feature over
several turns" — long multi-turn sessions without hitting context
limits quickly

[KV cache unquantized — full decode speed, plenty of free memory
 for this context]
────────────────────────────────────────────────────────────────
> Type to filter, ↑↓ to move, enter to select, esc to quit
```

Preview pane goes on the **bottom** (`fzf --preview-window down:50%:wrap`), chosen
over a right-side pane specifically for full-width prose wrapping on the long
`good_for`/`usage_examples` text — traded off against showing fewer model rows
above the fold at once, which is an acceptable cost since this catalog is small
(single digits of models).

**Implementation shape:**
- `model_advisor.py` gets three new modes, none of which change the existing
  scoring/fit logic (`recommend_kv_bits_and_context`, `build_recommendations`) —
  this is a presentation-layer change only:
  - `--list-compact`: tab-delimited rows (`id\tsize\tcontext\tspeed`) for *fitting*
    models only, for fzf's list — matches today's behavior where only fitting
    models get a selectable number; not-fitting models are summarized as a plain
    count/reason line instead (same as today's `not_fitting` line), not fed into
    fzf at all.
  - `--detail <id>`: prints one model's existing tagline/good_for/usage_examples/
    known_issues/kv-note block — refactored out of `print_menu` into a shared
    function so the detail *text* itself doesn't change, only where it's rendered
    (fzf's preview pane instead of inline in the main list).
  - `--resolve <id>`: emits the same final machine-readable JSON selection line the
    interactive path already prints (`id`, `huggingface`, `size_gb`,
    `recommended_context`, `kv_bits`, `estimated_tok_s`), parameterized by an
    already-chosen id instead of an interactive prompt — reuses the exact contract
    `start_litellm_proxy.sh` already expects (`SELECTION=$(python3
    model_advisor.py ...)`).
- `start_litellm_proxy.sh` detects `fzf` via `command -v fzf`. If present: pipes
  `--list-compact`'s output into
  `fzf --delimiter='\t' --with-nth=2.. --preview 'python3 model_advisor.py --detail {1}' --preview-window down:50%:wrap`,
  takes the selected line's first field (the id), and calls `--resolve <id>`. If
  `fzf` is missing: falls back to today's plain numbered-menu behavior unchanged —
  same graceful-degradation pattern already used elsewhere for missing Claude
  Code/OpenCode (grey out / skip, don't hard-require).

**Implemented and live-validated (2026-09-29)**, on `start_mlx_server.sh` (this
script's Phase 4 name) and `model_advisor.py`, with one correction to this section's
own fzf invocation:

- Added `model_advisor.py --list-compact` (tab rows for fitting models, compact
  banner/hidden-count summary to stderr), `--detail <id>`, and `--resolve <id>`,
  none of which touch `recommend_kv_bits_and_context`/`build_recommendations`.
  `format_model_body_lines()` is the shared good_for/examples/known_issues/kv-note
  block used by both the plain menu and `--detail`'s preview text; `emit_selection_json()`
  is the shared final-JSON emitter used by both the interactive prompt and
  `--resolve`. Verified all three flags directly, plus the plain-menu fallback's
  output after the refactor (ordering intentionally shifted slightly: the KV-cache
  note now prints after usage examples/known_issues instead of right after
  `good_for`, matching this phase's own mockup — a presentation-only change).
- **Correction, found live (fzf 0.74.4) while validating, not assumed:** this
  section's own `--with-nth=2..` breaks "type to filter" by model name — confirmed
  live that `--with-nth` restricts fuzzy MATCHING to the displayed fields, not just
  their presentation, so hiding the id column also makes it unsearchable. That
  directly contradicts this same section's own mockup, which shows the id as the
  first visible column and promises "Type to filter". Fixed by dropping
  `--with-nth` entirely (all 4 tab fields display and are searchable, matching the
  mockup) and extracting the id from the accepted line with `cut -f1` in bash
  rather than depending on `--accept-nth` (confirmed present in fzf 0.74.4, but not
  worth a version floor just to avoid one `cut`).
- Installed fzf via Homebrew on this machine (wasn't present) specifically to
  validate this phase live rather than trust it unverified. Confirmed:
  `--list-compact`'s tab rows are well-formed for `--delimiter`; `--filter` (fzf's
  non-interactive test mode) correctly selects a model by typing its id, something
  the original `--with-nth=2..` invocation could not do; the `--detail {1}` preview
  command's substituted form runs correctly once `{1}` is a real id; and
  `--resolve <id>` returns the exact same JSON shape the interactive path already
  produced. The full interactive curses UI itself (arrow keys, live preview
  rendering) still needs a real terminal and wasn't exercised pixel-for-pixel — the
  three non-interactive python modes and the fzf command construction/piping are
  what's been verified.

### Phase 7 — GitHub Pages `docs/` site with a no-clone-needed download

Added 2026-09-29. Sequenced **after Phase 4**, since the zip and `index.html` need
to reference the final script names/trimmed docs Phase 4 produces, not today's
names.

**Prerequisite, not scriptable from here:** this repo has no GitHub remote
configured yet (`git remote -v` is empty). GitHub Pages requires an actual GitHub
repo with Pages enabled in its settings, pointed at the `docs/` folder on `main` —
a one-time manual step (push the repo, then flip the setting) that has to happen
before this phase's `index.html` is reachable at a real URL, though the folder and
its contents can still be built and committed before that.

**Structure:**

```
docs/
├── index.html              # GitHub Pages landing page
├── bushpilot.zip   # packaged release bundle
└── build_release.sh        # regenerates the zip — run manually before publishing an update
```

- **`index.html`**: project tagline and what it does (run Claude Code/OpenCode
  against a local MLX-served LLM instead of a cloud API, hardware-aware so it
  won't recommend a model/context that crashes the machine), requirements (Apple
  Silicon Mac, macOS, Homebrew, Python 3), a short user guide (download → unzip →
  run the supervisor script → run the launcher script), the zip's download link,
  and a link back to the GitHub repo itself for anyone who'd rather clone/contribute.
- **`bushpilot.zip`**: end-user runtime files only, not this project's
  internal planning/research docs — the renamed supervisor script and
  `launch_tool` script (Phase 4's final names), `model_advisor.py`, `models.json`,
  and a short standalone `README.md` written specifically for zip users (distinct
  from `PHASE_PLAN.md`/`PROJECT_STATUS.md`, which stay working docs, not shipped).
- **`build_release.sh`**: a small manual script that zips exactly that file list
  into `docs/bushpilot.zip` — no CI/automation, matching this project's
  existing plain-shell-scripts-only approach elsewhere.

**Implemented and live-validated (2026-09-29):**

- **The "not scriptable from here" prerequisite got resolved this session**, not
  deferred: pushed this repo to a real remote —
  `https://github.com/frank321/bushpilot` (created public via `gh
  repo create`, at the user's direction) — and enabled GitHub Pages on it via the
  API (`POST /repos/frank-yan/bushpilot/pages`, `source.branch=main`,
  `source.path=/docs`), rather than leaving it as a manual step. `index.html` and
  the footer link both point at that real repo URL.
- Built `docs/index.html` (tagline, requirements, download button, repo link, a
  3-step get-running guide mirroring `QUICK_START.md`), `docs/release-README.md`
  (the zip's standalone `README.md` source, distinct from this doc and
  `PROJECT_STATUS.md`), and `docs/build_release.sh` (zips
  `start_mlx_server.sh`/`launch_tool.sh`/`model_advisor.py`/`models.json` +
  `release-README.md` into `docs/bushpilot.zip`, from a temp dir so the
  zip's top-level folder is `bushpilot/`, not loose files).
- **Validated, not just built:** ran `build_release.sh` for real; unzipped the
  actual output into an isolated scratch directory (no other repo files present)
  and confirmed `start_mlx_server.sh`/`launch_tool.sh` both pass `bash -n`,
  `model_advisor.py` imports and its `--list-compact` runs correctly there — the
  bundle is genuinely self-contained, not just assumed to be from the file list.
  Also lint-checked `index.html` (`tidy -utf8`, after ruling out the installed
  `tidy`'s HTML5-element/encoding false positives) and served `docs/` locally to
  confirm the zip's relative download link actually resolves next to the page.

### Phase 8 — Speculative-decoding backend (`mlx-dspark`)

Added 2026-09-29. Real measured decode-speed gains (1.5x-3.1x depending on model) are available
via speculative decoding — a small "drafter" model proposes several tokens per round, the target
verifies them in one pass — but getting them requires a second serving backend, not just a flag on
the existing one: `mlx_vlm.server`'s own native `--draft-model` support works, but underperforms a
separate third-party package, `mlx-dspark`, by ~15-20% on this project's actual hardware (measured,
not assumed — see `HARDWARE_SPEED_ESTIMATION.md` §18 for the full methodology and per-model
numbers), because `mlx-dspark` runs a one-time per-machine kernel calibration `mlx_vlm`'s
implementation doesn't have.

**Decision: two backends, selected per-`models.json`-entry, not a global setting.**
`mlx-dspark` cannot handle image/video input at all (no CLI support, silently drops the target's
vision tower on load) — so it's only used for a text-only "fast" tier. `mlx_vlm`'s native draft
support is kept for the two things `mlx-dspark` structurally can't do: a multimodal-capable "some
speedup, keeps vision" tier, and Gemma-4-E4B's MTP drafter specifically (Google's own architecture,
not a DFlash checkpoint — no `mlx-dspark`-compatible drafter exists for this model at all). Every
model that has a real drafter checkpoint gets up to 3 selectable `models.json` entries: the
original plain entry (safest, always multimodal-capable, no speedup), an `mlx-dspark`-backed
text-only speed variant (where a compatible drafter exists), and — only where multimodal+drafter
was actually tested — an `mlx_vlm`-backed multimodal speed variant.

**Decision: `models.json` gets a new optional `serving` block, not a command template.**
Considered making the launcher fully data-driven (`models.json` specifies the literal command to
run) and rejected it: letting a data file control an executed shell command is an injection-shaped
risk, and several flags (`--max-kv-size`/`--kv-bits`/`--max-num-seqs`) genuinely have to be computed
from the *live* free memory at launch time, which can't be frozen into a static file. Landed on
structured fields instead (`backend`, `mode`/`drafter` for `mlx-dspark`; `draft_model`/`draft_kind`/
`draft_block_size` for `mlx_vlm`; plus informational `measured_speedup`/`drafter_size_gb`/
`multimodal_capable`/`greedy_divergence_observed`) — `start_mlx_server.sh` branches only on the
`backend` enum, never on a specific model id, so adding model #18 later never requires touching the
launcher script.

**Implemented and live-validated (2026-09-29)** — every item below was tested against a real
running server (`mlx-dspark serve` and `mlx_vlm.server` both), with real inference requests, not
just written and assumed correct:

- `models.json`: 11 new drafter-backed entries alongside the original 8 (19 total), each with real
  measured `serving.measured_speedup`/`greedy_divergence_observed` data — 9 covering the text-only
  `mlx-dspark` tier and the 8-bit multimodal `mlx_vlm` tier in the main session, plus the two 4-bit
  multimodal variants (`qwen3.5-9b-mlx-4bit-dflash-vlm`, `gemma-4-12b-it-4bit-dflash-vlm`) in a
  same-day follow-up that ran the full baseline-vs-drafter image-request methodology on both,
  including a manual `--draft-block-size` sweep (3/4/5/6) and live tool-call checks with the drafter
  attached. See `HARDWARE_SPEED_ESTIMATION.md` §18 for the full table and methodology.
  `_schema_notes.serving` documents the new field set.
- `model_advisor.py`: `effective_size_gb()` folds `serving.drafter_size_gb` into the existing
  memory-fit checks (a drafter is a real, separate download sitting in memory alongside the
  target); `estimate_speed_tok_s()` applies `serving.measured_speedup` as a multiplier on top of
  the existing chip-scaled bandwidth formula, verified to reproduce real measured tok/s closely on
  this machine.
- `start_mlx_server.sh`: `resolve_serving_config()` (single source of truth reading `models.json`
  by id, used by both the interactive picker and the fixed-model CLI form so they can't drift out
  of sync), `ensure_backend_installed()` (installs/upgrades `mlx-dspark` only when a
  selected model needs it, with a minimum-version floor — `MLX_DSPARK_MIN_VERSION`,
  mirroring `MLX_VLM_MIN_VERSION`'s upgrade-if-older pattern; added by the same-day
  follow-up along with the 4-bit vlm entries), a generalized `ensure_repo_downloaded()`
  (pre-downloads the drafter too, same as the target), backend-aware process-identity
  matching, two full argument-building branches (each
  backend's own flag set — confirmed via each backend's own `--help`, not guessed), a
  backend-aware startup smoke check (`mlx-dspark` has no `/v1/cache/stats` endpoint and a different
  `/metrics` shape, confirmed live), and a backend-aware `render_dashboard()` with an
  `mlx-dspark`-specific stats section (live speculative-decoding accept rate, per-position
  acceptance decay, real peak-memory figures, swap usage, memory-guard shed warnings — richer than
  what `mlx_vlm`'s own API exposes for some of these).
- **A real bash bug found and fixed along the way, not just a script bug**: macOS's shipped
  `/bin/bash` (3.2.57, pre-GPLv3) has a genuine parser bug with `eval "$(cmd <<'EOF' ... EOF)"`
  nested inside a function body — it silently fed the heredoc's Python source text back as shell
  code instead of running it. Fixed via the two-step capture-then-`eval` form. Wouldn't have
  surfaced from `bash -n` alone; only found by actually running the script.
- **A second bash 3.2 bug, found by the config.json follow-up's live testing**: `set -u` +
  `"${arr[@]}"` on an EMPTY array is an "unbound variable" error on bash < 4.4 — which meant any
  interactive launch whose config.json set `preferred_model` but not `concurrency` crashed with
  `CONCURRENCY_FLAG[@]: unbound variable` (reproduced against the committed pre-fix script).
  Fixed by renaming the array to `ADVISOR_FLAGS` and using the empty-safe idiom
  `${ADVISOR_FLAGS[@]+"${ADVISOR_FLAGS[@]}"}` at every call site, plus a conditional append
  inside the fzf preview string where the idiom can't be used inline.
- **Tool-calling verified live** through all three paths — `mlx-dspark` with both the `gemma4` and
  `qwen3_coder` tool-parser families, and `mlx_vlm`+MTP — via real `/v1/chat/completions` requests
  with a `web_search` tool schema attached. All three correctly returned `finish_reason: tool_calls`
  with a well-formed function call, with the drafter confirmed genuinely active in the same
  response (not silently bypassed for tool-call requests). `models.json`'s `tool_calling` blocks for
  the tested entries were updated with the real method/timestamp rather than left inherited from
  the base entry.
- `launch_tool.sh`: checked, needs no changes — it only checks `/health` returns 200 and points
  `ANTHROPIC_BASE_URL` at host:port, never inspects response shape, so it's already
  backend-agnostic.
- Investigated but deliberately NOT integrated: **Splash**, a purpose-built third-party inference
  engine for the same Qwen3.8-27B target — real, well-provenanced (same publisher as the DFlash2
  drafter already in use), but too young a project at investigation time (~2 weeks old, single
  committer, a documented benchmark/shipped-defaults discrepancy in its own launch post, no
  fallback path, only covers one model already covered by `mlx-dspark`) to justify a third backend.

**Remaining work**: none from the original Phase 8 list — the 4-bit multimodal+drafter
combos, the `mlx-dspark` minimum-version pin (`MLX_DSPARK_MIN_VERSION`), and the
`config.json` extensions (`max_context_tokens` cap; `dspark_prefix_cache_*` tuning
keys) were all completed in same-day follow-up sessions. The only still-open
publishing step is rebuilding the release zip via `docs/build_release.sh` and pushing
to the remote (the zip's file list picks up all of this automatically on rebuild).
