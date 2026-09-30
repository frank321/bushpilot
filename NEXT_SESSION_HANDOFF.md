# Handoff — mlx-dspark / speculative-decoding integration

**Written:** 2026-09-29, updated same day after a documentation pass, then again late the same
evening (local time; 2026-09-30 UTC) by a follow-up session that completed remaining-work item #1
(the missing 4-bit multimodal+drafter combos). The original session
added a second serving backend (`mlx-dspark`) alongside the existing `mlx_vlm` one,
wired it fully through `models.json` → `model_advisor.py` → `start_mlx_server.sh`
(including its own dashboard stats), live-verified all of it — real launches, real
inference requests, real tool-calling checks — and updated every project doc that
was stale as a result (`PHASE_PLAN.md` Phase 8, `HARDWARE_SPEED_ESTIMATION.md` §18,
`PROJECT_STATUS.md`'s staleness note, `docs/index.html`). Everything described as
"done" below was tested against a real running server on this machine, not just
written and assumed correct.

If you're picking this up in a new session, **read this whole file before touching
code** — it front-loads the gotchas that cost real time this session, so you don't
rediscover them. `PHASE_PLAN.md` Phase 8 and `HARDWARE_SPEED_ESTIMATION.md` §18 have
the fuller decision-record/research version of what's summarized here.

---

## What's done (fully implemented AND live-tested)

1. **`models.json`**: new optional `serving` block (documented in
   `_schema_notes.serving`) + 11 drafter-backed model entries alongside the
   original 8 plain entries (19 total; 9 from the original session, 2 more from the
   follow-up). Each carries real measured
   `speedup`/`greedy_divergence_observed`/`multimodal_capable` data, not estimates.
   - Text-only speed tier (`mlx_dspark` backend): `qwen3.8-27b-4bit-dflash`,
     `qwen3.5-9b-mlx-8bit-dflash`, `qwen3.5-9b-mlx-4bit-dflash`,
     `gemma-4-12b-it-8bit-dflash`, `gemma-4-12b-it-4bit-dflash`
   - Multimodal-capable speed tier (`mlx_vlm` + `--draft-model`):
     `qwen3.8-27b-4bit-dflash-vlm`, `qwen3.5-9b-mlx-8bit-dflash-vlm`,
     `gemma-4-12b-it-8bit-dflash-vlm`, `gemma-4-e4b-it-4bit-mtp` (Google's own MTP,
     not DFlash — no DFlash checkpoint exists for E4B); plus, from the follow-up
     session, `qwen3.5-9b-mlx-4bit-dflash-vlm` and `gemma-4-12b-it-4bit-dflash-vlm`
   - Deliberately NOT added: a drafter variant for `gemma-4-26b-a4b-it-4bit` (MoE) —
     measured only 1.08x with MTP, not worth the complexity/download.
   - Originally NOT added: `-dflash-vlm` variants for the two 4-bit-only combos
     (`qwen3.5-9b-mlx-4bit`, `gemma-4-12b-it-4bit`) — only 8-bit had been tested for the
     multimodal+drafter tier. **Completed by the follow-up session — see remaining-work
     item #1 below for the results.**

2. **`model_advisor.py`**: `effective_size_gb()` (adds `serving.drafter_size_gb` to
   memory-fit checks) and `estimate_speed_tok_s()` (applies `serving.measured_speedup`
   as a multiplier on top of the existing chip-scaled bandwidth formula, not a flat
   number — verified this reproduces real measured tok/s closely on this machine).

3. **`start_mlx_server.sh`**: `resolve_serving_config()` (reads `models.json` by
   `$LOCAL_MODEL` id, single source of truth for both interactive and fixed-model CLI
   paths), `ensure_backend_installed()`, generalized `ensure_repo_downloaded()` (fetches
   the drafter too), two full argument-building branches, backend-aware process-identity
   pattern, backend-aware startup smoke check, and a backend-aware `render_dashboard()`
   with a rich `mlx-dspark`-specific stats section (SPECULATION, PREFIX CACHE, MEMORY,
   memory-guard-shed warnings).

4. **Tool-calling verified live** through all three paths (not assumed): `mlx-dspark`
   with both `gemma4` and `qwen3_coder` tool parsers, and `mlx_vlm`+MTP. All three
   correctly returned `finish_reason: tool_calls` with well-formed function calls,
   confirmed via real `/v1/chat/completions` requests with a `web_search` tool schema
   attached — `models.json`'s `tool_calling` blocks for the tested entries were updated
   with the real verification method/timestamp.

5. **`launch_tool.sh`**: checked, needs NO changes — it only checks `/health` returns
   200 and points `ANTHROPIC_BASE_URL` at host:port; never inspects response shape, so
   it's already backend-agnostic.

---

## Gotchas discovered this session (don't rediscover these)

- **macOS's `/bin/bash` is 3.2.57** (ancient, pre-GPLv3). It has a real parser bug:
  `eval "$(cmd <<'EOF' ... EOF)"` — heredoc nested directly inside a command
  substitution inside a function body — silently prints back the heredoc's SOURCE TEXT
  instead of running it, then fails with `syntax error near unexpected token '('`. Fix:
  two-step form (`sub="$(cmd <<'EOF' ... EOF)"; eval "$sub"`) works fine in the same
  context. The pre-existing top-of-file config.json eval was safe only because it runs
  at script top-level, not inside a function — this bug only bites inside functions.
  See `resolve_serving_config()` for the fixed pattern; copy that style for any future
  eval+heredoc inside a function.

- **This dev sandbox sits behind a corporate TLS-intercepting proxy.**
  Historically this broke every Python HTTPS download (`huggingface_hub`/
  `httpx`) two ways: `unable to get local issuer certificate` (bundle without
  the intercepting corporate root) or `Basic Constraints of CA cert not marked
  critical` (bundle with it). **Root cause was two-fold and BOTH halves are now
  fixed (2026-09-30):** (1) Python 3.14 enables `VERIFY_X509_STRICT` by default and
  the intercepting corporate root CA is non-compliant (non-critical Basic
  Constraints, no keyUsage) — fixed by patching the copy of that root in the
  machine-local CA bundle pointed to by `~/.zshrc`'s `SSL_CERT_FILE`
  (BC marked critical, critical keyCertSign keyUsage added; trust-anchor
  self-signatures aren't checked by OpenSSL, so the DER surgery is safe —
  verified by a live strict-mode handshake to huggingface.co; the original
  bundle is backed up alongside it). This bundle is machine-local and must
  NEVER be committed to the repo. (2) `start_mlx_server.sh` used to
  clobber the user's `SSL_CERT_FILE` (pointing at that interception-aware
  bundle) with certifi's — it now respects an already-set, valid `SSL_CERT_FILE`,
  and falls back to certifi / the system bundle on machines where it is unset,
  so the script stays fully portable to non-intercepted networks.
  Plain `curl` always worked (LibreSSL is lenient). The old hand-built
  curl-cache workaround should no longer be needed — `snapshot_download` now
  works directly.

- **`mlx-dspark`'s `verdict.headline` field is unreliable** — observed live saying
  "No decode measured yet" in the same `/metrics` response where
  `mean_decode_tokens_per_sec` and `mean_accept_len` were already populated.
  Deliberately excluded from the dashboard for this reason.

- **`mlx-dspark`'s `auto_cap` metrics field is always `null`** in this project's
  wiring, because we never pass `--max-draft auto` (mlx-dspark self-calibrates a
  static cap by default, which is the whole point of using it). Not worth showing on
  the dashboard as a permanent "—".

- **`mlx_vlm`'s native `--draft-kind dflash` support works but underperforms
  `mlx-dspark` by ~15-20%** even after manually matching `--draft-block-size` to what
  `mlx-dspark` auto-discovers — confirmed on 3 separate models. `mlx-dspark` does
  per-machine kernel calibration (`small_m`, `sdpa_split`, `multirow_attn`, wide-GEMM
  crossover) that `mlx_vlm`'s implementation doesn't have. This is WHY the project
  uses two backends instead of just extending `mlx_vlm` everywhere.

- **Greedy-decoding divergence from baseline is real but has no clean predictive
  rule.** Confirmed via live temperature=0/matched-seed testing: Qwen3.8-27B-4bit and
  Qwen3.5-9B-4bit diverge under `mlx-dspark`; Qwen3.5-9B-8bit and both Gemma-4-12B
  quants don't. Then, on the SAME Gemma-4-12B model, the `mlx_vlm`+DFlash (not
  `mlx-dspark`) image-request path DOES diverge while the `mlx-dspark` text path
  doesn't. So divergence risk depends on model+quant+backend+content combination, not
  any single one of those — it's tracked per-`models.json`-entry
  (`greedy_divergence_observed`), not assumed.

---

## Remaining work (not started, or started-but-incomplete)

Prioritize roughly in this order:

### 1. ~~Test the missing 4-bit multimodal+drafter combos~~ — DONE (2026-09-29 local / 2026-09-30 UTC, follow-up session)
Both 4-bit combos were tested with the full methodology (three real image prompts of
91-900-token completions each via a PIL-generated text-bearing test image,
`temperature=0`, matched seed, baseline vs. `--draft-model`/`--draft-kind dflash`/
`--draft-block-size 4`), plus a manual `--draft-block-size` sweep (3/4/5/6 — 4 held up
as the right value on both) and live tool-call checks with the drafter attached (both
returned `finish_reason: tool_calls` with the drafter confirmed active in the same
response). The `models.json` entries `qwen3.5-9b-mlx-4bit-dflash-vlm`
(`measured_speedup: 1.3`, `greedy_divergence_observed: false`) and
`gemma-4-12b-it-4bit-dflash-vlm` (`measured_speedup: 1.55`,
`greedy_divergence_observed: true`) were added following the 8-bit siblings' pattern,
and `start_mlx_server.sh` was verified end-to-end against the new qwen entry
(launcher resolved the serving config, served a real image request with draft stats
present, tore down cleanly). Notable finding: each 4-bit quant flipped its divergence
outcome vs. its own mlx-dspark result — Qwen 4-bit diverged under mlx-dspark but
stayed byte-identical under mlx_vlm on all three image prompts; Gemma 4-bit stayed
identical under mlx-dspark but diverged under mlx_vlm on two of three prompts —
strengthening the model+quant+backend+content rule (documented in
`HARDWARE_SPEED_ESTIMATION.md` §18, which also gained the two new table rows).

### 2. ~~`mlx-dspark` minimum-version pinning~~ — DONE (2026-09-29 local / 2026-09-30 UTC, follow-up session)
Added `MLX_DSPARK_MIN_VERSION="0.20.1"` right next to `MLX_VLM_MIN_VERSION` (with the
same verified-against rationale in its comment), and `ensure_backend_installed()` now
checks the installed version against that floor using the existing generic
`version_at_least`/`installed_version` helpers — pip-upgrading with a
`mlx-dspark>=${MLX_DSPARK_MIN_VERSION}` constraint when older (mirroring
`install_packages_if_needed`'s mlx-vlm upgrade path exactly), and the fresh-install
path now carries the same constraint instead of unbounded `mlx-dspark`. Verified
three ways: `bash -n`; a behavioral harness running the real shipped function
against a stubbed pip (already-installed → no-op, older → correct
`--upgrade`/constraint args, not-importable → correct install args, non-dspark
backend → complete no-op — 6/6 checks pass); and a live end-to-end launcher run
with an `mlx_dspark`-backed model (`qwen3.5-9b-mlx-4bit-dflash`), which logged the
floor check, found 0.20.1, and passed the startup smoke check cleanly.

### 3. ~~`config.json` extensions~~ — DONE (2026-09-29 local / 2026-09-30 UTC, follow-up session)
Both extensions implemented and live-tested end-to-end:
- **`max_context_tokens`**: a persistent UPPER BOUND on the interactive picker's
  computed context — deliberately a cap, not an override. Threaded as
  `model_advisor.py --max-context-tokens` through a new shared `ADVISOR_FLAGS`
  array (which replaced `CONCURRENCY_FLAG` and now carries both advisor-affecting
  config keys uniformly across every advisor call site: fzf picker, detail
  preview, `--resolve`, and `preferred_model`). The advisor's free-memory fit
  math still runs underneath the cap (applied in `recommend_max_context` before
  tier rounding), so it can only shrink the result, never grow it past what the
  host can afford — the memory-fit-safe alternative to the fixed-model CLI
  form's 3rd positional arg. Invalid values fall back to no cap; a cap below the
  usability floor greys the model out with the existing reason string. Verified
  live: cap 65536 → picker selects 65536 with the log line saying so; cap 10000
  → model doesn't fit (exit 3); garbage → no cap.
- **`dspark_prefix_cache_slots` / `_rungs` / `_dir` / `_max_ram_mb`**: forwarded to
  mlx-dspark's own `--prefix-cache-*` flags (names confirmed against its `--help`)
  only when set — unset means its own defaults, unchanged. The
  max-ram-mb-requires-dir rule from mlx-dspark's own docs is enforced in the
  launcher with a clear error message. Verified live: all three set flags appear on
  the running `mlx-dspark serve` process's command line, server healthy, smoke
  check passed; the requires-dir violation exits 1 before spawning.
- **Bonus — a real pre-existing bug found and fixed by the testing**: the committed
  script crashed with `CONCURRENCY_FLAG[@]: unbound variable` on ANY interactive
  launch whose config.json set `preferred_model` but NOT `concurrency` — macOS
  bash 3.2 + `set -u` treats `"${arr[@]}"` on an empty array as an unbound
  variable (fixed upstream only in bash 4.4). Reproduced against the pre-fix
  committed script, then fixed at every call site with the empty-safe idiom
  `${ADVISOR_FLAGS[@]+"${ADVISOR_FLAGS[@]}"}` (and a conditional append inside
  the fzf `preview_cmd` string, where the idiom can't be used inline). The
  previously-crashing `preferred_model`-only config now launches cleanly.

### 4. ~~Update project docs~~ — DONE (2026-09-29, same session)
`PROJECT_STATUS.md` (staleness note added), `PHASE_PLAN.md` (new Phase 8),
`HARDWARE_SPEED_ESTIMATION.md` (new §18, full measured-speedup table + methodology +
its own open-items list), and `docs/index.html` (new Feature 7 section, new
Architecture subsection, updated catalog list/Components table/meta description) were
all updated to document the dual-backend architecture, following each doc's existing
style. Structurally validated (`docs/index.html`'s tag balance checked, `tidy`'s
pre-existing HTML5/UTF-8 false positives — already documented in `PHASE_PLAN.md`
Phase 7 — confirmed unrelated to these edits).

**Not done as part of this**: the release zip (`docs/bushpilot.zip`) was not
rebuilt via `docs/build_release.sh`, and nothing was pushed to the GitHub remote —
the zip's file list already includes `models.json`/`model_advisor.py`/
`start_mlx_server.sh` by name, so a rebuild would pick up all of this session's
changes automatically, but that's a separate manual publish step or a follow-up
task, not something instructed this session.

### 5. (Lower priority / optional) Splash
Investigated but deliberately NOT integrated — see the conversation history for the
full reasoning (too young a project — ~2 weeks old at investigation time, single
committer, documented benchmark/default-settings discrepancy, no fallback path, only
covers Qwen3.8-27B which is already covered by `mlx-dspark`). Worth revisiting in a
month or two if curious, not urgent.

---

## Quick orientation for a new session

- Real drafters are downloaded into the standard HF cache
  (`~/.bushpilot/models/hub/models--{org}--{repo}/`) for: `z-lab/gemma4-12B-it-DFlash`,
  `z-lab/Qwen3.5-9B-DFlash`, `incoai/Qwen3.8-27B-DFlash2`,
  `mlx-community/gemma-4-E4B-it-assistant-bf16`, `mlx-community/gemma-4-26B-A4B-it-assistant-bf16`
  — all checksum-verified against HF's own reported SHA-256 at download time. You should
  NOT need to re-download these to keep testing.
- `~/.bushpilot/venv` already has `mlx-dspark==0.20.1` installed alongside
  `mlx-lm`/`mlx-vlm`.
- To manually smoke-test either backend without the full script:
  `~/.bushpilot/venv/bin/mlx-dspark serve --model <repo> --drafter <repo> --mode dflash --port 38999`
  or `~/.bushpilot/venv/bin/python -m mlx_vlm.server --model <path> --draft-model <repo> --draft-kind {dflash,mtp} --port 38999`,
  then `curl http://127.0.0.1:38999/health` / `/metrics` / `/v1/chat/completions`.
