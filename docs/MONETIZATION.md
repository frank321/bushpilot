# Monetization — plan, not current state

**Status (2026-09-30): nothing is monetized. This file records the strategy
decided during the bushpilot rename session, so future-us doesn't re-derive it.**

Positioning recap (why any of this could work): bushpilot's wedge is *not* "another
local LLM server" — ollama/LM Studio/llama.cpp own that ground. It is: **the way
to run coding agents (Claude Code / OpenCode) against local models on a Mac,
with the math to prove what actually fits.** Four things nobody else combines:

1. Exact context-size math from each model's real KV config (hybrid layers,
   sliding windows, MoE active params), against live free memory.
2. Server-side caps that are actually enforced (`--max-kv-size` rejected with
   HTTP 400; Ollama's MLX runner cannot do this — maintainers rejected the fix
   upstream twice).
3. A verified catalog (`models.json`): memory-admission tests + tool-calling
   verdicts against real coding agents.
4. Agent-first launch flow (`launch_tool.sh`), not model-first.

The moat is the **data and research** (verified catalog,
`HARDWARE_SPEED_ESTIMATION.md`), not the wrapper code. Monetize the moat, never
the launcher.

## Guiding principle

**The launcher stays free and open forever — it is the funnel.** What gets
monetized is what the funnel accumulates: the verified dataset, fleet
management, and hand-holding. Individual developers do not pay for CLIs;
companies pay for fleet software, certified data, and support.

## Options, weakest to strongest

### 1. Donations (GitHub Sponsors / Open Collective / Ko-fi) — tip jar

Set up on day one (it's free, and some users genuinely want to give back), but
set expectations: niche CLI tools typically earn $0–100/month. This is goodwill
capture, not a business.

### 2. Open core / Pro tier

Precedent: llmpilot (macOS Claude Code companion, seen during the 2026-09 naming
research) — free open core, one-time Pro purchase for automation features.

Candidate Pro features for bushpilot (convenience, never core correctness):

- Polished menu-bar app / GUI over the two scripts
- Auto-updating verified catalog (new models tested & shipped to subscribers)
- Scheduled / automatic model rotation and cache management

### 3. Mac mini fleet — the strongest play

Companies are buying Mac minis as inference boxes right now (privacy + cost +
Apple Silicon performance per watt). A "bushpilot for teams" — one dashboard
managing a rack of Macs, shared catalog, health alerts, SSO — is infrastructure
software, and companies reliably pay for infrastructure.

**Trigger signal:** the moment someone asks "can I run this on 20 Mac minis for
my team," the fleet tier has buyers. Until then, don't build it.

### 4. Verification as a service / data licensing

The catalog is the least copyable asset. Model publishers and quantizers
(including mlx-community-style orgs and model vendors) might pay for certified
inclusion — a "bushpilot-verified" badge backed by our memory-admission and
tool-calling results on their model pages. The hardware × model × tok/s
benchmark matrix could also be licensed as a dataset/API that other tools query.

### 5. Enterprise support contracts

Paid SLAs, setup help, and deployment consulting for teams running local coding
agents. Boring, highest margin per customer, only viable with visible traction.

## Sequencing

1. **Now → launch OSS.** Public repo, README with the "Ollama can't cap context
   on MLX — we can" hook, donations link live.
2. **Build audience.** Stars, issues, real users. No paywall, no Pro.
3. **Watch for the fleet question.** That's the market telling us tier 3 is real.
4. **Then** add the paid layer that matches the pull (fleet > verification >
   Pro convenience > support).

Do not invert this order: monetizing before there's pull burns the niche
goodwill that the whole strategy depends on.
