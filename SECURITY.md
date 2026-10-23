# Security Policy

## Scope

Bushpilot runs a local model server (MLX) on your machine and attaches coding
tools to it. It does not send data to any cloud service by design. The local
server is intended to be reachable only from `localhost`.

Relevant classes of issues include:

- The server binding to a non-loopback interface, or otherwise being reachable
  from the network when it shouldn't be.
- The memory/liveness backstops in `start_mlx_server.sh` failing to tear down a
  server under memory pressure (system-stability impact).
- Command injection or path traversal in the launcher scripts from a malicious
  model id or catalog entry.
- Anything that would write outside `~/.bushpilot/` or the project directory.

Out of scope: vulnerabilities in MLX / `mlx_vlm` / MLX-DSpark themselves, in the
attached coding tools (Claude Code, OpenCode, Aider, Cline, Pi), or in models you
download — please report those upstream.

## Reporting

Please **do not** open a public issue for a security problem.

Use GitHub's [private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
via the repo's **Security** tab → "Report a vulnerability".

Include: a description, steps to reproduce, the affected script/module, your
macOS version and chip, and the impact you believe it has.

## Supported versions

This is an actively developed project; fixes land on `main`. Please test against
the latest `main` before reporting.
