#!/usr/bin/env bash
set -uo pipefail

# ci.sh — runs everything that can be checked without Apple Silicon or a
# downloaded model: the pytest suite, the tests/shell/ bats suite, ruff, and the
# ShellCheck linter. Each tool is skipped (not failed) when not installed locally,
# so this is safe to run on any machine; .github/workflows/ci.yml runs the
# same script with every tool installed, so nothing is skipped there.
#
# What this deliberately does NOT run: verify_max_kv_size.sh and
# verify_long_context_memory.sh (need a real Mac + a downloaded model --
# see their own headers), and anything that talks to a live mlx_vlm.server.

cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

FAILED=0
run_step() {
    local name="$1"; shift
    echo "=== $name ==="
    if "$@"; then
        echo "--- $name: OK"
    else
        echo "--- $name: FAILED"
        FAILED=1
    fi
    echo
}

skip_step() {
    echo "=== $1 ==="
    echo "--- $1: SKIPPED ($2 not found -- install it to run this locally)"
    echo
}

if command -v pytest >/dev/null 2>&1; then
    run_step "pytest" pytest
elif python3 -c "import pytest" >/dev/null 2>&1; then
    run_step "pytest" python3 -m pytest
else
    skip_step "pytest" "pytest (pip install -r requirements-dev.txt)"
fi

if command -v ruff >/dev/null 2>&1; then
    run_step "ruff" ruff check .
else
    skip_step "ruff" "ruff (pip install ruff)"
fi

if command -v shellcheck >/dev/null 2>&1; then
    run_step "shellcheck" shellcheck start_mlx_server.sh launch_tool.sh uninstall.sh \
        verify_max_kv_size.sh verify_long_context_memory.sh docs/build_release.sh
else
    skip_step "shellcheck" "shellcheck"
fi

if command -v bats >/dev/null 2>&1; then
    if [ -d tests/shell ]; then
        run_step "bats" bats tests/shell
    fi
else
    skip_step "bats (tests/shell)" "bats"
fi

run_step "models.json is valid JSON" python3 -c "import json; json.load(open('models.json'))"

if [ "$FAILED" -ne 0 ]; then
    echo "ci.sh: one or more checks FAILED (see above)"
    exit 1
fi
echo "ci.sh: all checks that ran, passed"
