#!/usr/bin/env bash
set -euo pipefail

# build_release.sh — regenerates docs/local-model-setup.zip from the current runtime
# files in the repo root. Manual, run-before-publishing step -- no CI/automation,
# matching this project's plain-shell-scripts-only approach elsewhere (PHASE_PLAN.md
# Phase 7).
#
# Usage: ./docs/build_release.sh   (paths below are resolved relative to this
#   script's own location, so it works regardless of the caller's cwd)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
ZIP_PATH="$SCRIPT_DIR/local-model-setup.zip"

# End-user runtime files only -- not this project's internal planning/research docs
# (PHASE_PLAN.md, PROJECT_STATUS.md, etc. stay working docs, never shipped).
RUNTIME_FILES=(
    start_mlx_server.sh
    launch_tool.sh
    model_advisor.py
    models.json
)

for f in "${RUNTIME_FILES[@]}"; do
    if [ ! -f "$REPO_ROOT/$f" ]; then
        echo "[ERROR] Expected runtime file missing: $REPO_ROOT/$f" >&2
        exit 1
    fi
done

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

BUNDLE_DIR="$WORKDIR/local-model-setup"
mkdir -p "$BUNDLE_DIR"
for f in "${RUNTIME_FILES[@]}"; do
    cp "$REPO_ROOT/$f" "$BUNDLE_DIR/"
done
cp "$SCRIPT_DIR/release-README.md" "$BUNDLE_DIR/README.md"
chmod +x "$BUNDLE_DIR/start_mlx_server.sh" "$BUNDLE_DIR/launch_tool.sh"

rm -f "$ZIP_PATH"
(cd "$WORKDIR" && zip -rq "$ZIP_PATH" "local-model-setup")

echo "[OK] Built $ZIP_PATH"
echo "     Contents:"
(cd "$WORKDIR" && find "local-model-setup" -type f | sort | sed 's/^/       /')
