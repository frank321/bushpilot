#!/usr/bin/env bash
set -euo pipefail

# build_release.sh — builds docs/bushpilot.zip from the current runtime files in
# the repo root. Run automatically by .github/workflows/release.yml on a `v*` tag
# push (which then attaches the zip to a GitHub Release); can also be run by hand
# to inspect the bundle locally. The zip is a build artifact -- gitignored, never
# committed. See docs/README.md and PHASE_PLAN.md Phase 7.
#
# Usage: ./docs/build_release.sh   (paths below are resolved relative to this
#   script's own location, so it works regardless of the caller's cwd)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
ZIP_PATH="$SCRIPT_DIR/bushpilot.zip"

# End-user runtime files only -- not this project's internal planning/research docs
# (PHASE_PLAN.md, PROJECT_STATUS.md, etc. stay working docs, never shipped).
RUNTIME_FILES=(
    start_mlx_server.sh
    launch_tool.sh
    model_advisor.py
    models.json
    uninstall.sh
)
# model_advisor.py is a thin shim over this package (see bushpilot/__init__.py) --
# both or neither ship, never one without the other.
RUNTIME_DIR="bushpilot"

for f in "${RUNTIME_FILES[@]}"; do
    if [ ! -f "$REPO_ROOT/$f" ]; then
        echo "[ERROR] Expected runtime file missing: $REPO_ROOT/$f" >&2
        exit 1
    fi
done
if [ ! -d "$REPO_ROOT/$RUNTIME_DIR" ]; then
    echo "[ERROR] Expected runtime directory missing: $REPO_ROOT/$RUNTIME_DIR" >&2
    exit 1
fi

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

BUNDLE_DIR="$WORKDIR/bushpilot"
mkdir -p "$BUNDLE_DIR"
for f in "${RUNTIME_FILES[@]}"; do
    cp "$REPO_ROOT/$f" "$BUNDLE_DIR/"
done
cp -R "$REPO_ROOT/$RUNTIME_DIR" "$BUNDLE_DIR/$RUNTIME_DIR"
find "$BUNDLE_DIR/$RUNTIME_DIR" -name "__pycache__" -type d -exec rm -rf {} +
cp "$SCRIPT_DIR/release-README.md" "$BUNDLE_DIR/README.md"
chmod +x "$BUNDLE_DIR/start_mlx_server.sh" "$BUNDLE_DIR/launch_tool.sh" "$BUNDLE_DIR/uninstall.sh"

rm -f "$ZIP_PATH"
(cd "$WORKDIR" && zip -rq "$ZIP_PATH" "bushpilot")

echo "[OK] Built $ZIP_PATH"
echo "     Contents:"
(cd "$WORKDIR" && find "bushpilot" -type f | sort | sed 's/^/       /')
