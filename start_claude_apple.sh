#!/usr/bin/env bash
set -e

# Launches Claude Code through a local passthrough proxy that exposes Apple's
# on-device Foundation Models framework as the "apple-foundation-model" model.
# The bridge forwards only the latest user message (no history, no system
# prompt, no tools) because Apple's on-device model has a hard 4096-token
# ceiling and Claude Code's baseline is ~10k tokens. This is for single-turn
# capability evaluation only.
#
# Real Claude models (sonnet/opus/haiku) pass through unmodified to Anthropic.
# Claude Code's normal subscription auth is used (no separate API key needed).
#
# Usage:
#   ~/start_claude_apple.sh              — interactive session
#   ~/start_claude_apple.sh -p "prompt"  — single prompt
#   ~/start_claude_apple.sh --model sonnet -p "..."  — real Claude model

# ============================================================================
# VALIDATIONS
# ============================================================================

validate_macos_version() {
    local major=$(sw_vers -productVersion | cut -d. -f1)
    if [ "$major" -lt 15 ]; then
        echo "❌ macOS 15.0+ required (Foundation Models framework)"
        echo "   Current: $(sw_vers -productVersion)"
        echo "   Fix: Update macOS in System Settings → General → Software Update"
        exit 1
    fi
}

validate_apple_silicon() {
    local arch=$(uname -m)
    if [ "$arch" != "arm64" ]; then
        echo "❌ Apple Silicon (M-series chip) required"
        echo "   Current: $arch"
        echo "   Foundation Models only run on ARM64 Macs"
        exit 1
    fi
}

validate_apple_intelligence() {
    if ! swift -e 'import FoundationModels; let m = SystemLanguageModel.default; print(m.availability)' 2>&1 | grep -q "available"; then
        echo "❌ Apple Intelligence not enabled or model not ready"
        echo "   Fix:"
        echo "   1. Go to System Settings → Apple Intelligence & Siri"
        echo "   2. Enable Apple Intelligence"
        echo "   3. Wait for on-device model to download (can take a few minutes)"
        echo "   4. Retry this script"
        exit 1
    fi
}

validate_swift() {
    if ! command -v swift &>/dev/null; then
        echo "❌ Swift not found"
        echo "   Fix: Install Xcode Command Line Tools:"
        echo "        xcode-select --install"
        exit 1
    fi
}

validate_python3() {
    if ! command -v python3 &>/dev/null; then
        echo "❌ Python 3 not found"
        echo "   Fix: Install from https://www.python.org or via Homebrew:"
        echo "        brew install python3"
        exit 1
    fi
}

validate_ports() {
    local bridge_port=8790
    local proxy_port=8791

    if lsof -nP -iTCP:"$bridge_port" -sTCP:LISTEN 2>/dev/null | grep -q .; then
        echo "❌ Port $bridge_port already in use"
        echo "   Fix: Either wait for it to free up, or run:"
        echo "        lsof -nP -iTCP:$bridge_port -sTCP:LISTEN | awk 'NR>1 {print \$2}' | xargs kill"
        exit 1
    fi
    if lsof -nP -iTCP:"$proxy_port" -sTCP:LISTEN 2>/dev/null | grep -q .; then
        echo "❌ Port $proxy_port already in use"
        echo "   Fix: Either wait for it to free up, or run:"
        echo "        lsof -nP -iTCP:$proxy_port -sTCP:LISTEN | awk 'NR>1 {print \$2}' | xargs kill"
        exit 1
    fi
}

validate_bridge_files() {
    local bridge_dir="$HOME/.apple-fm-bridge"

    if [ ! -f "$bridge_dir/bridge.swift" ]; then
        echo "❌ Bridge script not found: $bridge_dir/bridge.swift"
        echo "   Fix: Clone or download the apple-fm-bridge from the repository"
        exit 1
    fi
    if [ ! -f "$bridge_dir/proxy.py" ]; then
        echo "❌ Proxy script not found: $bridge_dir/proxy.py"
        echo "   Fix: Clone or download the apple-fm-bridge from the repository"
        exit 1
    fi
}

validate_claude_code() {
    if ! command -v claude &>/dev/null; then
        echo "❌ Claude Code not found"
        echo "   Fix: Install from https://claude.com/download or run:"
        echo "        npm install -g @anthropic-ai/claude"
        exit 1
    fi
}

# Run all validations
echo "🔍 Validating environment..."
validate_macos_version
validate_apple_silicon
validate_swift
validate_python3
validate_claude_code
validate_bridge_files
validate_ports
validate_apple_intelligence
echo "✅ All checks passed"
echo ""

BRIDGE_DIR="$HOME/.apple-fm-bridge"
BRIDGE_PORT=8790
PROXY_PORT=8791

echo "🍎 Starting Apple Foundation Models bridge (fresh each launch -- these are"
echo "   lightweight dev processes, not worth keeping stale copies of around)..."

# 1. ALWAYS RESTART THE SWIFT BRIDGE. These two helper scripts get edited
# often during development; silently reusing whatever's already bound to the
# port previously caused a real, confusing bug (stale pre-fix code kept
# running while the on-disk script had already moved on). Killing by exact
# script path, not just by port, so we don't clobber an unrelated process
# that happens to be using these ports.
pkill -f "$BRIDGE_DIR/bridge.swift" > /dev/null 2>&1 || true
pkill -f "$BRIDGE_DIR/proxy.py" > /dev/null 2>&1 || true
sleep 1

echo "🌐 Starting apple-fm-bridge on port $BRIDGE_PORT..."
swift "$BRIDGE_DIR/bridge.swift" > /tmp/apple-fm-bridge.log 2>&1 &
for i in $(seq 1 30); do
    if lsof -nP -iTCP:"$BRIDGE_PORT" -sTCP:LISTEN > /dev/null 2>&1; then
        break
    fi
    sleep 1
done
if ! lsof -nP -iTCP:"$BRIDGE_PORT" -sTCP:LISTEN > /dev/null 2>&1; then
    echo "❌ Error: apple-fm-bridge failed to start. Check /tmp/apple-fm-bridge.log"
    exit 1
fi
echo "✅ apple-fm-bridge ready on port $BRIDGE_PORT"

# 2. ALWAYS RESTART THE PASSTHROUGH PROXY (same reasoning as above).
echo "🌐 Starting apple-fm-proxy on port $PROXY_PORT..."
python3 "$BRIDGE_DIR/proxy.py" "$PROXY_PORT" > /tmp/apple-fm-proxy.log 2>&1 &
for i in $(seq 1 30); do
    if curl -s "http://127.0.0.1:$PROXY_PORT/health" > /dev/null 2>&1; then
        break
    fi
    sleep 1
done
if ! curl -s "http://127.0.0.1:$PROXY_PORT/health" > /dev/null 2>&1; then
    echo "❌ Error: apple-fm-proxy failed to start. Check /tmp/apple-fm-proxy.log"
    exit 1
fi
echo "✅ apple-fm-proxy ready on port $PROXY_PORT"

# 3. POINT CLAUDE CODE AT THE PROXY. Deliberately NOT setting
# ANTHROPIC_API_KEY/ANTHROPIC_AUTH_TOKEN here -- see header comment. Claude
# Code keeps using its normal login for real models.
export ANTHROPIC_BASE_URL="http://127.0.0.1:$PROXY_PORT"

# 4. LAUNCH INTO THE ON-DEVICE MODEL. Apple's on-device model has a hard
# 4096-token ceiling. Claude Code's baseline system prompt + tools are ~10k
# tokens, so the bridge strips everything and sends only the latest user
# message. This is single-turn capability: one prompt, one answer.
DEFAULT_MODEL="apple-foundation-model"

HAS_MODEL_FLAG=0
for arg in "$@"; do
    case "$arg" in
        --model|--model=*) HAS_MODEL_FLAG=1 ;;
    esac
done

echo "🎯 Opening Claude Code (real models -> Anthropic directly, unmodified;"
echo "   'apple-foundation-model' -> on-device Foundation Models, single-turn)"
echo "   (4096-token ceiling, latest message only)"
echo "--------------------------------------------------------"

if [ "$HAS_MODEL_FLAG" -eq 0 ]; then
    claude --model "$DEFAULT_MODEL" "$@"
else
    claude "$@"
fi
