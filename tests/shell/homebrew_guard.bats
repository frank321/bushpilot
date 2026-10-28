#!/usr/bin/env bats
# Tests for the Homebrew-formula guard in ensure_symlinked_on_path -- both in
# start_mlx_server.sh and (via a mirrored extraction) launch_tool.sh.
#
# When installed by the frank321/homebrew-bushpilot formula, the runtime files
# live read-only under the Cellar and the formula already owns the bin wrappers.
# The scripts must NOT self-symlink in that case (it would clobber the formula's
# entries or try to write into a Cellar path Homebrew owns). Detected two ways:
# the Cellar path pattern (always reliable) and HOMEBREW_PREFIX-prefix (only when
# the variable happens to be exported into the wrapper's environment).

load helpers.bash

# is_command_available / log_success / log_info are the only other functions
# ensure_symlinked_on_path calls; stub them so the test is hermetic and so we can
# observe whether the symlink branch was reached.
setup() {
    TEST_TMPDIR="$(mktemp -d)"
    HOME="$TEST_TMPDIR/home"
    mkdir -p "$HOME/.local/bin"   # a writable candidate dir, so PROCEED would symlink

    source_backstop_functions ensure_symlinked_on_path

    # Record calls instead of touching the real filesystem/terminal.
    is_command_available() { return 1; }        # target not yet on PATH
    log_success() { echo "LOG_SUCCESS: $*" >&2; }
    log_info()    { echo "LOG_INFO: $*" >&2; }
}

teardown() {
    rm -rf "$TEST_TMPDIR"
}

# --- start_mlx_server.sh ---------------------------------------------------

@test "guard: Cellar path skips self-symlinking (start_mlx_server.sh)" {
    SCRIPT_PATH="/opt/homebrew/Cellar/bushpilot/1.0.0/libexec/start_mlx_server.sh"
    run ensure_symlinked_on_path
    [ "$status" -eq 0 ]
    # No symlink created anywhere under the fake HOME.
    [ -z "$(find "$HOME" -type l -print -quit)" ]
    [[ "$output" != *"LOG_SUCCESS"* ]]
}

@test "guard: HOMEBREW_PREFIX-prefixed path skips self-symlinking" {
    SCRIPT_PATH="/opt/homebrew/Cellar/bushpilot/1.0.0/libexec/launch_tool.sh"
    HOMEBREW_PREFIX="/opt/homebrew"
    run ensure_symlinked_on_path
    [ "$status" -eq 0 ]
    [ -z "$(find "$HOME" -type l -print -quit)" ]
}

@test "guard: Cellar path skips even when HOMEBREW_PREFIX is unset" {
    SCRIPT_PATH="/opt/homebrew/Cellar/bushpilot/1.0.0/libexec/start_mlx_server.sh"
    unset HOMEBREW_PREFIX
    run ensure_symlinked_on_path
    [ "$status" -eq 0 ]
    [ -z "$(find "$HOME" -type l -print -quit)" ]
}

@test "guard: git-clone path still self-symlinks (guard does not over-fire)" {
    SCRIPT_PATH="$HOME/src/bushpilot/start_mlx_server.sh"
    mkdir -p "$HOME/src/bushpilot"
    touch "$SCRIPT_PATH"
    HOMEBREW_PREFIX="/opt/homebrew"
    run ensure_symlinked_on_path
    [ "$status" -eq 0 ]
    [ -L "$HOME/.local/bin/start_mlx_server.sh" ]
    [[ "$output" == *"LOG_SUCCESS"* ]]
}

@test "guard: zip-in-home path still self-symlinks (guard does not over-fire)" {
    SCRIPT_PATH="$HOME/Downloads/bushpilot/launch_tool.sh"
    mkdir -p "$HOME/Downloads/bushpilot"
    touch "$SCRIPT_PATH"
    unset HOMEBREW_PREFIX
    run ensure_symlinked_on_path
    [ "$status" -eq 0 ]
    [ -L "$HOME/.local/bin/launch_tool.sh" ]
}

# --- launch_tool.sh (mirrored guard) ---------------------------------------

@test "guard: launch_tool.sh has the same Cellar skip" {
    run grep -q 'Cellar/bushpilot' "$REPO_ROOT/launch_tool.sh"
    [ "$status" -eq 0 ]
}

@test "guard: launch_tool.sh skips self-symlinking from a Cellar path" {
    # Extract launch_tool.sh's own copy of the function and exercise it, so a
    # divergence between the two scripts is caught here too.
    local tmpf
    tmpf="$(mktemp)"
    awk '$0 == "ensure_symlinked_on_path() {" { p=1 } p { print } p && $0 == "}" { exit }' \
        "$REPO_ROOT/launch_tool.sh" > "$tmpf"
    [ -s "$tmpf" ]
    # shellcheck disable=SC1090
    source "$tmpf"
    rm -f "$tmpf"

    SCRIPT_PATH="/opt/homebrew/Cellar/bushpilot/1.0.0/libexec/launch_tool.sh"
    unset HOMEBREW_PREFIX
    run ensure_symlinked_on_path
    [ "$status" -eq 0 ]
    [ -z "$(find "$HOME" -type l -print -quit)" ]
}
