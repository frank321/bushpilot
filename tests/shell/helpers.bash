# Shared bats helpers for tests/shell/.
#
# These tests exercise start_mlx_server.sh's safety-critical backstop functions
# (check_memory_backstop, check_backend_alive, mlx_port_owner_is_expected)
# WITHOUT sourcing the whole script. Sourcing the real file runs ~1000 lines of
# real top-level setup first (mkdir -p under the real $HOME/.bushpilot, reading
# the real ~/.bushpilot/config.json through `eval "$(python3 ...)"`, probing the
# filesystem for SSL certs) before the first function is even defined -- fine for
# the script's one real entry point, but not something a hermetic unit test
# should trigger just to reach three pure-ish functions. extract_function below
# pulls each one's REAL body (not a reimplementation that could drift from it)
# out of the source file by exact line range, so these tests fail the moment
# the real function's behavior changes, same guarantee a full `source` would
# give, without the side effects or the macOS-only sysctl dependency the rest of
# the script has (get_percent_free/get_pressure_level are stubbed per-test
# instead -- see each .bats file).

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
START_MLX_SERVER_SH="$REPO_ROOT/start_mlx_server.sh"

# extract_function NAME -- prints NAME's definition (`NAME() { ... }`, exactly as
# written in start_mlx_server.sh) to stdout. Relies on every targeted function
# being written as `NAME() {` on its own line through a standalone closing `}`
# line -- true for all of this project's multi-line function definitions (the
# few one-liners like get_percent_free are never extracted this way; tests stub
# them directly instead).
extract_function() {
    local name="$1"
    sed -n "/^${name}() {/,/^}/p" "$START_MLX_SERVER_SH"
}

# source_backstop_functions NAME... -- defines each named real function (via
# extract_function) in the current shell, for a bats test's `setup()` to call
# before stubbing out its external dependencies (sysctl/curl/lsof/ps).
source_backstop_functions() {
    local name
    for name in "$@"; do
        local body
        body="$(extract_function "$name")"
        if [ -z "$body" ]; then
            echo "extract_function: '$name' not found in $START_MLX_SERVER_SH -- did it get renamed?" >&2
            return 1
        fi
        eval "$body"
    done
}

# call NAME [args...] -- like bats' `run`, but executes in the CURRENT shell, not
# a subshell. bats' `run` captures output via `$(...)`, which forks a subshell --
# any global variable a command assigns inside `run` (MEM_ABORT_STREAK,
# TEARDOWN_REASON, ...) is then lost the instant `run` returns. The backstop
# functions here are deliberately mutate-shared-state-and-return-a-status
# functions (that IS the debounce logic under test), so every test below uses
# `call` instead of `run` and inspects `$status` plus whichever globals it cares
# about directly, in the same shell the mutation happened in.
call() {
    "$@" && status=0 || status=$?
}
