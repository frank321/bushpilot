#!/usr/bin/env bats
# Tests for start_mlx_server.sh's backend-liveness backstop: mlx_port_owner_is_expected
# (does the real mlx_vlm.server/mlx-dspark process still own $MLX_PORT) and
# check_backend_alive (the debounced check built on top of it + a /health probe).
# See AGENTS.md #6 and this backstop's own in-source comment (the 2026-10-03 live
# incident it was added for: a dead backend going unnoticed for 2.5+ hours because
# the memory backstop alone never looks at whether the backend itself is still there).

load helpers.bash

setup() {
    source_backstop_functions mlx_port_owner_is_expected check_backend_alive
    MLX_HOST="127.0.0.1"
    MLX_PORT="38472"
    MLX_PROC_PATTERN="mlx_vlm.server"
    MLX_LOG_FILE="/tmp/bats-mlx-server.log"
    BACKEND_DOWN_CONSECUTIVE_POLLS=2
    BACKEND_DOWN_STREAK=0
    TEARDOWN_REASON=""
}

# --- mlx_port_owner_is_expected ------------------------------------------------------

@test "no process owns the port: not the expected owner" {
    lsof() { echo ""; }
    call mlx_port_owner_is_expected
    [ "$status" -eq 1 ]
}

@test "the expected process owns the port" {
    lsof() { echo "4242"; }
    ps() { echo "/usr/bin/python3 -m mlx_vlm.server --host 127.0.0.1"; }
    call mlx_port_owner_is_expected
    [ "$status" -eq 0 ]
}

@test "port stolen: a different process now owns the port" {
    lsof() { echo "9999"; }
    ps() { echo "/usr/sbin/some-other-daemon"; }
    call mlx_port_owner_is_expected
    [ "$status" -eq 1 ]
}

@test "multiple PIDs on the port: expected process is among them" {
    lsof() { printf '1111\n2222\n'; }
    ps() {
        # mlx_port_owner_is_expected calls `ps -p "$pid" -o command=`; $2 is the pid.
        if [ "$2" = "2222" ]; then echo "/usr/bin/python3 -m mlx_vlm.server"; else echo "/usr/bin/unrelated"; fi
    }
    call mlx_port_owner_is_expected
    [ "$status" -eq 0 ]
}

@test "multiple PIDs on the port: none of them match" {
    lsof() { printf '1111\n2222\n'; }
    ps() { echo "/usr/bin/unrelated"; }
    call mlx_port_owner_is_expected
    [ "$status" -eq 1 ]
}

# --- check_backend_alive --------------------------------------------------------------

stub_backend() {
    # $1 = "up" or "down" (curl's /health probe), $2 = "owned" or "stolen" (port owner)
    if [ "$1" = "up" ]; then
        curl() { return 0; }
    else
        curl() { return 7; }  # curl's real exit code for "couldn't connect"
    fi
    if [ "$2" = "owned" ]; then
        lsof() { echo "4242"; }
        ps() { echo "/usr/bin/python3 -m mlx_vlm.server"; }
    else
        lsof() { echo ""; }
        ps() { echo ""; }
    fi
}

@test "healthy backend: returns 0 and resets the streak" {
    stub_backend up owned
    BACKEND_DOWN_STREAK=1
    call check_backend_alive
    [ "$status" -eq 0 ]
    [ "$BACKEND_DOWN_STREAK" -eq 0 ]
}

@test "a single health-check failure is debounced, not an immediate teardown" {
    stub_backend down owned
    call check_backend_alive
    [ "$status" -eq 0 ]
    [ -z "$TEARDOWN_REASON" ]
    [ "$BACKEND_DOWN_STREAK" -eq 1 ]
}

@test "backend down for BACKEND_DOWN_CONSECUTIVE_POLLS consecutive checks tears down" {
    stub_backend down owned
    check_backend_alive
    call check_backend_alive
    [ "$status" -eq 1 ]
    [[ "$TEARDOWN_REASON" == *"127.0.0.1:38472"* ]]
    [[ "$TEARDOWN_REASON" == *"2 consecutive checks"* ]]
    [[ "$TEARDOWN_REASON" == *"/tmp/bats-mlx-server.log"* ]]
}

@test "health check passes but the port was stolen: still treated as down" {
    stub_backend up stolen
    check_backend_alive
    call check_backend_alive
    [ "$status" -eq 1 ]
    [[ "$TEARDOWN_REASON" == *"process died or port was taken over"* ]]
}

@test "a recovery between failures resets the streak (no teardown)" {
    stub_backend down owned
    check_backend_alive
    stub_backend up owned
    check_backend_alive
    [ "$BACKEND_DOWN_STREAK" -eq 0 ]
    stub_backend down owned
    call check_backend_alive
    [ "$status" -eq 0 ]
    [ "$BACKEND_DOWN_STREAK" -eq 1 ]
    [ -z "$TEARDOWN_REASON" ]
}
