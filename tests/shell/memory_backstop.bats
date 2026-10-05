#!/usr/bin/env bats
# Tests for start_mlx_server.sh's check_memory_backstop -- the consecutive-polls
# debounce is the invariant that matters most here (AGENTS.md #6): a single
# transient low-memory/pressure reading must never tear down a healthy session.

load helpers.bash

setup() {
    source_backstop_functions check_memory_backstop
    MEM_ABORT_BELOW_PCT=10
    MEM_ABORT_CONSECUTIVE_POLLS=3
    MEM_CRITICAL_CONSECUTIVE_POLLS=3
    MEM_ABORT_STREAK=0
    MEM_CRITICAL_STREAK=0
    TEARDOWN_REASON=""
    # Stubbed per test below -- real get_percent_free/get_pressure_level shell
    # out to macOS-only sysctl, not available (or deterministic) in CI.
}

stub_memory() {
    # $1 = percent free (or "" for "sysctl unavailable"), $2 = pressure level (or "")
    #
    # Bakes the literal value into the function body text via eval, rather than
    # defining `get_percent_free() { echo "$pct"; }` directly: that would reference
    # $pct by NAME, and $pct is local to this function -- gone by the time
    # get_percent_free is actually CALLED later (bash resolves a function body's
    # variables at call time, not at definition time), so every stubbed reading
    # would silently come back empty regardless of what was asked for here.
    local pct="$1" pressure="$2"
    eval "get_percent_free() { echo '${pct}'; }"
    eval "get_pressure_level() { echo '${pressure}'; }"
}

@test "healthy system: returns 0 and never sets TEARDOWN_REASON" {
    stub_memory 80 0
    call check_memory_backstop
    [ "$status" -eq 0 ]
    [ -z "$TEARDOWN_REASON" ]
}

@test "a single low-memory blip is debounced, not an immediate teardown" {
    stub_memory 5 0
    call check_memory_backstop
    [ "$status" -eq 0 ]
    [ -z "$TEARDOWN_REASON" ]
    [ "$MEM_ABORT_STREAK" -eq 1 ]
}

@test "low memory for MEM_ABORT_CONSECUTIVE_POLLS consecutive checks tears down" {
    stub_memory 5 0
    check_memory_backstop
    check_memory_backstop
    call check_memory_backstop
    [ "$status" -eq 1 ]
    [[ "$TEARDOWN_REASON" == *"below 10%"* ]]
    [[ "$TEARDOWN_REASON" == *"3 consecutive checks"* ]]
}

@test "a recovery between low readings resets the streak (no teardown)" {
    stub_memory 5 0
    check_memory_backstop
    check_memory_backstop
    stub_memory 80 0
    check_memory_backstop
    [ "$MEM_ABORT_STREAK" -eq 0 ]
    stub_memory 5 0
    check_memory_backstop
    [ "$MEM_ABORT_STREAK" -eq 1 ]
    [ -z "$TEARDOWN_REASON" ]
}

@test "a single CRITICAL pressure blip is debounced" {
    stub_memory 80 4
    call check_memory_backstop
    [ "$status" -eq 0 ]
    [ "$MEM_CRITICAL_STREAK" -eq 1 ]
}

@test "CRITICAL pressure for MEM_CRITICAL_CONSECUTIVE_POLLS consecutive checks tears down" {
    stub_memory 80 4
    check_memory_backstop
    check_memory_backstop
    call check_memory_backstop
    [ "$status" -eq 1 ]
    [[ "$TEARDOWN_REASON" == *"CRITICAL"* ]]
    [[ "$TEARDOWN_REASON" == *"3 consecutive checks"* ]]
}

@test "CRITICAL pressure is checked before the percent-free threshold" {
    # Both conditions true at once -- the function must still return 1 (teardown),
    # and report the pressure reason since that branch runs first.
    stub_memory 5 4
    check_memory_backstop
    check_memory_backstop
    call check_memory_backstop
    [ "$status" -eq 1 ]
    [[ "$TEARDOWN_REASON" == *"CRITICAL"* ]]
}

@test "sysctl unavailable (empty readings) is treated as healthy, not as a false teardown" {
    stub_memory "" ""
    check_memory_backstop
    check_memory_backstop
    call check_memory_backstop
    [ "$status" -eq 0 ]
    [ -z "$TEARDOWN_REASON" ]
    [ "$MEM_ABORT_STREAK" -eq 0 ]
    [ "$MEM_CRITICAL_STREAK" -eq 0 ]
}

@test "exactly at the threshold (not below it) does not count as low" {
    stub_memory 10 0
    call check_memory_backstop
    [ "$status" -eq 0 ]
    [ "$MEM_ABORT_STREAK" -eq 0 ]
}
