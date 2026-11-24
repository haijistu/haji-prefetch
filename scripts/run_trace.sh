#!/usr/bin/env bash
set -euo pipefail

log_dir=$1
binary=$2
trace=$3
warmup=$4
simulation=$5
trace_name=${trace##*/}

mkdir -p -- "$log_dir"
partial=$(mktemp -- "$log_dir/$trace_name.XXXXXXXX.log.partial")

if "$binary" --warmup-instructions "$warmup" --simulation-instructions "$simulation" -- "$trace" 2>&1 | tee -p -- "$partial"; then
    # Only run-all publishes the exact <trace-name>.log completion marker.
    log=${partial%.partial}
    ln -T -- "$partial" "$log"
    rm -- "$partial"
    printf 'Log saved to %s\n' "$log"
else
    status=$?
    printf 'Trace failed (exit %s). Partial log: %s\n' "$status" "$partial" >&2
    exit "$status"
fi
