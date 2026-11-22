"""Run a batch of ChampSim traces with bounded concurrency and resumable logs."""

import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def positive_int(text):
    try:
        value = int(text)
    except ValueError:
        value = 0
    if value <= 0:
        raise argparse.ArgumentTypeError("JOBS must be a positive integer")
    return value


def signal_workers(workers, signum):
    for process in workers:
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass


def stop_workers(workers):
    signal_workers(workers, signal.SIGTERM)
    deadline = time.monotonic() + 2
    for process in workers:
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            signal_workers(workers, signal.SIGKILL)
            break
    for process in workers:
        process.wait()


def run_traces(args, traces, lock):
    interrupted = 0

    def interrupt(signum, _frame):
        nonlocal interrupted
        interrupted = interrupted or signum

    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    command = [str(args.binary.absolute()), "--warmup-instructions", args.warmup,
               "--simulation-instructions", args.simulation, "--"]
    remaining = iter(traces)
    active = {}
    failure = 0

    try:
        while not interrupted:
            for process, (trace, partial, final) in tuple(active.items()):
                status = process.poll()
                if status is None:
                    continue
                del active[process]
                if status != 0:
                    print(f"Trace failed (exit {status}): {trace}. Partial log: {partial}", file=sys.stderr, flush=True)
                    failure = failure or (status if status > 0 else 128 - status)
                else:
                    try:
                        partial.replace(final)
                    except OSError as error:
                        print(f"Cannot publish {final}: {error}", file=sys.stderr, flush=True)
                        failure = failure or 1

            while not interrupted and not failure and len(active) < args.jobs:
                trace = next(remaining, None)
                if trace is None:
                    break
                final = args.log_dir / (trace.name + ".log")
                if final.is_file():
                    print(f"Skip {trace} (completed)", flush=True)
                    continue
                partial = final.with_name(final.name + ".partial")
                print(f"Run {trace} -> {final}", flush=True)
                try:
                    with partial.open("wb") as log:
                        # Workers retain the lock if the coordinator is killed.
                        process = subprocess.Popen([*command, str(trace)], stdout=log, stderr=subprocess.STDOUT,
                                                   pass_fds=(lock.fileno(),), start_new_session=True)
                except OSError as error:
                    print(f"Cannot start {trace}: {error}", file=sys.stderr, flush=True)
                    failure = 1
                else:
                    active[process] = (trace, partial, final)

            if not active and not interrupted:
                return failure
            if not interrupted:
                time.sleep(0.05)
    finally:
        if active:
            stop_workers(active)
    return 128 + interrupted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("warmup")
    parser.add_argument("simulation")
    parser.add_argument("jobs", type=positive_int)
    args = parser.parse_args()

    if not args.binary.is_file() or not os.access(args.binary, os.X_OK):
        parser.exit(2, f"Cannot execute {args.binary}. Build the selected program first.\n")
    if not args.trace_dir.is_dir() or not os.access(args.trace_dir, os.R_OK | os.X_OK):
        parser.exit(2, f"Cannot read trace directory: {args.trace_dir}\n")

    try:
        args.log_dir.mkdir(parents=True, exist_ok=True)
        with (args.log_dir / ".run-all.lock").open("ab") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                parser.exit(2, f"Another run-all is using {args.log_dir}.\n")

            traces = sorted((path for path in args.trace_dir.iterdir() if not path.name.startswith(".") and path.is_file()),
                            key=lambda path: os.fsencode(path.name))
            if not traces:
                parser.exit(2, f"No trace files found in {args.trace_dir}.\n")
            return run_traces(args, traces, lock)
    except OSError as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    sys.exit(main())
