#!/usr/bin/env python3
"""Idle GPU baseline workload.

Does nothing for a configurable duration. Used to establish baseline telemetry
(idle power draw, idle temperature, zero utilization).
"""

import argparse
import signal
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="Idle GPU baseline workload")
    parser.add_argument(
        "--duration",
        type=int,
        default=120,
        help="Duration in seconds (default: 120)",
    )
    args = parser.parse_args()

    # Handle SIGTERM gracefully
    stopped = False

    def handle_signal(signum, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, handle_signal)

    print(f"Idle workload: sleeping for {args.duration}s")
    start = time.time()
    elapsed = 0
    while elapsed < args.duration and not stopped:
        time.sleep(1)
        elapsed = time.time() - start

    if stopped:
        print(f"Idle workload: terminated after {elapsed:.0f}s")
    else:
        print(f"Idle workload complete ({args.duration}s)")


if __name__ == "__main__":
    main()
