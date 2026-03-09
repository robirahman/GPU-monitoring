#!/usr/bin/env python3
"""FFmpeg NVENC video encoding workload.

Generates a synthetic test video and encodes it using NVIDIA's hardware
encoder. Produces a distinct telemetry signature: encoder utilization high,
GPU compute utilization low.

Usage:
    python ffmpeg_nvenc.py --duration 300
"""

import argparse
import os
import signal
import subprocess
import sys
import time


def check_ffmpeg():
    """Check that ffmpeg is available with NVENC support."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-encoders"],
            capture_output=True, text=True, timeout=10,
        )
        if "h264_nvenc" not in result.stdout:
            print("ERROR: ffmpeg does not have h264_nvenc support")
            sys.exit(1)
    except FileNotFoundError:
        print("ERROR: ffmpeg not found. Install with: apt-get install ffmpeg")
        sys.exit(1)


def run_encode_loop(duration, resolution="1920x1080", fps=30):
    """Generate synthetic video and encode in a loop."""
    stopped = False

    def handle_signal(signum, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, handle_signal)

    output_dir = "/tmp/nvenc_workload"
    os.makedirs(output_dir, exist_ok=True)

    wall_start = time.time()
    iteration = 0

    while (time.time() - wall_start) < duration and not stopped:
        output_path = os.path.join(output_dir, f"output_{iteration}.mp4")
        # Generate 30s of synthetic video (testsrc pattern) and encode with NVENC
        # This creates a visually complex pattern that exercises the encoder
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"testsrc=duration=30:size={resolution}:rate={fps}",
            "-c:v", "h264_nvenc",
            "-preset", "fast",
            "-b:v", "10M",
            output_path,
        ]

        iter_start = time.time()
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
        )
        iter_time = time.time() - iter_start

        if proc.returncode != 0:
            print(f"  Encode failed: {proc.stderr[-200:]}")
            break

        # Get output file size
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  Iteration {iteration + 1}: {iter_time:.1f}s, {size_mb:.1f} MB")

        # Clean up to save disk
        os.remove(output_path)
        iteration += 1

    wall_time = time.time() - wall_start
    print(f"NVENC encoding complete. {iteration} encodes in {wall_time:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="FFmpeg NVENC encoding workload")
    parser.add_argument("--duration", type=int, default=300, help="Duration in seconds")
    parser.add_argument("--resolution", type=str, default="1920x1080", help="Video resolution")
    args = parser.parse_args()

    print(f"FFmpeg NVENC encoding workload")
    print(f"  Resolution: {args.resolution}, Duration: {args.duration}s")

    check_ffmpeg()
    run_encode_loop(args.duration, resolution=args.resolution)


if __name__ == "__main__":
    main()
