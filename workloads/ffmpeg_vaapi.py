#!/usr/bin/env python3
"""FFmpeg VA-API video encoding workload (AMD analogue of ffmpeg_nvenc.py).

Generates a synthetic test video and encodes it using the AMD VCN engine
through VA-API. Produces a distinct telemetry signature: the multimedia
engine activity (mm_activity in amdsmi) goes high while the GFX engine
stays mostly idle.

Requirements:
    apt install ffmpeg vainfo mesa-va-drivers
    # Verify VA-API can see the AMD GPU:
    vainfo --display drm --device /dev/dri/renderD128

Usage:
    python ffmpeg_vaapi.py --duration 300
"""

import argparse
import glob
import os
import signal
import subprocess
import sys
import tempfile
import time


def _find_render_node() -> str:
    """Pick the renderD* node that VA-API should use.

    On a single-GPU system this is /dev/dri/renderD128. If multiple are
    present we let VAAPI_DEVICE override; otherwise pick the lowest.
    """
    override = os.environ.get("VAAPI_DEVICE")
    if override:
        return override
    nodes = sorted(glob.glob("/dev/dri/renderD*"))
    if not nodes:
        print("ERROR: no /dev/dri/renderD* device found. "
              "Is the kernel amdgpu driver loaded and your user in the 'render' group?",
              file=sys.stderr)
        sys.exit(1)
    return nodes[0]


def check_ffmpeg(render_node: str):
    """Verify ffmpeg has VA-API support and the device is usable."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10,
        )
        if "h264_vaapi" not in result.stdout:
            print("ERROR: ffmpeg does not have h264_vaapi support. "
                  "Install a VA-API-enabled build (`apt install ffmpeg`).",
                  file=sys.stderr)
            sys.exit(1)
    except FileNotFoundError:
        print("ERROR: ffmpeg not found. Install with: apt install ffmpeg",
              file=sys.stderr)
        sys.exit(1)

    # Best-effort vainfo check; not fatal because vainfo isn't always installed.
    try:
        vi = subprocess.run(
            ["vainfo", "--display", "drm", "--device", render_node],
            capture_output=True, text=True, timeout=10,
        )
        if vi.returncode != 0:
            print(f"WARNING: vainfo on {render_node} returned non-zero. "
                  f"Encoding may still work, but check 'vainfo' output if not.")
    except FileNotFoundError:
        pass  # vainfo is optional


def run_encode_loop(duration: int, render_node: str, resolution: str = "1920x1080", fps: int = 30):
    stopped = False

    def handle_signal(signum, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, handle_signal)

    output_dir = os.path.join(tempfile.gettempdir(), "vaapi_workload")
    os.makedirs(output_dir, exist_ok=True)

    wall_start = time.time()
    iteration = 0

    while (time.time() - wall_start) < duration and not stopped:
        output_path = os.path.join(output_dir, f"output_{iteration}.mp4")
        # Generate 30s of testsrc, upload to GPU as NV12 in VAAPI memory,
        # then encode with h264_vaapi. Mirrors the NVENC workload's clip
        # length and bitrate so output sizes are roughly comparable.
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-vaapi_device", render_node,
            "-f", "lavfi",
            "-i", f"testsrc=duration=30:size={resolution}:rate={fps}",
            "-vf", "format=nv12,hwupload",
            "-c:v", "h264_vaapi",
            "-b:v", "10M",
            output_path,
        ]

        iter_start = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        iter_time = time.time() - iter_start

        if proc.returncode != 0:
            print(f"  Encode failed: {proc.stderr[-200:]}")
            break

        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  Iteration {iteration + 1}: {iter_time:.1f}s, {size_mb:.1f} MB")
        os.remove(output_path)
        iteration += 1

    wall_time = time.time() - wall_start
    print(f"VA-API encoding complete. {iteration} encodes in {wall_time:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="FFmpeg VA-API encoding workload (AMD VCN)")
    parser.add_argument("--duration", type=int, default=300, help="Duration in seconds")
    parser.add_argument("--resolution", type=str, default="1920x1080")
    parser.add_argument("--render-node", type=str, default=None,
                        help="VA-API render node (default: auto-pick from /dev/dri/renderD*)")
    args = parser.parse_args()

    render_node = args.render_node or _find_render_node()

    print("FFmpeg VA-API encoding workload")
    print(f"  Render node: {render_node}")
    print(f"  Resolution:  {args.resolution}, Duration: {args.duration}s")

    check_ffmpeg(render_node)
    run_encode_loop(args.duration, render_node, resolution=args.resolution)


if __name__ == "__main__":
    main()
