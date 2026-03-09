#!/usr/bin/env python3
"""Blender Cycles GPU rendering workload.

Downloads the BMW benchmark scene and renders it with CUDA. Produces a
rendering-specific telemetry signature: sustained high GPU utilization,
no tensor cores, different memory pattern than ML workloads.

Usage:
    python blender_render.py --samples 128
    python blender_render.py --samples 128 --loops 3
"""

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.request


BMW_BLEND_URL = "https://download.blender.org/demo/test/BMW27_2.blend.zip"
BMW_BLEND_DIR = "/tmp/blender_benchmark"
BMW_BLEND_FILE = os.path.join(BMW_BLEND_DIR, "bmw27", "bmw27_gpu.blend")


def check_blender():
    """Check that blender is available."""
    try:
        result = subprocess.run(
            ["blender", "--version"],
            capture_output=True, text=True, timeout=10,
        )
        print(f"  Blender: {result.stdout.strip().splitlines()[0]}")
    except FileNotFoundError:
        print("ERROR: blender not found. Install with: apt-get install blender")
        sys.exit(1)


def download_scene():
    """Download the BMW benchmark scene if not present."""
    if os.path.exists(BMW_BLEND_FILE):
        print(f"  Scene already downloaded: {BMW_BLEND_FILE}")
        return BMW_BLEND_FILE

    os.makedirs(BMW_BLEND_DIR, exist_ok=True)
    zip_path = os.path.join(BMW_BLEND_DIR, "bmw.zip")

    if not os.path.exists(zip_path):
        print(f"  Downloading BMW benchmark scene...")
        urllib.request.urlretrieve(BMW_BLEND_URL, zip_path)
        print(f"  Downloaded to {zip_path}")

    print(f"  Extracting...")
    subprocess.run(["unzip", "-o", zip_path, "-d", BMW_BLEND_DIR],
                    capture_output=True, check=True)

    # The zip may have a different internal structure
    # Find the .blend file
    for root, dirs, files in os.walk(BMW_BLEND_DIR):
        for f in files:
            if f.endswith(".blend"):
                blend_path = os.path.join(root, f)
                print(f"  Found scene: {blend_path}")
                return blend_path

    print("ERROR: Could not find .blend file in archive")
    sys.exit(1)


def render(blend_file, samples, output_prefix="/tmp/blender_render_"):
    """Render one frame with Blender Cycles CUDA."""
    # Blender Python script to configure Cycles + CUDA
    setup_script = (
        "import bpy;"
        "bpy.context.scene.render.engine = 'CYCLES';"
        f"bpy.context.scene.cycles.samples = {samples};"
        "prefs = bpy.context.preferences.addons['cycles'].preferences;"
        "prefs.compute_device_type = 'CUDA';"
        "prefs.get_devices();"
        "for d in prefs.devices: d.use = d.type == 'CUDA';"
        "bpy.context.scene.cycles.device = 'GPU';"
    )

    cmd = [
        "blender", "-b", blend_file,
        "--python-expr", setup_script,
        "-o", output_prefix,
        "-f", "1",
    ]

    return subprocess.run(cmd, capture_output=True, text=True, timeout=600)


def main():
    parser = argparse.ArgumentParser(description="Blender Cycles rendering workload")
    parser.add_argument("--samples", type=int, default=128, help="Render samples")
    parser.add_argument("--loops", type=int, default=3, help="Number of render passes")
    args = parser.parse_args()

    print(f"Blender Cycles rendering workload")
    print(f"  Samples: {args.samples}, Loops: {args.loops}")

    check_blender()
    blend_file = download_scene()

    stopped = False

    def handle_signal(signum, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, handle_signal)

    wall_start = time.time()
    for i in range(args.loops):
        if stopped:
            break
        iter_start = time.time()
        print(f"  Render {i + 1}/{args.loops}...")
        result = render(blend_file, args.samples)
        iter_time = time.time() - iter_start
        if result.returncode != 0:
            print(f"  Render failed: {result.stderr[-300:]}")
            break
        print(f"  Render {i + 1} complete: {iter_time:.1f}s")

    wall_time = time.time() - wall_start
    print(f"Blender rendering complete. Total wall time: {wall_time:.1f}s")


if __name__ == "__main__":
    main()
