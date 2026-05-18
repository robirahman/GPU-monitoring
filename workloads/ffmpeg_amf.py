"""FFmpeg AMF video encoding workload (Windows-AMD analogue of ffmpeg_vaapi /
ffmpeg_nvenc).

Encodes a synthetic test video using the AMD VCN (Video Codec Engine /
Video Codec 0) through the AMF encoder family in ffmpeg. Produces a
distinctive telemetry signature: ``gpu_engine_video_pct`` (or
``encoder_util_pct`` in cross-vendor schema) goes high while
``gpu_engine_compute_pct`` and ``gpu_engine_3d_pct`` stay low.

Important — adapter selection on dual-GPU systems:
    On a box with both an iGPU and a dGPU, AMF auto-selects an adapter that
    is *not* always the dGPU. ffmpeg's ``-init_hw_device d3d11va=d3d11:N``
    passes adapter index N to D3D11; whether N is the dGPU depends on
    DXGI's enumeration order, which depends in turn on Windows 'Graphics
    settings' high-performance preference. Verify after the first run by
    checking which LUID lit up in the PDH telemetry. If the wrong adapter
    is picked, try the other index, or pin the dGPU as the system-wide
    high-performance default in Settings -> Display -> Graphics.

Requirements:
    winget install Gyan.FFmpeg
    # ffmpeg -hide_banner -encoders | findstr amf
    #   should list h264_amf, hevc_amf, av1_amf

Usage:
    python ffmpeg_amf.py --duration 300
    python ffmpeg_amf.py --duration 300 --codec hevc_amf --resolution 3840x2160
    python ffmpeg_amf.py --duration 300 --adapter 1
"""

import argparse
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time


_VALID_CODECS = ("h264_amf", "hevc_amf", "av1_amf")


_FFMPEG_FALLBACK_PATHS = [
    # Gyan.FFmpeg (winget) drops the binary at a versioned path under WinGet.
    # Probe both Machine-scope and User-scope install locations.
    r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
    r"C:\ffmpeg\bin\ffmpeg.exe",
]


def _find_ffmpeg() -> str | None:
    found = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    if found:
        return found
    # Probe WinGet's versioned package layout
    import glob as _glob
    for base in (
        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages"),
        os.path.expandvars(r"%ProgramFiles%\WinGet\Packages"),
    ):
        if not os.path.isdir(base):
            continue
        matches = _glob.glob(os.path.join(base, "Gyan.FFmpeg*", "ffmpeg-*", "bin", "ffmpeg.exe"))
        if matches:
            return sorted(matches)[-1]   # latest version
    for p in _FFMPEG_FALLBACK_PATHS:
        if os.path.isfile(p):
            return p
    return None


def check_ffmpeg(codec: str) -> str:
    ffmpeg = _find_ffmpeg()
    if ffmpeg is None:
        print("ERROR: ffmpeg not found on PATH or in WinGet package dir. "
              "Install with: winget install Gyan.FFmpeg, then open a new shell.",
              file=sys.stderr)
        sys.exit(1)
    res = subprocess.run([ffmpeg, "-hide_banner", "-encoders"],
                         capture_output=True, text=True, timeout=10)
    if codec not in res.stdout:
        print(f"ERROR: ffmpeg lacks {codec}. AMF encoders should ship with the "
              "Gyan/BtbN Windows builds. Available encoders contain "
              f"these AMF entries:\n  " +
              "\n  ".join(line for line in res.stdout.splitlines() if "amf" in line),
              file=sys.stderr)
        sys.exit(1)
    return ffmpeg


def run_encode_loop(
    ffmpeg: str,
    duration: int,
    codec: str,
    resolution: str,
    fps: int,
    bitrate: str,
    adapter: int | None,
):
    stopped = False

    def handle_signal(signum, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, handle_signal)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, handle_signal)

    out_dir = os.path.join(tempfile.gettempdir(), "amf_workload")
    os.makedirs(out_dir, exist_ok=True)

    wall_start = time.time()
    iteration = 0
    while (time.time() - wall_start) < duration and not stopped:
        out_path = os.path.join(out_dir, f"output_{iteration}.mp4")
        # Match ffmpeg_vaapi clip length (30s) for cross-vendor comparability.
        cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
        if adapter is not None:
            cmd += ["-init_hw_device", f"d3d11va=d3d11:{adapter}",
                    "-filter_hw_device", "d3d11"]
        cmd += [
            "-f", "lavfi",
            "-i", f"testsrc=duration=30:size={resolution}:rate={fps}",
            "-c:v", codec,
            "-b:v", bitrate,
            out_path,
        ]
        iter_start = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        iter_time = time.time() - iter_start

        if proc.returncode != 0:
            print(f"  Encode failed: {proc.stderr.strip()[-300:]}")
            break

        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"  Iter {iteration + 1}: {iter_time:.1f}s, {size_mb:.1f} MB")
        try:
            os.remove(out_path)
        except OSError:
            pass
        iteration += 1

    print(f"AMF encode complete. {iteration} clips in {time.time() - wall_start:.1f}s")


def main():
    ap = argparse.ArgumentParser(description="FFmpeg AMF encoding workload (AMD VCN, Windows)")
    ap.add_argument("--duration", type=int, default=300)
    ap.add_argument("--codec", choices=_VALID_CODECS, default="h264_amf")
    ap.add_argument("--resolution", type=str, default="1920x1080")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--bitrate", type=str, default="10M")
    ap.add_argument("--adapter", type=int, default=None,
                    help="DXGI adapter index for D3D11 hw device "
                         "(default: AMF auto-pick — may select iGPU on dual-GPU boxes)")
    args = ap.parse_args()

    ffmpeg = check_ffmpeg(args.codec)
    print("FFmpeg AMF encoding workload (Windows / AMD VCN)")
    print(f"  Codec:      {args.codec}")
    print(f"  Resolution: {args.resolution} @ {args.fps} fps, {args.bitrate}")
    print(f"  Adapter:    {'auto' if args.adapter is None else args.adapter}")
    print(f"  Duration:   {args.duration}s")
    run_encode_loop(ffmpeg, args.duration, args.codec, args.resolution,
                    args.fps, args.bitrate, args.adapter)


if __name__ == "__main__":
    main()
