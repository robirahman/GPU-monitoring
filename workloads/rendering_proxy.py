#!/usr/bin/env python3
"""GPU rendering proxy workload for SPAR telemetry collection.

Simulates GPU ray-tracing / path-tracing patterns (like Blender Cycles) using
PyTorch CUDA kernels. Characteristics:
  - Per-ray random walk (Monte Carlo integration)
  - Irregular memory access patterns (unlike regular ML tensor ops)
  - Mixed FP32 compute (dot products, sqrt, transcendentals)
  - Low tensor-core utilization (no large GEMMs)

Used as a rendering workload proxy since A100 data-center GPUs lack NVENC.

Usage:
    python rendering_proxy.py --duration 600
    python rendering_proxy.py --rays 65536 --bounces 8 --duration 600
"""

import argparse
import time

import torch


def run_path_tracing(duration: int, device: torch.device, n_rays: int, n_bounces: int):
    """Monte Carlo path tracing simulation.

    Each 'ray' bounces through a random scene, accumulating radiance.
    This is compute-bound with irregular per-ray divergence patterns.
    """
    print(f"  Path tracing proxy: {n_rays} rays, {n_bounces} bounces, {duration}s")

    wall_start = time.time()
    frame = 0

    while time.time() - wall_start < duration:
        # Ray origins and directions (random hemisphere sampling)
        ray_orig = torch.randn(n_rays, 3, device=device)
        ray_dir = torch.nn.functional.normalize(
            torch.randn(n_rays, 3, device=device), dim=-1
        )

        # Accumulated radiance per ray
        radiance = torch.zeros(n_rays, 3, device=device)
        throughput = torch.ones(n_rays, 3, device=device)

        for bounce in range(n_bounces):
            # Simulate ray-sphere intersection (simplified)
            # Random sphere centers as scene geometry
            sphere_centers = torch.randn(64, 3, device=device) * 5
            sphere_radii = torch.rand(64, device=device) * 0.5 + 0.1

            # Compute ray-sphere distances: (n_rays, 64) hit distances
            oc = ray_orig.unsqueeze(1) - sphere_centers.unsqueeze(0)  # (R, 64, 3)
            b = (oc * ray_dir.unsqueeze(1)).sum(dim=-1)               # (R, 64)
            c = (oc * oc).sum(dim=-1) - sphere_radii.unsqueeze(0) ** 2  # (R, 64)
            discriminant = b * b - c                                    # (R, 64)

            hit = discriminant > 0
            t = torch.where(hit, -b - torch.sqrt(torch.clamp(discriminant, min=0)),
                           torch.full_like(b, 1e10))

            # Find nearest hit per ray
            t_min, hit_sphere_idx = t.min(dim=-1)  # (R,)
            any_hit = t_min < 1e9

            # Compute surface normal at hit point
            hit_point = ray_orig + ray_dir * t_min.unsqueeze(-1)
            normal_centers = sphere_centers[hit_sphere_idx]
            normal = torch.nn.functional.normalize(hit_point - normal_centers, dim=-1)
            normal = torch.where(any_hit.unsqueeze(-1), normal, ray_dir)

            # Random BRDF: Lambert + random scatter
            rand_dir = torch.nn.functional.normalize(
                normal + torch.randn_like(normal) * 0.5, dim=-1
            )

            # Emission from emissive spheres (random subset)
            emission_mask = (hit_sphere_idx % 8 == 0) & any_hit
            emission = torch.where(
                emission_mask.unsqueeze(-1),
                torch.ones(n_rays, 3, device=device) * 5.0,
                torch.zeros(n_rays, 3, device=device)
            )

            # Accumulate radiance and update throughput
            radiance = radiance + throughput * emission
            albedo = 0.7
            throughput = throughput * albedo * torch.where(
                any_hit.unsqueeze(-1),
                (normal * ray_dir).sum(dim=-1).abs().unsqueeze(-1).expand(-1, 3),
                torch.zeros(n_rays, 3, device=device)
            )

            # Update ray for next bounce
            ray_orig = hit_point + normal * 1e-4
            ray_dir = rand_dir

        # Tone-map and compute final pixel values
        pixels = radiance.clamp(0, 1).mean()
        torch.cuda.synchronize()

        frame += 1
        if frame % 5 == 0:
            elapsed = time.time() - wall_start
            fps = frame / elapsed
            print(f"  Frame {frame}: {fps:.2f} fps, {elapsed:.1f}s elapsed")

    wall_time = time.time() - wall_start
    print(f"Rendering proxy complete. Frames: {frame}, {frame/wall_time:.2f} fps, {wall_time:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="GPU rendering proxy workload")
    parser.add_argument("--duration", type=int, default=600,
                        help="Run duration in seconds (default: 600)")
    parser.add_argument("--rays", type=int, default=32768,
                        help="Rays per frame (default: 32768)")
    parser.add_argument("--bounces", type=int, default=4,
                        help="Max bounces per ray (default: 4)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        args.device = "cpu"
    device = torch.device(args.device)

    print(f"GPU rendering proxy (Monte Carlo path tracing)")
    print(f"  Device: {device}")

    run_path_tracing(args.duration, device, args.rays, args.bounces)


if __name__ == "__main__":
    main()
