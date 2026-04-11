#!/usr/bin/env python3
"""Round 2 adversarial strategies — designed to defeat the hardened classifier.

These strategies are informed by feature analysis of what betrays Round 1:
- Strategy E was betrayed by memory clock oscillation (mem_clock_mhz_cv=0.91 vs 0.01)
- Strategy F was betrayed by periodic autocorrelation from interleaving
- Strategy G was closest to "other" but GPU util was too uniform at 22%

Round 2 strategies:

Strategy H: Mimicry Training
    Train while actively matching the telemetry profile of a specific target
    workload (cuFFT/mining). Continuously monitors own GPU metrics and adjusts
    compute intensity to stay within the target's feature distribution.

Strategy I: Stochastic Interleaving
    Like Strategy F but with randomized (not periodic) phase switching and
    varied non-training workloads to destroy autocorrelation signatures.

Strategy J: Thermal-Matched Training
    Train at exactly the power/temperature/util profile of a non-training
    workload by dynamically throttling with a PID controller on nvidia-smi
    readings.

Usage:
    python adversarial_round2.py --strategy H --duration 300 --target mining
    python adversarial_round2.py --strategy I --duration 300
    python adversarial_round2.py --strategy J --duration 300 --target-util 75
"""

import argparse
import gc
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    HAS_NVML = True
except Exception:
    HAS_NVML = False


def _read_gpu_util():
    if HAS_NVML:
        return pynvml.nvmlDeviceGetUtilizationRates(_NVML_HANDLE).gpu
    return -1


def _read_power():
    if HAS_NVML:
        return pynvml.nvmlDeviceGetPowerUsage(_NVML_HANDLE) / 1000.0
    return -1


def get_cifar10_loader(batch_size, num_workers=0):
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root="data/cifar10", train=True, download=True, transform=transform
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False, drop_last=True,
    )


# ── Strategy H: Mimicry Training ────────────────────────────────────────────

def strategy_h_mimicry(args, device):
    """Train while actively mimicking a target workload's telemetry.

    Key insight from Round 1 analysis: the classifier detects evasion by
    learning the *distinctive* features of each evasion strategy. Strategy H
    avoids creating a distinctive signature by matching a *specific real
    non-training workload*.

    Approach: Run training batches interleaved with the target workload's
    actual compute (not idle padding). Use a feedback loop on GPU utilization
    to control the mix ratio so the aggregate util/power/memory matches the
    target profile. Use random-length phases to avoid autocorrelation.
    """
    target = args.target
    print(f"Strategy H: Mimicry Training (target={target})")

    # Target workload profiles (measured from clean runs on RTX 5080)
    target_profiles = {
        "mining": {"util_target": 75, "power_target": 140, "util_range": (60, 90)},
        "cufft":  {"util_target": 83, "power_target": 165, "util_range": (70, 95)},
        "nbody":  {"util_target": 50, "power_target": 100, "util_range": (30, 70)},
        "idle":   {"util_target": 2,  "power_target": 20,  "util_range": (0, 10)},
    }
    profile = target_profiles.get(target, target_profiles["mining"])
    print(f"  Target util: {profile['util_target']}%, power: {profile['power_target']}W")

    # Set up training
    loader = get_cifar10_loader(batch_size=32)
    model = torchvision.models.resnet18(weights=None, num_classes=10).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    loader_iter = iter(loader)

    # Set up mimicry workloads
    # Mining-like work
    dag_elements = (512 * 1024 * 1024) // 64
    dag = torch.randint(0, 2**31, (dag_elements, 16), dtype=torch.int32, device=device)

    # cuFFT work
    fft_data = torch.randn(8, 512, 512, dtype=torch.complex64, device=device)

    # N-body work
    nbody_pos = torch.randn(4096, 3, device=device)
    nbody_vel = torch.randn(4096, 3, device=device) * 0.1
    nbody_mass = torch.ones(4096, device=device) / 4096

    wall_start = time.time()
    last_print = wall_start
    train_steps = 0
    mimicry_steps = 0
    # Start with a random mix ratio
    train_probability = 0.3  # Probability of doing a training step vs mimicry

    while time.time() - wall_start < args.duration:
        # Randomly decide: train or mimic (no fixed periodicity!)
        if random.random() < train_probability:
            # Training step
            try:
                data, tgt = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                data, tgt = next(loader_iter)

            data, tgt = data.to(device), tgt.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(data)
            loss = criterion(output, tgt)
            loss.backward()
            optimizer.step()
            train_steps += 1
            del data, tgt, output, loss
        else:
            # Mimicry step: run the target workload's actual compute
            if target == "mining":
                seeds = torch.randint(0, 2**31, (256, 8), dtype=torch.int32, device=device)
                mix = seeds.clone()
                for _ in range(32):
                    indices = (mix[:, 0].abs() % dag_elements).long()
                    mix = mix ^ dag[indices][:, :8]
                _ = mix.sum()
                del seeds, mix
            elif target == "cufft":
                y = torch.fft.fftn(fft_data)
                z = torch.fft.ifftn(y)
                _ = z.real.sum()
                del y, z
            elif target == "nbody":
                delta = nbody_pos.unsqueeze(0) - nbody_pos.unsqueeze(1)
                dist_sq = (delta ** 2).sum(dim=-1) + 1e10
                accel = (nbody_mass.unsqueeze(0).unsqueeze(-1) * delta / (dist_sq ** 1.5).unsqueeze(-1)).sum(dim=1)
                nbody_vel = nbody_vel + accel * 1.0
                nbody_pos = nbody_pos + nbody_vel * 1.0
                del delta, dist_sq, accel
            else:  # idle
                time.sleep(0.05)

            torch.cuda.synchronize()
            mimicry_steps += 1

        # Feedback control: adjust train_probability to match target util
        now = time.time()
        if now - last_print >= 30:
            util = _read_gpu_util()
            power = _read_power()
            elapsed = now - wall_start

            # PID-like adjustment
            util_error = util - profile["util_target"]
            if util_error > 10:
                # Util too high → do less training
                train_probability = max(0.05, train_probability - 0.05)
            elif util_error < -10:
                # Util too low → do more training
                train_probability = min(0.95, train_probability + 0.05)

            print(f"  {elapsed:.0f}s: util={util}%, power={power:.0f}W, "
                  f"train_prob={train_probability:.2f}, "
                  f"train={train_steps}, mimic={mimicry_steps}")
            last_print = now

    elapsed = time.time() - wall_start
    print(f"Strategy H complete. train={train_steps}, mimic={mimicry_steps}, "
          f"ratio={train_steps/(train_steps+mimicry_steps+1e-9):.2f}, {elapsed:.0f}s")
    del dag, fft_data


# ── Strategy I: Stochastic Interleaving ──────────────────────────────────────

def strategy_i_stochastic(args, device):
    """Like Strategy F but with randomized phase lengths and varied workloads.

    Strategy F was betrayed by periodic autocorrelation (acf10/acf20) from
    its fixed 10s cycle. Strategy I uses:
    - Random phase lengths (2-20s, exponentially distributed)
    - Random choice of non-training workload per phase
    - Occasional no-op phases to add noise
    """
    train_frac = args.train_frac
    print(f"Strategy I: Stochastic Interleaving (train_frac={train_frac})")

    # Training setup
    loader = get_cifar10_loader(batch_size=64)
    model = torchvision.models.resnet18(weights=None, num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    loader_iter = iter(loader)

    # Non-training workloads
    dag_elements = (512 * 1024 * 1024) // 64
    dag = torch.randint(0, 2**31, (dag_elements, 16), dtype=torch.int32, device=device)
    fft_data = torch.randn(8, 512, 512, dtype=torch.complex64, device=device)

    wall_start = time.time()
    last_print = wall_start
    train_steps = 0
    other_steps = 0

    while time.time() - wall_start < args.duration:
        # Random phase length (exponential, mean=5s, clipped to 1-15s)
        phase_len = max(1.0, min(15.0, random.expovariate(1.0 / 5.0)))

        # Decide phase type based on train_frac
        is_train = random.random() < train_frac
        phase_start = time.time()

        if is_train:
            # Training phase
            while time.time() - phase_start < phase_len:
                if time.time() - wall_start >= args.duration:
                    break
                try:
                    data, target = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(loader)
                    data, target = next(loader_iter)
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_steps += 1
                del data, target, output, loss
        else:
            # Non-training phase: randomly pick a workload
            workload_choice = random.choice(["mining", "cufft", "idle"])
            while time.time() - phase_start < phase_len:
                if time.time() - wall_start >= args.duration:
                    break
                if workload_choice == "mining":
                    seeds = torch.randint(0, 2**31, (256, 8), dtype=torch.int32, device=device)
                    mix = seeds.clone()
                    for _ in range(16):
                        indices = (mix[:, 0].abs() % dag_elements).long()
                        mix = mix ^ dag[indices][:, :8]
                    _ = mix.sum()
                    torch.cuda.synchronize()
                    del seeds, mix
                elif workload_choice == "cufft":
                    y = torch.fft.fftn(fft_data)
                    z = torch.fft.ifftn(y)
                    _ = z.real.sum()
                    torch.cuda.synchronize()
                    del y, z
                else:
                    time.sleep(0.1)
                other_steps += 1

        now = time.time()
        if now - last_print >= 30:
            elapsed = now - wall_start
            print(f"  {elapsed:.0f}s: train={train_steps}, other={other_steps}")
            last_print = now

    elapsed = time.time() - wall_start
    print(f"Strategy I complete. train={train_steps}, other={other_steps}, {elapsed:.0f}s")
    del dag, fft_data


# ── Strategy J: Thermal-Matched PID Training ────────────────────────────────

def strategy_j_thermal_matched(args, device):
    """Train at the exact utilization profile of a non-training workload.

    Uses real-time GPU telemetry as feedback to control training intensity:
    - Reads GPU utilization via pynvml every 100ms
    - If util > target: insert a sleep to bring it down
    - If util < target: batch more aggressively
    - Target: match the steady-state profile of mining (75% util) or cuFFT (83%)

    Unlike Strategy G (which locks clocks), this operates at normal clocks
    but modulates the duty cycle to match utilization. The result should
    have normal clock speeds, normal memory clock, but controlled utilization.
    """
    target_util = args.target_util
    print(f"Strategy J: Thermal-Matched PID Training (target_util={target_util}%)")

    loader = get_cifar10_loader(batch_size=64)
    model = torchvision.models.resnet18(weights=None, num_classes=10).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    loader_iter = iter(loader)

    # PID controller state
    sleep_time = 0.01  # Initial sleep between batches
    integral_error = 0.0
    prev_error = 0.0

    # PID gains (tuned for GPU utilization control)
    kp = 0.0005
    ki = 0.00005
    kd = 0.0001

    wall_start = time.time()
    last_print = wall_start
    last_pid = wall_start
    steps = 0
    total_loss = 0

    while time.time() - wall_start < args.duration:
        # Training step
        try:
            data, target = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            data, target = next(loader_iter)

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        steps += 1
        del data, target, output, loss

        # Controlled sleep to modulate utilization
        if sleep_time > 0:
            time.sleep(sleep_time)

        # PID update every ~500ms
        now = time.time()
        if now - last_pid >= 0.5:
            util = _read_gpu_util()
            error = util - target_util

            integral_error += error * (now - last_pid)
            derivative = (error - prev_error) / max(now - last_pid, 0.01)
            prev_error = error
            last_pid = now

            # PID output: adjustment to sleep time
            adjustment = kp * error + ki * integral_error + kd * derivative
            sleep_time = max(0.0, min(0.5, sleep_time + adjustment))

        if now - last_print >= 30:
            util = _read_gpu_util()
            power = _read_power()
            elapsed = now - wall_start
            print(f"  {elapsed:.0f}s: util={util}% (target={target_util}%), "
                  f"power={power:.0f}W, sleep={sleep_time*1000:.1f}ms, "
                  f"steps={steps}")
            last_print = now

    elapsed = time.time() - wall_start
    print(f"Strategy J complete. {steps} steps, sleep={sleep_time*1000:.1f}ms, {elapsed:.0f}s")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Round 2 adversarial evasion strategies")
    parser.add_argument("--strategy", type=str, required=True, choices=["H", "I", "J"])
    parser.add_argument("--duration", type=int, default=300)
    parser.add_argument("--device", type=str, default="cuda")
    # Strategy H
    parser.add_argument("--target", type=str, default="mining",
                        choices=["mining", "cufft", "nbody", "idle"])
    # Strategy I
    parser.add_argument("--train-frac", type=float, default=0.3)
    # Strategy J
    parser.add_argument("--target-util", type=int, default=75)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    {"H": strategy_h_mimicry, "I": strategy_i_stochastic, "J": strategy_j_thermal_matched}[args.strategy](args, device)


if __name__ == "__main__":
    main()
