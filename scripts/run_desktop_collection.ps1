# SPAR Week 8: RTX 5080 Desktop Full Workload Collection
# ======================================================
# Runs 14 workloads x 3 reps = 42 total runs with Tier 1 NVML + Tier 5 wall power.
# Can run unattended; takes ~10-14 hours total.
#
# Usage:
#   # With Shelly Plug wall-power logging:
#   .\scripts\run_desktop_collection.ps1 -ShellyIP "192.168.1.100"
#
#   # Without Shelly (GPU telemetry only):
#   .\scripts\run_desktop_collection.ps1
#
#   # Resume from a specific workload (if interrupted):
#   .\scripts\run_desktop_collection.ps1 -ShellyIP "192.168.1.100" -StartFrom "cufft_benchmark"

param(
    [string]$ShellyIP = "",
    [string]$StartFrom = "",
    [int]$Reps = 3,
    [int]$CooldownSec = 10
)

$ErrorActionPreference = "Continue"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

# Activate venv if present
$venvActivate = Join-Path $RepoRoot ".venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    & $venvActivate
}

# Determine which launcher to use
if ($ShellyIP -ne "") {
    $launcher = "scripts\run_workload_with_power.py"
    $baseArgs = @("--no-dcgm", "--shelly-ip", $ShellyIP)
    Write-Host "Using wall-power logging (Shelly at $ShellyIP)" -ForegroundColor Green
} else {
    $launcher = "scripts\run_workload.py"
    $baseArgs = @("--no-dcgm")
    Write-Host "GPU telemetry only (no Shelly IP provided)" -ForegroundColor Yellow
}

# Workload list in collection order (ML training first = highest priority)
$workloads = @(
    # ML Training (8 workloads)
    "pytorch_resnet_cifar10",
    "pytorch_resnet_cifar10_amp",
    "pytorch_mlp_cifar10",
    "gpt2_wikitext2",
    "gpt2_wikitext2_amp",
    "bert_sst2",
    "bert_sst2_amp",
    # ML Inference (1 workload)
    "resnet50_inference",
    # Scientific HPC (2 workloads)
    "cufft_benchmark",
    "nbody_sim",
    # Crypto Mining (1 workload)
    "mining_ethash_proxy",
    # Rendering (1 workload)
    "rendering_proxy",
    # Other (2 workloads)
    "ffmpeg_nvenc",
    "idle"
)

$totalRuns = $workloads.Count * $Reps
$completedRuns = 0
$failedRuns = @()
$startTime = Get-Date
$skipping = ($StartFrom -ne "")

Write-Host ""
Write-Host "=" * 60
Write-Host "SPAR RTX 5080 Desktop Collection"
Write-Host "=" * 60
Write-Host "  Workloads:  $($workloads.Count)"
Write-Host "  Reps:       $Reps"
Write-Host "  Total runs: $totalRuns"
Write-Host "  Cooldown:   ${CooldownSec}s between runs"
Write-Host "  Started:    $startTime"
Write-Host "=" * 60
Write-Host ""

foreach ($workload in $workloads) {
    # Skip until we reach the StartFrom workload
    if ($skipping) {
        if ($workload -eq $StartFrom) {
            $skipping = $false
        } else {
            Write-Host "Skipping $workload (resuming from $StartFrom)" -ForegroundColor DarkGray
            continue
        }
    }

    for ($rep = 1; $rep -le $Reps; $rep++) {
        $completedRuns++
        $elapsed = (Get-Date) - $startTime
        $pct = [math]::Round(($completedRuns / $totalRuns) * 100, 1)

        Write-Host ""
        Write-Host "[$completedRuns/$totalRuns] ($pct%) $workload (rep $rep/$Reps)" -ForegroundColor Cyan
        Write-Host "  Elapsed: $($elapsed.ToString('hh\:mm\:ss'))"

        $runArgs = @("python", $launcher, "--workload", $workload) + $baseArgs

        try {
            & python $launcher --workload $workload @baseArgs
            if ($LASTEXITCODE -ne 0) {
                Write-Host "  WARNING: $workload rep $rep exited with code $LASTEXITCODE" -ForegroundColor Yellow
                $failedRuns += "$workload (rep $rep, exit $LASTEXITCODE)"
            }
        } catch {
            Write-Host "  ERROR: $workload rep $rep failed: $_" -ForegroundColor Red
            $failedRuns += "$workload (rep $rep, exception)"
        }

        # Cooldown between runs
        if ($rep -lt $Reps -or $workload -ne $workloads[-1]) {
            Write-Host "  Cooldown ${CooldownSec}s..." -ForegroundColor DarkGray
            Start-Sleep -Seconds $CooldownSec
        }
    }
}

$totalElapsed = (Get-Date) - $startTime

Write-Host ""
Write-Host "=" * 60
Write-Host "COLLECTION COMPLETE"
Write-Host "=" * 60
Write-Host "  Total time:   $($totalElapsed.ToString('hh\:mm\:ss'))"
Write-Host "  Runs:         $completedRuns / $totalRuns"
Write-Host "  Failed:       $($failedRuns.Count)"

if ($failedRuns.Count -gt 0) {
    Write-Host ""
    Write-Host "Failed runs:" -ForegroundColor Red
    foreach ($f in $failedRuns) {
        Write-Host "  - $f" -ForegroundColor Red
    }
}

# Count output files
$parquetFiles = Get-ChildItem -Path (Join-Path $RepoRoot "data") -Filter "*RTX_5080*.parquet" -ErrorAction SilentlyContinue
$powerFiles = Get-ChildItem -Path (Join-Path $RepoRoot "data\wall_power") -Filter "*RTX_5080*.parquet" -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "Output files:"
Write-Host "  GPU telemetry: $($parquetFiles.Count) parquet files in data/"
if ($ShellyIP -ne "") {
    Write-Host "  Wall power:    $($powerFiles.Count) parquet files in data/wall_power/"
}
Write-Host "=" * 60
