# SPAR Week 8: Overnight Non-ML Workload Collection (RTX 5080)
# ==============================================================
# Runs non-ML workloads overnight without Shelly Plug.
# ML training workloads will run tomorrow with wall-power logging.
#
# Usage:
#   .venv\Scripts\Activate.ps1
#   .\scripts\run_overnight_nonml.ps1

$ErrorActionPreference = "Continue"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$launcher = Join-Path $RepoRoot "scripts\run_workload.py"
$Reps = 3
$CooldownSec = 15

# Non-ML workloads only (ML training runs tomorrow with Shelly Plug)
$workloads = @(
    # ML Inference
    "resnet50_inference",
    # Scientific HPC
    "cufft_benchmark",
    "nbody_sim",
    # Crypto Mining
    "mining_ethash_proxy",
    # Rendering
    "rendering_proxy",
    # Baseline
    "idle"
)

# Note: ffmpeg_nvenc excluded — requires ffmpeg installation

$totalRuns = $workloads.Count * $Reps
$completedRuns = 0
$failedRuns = @()
$startTime = Get-Date

Write-Host ""
Write-Host "=" * 60
Write-Host "SPAR Overnight Non-ML Collection (RTX 5080)"
Write-Host "=" * 60
Write-Host "  Workloads:  $($workloads.Count) (non-ML only)"
Write-Host "  Reps:       $Reps"
Write-Host "  Total runs: $totalRuns"
Write-Host "  Cooldown:   ${CooldownSec}s between runs"
Write-Host "  Started:    $startTime"
Write-Host "  Note:       No Shelly Plug (wall-power logging disabled)"
Write-Host "  Note:       ffmpeg_nvenc skipped (ffmpeg not installed)"
Write-Host "=" * 60
Write-Host ""

foreach ($workload in $workloads) {
    for ($rep = 1; $rep -le $Reps; $rep++) {
        $completedRuns++
        $elapsed = (Get-Date) - $startTime
        $pct = [math]::Round(($completedRuns / $totalRuns) * 100, 1)

        Write-Host ""
        Write-Host "[$completedRuns/$totalRuns] ($pct%) $workload (rep $rep/$Reps)" -ForegroundColor Cyan
        Write-Host "  Elapsed: $($elapsed.ToString('hh\:mm\:ss'))"

        try {
            & $python $launcher --workload $workload --no-dcgm
            if ($LASTEXITCODE -ne 0) {
                Write-Host "  WARNING: $workload rep $rep exited with code $LASTEXITCODE" -ForegroundColor Yellow
                $failedRuns += "$workload (rep $rep, exit $LASTEXITCODE)"
            } else {
                Write-Host "  OK" -ForegroundColor Green
            }
        } catch {
            Write-Host "  ERROR: $workload rep $rep failed: $_" -ForegroundColor Red
            $failedRuns += "$workload (rep $rep, exception)"
        }

        # Cooldown between runs (let GPU temps settle)
        if ($completedRuns -lt $totalRuns) {
            Write-Host "  Cooldown ${CooldownSec}s..." -ForegroundColor DarkGray
            Start-Sleep -Seconds $CooldownSec
        }
    }
}

$totalElapsed = (Get-Date) - $startTime

Write-Host ""
Write-Host "=" * 60
Write-Host "OVERNIGHT COLLECTION COMPLETE"
Write-Host "=" * 60
Write-Host "  Total time: $($totalElapsed.ToString('hh\:mm\:ss'))"
Write-Host "  Runs:       $completedRuns / $totalRuns"
Write-Host "  Failed:     $($failedRuns.Count)"

if ($failedRuns.Count -gt 0) {
    Write-Host ""
    Write-Host "Failed runs:" -ForegroundColor Red
    foreach ($f in $failedRuns) {
        Write-Host "  - $f" -ForegroundColor Red
    }
}

# Count output files
$parquetFiles = Get-ChildItem -Path (Join-Path $RepoRoot "data") -Filter "*.parquet" -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match "RTX_5080|GeForce" }

Write-Host ""
Write-Host "Output files: $($parquetFiles.Count) parquet files in data/"
Write-Host "=" * 60
Write-Host ""
Write-Host "NEXT: Run ML training workloads tomorrow with Shelly Plug:" -ForegroundColor Yellow
Write-Host '  .\scripts\run_desktop_collection.ps1 -ShellyIP "<IP>" -StartFrom "pytorch_resnet_cifar10"' -ForegroundColor Yellow
