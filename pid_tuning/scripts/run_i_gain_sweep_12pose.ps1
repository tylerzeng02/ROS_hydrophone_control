# Automated I-gain sweep for shoulder_pitch (motor 1): cycles through every
# candidate value, running the real accuracy check at each one, saving each
# candidate's results under a distinct filename so nothing gets overwritten.
#
# Run from anywhere: .\run_i_gain_sweep_12pose.ps1
# (moved into pid_tuning/ 2026-08-19; the .exe tools stay in build\, so this
# script sets its working directory there itself.)

$BuildDir = Join-Path $PSScriptRoot "..\..\build"
$DataDir = Join-Path $PSScriptRoot "..\data"

$candidates = @(0, 4, 8, 12, 16, 20, 24, 30)
$inputCsv = Join-Path $DataDir "i_gain_sweep_12pose_spread_across_range_input.csv"
$motorId = 1

Push-Location $BuildDir

foreach ($i in $candidates) {
    Write-Host "`n=== Setting I=$i on motor $motorId ===" -ForegroundColor Cyan
    & "$BuildDir\set_i_gain.exe" $motorId $i
    if ($LASTEXITCODE -ne 0) {
        Write-Host "set_i_gain FAILED for I=$i -- stopping sweep." -ForegroundColor Red
        Pop-Location
        exit 1
    }

    Write-Host "=== Running validate on $inputCsv (I=$i) ===" -ForegroundColor Cyan
    & "$BuildDir\test_five_pose_ndi_capture.exe" --validate $inputCsv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "validate run FAILED for I=$i -- stopping sweep." -ForegroundColor Red
        Pop-Location
        exit 1
    }

    $outFile = Join-Path $DataDir "i_gain_${i}_sweep_12pose_results.csv"
    Copy-Item "$BuildDir\validation_results.csv" $outFile -Force
    Write-Host "Saved results to $outFile" -ForegroundColor Green
}

Write-Host "`n=== Sweep complete. Files: ===" -ForegroundColor Yellow
foreach ($i in $candidates) {
    Write-Host "  $(Join-Path $DataDir "i_gain_${i}_sweep_12pose_results.csv")"
}

Write-Host "`n=== All candidates done. Torque has been held enabled the whole time. ===" -ForegroundColor Cyan
& "$BuildDir\disable_torque_prompt.exe"

Pop-Location
