# Automated I-gain sweep for shoulder_pitch (motor 1), using the freshly
# hand-recorded 45-pose set (i_gain_final_sweep_45pose_hand_posed_dataset.csv -> pid_tuning_data_
# validate_input.csv) instead of a subset of the old 374-pose dataset.
# I=0 is deliberately the first candidate run -- since these poses have no
# pre-existing NDI ground truth, this pass IS what establishes the fresh
# baseline everything else gets compared against, no separate step needed.
# Torque stays enabled across all 8 candidates -- only disabled once, at
# the very end, after an explicit Enter confirmation.
#
# Run from anywhere: .\run_i_gain_sweep_final_45pose.ps1
# (the .exe tools live in build\, input/output data lives in
# pid_tuning\data\ -- this script no longer needs to be run from build\
# itself, moved here 2026-08-19 as part of the pid_tuning/ reorganization.)

$BuildDir = Join-Path $PSScriptRoot "..\..\build"
$DataDir = Join-Path $PSScriptRoot "..\data"

$candidates = @(0, 4, 8, 12, 16, 20, 24, 30)
$inputCsv = Join-Path $DataDir "i_gain_final_sweep_45pose_validate_input.csv"
$motorId = 1

# The .exe writes validation_results.csv relative to the working directory,
# not its own location -- run everything from build\ so that lands where
# expected, regardless of where this script itself was launched from.
Push-Location $BuildDir

foreach ($i in $candidates) {
    Write-Host "`n=== Setting I=$i on motor $motorId ===" -ForegroundColor Cyan
    & "$BuildDir\set_i_gain.exe" $motorId $i
    if ($LASTEXITCODE -ne 0) {
        Write-Host "set_i_gain FAILED for I=$i -- stopping sweep." -ForegroundColor Red
        exit 1
    }

    Write-Host "=== Running validate on $inputCsv (I=$i) ===" -ForegroundColor Cyan
    & "$BuildDir\test_five_pose_ndi_capture.exe" --validate $inputCsv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "validate run FAILED for I=$i -- stopping sweep." -ForegroundColor Red
        exit 1
    }

    $outFile = Join-Path $DataDir "i_gain_${i}_final_sweep_accuracy_results.csv"
    Copy-Item "$BuildDir\validation_results.csv" $outFile -Force
    Write-Host "Saved results to $outFile" -ForegroundColor Green
}

Write-Host "`n=== Sweep complete. Files: ===" -ForegroundColor Yellow
foreach ($i in $candidates) {
    Write-Host "  $(Join-Path $DataDir "i_gain_${i}_final_sweep_accuracy_results.csv")"
}

Write-Host "`n=== All candidates done. Torque has been held enabled the whole time. ===" -ForegroundColor Cyan
& "$BuildDir\disable_torque_prompt.exe"

Pop-Location
