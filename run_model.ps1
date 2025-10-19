# run_model.ps1 - PowerShell runner (auto pipeline)
# ==================================================
# 1. Activate venv
# 2. Train calibrated ensemble model
# 3. Generate full PDF report
# 4. Open results folder

$here = Get-Location
$venvActivate = Join-Path $here "venv\Scripts\Activate.ps1"

if (-not (Test-Path $venvActivate)) {
    Write-Error "Virtual environment not found. Create it first: python -m venv venv"
    exit 1
}

Write-Output "[STEP] Activating virtual environment..."
. $venvActivate

Write-Output "`n[STEP] Running training pipeline..."
python -B -m examples.train_diabetes
if ($LASTEXITCODE -ne 0) {
    Write-Error "Training script failed!"
    exit $LASTEXITCODE
}

Write-Output "`n[STEP] Generating PDF report..."
python -B -m reports.generate_report
if ($LASTEXITCODE -ne 0) {
    Write-Error "Report generation failed!"
    exit $LASTEXITCODE
}

Write-Output "`n[OK] All tasks completed successfully!"
Start-Process -FilePath (Join-Path $here "results")
Write-Output "Results folder opened. Check model_report.pdf."
