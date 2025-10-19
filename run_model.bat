@echo off
echo =======================================================
echo Diabetes Ensemble Training & Report Generation
echo =======================================================
echo.

:: Train model & generate visualizations
echo [1/2] Running model training...
python -B -m examples.train_diabetes
if %errorlevel% neq 0 (
    echo  Training failed. Please check errors above.
    pause
    exit /b
)

echo.
:: Step 2 - Generate structured PDF report
echo [2/2] Generating PDF report...
python -B -m reports.generate_report
if %errorlevel% neq 0 (
    echo  Report generation failed. Please check errors above.
    pause
    exit /b
)

echo.
echo  All tasks completed successfully!
echo  Report saved to: results\model_report.pdf
echo.

:: Optional: open PDF automatically
start results\model_report.pdf

pause
