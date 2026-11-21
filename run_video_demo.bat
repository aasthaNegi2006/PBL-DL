@echo off
setlocal
cd /d "%~dp0"

echo ========================================================
echo Starting ISL Video Demo Runner (Pre-recorded Clip)
echo ========================================================

if not exist ".\.venv\Scripts\python.exe" (
    echo [ERROR] Python virtual environment not found in .venv!
    echo Please set up the environment with:
    echo   python -m venv .venv
    echo   .\.venv\Scripts\activate
    echo   pip install -r requirements.txt
    echo ========================================================
    pause
    exit /b 1
)

if not exist "models\best_isl_model.pth" (
    echo [ERROR] Model checkpoint not found at models\best_isl_model.pth!
    echo Please train or place the model file first:
    echo   python -m src.train
    echo ========================================================
    pause
    exit /b 1
)

set "SAMPLE_VIDEO=data\raw\help\help_Crop_001_01.avi"
if not exist "%SAMPLE_VIDEO%" (
    echo [WARNING] Default clip %SAMPLE_VIDEO% not found.
    echo Searching for any available .avi clip in data\raw ...
    for /r "data\raw" %%f in (*.avi) do (
        set "SAMPLE_VIDEO=%%f"
        goto :found_clip
    )
)

:found_clip
if not exist "%SAMPLE_VIDEO%" (
    echo [ERROR] No video clips found in data\raw!
    echo Please ensure the dataset is placed in data\raw.
    pause
    exit /b 1
)

echo Running offline video demonstration on:
echo   %SAMPLE_VIDEO%
echo ========================================================

".\.venv\Scripts\python.exe" -m src.real_time_webcam --source "%SAMPLE_VIDEO%"
pause
