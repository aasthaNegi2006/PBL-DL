@echo off
setlocal
cd /d "%~dp0"

echo ========================================================
echo Starting ISL Real-Time Desktop Webcam Recognizer...
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

echo Controls:
echo   - Position your upper body inside the yellow ROI box (head at top, hands in center)
echo   - Raise hand fully to ear/cheek for CALL; twist wrist slowly for PAIN
echo   - Press 'S' to toggle voice output ON/OFF
echo   - Press 'C' to clear recognition history
echo   - Press 'Q' to quit
echo ========================================================

".\.venv\Scripts\python.exe" -m src.real_time_webcam --source 0
pause
