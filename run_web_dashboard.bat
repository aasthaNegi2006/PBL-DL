@echo off
setlocal
cd /d "%~dp0"

echo ========================================================
echo Starting ISL Emergency Gesture Recognition Web Dashboard
echo ========================================================

if not exist ".\.venv\Scripts\uvicorn.exe" (
    echo [ERROR] Uvicorn server binary not found in .venv!
    echo Please set up the environment with:
    echo   python -m venv .venv
    echo   .\.venv\Scripts\activate
    echo   pip install -r requirements.txt
    echo ========================================================
    pause
    exit /b 1
)

if not exist "models\best_isl_model.pth" (
    echo [WARNING] Model checkpoint not found at models\best_isl_model.pth!
    echo The web dashboard will start, but inference will return HTTP 503
    echo until a model is trained: python -m src.train
    echo ========================================================
)

netstat -ano | findstr :8000 >nul 2>&1
if %errorlevel% neq 0 (
    echo Starting FastAPI / Uvicorn server on http://127.0.0.1:8000 ...
    start "ISL Model Server" /min cmd /c ".\.venv\Scripts\uvicorn.exe app.main:app --host 127.0.0.1 --port 8000"
    timeout /t 2 >nul
) else (
    echo Model server is already running on port 8000.
)

echo Opening http://127.0.0.1:8000 in your default browser...
start http://127.0.0.1:8000

echo.
echo Dashboard is live at http://127.0.0.1:8000
echo Press any key to close this launcher window (server will keep running).
pause >nul
