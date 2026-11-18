@echo off
cd /d "%~dp0"
echo ========================================================
echo Starting ISL Emergency Gesture Recognition Web Dashboard
echo ========================================================

netstat -ano | findstr :8000 >nul 2>&1
if %errorlevel% neq 0 (
    echo Starting FastAPI / Uvicorn server on http://localhost:8000 ...
    start "ISL Model Server" /min cmd /c ".\.venv\Scripts\uvicorn.exe app.main:app --host 0.0.0.0 --port 8000"
    timeout /t 2 >nul
) else (
    echo Model server is already running on port 8000.
)

echo Opening http://localhost:8000 in your browser...
start http://localhost:8000

echo.
echo Dashboard is live!
echo Press any key to close this launcher window (server will keep running).
pause >nul
