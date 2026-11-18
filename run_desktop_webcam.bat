@echo off
cd /d "%~dp0"
echo ========================================================
echo Starting ISL Real-Time Desktop Webcam Recognizer...
echo ========================================================
echo Controls:
echo   - Sign inside the yellow ROI box on screen
echo   - Press 'S' to toggle voice output ON/OFF
echo   - Press 'C' to clear recognition history
echo   - Press 'Q' to quit
echo ========================================================
.\.venv\Scripts\python.exe -m src.real_time_webcam --source 0
pause
