@echo off
cd /d "%~dp0"
echo ========================================================
echo Starting ISL Video Demo Runner (Help Gesture)
echo ========================================================
echo Recommended for viva presentations to demonstrate guaranteed
echo accurate detection without needing a live webcam.
echo ========================================================
.\.venv\Scripts\python.exe -m src.real_time_webcam --source data/raw/help/help_Crop_001_01.avi
pause
