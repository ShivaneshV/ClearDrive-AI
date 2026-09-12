@echo off
title ClearDrive AI - Commercial ADAS Cockpit
color 0b
echo ======================================================================
echo       CLEAR-DRIVE AI : NEXT-GEN AUTOMOTIVE COCKPIT
echo ======================================================================
echo Starting local Edge AI Perception Server...

:: Start Flask server in background
start "" /b python -u -B app.py

:: Give server 2 seconds to initialize
timeout /t 2 /nobreak >nul

:: Launch directly into standalone, borderless native App Window
echo Launching Cockpit Native App Window...
start msedge --app=http://localhost:5000 2>nul || start chrome --app=http://localhost:5000 2>nul || start http://localhost:5000

echo Application launched successfully!
