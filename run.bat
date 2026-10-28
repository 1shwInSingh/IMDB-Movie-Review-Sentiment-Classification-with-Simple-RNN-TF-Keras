@echo off
setlocal
title IMDB Sentiment Classifier Web App

:: Always navigate to the script's root directory
cd /d "%~dp0"

:: Set UTF-8 encoding for Python IO
set PYTHONUTF8=1

echo ==========================================================
echo   IMDB Sentiment Classification Web Application
echo ==========================================================
echo.

:: Detect Python environment (.venv first, then system python, then py launcher)
set "PYTHON_EXE="
if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
    echo [*] Found virtual environment: .venv
) else (
    where python >nul 2>nul
    if %ERRORLEVEL% equ 0 (
        set "PYTHON_EXE=python"
        echo [*] Using system Python
    ) else (
        where py >nul 2>nul
        if %ERRORLEVEL% equ 0 (
            set "PYTHON_EXE=py"
            echo [*] Using Python launcher (py)
        )
    )
)

if "%PYTHON_EXE%"=="" (
    echo [!] ERROR: Python was not found on your system!
    echo     Please install Python or create a virtual environment in .venv.
    echo.
    pause
    exit /b 1
)

echo [*] Opening browser at http://127.0.0.1:5000 ...
start "" "http://127.0.0.1:5000"

echo [*] Starting Flask web server (app.py)...
echo [*] Press Ctrl+C in this window to stop the server.
echo.

"%PYTHON_EXE%" app.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo [!] Server stopped with exit code %ERRORLEVEL%.
    pause
)
