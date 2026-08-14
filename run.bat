@echo off
REM DeepS&P — one-command local launch.
REM Creates a virtual environment on first run, installs dependencies, starts the app.
REM The first install pulls PyTorch (CPU build, ~200 MB) and takes a few minutes.
setlocal
cd /d "%~dp0"

set VENV=.venv
set PY=%VENV%\Scripts\python.exe

if not exist "%PY%" (
    echo Creating virtual environment...
    py -3.12 -m venv %VENV% 2>nul || py -3 -m venv %VENV% || python -m venv %VENV%
    if not exist "%PY%" (
        echo.
        echo Could not create a virtual environment. Is Python installed and on PATH?
        pause
        exit /b 1
    )
    "%PY%" -m pip install --upgrade pip
    echo Installing PyTorch ^(CPU build^)...
    "%PY%" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    echo Installing dependencies...
    "%PY%" -m pip install -r requirements.txt
)

echo.
echo Starting DeepS^&P at http://localhost:8502
echo Press Ctrl+C to stop.
echo.
"%PY%" -m streamlit run sp500.py --server.port 8502
endlocal
