@echo off
setlocal
cd /d "%~dp0"
echo ==========================================
echo LAB 02 - VS CODE PROJECT SETUP
echo ==========================================

where py >nul 2>nul
if %errorlevel%==0 (
    py -3.11 --version >nul 2>nul
    if %errorlevel%==0 (
        py -3.11 -m venv .venv
    ) else (
        py -3 -m venv .venv
    )
) else (
    python -m venv .venv
)

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Python virtual environment could not be created.
    echo Install Python 3.11 from python.org and enable Add Python to PATH.
    pause
    exit /b 1
)

call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Package installation failed. Check internet connection.
    pause
    exit /b 1
)

python -m spacy download en_core_web_sm
if %errorlevel% neq 0 (
    echo ERROR: spaCy English model download failed.
    pause
    exit /b 1
)

echo.
echo SETUP COMPLETED.
echo In VS Code select interpreter: .venv\Scripts\python.exe
echo Then run: python main.py
pause
