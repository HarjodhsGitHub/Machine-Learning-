@echo off
REM DJ Recommender Startup Script (Windows)

echo.
echo 🎵 Starting DJ Transition Recommender...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is required but not installed.
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo 📥 Installing dependencies...
pip install -q -r requirements.txt

REM Create necessary directories
if not exist "uploads" mkdir uploads
if not exist "data" mkdir data

REM Start backend
echo.
echo 🚀 Starting FastAPI backend on http://localhost:8000
echo.
echo Press Ctrl+C to stop the server
echo.

cd backend
python main.py

pause
