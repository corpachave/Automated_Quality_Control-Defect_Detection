@echo off
REM Startup script for VisionSpec QC - PCB Inspection

echo Starting VisionSpec QC System...
echo.

REM Get the project directory
cd /d "e:\Learning\Zaamila Development\Projects\Automated_Quality_Control-Defect_Detection"

REM Activate virtual environment and start FastAPI backend
echo Launching FastAPI Backend (http://127.0.0.1:8000)...
start "FastAPI Backend" cmd /k "call .venv\Scripts\activate.bat && uvicorn main:app --reload"

REM Wait a moment for backend to start
timeout /t 3 /nobreak

REM Start Streamlit frontend
echo Launching Streamlit Frontend (http://localhost:8501)...
start "Streamlit Frontend" cmd /k "call .venv\Scripts\activate.bat && streamlit run app.py"

echo.
echo Both services are starting...
echo - FastAPI Docs: http://127.0.0.1:8000/docs
echo - Streamlit UI: http://localhost:8501
echo.
pause