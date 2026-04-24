# Startup script for VisionSpec QC - PCB Inspection

Write-Host "Starting VisionSpec QC System..." -ForegroundColor Cyan
Write-Host ""

# Navigate to project directory
Set-Location "e:\Learning\Zaamila Development\Projects\Automated_Quality_Control-Defect_Detection"

# Start FastAPI backend in a new PowerShell window
Write-Host "Launching FastAPI Backend (http://127.0.0.1:8000)..." -ForegroundColor Green
$backendScript = @"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned -Force
& ".\.venv\Scripts\Activate.ps1"
uvicorn main:app --reload
"@

Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendScript -WindowStyle Normal

# Wait for backend to start
Start-Sleep -Seconds 3

# Start Streamlit frontend in a new PowerShell window
Write-Host "Launching Streamlit Frontend (http://localhost:8501)..." -ForegroundColor Green
$frontendScript = @"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned -Force
& ".\.venv\Scripts\Activate.ps1"
streamlit run app.py
"@

Start-Process powershell -ArgumentList "-NoExit", "-Command", $frontendScript -WindowStyle Normal

Write-Host ""
Write-Host "Both services are starting..." -ForegroundColor Yellow
Write-Host "- FastAPI Docs: http://127.0.0.1:8000/docs" -ForegroundColor Cyan
Write-Host "- Streamlit UI: http://localhost:8501" -ForegroundColor Cyan
Write-Host ""