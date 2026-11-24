# ==============================================================================
# Invictus Vehicle Trace — Dual-Terminal Live Launcher
# Launches:
#   Terminal 1: FastAPI Backend + Live Forensic Pipeline Monitor (Port 8000)
#   Terminal 2: Vite React Frontend Development Server (Port 5173)
# ==============================================================================

$Root = $PSScriptRoot
Set-Location $Root

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "   INVICTUS VEHICLE TRACEABILITY SYSTEM — FULL-STACK LAUNCHER    " -ForegroundColor Green
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "-> Launching Terminal 1: FastAPI Backend & Live Pipeline Monitor..." -ForegroundColor Yellow

$BackendCmd = "`$host.UI.RawUI.WindowTitle = 'Invictus Backend & Live Pipeline Logs'; Set-Location '$Root'; .\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $BackendCmd

Start-Sleep -Seconds 2

Write-Host "-> Launching Terminal 2: Frontend Vite Development Server..." -ForegroundColor Yellow

$FrontendCmd = "`$host.UI.RawUI.WindowTitle = 'Invictus Frontend (http://localhost:5173)'; Set-Location '$Root\frontend'; npm run dev"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $FrontendCmd

Write-Host ""
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "   SERVICES ONLINE AND RUNNING:                                  " -ForegroundColor Green
Write-Host "   * Frontend UI:    http://localhost:5173" -ForegroundColor White
Write-Host "   * Backend API:    http://127.0.0.1:8000" -ForegroundColor White
Write-Host "   * Swagger Docs:   http://127.0.0.1:8000/docs" -ForegroundColor White
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host ""
