# Live LLM - Start Script
# Run this to start both backend and frontend

Write-Host "================================" -ForegroundColor Cyan
Write-Host "  Live LLM - Starting Servers   " -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Start backend
Write-Host "[1/2] Starting Python backend on port 8000..." -ForegroundColor Yellow
Start-Process -FilePath "python" -ArgumentList "main.py" -WorkingDirectory "$PSScriptRoot\server" -WindowStyle Normal

Start-Sleep -Seconds 2

# Start frontend
Write-Host "[2/2] Starting Next.js frontend on port 3000..." -ForegroundColor Yellow
Start-Process -FilePath "npm" -ArgumentList "run", "dev" -WorkingDirectory "$PSScriptRoot\web" -WindowStyle Normal

Write-Host ""
Write-Host "================================" -ForegroundColor Green
Write-Host "  Servers Started!              " -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green
Write-Host ""
Write-Host "Backend API:  http://localhost:8000" -ForegroundColor White
Write-Host "Frontend:     http://localhost:3000" -ForegroundColor White
Write-Host "API Docs:     http://localhost:8000/docs" -ForegroundColor White
Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
