@echo off
echo =======================================================
echo Starting LIC Techathon Backend and Frontend Pipeline
echo =======================================================

echo Starting FastAPI Backend Pipeline...
start cmd /k "python -m uvicorn src.api:app --host 0.0.0.0 --port 8000"

echo Starting Next.js Frontend...
cd frontend\frontend
start cmd /k "npm run dev"

echo Both backend and frontend are starting.
echo Backend URL: http://localhost:8000
echo Frontend URL: http://localhost:3000
echo.
echo Leave these windows open to keep the services running.
pause
