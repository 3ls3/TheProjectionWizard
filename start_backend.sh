#!/bin/bash

# Script to start the FastAPI backend server
# Make sure to activate the virtual environment first

echo "Starting The Projection Wizard API backend..."
echo "Server will be available at: http://localhost:8000"
echo "API docs will be available at: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Unset old service account credentials to use Application Default Credentials (ADC)
unset GOOGLE_APPLICATION_CREDENTIALS

# Set CORS origins including current ngrok URL (update this when ngrok URL changes)
export ALLOWED_ORIGINS="https://www.predictingwizard.com,https://lovable.dev,http://localhost:3001,https://cdba-2003-ec-df23-4700-dda-bd6a-6957-367.ngrok-free.app"

# Activate virtual environment and start the server
source .venv/bin/activate
uvicorn api.main:app --host 0.0.0.0 --port 8000 