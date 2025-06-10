#!/bin/bash

# Script to start ngrok tunnel for backend development
# This exposes the local backend (port 8000) to the internet
# so that the deployed frontend can access it

echo "Starting ngrok tunnel for backend development..."
echo "Backend will be accessible at: https://958d-2003-ec-df23-4700-dda-bd6a-6957-367.ngrok-free.app"
echo "Make sure your backend is running on port 8000 first!"
echo ""
echo "To start the backend, run:"
echo "  source .venv/bin/activate && cd api && python main.py"
echo ""
echo "Press Ctrl+C to stop ngrok"
echo ""

ngrok http 8000 