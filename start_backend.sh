#!/bin/bash

# Script to start the FastAPI backend server for The Projection Wizard
# Make sure to activate the virtual environment first

echo "🔮 Starting The Projection Wizard API backend..."
echo "Server will be available at: http://localhost:8000" 
echo "API docs will be available at: http://localhost:8000/docs"
echo ""
echo "✅ Fixed background pipeline logging issues"
echo "✅ Using Application Default Credentials for GCS"
echo "✅ CORS configured for frontend domain"
echo ""

# ================================================================================
# LOCAL DEVELOPMENT LOGGING CONFIGURATION
# ================================================================================
# Uncomment the line below to enable local file logging for pipeline debugging:
# export LOCAL_DEV_LOGGING=true
#
# When LOCAL_DEV_LOGGING=true:
# - Pipeline logs are written to data/runs/{run_id}/logs/ directory  
# - Human-readable logs: {stage}.log files
# - Structured JSON logs: {stage}_structured.jsonl files
# - Console output remains active for immediate feedback
# - Log files persist across server restarts for easier debugging
#
# When LOCAL_DEV_LOGGING=false (default):
# - All logs go to stdout/stderr for cloud logging (production behavior)
# - No local files are created
# ================================================================================

if [ "${LOCAL_DEV_LOGGING:-false}" = "true" ]; then
    echo "🗃️  Local development logging ENABLED"
    echo "   📁 Log files will be saved to: data/runs/{run_id}/logs/"
    echo "   📝 Human-readable logs: {stage}.log"
    echo "   📊 Structured JSON logs: {stage}_structured.jsonl"
    echo ""
else
    echo "☁️  Cloud logging mode (production)"
    echo "   💡 To enable local file logging, set: export LOCAL_DEV_LOGGING=true"
    echo ""
fi

echo "Press Ctrl+C to stop the server"
echo ""

# Unset old service account credentials to use Application Default Credentials (ADC)
unset GOOGLE_APPLICATION_CREDENTIALS

# Set CORS origins including current ngrok URL (update this when ngrok URL changes)
export ALLOWED_ORIGINS="https://www.predictingwizard.com,https://lovable.dev,https://lovable.dev/projects/dca90495-2ee9-4de4-86ef-b619f83fc331,http://localhost:3001,https://b09d-2a02-8109-d90b-2f00-5c57-9e7b-c6-50ca.ngrok-free.app,https://id-preview--dca90495-2ee9-4de4-86ef-b619f83fc331.lovable.app"

echo "🔗 Current ngrok URL: https://b09d-2a02-8109-d90b-2f00-5c57-9e7b-c6-50ca.ngrok-free.app"
echo "📝 Update VITE_API_URL in your frontend to use the ngrok URL above"
echo ""

# Activate virtual environment and start the server
source .venv/bin/activate
uvicorn api.main:app --host 0.0.0.0 --port 8000 
