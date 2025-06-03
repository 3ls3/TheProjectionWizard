#!/bin/bash
# Run Main Streamlit App (Team B)
echo "🧙 Starting The Projection Wizard Main App..."
echo "Opening http://localhost:8502 in browser"

# Activate virtual environment and run streamlit on a different port
source .venv/bin/activate
streamlit run app/main.py --server.port 8502 