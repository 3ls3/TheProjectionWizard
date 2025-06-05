#!/bin/bash
# Run Team A Streamlit App
echo "🔍 Starting Team A EDA & Validation Pipeline..."
echo "Opening http://localhost:8501 in browser"

# Activate virtual environment and run streamlit
source .venv/bin/activate
streamlit run app/team_a/main-part1.py 