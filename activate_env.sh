#!/bin/bash
# The Projection Wizard - Development Environment Activation Script
# 
# This script activates the virtual environment and provides helpful shortcuts

echo "🧙‍♂️ The Projection Wizard - Activating Development Environment..."
echo

# Activate virtual environment
source .venv/bin/activate

# Verify Python version
python_version=$(python --version)
echo "✅ Virtual environment activated"
echo "🐍 Python version: $python_version"

# Check for PyCaret compatibility and test import
if python -c "import pycaret" 2>/dev/null; then
    echo "✅ Python version compatible with PyCaret - all dependencies working!"
else
    echo "⚠️  WARNING: PyCaret import failed"
    echo "   This may indicate a compatibility issue"
fi

echo
echo "📁 Project directory: $(pwd)"
echo "🌐 Virtual environment: $(which python)"
echo
echo "🚀 Available commands:"
echo "   • Start API server:     cd api && uvicorn main:app --reload"
echo "   • Run tests:           pytest"
echo "   • Format code:         black ."
echo "   • Lint code:           flake8 ."
echo "   • Type check:          mypy ."
echo "   • Start Streamlit:     streamlit run app/main.py"
echo "   • Check Python setup:  python setup_check.py"
echo
echo "📋 For new team members:"
echo "   Run 'python setup_check.py' to verify Python compatibility"
echo
echo "Happy coding! 🎉" 