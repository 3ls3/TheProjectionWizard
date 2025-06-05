#!/usr/bin/env python3
"""
Setup script for The Projection Wizard.
Initializes the project environment and validates dependencies.
"""

import subprocess
import sys
import os
from pathlib import Path


def check_python_version():
    """Check if Python version is 3.10 or higher."""
    if sys.version_info < (3, 10):
        print("Error: Python 3.10 or higher is required.")
        print(f"Current version: {sys.version}")
        return False
    print(f"✓ Python version: {sys.version.split()[0]}")
    return True


def create_virtual_environment():
    """Create a virtual environment if it doesn't exist."""
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("✓ Virtual environment already exists")
        return True
        
    try:
        print("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("✓ Virtual environment created")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error creating virtual environment: {e}")
        return False


def install_dependencies():
    """Install project dependencies."""
    venv_python = "venv/bin/python" if os.name != "nt" else "venv\\Scripts\\python.exe"
    
    if not Path(venv_python).exists():
        print("Error: Virtual environment not found. Run setup again.")
        return False
        
    try:
        print("Installing dependencies...")
        subprocess.run([
            venv_python, "-m", "pip", "install", "--upgrade", "pip"
        ], check=True)
        
        subprocess.run([
            venv_python, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True)
        
        print("✓ Dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False


def setup_directories():
    """Ensure all required directories exist."""
    directories = [
        "data/runs",
        "data/fixtures", 
        "step_1_ingest",
        "step_2_schema",
        "step_3_validation",
        "step_4_prep",
        "step_5_automl",
        "step_6_explain",
        "ui",
        "common",
        "scripts"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✓ Directory structure verified")
    return True


def main():
    """Main setup function."""
    print("Setting up The Projection Wizard...")
    print("=" * 50)
    
    if not check_python_version():
        return 1
        
    if not setup_directories():
        return 1
        
    if not create_virtual_environment():
        return 1
        
    if not install_dependencies():
        return 1
        
    print("\n" + "=" * 50)
    print("✓ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Activate the virtual environment:")
    
    if os.name == "nt":
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
        
    print("2. Run the application:")
    print("   streamlit run ui/main.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 