#!/usr/bin/env python3
"""
The Projection Wizard - Python Version Check Script

This script validates that the Python version is compatible with the project
before allowing setup to proceed. Run this before creating virtual environments.

Usage:
    python setup_check.py
"""

import sys
import platform
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible with the project requirements."""
    current_version = sys.version_info
    major, minor = current_version.major, current_version.minor
    
    print("🧙‍♂️ The Projection Wizard - Python Version Check")
    print("=" * 50)
    print(f"🐍 Detected Python version: {platform.python_version()}")
    print(f"📍 Python executable: {sys.executable}")
    
    # Check if version is in supported range
    if major == 3 and 10 <= minor <= 11:
        print("✅ Python version is COMPATIBLE!")
        print(f"   Python {major}.{minor} is supported by PyCaret and all dependencies")
        
        # Check if .python-version exists and matches
        python_version_file = Path(".python-version")
        if python_version_file.exists():
            expected_version = python_version_file.read_text().strip()
            current_version_str = f"{major}.{minor}.{current_version.micro}"
            if current_version_str.startswith(expected_version.split('.')[0] + '.' + expected_version.split('.')[1]):
                print(f"✅ Matches .python-version file: {expected_version}")
            else:
                print(f"⚠️  .python-version specifies: {expected_version}")
                print(f"   Consider using: pyenv local {expected_version}")
        
        print("\n🚀 Ready to proceed with setup!")
        print("   Next steps:")
        print("   1. python -m venv .venv")
        print("   2. source .venv/bin/activate")
        print("   3. pip install -r requirements.txt")
        return True
        
    else:
        print("❌ Python version is NOT COMPATIBLE!")
        print(f"   Python {major}.{minor} is not supported")
        print("\n💡 Required: Python 3.10.x or 3.11.x")
        print("   Reason: PyCaret (core ML library) doesn't support Python 3.12+")
        
        print("\n🔧 How to fix:")
        if major == 3 and minor >= 12:
            print("   Your Python is too NEW. Install an older version:")
            print("   1. Install pyenv: https://github.com/pyenv/pyenv")
            print("   2. Install Python 3.10.6: pyenv install 3.10.6")
            print("   3. Set local version: pyenv local 3.10.6")
            print("   4. Re-run this script to verify")
        elif major == 3 and minor < 10:
            print("   Your Python is too OLD. Install a newer version:")
            print("   1. Install pyenv: https://github.com/pyenv/pyenv")
            print("   2. Install Python 3.10.6: pyenv install 3.10.6")
            print("   3. Set local version: pyenv local 3.10.6")
            print("   4. Re-run this script to verify")
        else:
            print("   You're not using Python 3. Install Python 3.10.6:")
            print("   1. Install pyenv: https://github.com/pyenv/pyenv")
            print("   2. Install Python 3.10.6: pyenv install 3.10.6")
            print("   3. Set local version: pyenv local 3.10.6")
            print("   4. Re-run this script to verify")
            
        return False


def main():
    """Main entry point."""
    try:
        compatible = check_python_version()
        sys.exit(0 if compatible else 1)
    except Exception as e:
        print(f"❌ Error checking Python version: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 