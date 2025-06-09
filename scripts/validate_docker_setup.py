#!/usr/bin/env python3
"""
Validation script for Docker setup.
Checks that all required files are present and properly configured.
"""

import os
import sys
from pathlib import Path

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_status(message, status='info'):
    """Print colored status message."""
    colors = {
        'success': GREEN,
        'error': RED,
        'warning': YELLOW,
        'info': BLUE
    }
    color = colors.get(status, BLUE)
    print(f"{color}[{status.upper()}]{RESET} {message}")

def check_file_exists(filepath, description):
    """Check if a file exists and print status."""
    if os.path.exists(filepath):
        print_status(f"✓ {description}: {filepath}", 'success')
        return True
    else:
        print_status(f"✗ {description}: {filepath} (missing)", 'error')
        return False

def check_docker_setup():
    """Validate Docker setup files."""
    print_status("🐳 Validating Docker Setup for The Projection Wizard", 'info')
    print()
    
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    checks = []
    
    # Check root level Docker files
    checks.append(check_file_exists("Dockerfile", "Backend Dockerfile"))
    checks.append(check_file_exists("docker-compose.yml", "Docker Compose config"))
    checks.append(check_file_exists(".dockerignore", "Root .dockerignore"))
    checks.append(check_file_exists(".env", "Environment config"))
    
    # Check frontend Docker files
    checks.append(check_file_exists("frontend/Dockerfile", "Frontend Dockerfile"))
    checks.append(check_file_exists("frontend/.dockerignore", "Frontend .dockerignore"))
    checks.append(check_file_exists("frontend/package.json", "Frontend package.json"))
    checks.append(check_file_exists("frontend/vite.config.js", "Vite config"))
    
    # Check frontend source files
    checks.append(check_file_exists("frontend/index.html", "Frontend HTML"))
    checks.append(check_file_exists("frontend/src/main.jsx", "React entry point"))
    checks.append(check_file_exists("frontend/src/App.jsx", "React App component"))
    checks.append(check_file_exists("frontend/src/App.css", "App styles"))
    checks.append(check_file_exists("frontend/src/index.css", "Base styles"))
    
    # Check API files
    checks.append(check_file_exists("api/main.py", "FastAPI main"))
    
    print()
    
    # Summary
    passed = sum(checks)
    total = len(checks)
    
    if passed == total:
        print_status(f"🎉 All {total} validation checks passed!", 'success')
        print_status("You can now run: docker-compose up --build", 'info')
        return True
    else:
        failed = total - passed
        print_status(f"❌ {failed} out of {total} checks failed", 'error')
        print_status("Please fix the missing files before running Docker", 'warning')
        return False

def check_docker_commands():
    """Check if Docker commands are available."""
    print_status("🔧 Checking Docker installation", 'info')
    
    docker_available = os.system("docker --version > /dev/null 2>&1") == 0
    compose_available = os.system("docker-compose --version > /dev/null 2>&1") == 0
    
    if docker_available:
        print_status("✓ Docker is installed", 'success')
    else:
        print_status("✗ Docker is not available", 'error')
    
    if compose_available:
        print_status("✓ Docker Compose is installed", 'success')
    else:
        print_status("✗ Docker Compose is not available", 'error')
    
    return docker_available and compose_available

if __name__ == "__main__":
    print_status("The Projection Wizard - Docker Setup Validator", 'info')
    print("=" * 60)
    print()
    
    # Check Docker availability
    docker_ok = check_docker_commands()
    print()
    
    # Check file setup
    files_ok = check_docker_setup()
    print()
    
    if docker_ok and files_ok:
        print_status("🚀 Ready for Docker deployment!", 'success')
        print()
        print("Next steps:")
        print("1. docker-compose up --build")
        print("2. Open http://localhost:3000 (frontend)")
        print("3. Open http://localhost:8000/docs (API docs)")
        sys.exit(0)
    else:
        print_status("❌ Setup incomplete", 'error')
        if not docker_ok:
            print("- Install Docker and Docker Compose")
        if not files_ok:
            print("- Fix missing files listed above")
        sys.exit(1) 