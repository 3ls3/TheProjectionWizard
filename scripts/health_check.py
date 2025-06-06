#!/usr/bin/env python3
"""
Health check script for The Projection Wizard.
Validates project structure, dependencies, and core functionality.
"""

import sys
import subprocess
from pathlib import Path
from importlib import import_module

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_project_structure():
    """Check that all expected directories and files exist."""
    print("🔍 Checking project structure...")
    
    required_dirs = [
        "common", "data", "data/fixtures", "data/runs", 
        "step_1_ingest", "step_2_schema", "step_3_validation",
        "step_4_prep", "step_5_automl", "step_6_explain",
        "ui", "scripts"
    ]
    
    required_files = [
        "app.py", "README.md", "requirements.txt", "pyproject.toml",
        "common/__init__.py", "common/constants.py", "common/schemas.py",
        "common/storage.py", "common/logger.py", "common/utils.py",
        "scripts/run_pipeline_cli.py", "scripts/test_cli_runner.py",
        "data/fixtures/sample_classification.csv", "data/fixtures/sample_regression.csv"
    ]
    
    missing_dirs = []
    missing_files = []
    
    for dir_path in required_dirs:
        if not (project_root / dir_path).exists():
            missing_dirs.append(dir_path)
    
    for file_path in required_files:
        if not (project_root / file_path).exists():
            missing_files.append(file_path)
    
    if missing_dirs or missing_files:
        print("❌ Project structure issues found:")
        for dir_path in missing_dirs:
            print(f"   Missing directory: {dir_path}")
        for file_path in missing_files:
            print(f"   Missing file: {file_path}")
        return False
    
    print("✅ Project structure looks good")
    return True


def check_imports():
    """Check that core modules can be imported."""
    print("🔍 Checking core module imports...")
    
    modules_to_test = [
        "common.constants",
        "common.schemas", 
        "common.storage",
        "common.logger",
        "common.utils",
        "step_1_ingest.ingest_logic",
        "step_2_schema.target_definition_logic",
        "step_2_schema.feature_definition_logic"
    ]
    
    failed_imports = []
    
    for module_name in modules_to_test:
        try:
            import_module(module_name)
        except ImportError as e:
            failed_imports.append((module_name, str(e)))
    
    if failed_imports:
        print("❌ Import issues found:")
        for module_name, error in failed_imports:
            print(f"   {module_name}: {error}")
        return False
    
    print("✅ All core modules import successfully")
    return True


def check_dependencies():
    """Check that required dependencies are available."""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        "pandas", "numpy", "sklearn", "pycaret",
        "great_expectations", "ydata_profiling", "shap",
        "streamlit", "pydantic", "joblib"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing dependencies:")
        for package in missing_packages:
            print(f"   {package}")
        print("\n💡 Run: pip install -r requirements.txt")
        return False
    
    print("✅ All required dependencies available")
    return True


def check_sample_data():
    """Check that sample data files are valid."""
    print("🔍 Checking sample data...")
    
    try:
        import pandas as pd
        
        # Check classification sample
        clf_path = project_root / "data/fixtures/sample_classification.csv"
        clf_df = pd.read_csv(clf_path)
        if clf_df.empty:
            print("❌ Classification sample is empty")
            return False
        
        # Check regression sample  
        reg_path = project_root / "data/fixtures/sample_regression.csv"
        reg_df = pd.read_csv(reg_path)
        if reg_df.empty:
            print("❌ Regression sample is empty")
            return False
        
        print("✅ Sample data files are valid")
        return True
        
    except Exception as e:
        print(f"❌ Error checking sample data: {e}")
        return False


def check_common_utilities():
    """Test basic functionality of common utilities."""
    print("🔍 Testing common utilities...")
    
    try:
        from common.utils import generate_run_id
        from common.storage import get_run_dir
        from common.constants import INGEST_STAGE
        
        # Test run ID generation
        run_id = generate_run_id()
        if not run_id or len(run_id) < 10:
            print("❌ Run ID generation failed")
            return False
        
        # Test directory creation (but don't leave it)
        run_dir = get_run_dir(run_id)
        if run_dir.exists():
            # Clean up test directory
            import shutil
            shutil.rmtree(run_dir)
        
        print("✅ Common utilities working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Error testing common utilities: {e}")
        return False


def check_git_status():
    """Check git repository status and cleanliness."""
    print("🔍 Checking git status...")
    
    try:
        # Check if we're in a git repository
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=project_root
        )
        
        if result.returncode != 0:
            print("❌ Not in a git repository or git not available")
            return False
        
        # Check for untracked or modified files that might need attention
        status_output = result.stdout.strip()
        if status_output:
            untracked = [line for line in status_output.split('\n') if line.startswith('??')]
            modified = [line for line in status_output.split('\n') if line.startswith(' M') or line.startswith('M ')]
            
            if untracked:
                print("ℹ️ Untracked files found:")
                for line in untracked:
                    print(f"   {line}")
            
            if modified:
                print("ℹ️ Modified files found:")
                for line in modified:
                    print(f"   {line}")
        
        print("✅ Git status checked")
        return True
        
    except Exception as e:
        print(f"ℹ️ Could not check git status: {e}")
        return True  # Non-critical


def main():
    """Run all health checks."""
    print("🏥 The Projection Wizard - Health Check")
    print("=" * 50)
    
    checks = [
        ("Project Structure", check_project_structure),
        ("Core Imports", check_imports),
        ("Dependencies", check_dependencies),
        ("Sample Data", check_sample_data),
        ("Common Utilities", check_common_utilities),
        ("Git Status", check_git_status),
    ]
    
    results = []
    
    for check_name, check_func in checks:
        print(f"\n{check_name}:")
        try:
            success = check_func()
            results.append((check_name, success))
        except Exception as e:
            print(f"❌ {check_name} failed with exception: {e}")
            results.append((check_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Health Check Summary:")
    
    passed = 0
    total = len(results)
    
    for check_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}: {check_name}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 Project is healthy and ready to use!")
        print("\n💡 Next steps:")
        print("   - Run: streamlit run app.py")
        print("   - Or test CLI: python scripts/test_cli_runner.py")
        return 0
    else:
        print("⚠️ Some issues found. Please address them before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 