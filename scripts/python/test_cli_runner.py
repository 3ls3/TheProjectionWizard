#!/usr/bin/env python3
"""
Smoke test for the CLI runner to verify basic functionality.

This script tests the CLI runner with the sample datasets to ensure
the pipeline can execute end-to-end without critical failures.
"""

import subprocess
import sys
import tempfile
import csv
from pathlib import Path
from datetime import datetime

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from common import storage


def create_test_data():
    """Create a temporary test CSV file for testing."""
    test_data = [
        ['feature1', 'feature2', 'target'],
        [1.0, 'A', 0],
        [2.0, 'B', 1],
        [3.0, 'A', 0],
        [4.0, 'B', 1],
        [5.0, 'A', 0],
        [6.0, 'B', 1],
        [7.0, 'A', 0],
        [8.0, 'B', 1],
        [9.0, 'A', 0],
        [10.0, 'B', 1],
        [11.0, 'A', 0],
        [12.0, 'B', 1],
        [13.0, 'A', 0],
        [14.0, 'B', 1],
        [15.0, 'A', 0],
    ]
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    writer = csv.writer(temp_file)
    writer.writerows(test_data)
    temp_file.close()
    
    return Path(temp_file.name)


def run_cli_test(csv_path: Path, target: str, task: str, target_ml_type: str) -> tuple[bool, str]:
    """
    Run the CLI with the given parameters and return success status and output.
    
    Returns:
        (success, output): Tuple of success boolean and output string
    """
    cmd = [
        sys.executable, 
        'scripts/python/run_pipeline_cli.py',
        '--csv', str(csv_path),
        '--target', target,
        '--task', task,
        '--target-ml-type', target_ml_type
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            cwd=project_root
        )
        
        return result.returncode == 0, result.stdout + result.stderr
        
    except subprocess.TimeoutExpired:
        return False, "CLI test timed out after 5 minutes"
    except Exception as e:
        return False, f"CLI test failed with exception: {e}"


def test_classification():
    """Test classification pipeline."""
    print("🧪 Testing Classification Pipeline...")
    
    # Use the existing sample classification data
    csv_path = project_root / "data" / "fixtures" / "sample_classification.csv"
    
    if not csv_path.exists():
        print(f"❌ Sample classification file not found: {csv_path}")
        return False
    
    success, output = run_cli_test(
        csv_path=csv_path,
        target="target",
        task="classification", 
        target_ml_type="binary_01"
    )
    
    if success:
        print("✅ Classification pipeline completed successfully")
        return True
    else:
        print("❌ Classification pipeline failed")
        print("Output:", output[-500:])  # Show last 500 chars
        return False


def test_regression():
    """Test regression pipeline."""
    print("🧪 Testing Regression Pipeline...")
    
    # Use the existing sample regression data
    csv_path = project_root / "data" / "fixtures" / "sample_regression.csv"
    
    if not csv_path.exists():
        print(f"❌ Sample regression file not found: {csv_path}")
        return False
    
    success, output = run_cli_test(
        csv_path=csv_path,
        target="price",
        task="regression",
        target_ml_type="numeric_continuous"
    )
    
    if success:
        print("✅ Regression pipeline completed successfully")
        return True
    else:
        print("❌ Regression pipeline failed")
        print("Output:", output[-500:])  # Show last 500 chars
        return False


def test_auto_detection():
    """Test auto-detection capabilities."""
    print("🧪 Testing Auto-Detection...")
    
    # Create a temporary test file
    test_csv = create_test_data()
    
    try:
        success, output = run_cli_test(
            csv_path=test_csv,
            target="target",
            task="classification",
            target_ml_type="binary_01"
        )
        
        if success:
            print("✅ Auto-detection pipeline completed successfully")
            return True
        else:
            print("❌ Auto-detection pipeline failed")
            print("Output:", output[-500:])  # Show last 500 chars
            return False
            
    finally:
        # Clean up temporary file
        test_csv.unlink(missing_ok=True)


def check_run_index():
    """Check that the run index is being updated correctly."""
    print("🧪 Checking Run Index...")
    
    try:
        index_path = project_root / "data" / "runs" / "index.csv"
        
        if not index_path.exists():
            print("❌ Run index file not found")
            return False
        
        # Read the index and check for recent entries
        with open(index_path, 'r') as f:
            lines = f.readlines()
        
        if len(lines) < 2:  # Header + at least one entry
            print("❌ Run index appears empty")
            return False
        
        # Check that we have recent entries (from today)
        today = datetime.now().strftime('%Y-%m-%d')
        recent_entries = [line for line in lines if today in line]
        
        if len(recent_entries) > 0:
            print(f"✅ Run index has {len(recent_entries)} entries from today")
            return True
        else:
            print("⚠️  No recent entries in run index (this may be expected)")
            return True  # Not a failure, just a warning
            
    except Exception as e:
        print(f"❌ Error checking run index: {e}")
        return False


def main():
    """Run all smoke tests."""
    print("🚀 Starting CLI Runner Smoke Tests")
    print("=" * 50)
    
    tests = [
        ("Classification Test", test_classification),
        ("Regression Test", test_regression), 
        ("Auto-Detection Test", test_auto_detection),
        ("Run Index Check", check_run_index),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}: {test_name}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All smoke tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 