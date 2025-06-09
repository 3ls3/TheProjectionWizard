"""
Quick test script to demonstrate The Projection Wizard testing framework.

This script runs a simple test of the pipeline stages to validate the testing framework.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from tests.fixtures.fixture_generator import TestFixtureGenerator, create_all_stage_fixtures
from tests.unit.stage_tests.test_validation import run_validation_test
from tests.unit.stage_tests.test_prep import run_prep_test
from tests.unit.stage_tests.test_automl import run_automl_test


def run_quick_test():
    """Run a quick test of the pipeline stages."""
    print("🚀 Running quick test of The Projection Wizard pipeline...")
    
    # Generate a unique test run ID
    test_run_id = f"quick_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create test directory
    test_dir = Path("data/test_runs") / test_run_id
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate test fixtures for all stages
    print("\n🔧 Generating test fixtures...")
    fixtures = create_all_stage_fixtures("classification")
    print(f"✅ Generated fixtures for stages: {list(fixtures.keys())}")
    
    # Run validation stage test
    print("\n📋 Testing Validation Stage...")
    validation_result = run_validation_test(fixtures["validation"])
    print(f"Validation Test Result: {'✅ Success' if validation_result.success else '❌ Failed'}")
    if not validation_result.success:
        print(f"Error: {validation_result.error_message}")
    
    # Run prep stage test
    print("\n🧹 Testing Prep Stage...")
    prep_result = run_prep_test(fixtures["prep"])
    print(f"Prep Test Result: {'✅ Success' if prep_result.success else '❌ Failed'}")
    if not prep_result.success:
        print(f"Error: {prep_result.error_message}")
    
    # Run AutoML stage test
    print("\n🤖 Testing AutoML Stage...")
    automl_result = run_automl_test(fixtures["automl"])
    print(f"AutoML Test Result: {'✅ Success' if automl_result.success else '❌ Failed'}")
    if not automl_result.success:
        print(f"Error: {automl_result.error_message}")
    
    # Print summary
    print("\n📊 Test Summary:")
    print(f"Validation Stage: {'✅ Passed' if validation_result.success else '❌ Failed'}")
    print(f"Prep Stage: {'✅ Passed' if prep_result.success else '❌ Failed'}")
    print(f"AutoML Stage: {'✅ Passed' if automl_result.success else '❌ Failed'}")
    
    # Check if all tests passed
    all_passed = all([
        validation_result.success,
        prep_result.success,
        automl_result.success
    ])
    
    if all_passed:
        print("\n🎉 All tests passed successfully!")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    run_quick_test() 