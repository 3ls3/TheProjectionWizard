#!/usr/bin/env python3
"""
Quick test script for validation functionality
"""

import pandas as pd
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from eda_validation.validation import setup_expectations, run_validation

def test_validation_pipeline():
    """Test the complete validation pipeline with sample data"""
    
    # Load sample data
    sample_path = "data/mock/sample_data.csv"
    if not Path(sample_path).exists():
        print(f"❌ Sample data not found at {sample_path}")
        return False
    
    try:
        print("🔍 Loading sample data...")
        df = pd.read_csv(sample_path)
        print(f"✅ Loaded data with shape: {df.shape}")
        print(f"📋 Columns: {list(df.columns)}")
        print(f"🔢 Data types: {df.dtypes.to_dict()}")
        
        # Test expectation suite creation
        print("\n🎯 Creating expectation suite...")
        expectations = setup_expectations.create_typed_expectation_suite(df, "test_suite")
        print(f"✅ Created {len(expectations)} expectations")
        
        # Create expectation suite format
        expectation_suite = {
            "expectation_suite_name": "test_suite",
            "expectations": expectations
        }
        
        # Test validation
        print("\n🧪 Running validation...")
        success, results = run_validation.validate_dataframe_with_suite(df, expectation_suite)
        
        print(f"\n📊 Validation Results:")
        print(f"   Overall Success: {'✅ PASSED' if success else '❌ FAILED'}")
        print(f"   Total Expectations: {results.get('total_expectations', 0)}")
        print(f"   Successful: {results.get('successful_expectations', 0)}")
        print(f"   Failed: {results.get('failed_expectations', 0)}")
        
        if not success:
            print("\n❌ Failed Expectations:")
            for result in results.get('expectation_results', []):
                if not result.get('success', True):
                    print(f"   - {result.get('expectation_type', 'Unknown')}: {result.get('details', 'No details')}")
        
        return success
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Testing Great Expectations Validation Pipeline")
    print("=" * 50)
    
    success = test_validation_pipeline()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! Validation pipeline is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the validation logic.")
    
    exit(0 if success else 1) 