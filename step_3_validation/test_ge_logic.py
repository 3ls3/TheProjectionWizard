#!/usr/bin/env python3
"""
Unit tests for step_3_validation/ge_logic.py
Tests the Great Expectations suite generation and validation functions.
"""

import pytest
import pandas as pd
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from step_3_validation.ge_logic import generate_ge_suite_from_metadata, run_ge_validation_on_dataframe
from common.schemas import TargetInfo


class TestGenerateGESuiteFromMetadata:
    """Test the generate_ge_suite_from_metadata function."""
    
    def test_basic_suite_generation(self):
        """Test basic GE suite generation with mock feature schemas."""
        # Mock feature schemas
        feature_schemas = {
            'numeric_col': {
                'dtype': 'int64',
                'encoding_role': 'numeric-continuous',
                'source': 'user_confirmed'
            },
            'categorical_col': {
                'dtype': 'object',
                'encoding_role': 'categorical-nominal',
                'source': 'system_defaulted'
            },
            'boolean_col': {
                'dtype': 'bool',
                'encoding_role': 'boolean',
                'source': 'user_confirmed'
            }
        }
        
        # Mock target info
        target_info = TargetInfo(
            name='target_col',
            task_type='classification',
            ml_type='binary_01'
        )
        
        df_columns = ['numeric_col', 'categorical_col', 'boolean_col', 'target_col']
        
        # Generate suite
        ge_suite = generate_ge_suite_from_metadata(
            feature_schemas=feature_schemas,
            target_info=target_info,
            df_columns=df_columns,
            run_id="test_run"
        )
        
        # Verify suite structure
        assert isinstance(ge_suite, dict)
        assert "expectation_suite_name" in ge_suite
        assert "expectations" in ge_suite
        assert "data_asset_type" in ge_suite
        assert ge_suite["data_asset_type"] == "Dataset"
        
        # Verify suite name
        assert "test_run" in ge_suite["expectation_suite_name"]
        
        # Verify expectations exist
        expectations = ge_suite["expectations"]
        assert len(expectations) > 0
        
        # Check for table-level expectations
        table_expectations = [exp for exp in expectations if "table" in exp["expectation_type"]]
        assert len(table_expectations) >= 2  # columns and row count
        
        # Check for column-level expectations
        column_expectations = [exp for exp in expectations if "column" in exp["expectation_type"]]
        assert len(column_expectations) >= len(df_columns)  # At least one per column
        
        # Verify target-specific expectations for binary_01
        target_expectations = [exp for exp in expectations 
                             if exp.get("kwargs", {}).get("column") == "target_col"]
        assert len(target_expectations) > 0
        
        print(f"✓ Generated {len(expectations)} expectations for {len(df_columns)} columns")
    
    def test_missing_column_handling(self):
        """Test handling when feature_schemas is missing columns from df."""
        # Feature schemas missing one column
        feature_schemas = {
            'existing_col': {
                'dtype': 'int64',
                'encoding_role': 'numeric-continuous',
                'source': 'user_confirmed'
            }
        }
        
        df_columns = ['existing_col', 'missing_col']
        
        # Should handle gracefully
        ge_suite = generate_ge_suite_from_metadata(
            feature_schemas=feature_schemas,
            target_info=None,
            df_columns=df_columns,
            run_id="test_run"
        )
        
        # Should still generate basic expectations
        assert len(ge_suite["expectations"]) > 0
        
        # Should have table-level expectations
        table_expectations = [exp for exp in ge_suite["expectations"] 
                            if "table" in exp["expectation_type"]]
        assert len(table_expectations) >= 1
    
    def test_no_target_info(self):
        """Test suite generation without target info."""
        feature_schemas = {
            'col1': {'dtype': 'float64', 'encoding_role': 'numeric-continuous', 'source': 'user_confirmed'}
        }
        
        df_columns = ['col1']
        
        ge_suite = generate_ge_suite_from_metadata(
            feature_schemas=feature_schemas,
            target_info=None,
            df_columns=df_columns,
            run_id="test_run"
        )
        
        # Should still work without target info
        assert len(ge_suite["expectations"]) > 0
        
        # Should not have target-specific expectations
        expectations_with_target = [exp for exp in ge_suite["expectations"] 
                                  if "target" in str(exp).lower()]
        assert len(expectations_with_target) == 0


class TestRunGEValidationOnDataframe:
    """Test the run_ge_validation_on_dataframe function."""
    
    def test_successful_validation(self):
        """Test successful validation with a simple DataFrame and suite."""
        # Create simple test data
        df = pd.DataFrame({
            'numeric_col': [1, 2, 3, 4, 5],
            'string_col': ['a', 'b', 'c', 'd', 'e']
        })
        
        # Create simple GE suite
        ge_suite = {
            "expectation_suite_name": "test_suite",
            "ge_cloud_id": None,
            "expectations": [
                {
                    "expectation_type": "expect_table_row_count_to_be_between",
                    "kwargs": {"min_value": 1, "max_value": None}
                },
                {
                    "expectation_type": "expect_column_to_exist",
                    "kwargs": {"column": "numeric_col"}
                },
                {
                    "expectation_type": "expect_column_values_to_be_of_type",
                    "kwargs": {"column": "numeric_col", "type_": "int"}
                }
            ],
            "data_asset_type": "Dataset",
            "meta": {"great_expectations_version": "test"}
        }
        
        # Run validation
        results = run_ge_validation_on_dataframe(df, ge_suite, run_id="test_run")
        
        # Verify results structure
        assert isinstance(results, dict)
        assert "statistics" in results or "success" in results  # Different GE versions
        
        # Should have metadata
        assert "meta" in results
        assert results["meta"]["validation_run_id"] == "test_run"
        assert "dataframe_shape" in results["meta"]
        assert results["meta"]["dataframe_shape"] == [5, 2]
    
    @patch('great_expectations.from_pandas')
    def test_validation_error_handling(self, mock_from_pandas):
        """Test error handling when GE validation fails."""
        # Mock GE to raise an error
        mock_from_pandas.side_effect = Exception("GE validation failed")
        
        df = pd.DataFrame({'col': [1, 2, 3]})
        ge_suite = {"expectations": []}
        
        # Should return error result
        results = run_ge_validation_on_dataframe(df, ge_suite, run_id="test_run")
        
        # Should have error structure
        assert results["success"] is False
        assert "error" in results["meta"]
        assert "GE validation failed" in results["meta"]["error"]
    
    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame()
        
        ge_suite = {
            "expectation_suite_name": "test_suite",
            "expectations": [],
            "data_asset_type": "Dataset",
            "meta": {}
        }
        
        # Should handle gracefully (may pass or fail depending on expectations)
        results = run_ge_validation_on_dataframe(df, ge_suite, run_id="test_run")
        
        # Should return a result structure
        assert isinstance(results, dict)
        assert "meta" in results


def run_unit_tests():
    """Run all unit tests."""
    print("Running GE Logic Unit Tests...")
    
    # Test suite generation
    test_gen = TestGenerateGESuiteFromMetadata()
    test_gen.test_basic_suite_generation()
    test_gen.test_missing_column_handling()
    test_gen.test_no_target_info()
    print("✓ Suite generation tests passed")
    
    # Test validation execution
    test_val = TestRunGEValidationOnDataframe()
    test_val.test_successful_validation()
    test_val.test_validation_error_handling()
    test_val.test_empty_dataframe_handling()
    print("✓ Validation execution tests passed")
    
    print("✅ All GE logic unit tests passed!")


if __name__ == "__main__":
    run_unit_tests() 