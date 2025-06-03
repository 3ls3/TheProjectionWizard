"""
Test Suite for EDA Validation Module
===================================

Basic tests for Team A's EDA validation pipeline components.

Usage:
    python -m pytest tests/test_eda_validation.py
    python tests/test_eda_validation.py
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import json
from pathlib import Path
import sys

# Add the project root to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

from eda_validation import utils, cleaning, ydata_profile
from eda_validation.validation import setup_expectations, run_validation


class TestUtils(unittest.TestCase):
    """Test cases for utility functions."""
    
    def test_detect_file_type(self):
        """Test file type detection."""
        self.assertEqual(utils.detect_file_type("data.csv"), "csv")
        self.assertEqual(utils.detect_file_type("data.xlsx"), "excel")
        self.assertEqual(utils.detect_file_type("data.json"), "json")
        self.assertEqual(utils.detect_file_type("data.parquet"), "parquet")
        self.assertEqual(utils.detect_file_type("data.unknown"), "unknown")
    
    def test_get_dataframe_info(self):
        """Test DataFrame information extraction."""
        # Create test DataFrame
        df = pd.DataFrame({
            'numeric': [1, 2, 3, 4, 5],
            'text': ['a', 'b', 'c', 'd', 'e'],
            'missing': [1, 2, None, 4, None]
        })
        
        info = utils.get_dataframe_info(df)
        
        self.assertEqual(info['shape'], (5, 3))
        self.assertEqual(info['row_count'], 5)
        self.assertEqual(info['column_count'], 3)
        self.assertEqual(info['total_missing_values'], 2)
        
        # Check column-specific info
        self.assertIn('numeric', info['columns'])
        self.assertIn('text', info['columns'])
        self.assertIn('missing', info['columns'])
    
    def test_detect_column_types(self):
        """Test semantic column type detection."""
        df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'category': ['A', 'B', 'A', 'C', 'B'],
            'year': [2020, 2021, 2022, 2023, 2024],
            'binary': [0, 1, 0, 1, 1],
            'numeric': [1.5, 2.7, 3.1, 4.8, 5.2]
        })
        
        types = utils.detect_column_types(df)
        
        self.assertEqual(types['category'], 'categorical')
        self.assertEqual(types['year'], 'year')
        self.assertEqual(types['binary'], 'binary')
        self.assertEqual(types['numeric'], 'numeric')


class TestCleaning(unittest.TestCase):
    """Test cases for data cleaning functions."""
    
    def setUp(self):
        """Set up test data."""
        self.test_df = pd.DataFrame({
            'Clean Column': [1, 2, 3, 4, 5],
            'Missing Data': [1, None, 3, None, 5],
            'UPPERCASE': ['A', 'B', 'C', 'D', 'E'],
            'duplicate_row': [1, 2, 3, 3, 4]
        })
    
    def test_handle_missing_values_drop(self):
        """Test dropping missing values."""
        result = cleaning.handle_missing_values(
            self.test_df, 
            strategy="drop", 
            threshold=0.3
        )
        
        # Should drop rows with missing values
        self.assertLess(len(result), len(self.test_df))
        self.assertEqual(result.isnull().sum().sum(), 0)
    
    def test_handle_missing_values_fill_mean(self):
        """Test filling missing values with mean."""
        result = cleaning.handle_missing_values(
            self.test_df, 
            strategy="fill_mean"
        )
        
        # Missing Data column should have no nulls after mean fill
        self.assertEqual(result['Missing Data'].isnull().sum(), 0)
    
    def test_standardize_column_names(self):
        """Test column name standardization."""
        result = cleaning.standardize_column_names(
            self.test_df, 
            naming_convention="snake_case"
        )
        
        expected_columns = ['clean_column', 'missing_data', 'uppercase', 'duplicate_row']
        self.assertEqual(list(result.columns), expected_columns)
    
    def test_remove_duplicates(self):
        """Test duplicate removal."""
        # Add a duplicate row
        df_with_dups = pd.concat([self.test_df, self.test_df.iloc[[0]]], ignore_index=True)
        
        result = cleaning.remove_duplicates(df_with_dups)
        
        self.assertLess(len(result), len(df_with_dups))
    
    def test_clean_dataframe(self):
        """Test complete cleaning pipeline."""
        result_df, report = cleaning.clean_dataframe(
            self.test_df,
            missing_strategy="drop",
            standardize_columns=True,
            remove_dups=True
        )
        
        # Check that report contains expected keys
        self.assertIn('original_shape', report)
        self.assertIn('final_shape', report)
        self.assertIn('steps_performed', report)
        
        # Check that some processing was done
        self.assertGreater(len(report['steps_performed']), 0)


class TestYDataProfile(unittest.TestCase):
    """Test cases for YData profiling functions."""
    
    def setUp(self):
        """Set up test data."""
        self.test_df = pd.DataFrame({
            'numeric': np.random.randn(100),
            'categorical': np.random.choice(['A', 'B', 'C'], 100),
            'text': [f"text_{i}" for i in range(100)]
        })
    
    def test_generate_profile(self):
        """Test profile generation."""
        # This test will pass even if ydata-profiling is not installed
        profile = ydata_profile.generate_profile(
            self.test_df, 
            title="Test Profile"
        )
        
        # If ydata-profiling is available, profile should not be None
        # If not available, profile will be None and that's expected
        if ydata_profile.YDATA_AVAILABLE:
            self.assertIsNotNone(profile)
        else:
            self.assertIsNone(profile)


class TestValidationSetup(unittest.TestCase):
    """Test cases for validation setup functions."""
    
    def setUp(self):
        """Set up test data."""
        self.test_df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'age': [25, 30, 35, 40, 45],
            'score': [85.5, 90.0, 78.5, 92.0, 88.5]
        })
    
    def test_create_basic_expectation_suite(self):
        """Test basic expectation suite creation."""
        expectations = setup_expectations.create_basic_expectation_suite(
            self.test_df,
            "test_suite"
        )
        
        self.assertIsInstance(expectations, list)
        self.assertGreater(len(expectations), 0)
        
        # Check that we have basic table expectations
        expectation_types = [exp['expectation_type'] for exp in expectations]
        self.assertIn('expect_table_row_count_to_be_between', expectation_types)
        self.assertIn('expect_table_column_count_to_equal', expectation_types)
    
    def test_save_expectation_suite(self):
        """Test saving expectation suite."""
        expectations = [
            {
                "expectation_type": "expect_table_row_count_to_be_between",
                "kwargs": {"min_value": 1, "max_value": 10}
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            success = setup_expectations.save_expectation_suite(
                expectations,
                "test_suite",
                temp_path
            )
            
            self.assertTrue(success)
            self.assertTrue(Path(temp_path).exists())
            
            # Load and verify content
            with open(temp_path, 'r') as f:
                saved_suite = json.load(f)
            
            self.assertEqual(saved_suite['expectation_suite_name'], 'test_suite')
            self.assertEqual(len(saved_suite['expectations']), 1)
            
        finally:
            # Clean up
            Path(temp_path).unlink(missing_ok=True)


class TestValidationRunner(unittest.TestCase):
    """Test cases for validation runner functions."""
    
    def setUp(self):
        """Set up test data."""
        self.test_df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'age': [25, 30, 35, 40, 45]
        })
        
        self.test_suite = {
            "expectation_suite_name": "test_suite",
            "expectations": [
                {
                    "expectation_type": "expect_table_row_count_to_be_between",
                    "kwargs": {"min_value": 1, "max_value": 10}
                },
                {
                    "expectation_type": "expect_column_to_exist",
                    "kwargs": {"column": "id"}
                },
                {
                    "expectation_type": "expect_column_to_exist",
                    "kwargs": {"column": "nonexistent_column"}
                }
            ]
        }
    
    def test_validate_single_expectation(self):
        """Test individual expectation validation."""
        # Test successful expectation
        result = run_validation.validate_single_expectation(
            self.test_df,
            "expect_column_to_exist",
            {"column": "id"}
        )
        
        self.assertTrue(result['success'])
        self.assertEqual(result['expectation_type'], 'expect_column_to_exist')
        
        # Test failing expectation
        result = run_validation.validate_single_expectation(
            self.test_df,
            "expect_column_to_exist",
            {"column": "nonexistent_column"}
        )
        
        self.assertFalse(result['success'])
    
    def test_validate_dataframe_with_suite(self):
        """Test DataFrame validation with a full suite."""
        # Note: This test assumes GX is not available for simplified testing
        success, results = run_validation.validate_dataframe_with_suite(
            self.test_df,
            self.test_suite
        )
        
        if run_validation.GX_AVAILABLE:
            self.assertIsInstance(results, dict)
            self.assertIn('total_expectations', results)
            self.assertIn('successful_expectations', results)
            self.assertIn('failed_expectations', results)
        else:
            # If GX not available, should return error
            self.assertFalse(success)
            self.assertIn('error', results)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def test_create_summary_report(self):
        """Test creating a comprehensive summary report."""
        # Create test data
        df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'value': [10, 20, None, 40, 50]
        })
        
        df_info = utils.get_dataframe_info(df)
        
        cleaning_report = {
            'steps_performed': ['handled_missing_values'],
            'rows_removed': 1
        }
        
        validation_results = {
            'overall_success': True,
            'total_expectations': 5,
            'failed_expectations': 0
        }
        
        report = utils.create_summary_report(
            "test.csv",
            df_info,
            cleaning_report,
            validation_results
        )
        
        self.assertIn('pipeline_run', report)
        self.assertIn('data_summary', report)
        self.assertIn('cleaning_report', report)
        self.assertIn('validation_results', report)
        self.assertIn('quality_score', report)
        
        # Quality score should be reasonable
        self.assertGreaterEqual(report['quality_score'], 0)
        self.assertLessEqual(report['quality_score'], 100)


def main():
    """Run all tests."""
    # Create a test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestUtils,
        TestCleaning,
        TestYDataProfile,
        TestValidationSetup,
        TestValidationRunner,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success/failure for CLI usage
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    exit(main()) 