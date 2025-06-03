"""
Great Expectations Validation Runner
===================================

Handles running validation checks against user data using Great Expectations.

Usage:
    # As a module
    from eda_validation.validation.run_validation import validate_dataframe
    
    # As CLI
    python eda_validation/validation/run_validation.py data/raw/sample.csv
"""

import pandas as pd
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import great_expectations as gx
    GX_AVAILABLE = True
except ImportError:
    logger.warning("Great Expectations not installed. Install with: pip install great_expectations")
    GX_AVAILABLE = False
except Exception as e:
    logger.warning(f"Great Expectations import failed: {str(e)}")
    GX_AVAILABLE = False


def load_expectation_suite(suite_path: str) -> Optional[Dict[str, Any]]:
    """
    Load expectation suite from JSON file.
    
    Args:
        suite_path (str): Path to expectation suite JSON file
        
    Returns:
        Dict containing expectation suite or None if failed
        
    Example:
        >>> suite = load_expectation_suite("expectations/my_suite.json")
    """
    try:
        logger.info(f"Loading expectation suite from: {suite_path}")
        
        with open(suite_path, 'r') as f:
            suite_config = json.load(f)
        
        logger.info(f"Loaded suite: {suite_config.get('expectation_suite_name', 'Unknown')}")
        logger.info(f"Number of expectations: {len(suite_config.get('expectations', []))}")
        
        return suite_config
        
    except Exception as e:
        logger.error(f"Error loading expectation suite: {str(e)}")
        return None


def validate_dataframe_with_suite(
    df: pd.DataFrame,
    expectation_suite: Dict[str, Any]
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate DataFrame against expectation suite.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        expectation_suite (Dict): Expectation suite configuration
        
    Returns:
        Tuple of (success, validation_results)
        
    Example:
        >>> success, results = validate_dataframe_with_suite(df, suite)
    """
    if not GX_AVAILABLE:
        logger.warning("Great Expectations not available. Using simplified validation.")
        # Continue with simplified validation instead of failing
    
    try:
        logger.info("Starting data validation")
        logger.info(f"DataFrame shape: {df.shape}")
        
        # Initialize validation results
        validation_results = {
            "suite_name": expectation_suite.get("expectation_suite_name", "Unknown"),
            "total_expectations": len(expectation_suite.get("expectations", [])),
            "successful_expectations": 0,
            "failed_expectations": 0,
            "expectation_results": [],
            "overall_success": True
        }
        
        expectations = expectation_suite.get("expectations", [])
        
        for i, expectation in enumerate(expectations):
            expectation_type = expectation.get("expectation_type")
            kwargs = expectation.get("kwargs", {})
            
            logger.info(f"Running expectation {i+1}/{len(expectations)}: {expectation_type}")
            
            # Simulate validation (replace with actual GX validation)
            result = validate_single_expectation(df, expectation_type, kwargs)
            
            validation_results["expectation_results"].append(result)
            
            if result["success"]:
                validation_results["successful_expectations"] += 1
            else:
                validation_results["failed_expectations"] += 1
                validation_results["overall_success"] = False
        
        success_rate = validation_results["successful_expectations"] / validation_results["total_expectations"]
        validation_results["success_rate"] = success_rate
        
        logger.info(f"Validation completed. Success rate: {success_rate:.2%}")
        
        return validation_results["overall_success"], validation_results
        
    except Exception as e:
        logger.error(f"Error during validation: {str(e)}")
        return False, {"error": str(e)}


def validate_single_expectation(
    df: pd.DataFrame,
    expectation_type: str,
    kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate a single expectation against the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        expectation_type (str): Type of expectation
        kwargs (Dict): Expectation parameters
        
    Returns:
        Dict containing validation result
        
    Example:
        >>> result = validate_single_expectation(df, "expect_column_to_exist", {"column": "age"})
    """
    result = {
        "expectation_type": expectation_type,
        "kwargs": kwargs,
        "success": False,
        "details": ""
    }
    
    try:
        # Basic expectation validations (simplified implementation)
        if expectation_type == "expect_table_row_count_to_be_between":
            min_val = kwargs.get("min_value", 0)
            max_val = kwargs.get("max_value", float('inf'))
            row_count = len(df)
            result["success"] = min_val <= row_count <= max_val
            result["details"] = f"Row count: {row_count}, Expected: {min_val}-{max_val}"
            
        elif expectation_type == "expect_table_column_count_to_equal":
            expected_count = kwargs.get("value")
            actual_count = len(df.columns)
            result["success"] = actual_count == expected_count
            result["details"] = f"Column count: {actual_count}, Expected: {expected_count}"
            
        elif expectation_type == "expect_column_to_exist":
            column = kwargs.get("column")
            result["success"] = column in df.columns
            result["details"] = f"Column '{column}' exists: {result['success']}"
            
        elif expectation_type == "expect_column_values_to_not_be_null":
            column = kwargs.get("column")
            mostly = kwargs.get("mostly", 1.0)
            if column in df.columns:
                non_null_rate = df[column].notna().mean()
                result["success"] = non_null_rate >= mostly
                result["details"] = f"Non-null rate: {non_null_rate:.2%}, Required: {mostly:.2%}"
            else:
                result["success"] = False
                result["details"] = f"Column '{column}' not found"
                
        elif expectation_type == "expect_column_values_to_be_between":
            column = kwargs.get("column")
            min_val = kwargs.get("min_value")
            max_val = kwargs.get("max_value")
            if column in df.columns:
                col_dtype = str(df[column].dtype).lower()
                # Handle various numeric types including nullable integers
                if any(x in col_dtype for x in ['int', 'float', 'number']):
                    non_null_values = df[column].dropna()
                    if len(non_null_values) > 0:
                        values_in_range = non_null_values.between(min_val, max_val).all()
                        result["success"] = values_in_range
                        actual_min = non_null_values.min()
                        actual_max = non_null_values.max()
                        result["details"] = f"Range: {actual_min}-{actual_max}, Expected: {min_val}-{max_val}"
                    else:
                        result["success"] = True
                        result["details"] = "No non-null values to validate"
                elif 'datetime' in col_dtype:
                    # Handle datetime ranges
                    non_null_values = df[column].dropna()
                    if len(non_null_values) > 0:
                        min_date = pd.to_datetime(min_val)
                        max_date = pd.to_datetime(max_val)
                        values_in_range = (non_null_values >= min_date) & (non_null_values <= max_date)
                        result["success"] = values_in_range.all()
                        actual_min = non_null_values.min()
                        actual_max = non_null_values.max()
                        result["details"] = f"Date range: {actual_min}-{actual_max}, Expected: {min_date}-{max_date}"
                    else:
                        result["success"] = True
                        result["details"] = "No non-null datetime values to validate"
                else:
                    result["success"] = False
                    result["details"] = f"Column '{column}' is not numeric or datetime (type: {col_dtype})"
            else:
                result["success"] = False
                result["details"] = f"Column '{column}' not found"
                
        elif expectation_type == "expect_column_values_to_be_in_set":
            column = kwargs.get("column")
            value_set = set(kwargs.get("value_set", []))
            if column in df.columns:
                unique_values = set(df[column].dropna().unique())
                values_in_set = unique_values.issubset(value_set)
                result["success"] = values_in_set
                unexpected_values = unique_values - value_set
                result["details"] = f"Unexpected values: {list(unexpected_values) if unexpected_values else 'None'}"
            else:
                result["success"] = False
                result["details"] = f"Column '{column}' not found"
                
        elif expectation_type == "expect_column_value_lengths_to_be_between":
            column = kwargs.get("column")
            min_length = kwargs.get("min_value", 0)
            max_length = kwargs.get("max_value", float('inf'))
            if column in df.columns:
                lengths = df[column].dropna().astype(str).str.len()
                if len(lengths) > 0:
                    lengths_in_range = lengths.between(min_length, max_length).all()
                    result["success"] = lengths_in_range
                    actual_min = lengths.min()
                    actual_max = lengths.max()
                    result["details"] = f"Length range: {actual_min}-{actual_max}, Expected: {min_length}-{max_length}"
                else:
                    result["success"] = True
                    result["details"] = "No non-null values to validate"
            else:
                result["success"] = False
                result["details"] = f"Column '{column}' not found"
                
        else:
            result["success"] = False
            result["details"] = f"Unsupported expectation type: {expectation_type}"
            
    except Exception as e:
        result["success"] = False
        result["details"] = f"Error during validation: {str(e)}"
    
    return result


def save_validation_results(
    validation_results: Dict[str, Any],
    output_path: str
) -> bool:
    """
    Save validation results to JSON file.
    
    Args:
        validation_results (Dict): Validation results
        output_path (str): Path to save results
        
    Returns:
        bool: True if successful, False otherwise
        
    Example:
        >>> save_validation_results(results, "validation_report.json")
    """
    try:
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving validation results to: {output_path}")
        
        with open(output_path, 'w') as f:
            json.dump(validation_results, f, indent=2, default=str)
        
        logger.info("Validation results saved successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error saving validation results: {str(e)}")
        return False


def validate_csv_file(
    input_path: str,
    suite_path: str,
    output_path: Optional[str] = None
) -> bool:
    """
    Complete pipeline: load CSV, load expectation suite, validate, and save results.
    
    Args:
        input_path (str): Path to input CSV file
        suite_path (str): Path to expectation suite JSON file
        output_path (str, optional): Path to save validation results
        
    Returns:
        bool: True if validation passed, False otherwise
    """
    try:
        # Load data
        logger.info(f"Loading data from: {input_path}")
        df = pd.read_csv(input_path)
        logger.info(f"Loaded dataset with shape: {df.shape}")
        
        # Load expectation suite
        expectation_suite = load_expectation_suite(suite_path)
        if expectation_suite is None:
            return False
        
        # Run validation
        success, validation_results = validate_dataframe_with_suite(df, expectation_suite)
        
        # Determine output path
        if output_path is None:
            input_name = Path(input_path).stem
            output_path = f"data/processed/{input_name}_validation_results.json"
        
        # Save validation results
        save_validation_results(validation_results, output_path)
        
        # Print summary
        if success:
            logger.info("✅ All validations passed!")
        else:
            failed_count = validation_results.get("failed_expectations", 0)
            total_count = validation_results.get("total_expectations", 0)
            logger.warning(f"❌ {failed_count}/{total_count} validations failed")
        
        return success
        
    except Exception as e:
        logger.error(f"Error in validate_csv_file: {str(e)}")
        return False


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run data validation using Great Expectations suite"
    )
    parser.add_argument(
        "input_path",
        help="Path to input CSV file"
    )
    parser.add_argument(
        "-s", "--suite",
        required=True,
        help="Path to expectation suite JSON file"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output path for validation results (default: auto-generated)"
    )
    
    args = parser.parse_args()
    
    # Run validation
    success = validate_csv_file(
        input_path=args.input_path,
        suite_path=args.suite,
        output_path=args.output
    )
    
    if success:
        print("✅ Data validation completed successfully")
    else:
        print("❌ Data validation failed")
        exit(1)


if __name__ == "__main__":
    main() 