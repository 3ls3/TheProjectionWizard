"""
Great Expectations Setup Module
==============================

Handles setup and configuration of Great Expectations expectations for data validation.

Usage:
    # As a module
    from eda_validation.validation.setup_expectations import create_expectation_suite
    
    # As CLI
    python eda_validation/validation/setup_expectations.py data/raw/sample.csv
"""

import pandas as pd
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import great_expectations as gx
    from great_expectations.core.expectation_configuration import ExpectationConfiguration
    GX_AVAILABLE = True
except ImportError:
    logger.warning("Great Expectations not installed. Install with: pip install great_expectations")
    GX_AVAILABLE = False
except Exception as e:
    logger.warning(f"Great Expectations import failed: {str(e)}")
    GX_AVAILABLE = False


def initialize_great_expectations_context(project_root: str = ".") -> Optional[object]:
    """
    Initialize Great Expectations context for the project.
    
    Args:
        project_root (str): Root directory of the project
        
    Returns:
        Great Expectations context object or None if not available
        
    Example:
        >>> context = initialize_great_expectations_context()
    """
    if not GX_AVAILABLE:
        logger.error("Great Expectations not available. Cannot initialize context.")
        return None
    
    try:
        project_path = Path(project_root)
        gx_dir = project_path / "gx"
        
        logger.info(f"Initializing Great Expectations context in: {gx_dir}")
        
        # Initialize GX context
        if not gx_dir.exists():
            logger.info("Creating new Great Expectations project")
            context = gx.get_context(project_root_dir=str(project_path))
        else:
            logger.info("Using existing Great Expectations project")
            context = gx.get_context(context_root_dir=str(gx_dir))
        
        return context
        
    except Exception as e:
        logger.error(f"Error initializing Great Expectations context: {str(e)}")
        return None


def create_typed_expectation_suite(
    df: pd.DataFrame,
    suite_name: str = "typed_validation_suite"
) -> List[Dict[str, Any]]:
    """
    Create expectations based on user-confirmed data types in the DataFrame.
    This is optimized for DataFrames that have already been type-cast by users.
    
    Args:
        df (pd.DataFrame): DataFrame with user-confirmed types
        suite_name (str): Name for the expectation suite
        
    Returns:
        List of expectation configurations
        
    Example:
        >>> expectations = create_typed_expectation_suite(df, "user_typed_suite")
    """
    logger.info(f"Creating typed expectation suite: {suite_name}")
    logger.info(f"Analyzing DataFrame with shape: {df.shape}")
    
    expectations = []
    
    try:
        # Basic dataset expectations
        expectations.append({
            "expectation_type": "expect_table_row_count_to_be_between",
            "kwargs": {
                "min_value": max(1, len(df) // 2),  # At least half the current rows
                "max_value": len(df) * 3  # Allow for some growth
            }
        })
        
        expectations.append({
            "expectation_type": "expect_table_column_count_to_equal",
            "kwargs": {
                "value": len(df.columns)
            }
        })
        
        # Column-specific expectations based on confirmed types
        for col in df.columns:
            col_data = df[col]
            col_dtype = str(col_data.dtype).lower()
            
            # Basic column existence
            expectations.append({
                "expectation_type": "expect_column_to_exist",
                "kwargs": {
                    "column": col
                }
            })
            
            # Type-specific expectations
            if 'int' in col_dtype:
                # Integer column expectations
                non_null_data = col_data.dropna()
                if len(non_null_data) > 0:
                    min_val = int(non_null_data.min())
                    max_val = int(non_null_data.max())
                    
                    expectations.append({
                        "expectation_type": "expect_column_values_to_be_between",
                        "kwargs": {
                            "column": col,
                            "min_value": min_val - abs(min_val * 0.1),  # 10% buffer
                            "max_value": max_val + abs(max_val * 0.1)
                        }
                    })
                
            elif 'float' in col_dtype:
                # Float column expectations
                non_null_data = col_data.dropna()
                if len(non_null_data) > 0:
                    min_val = float(non_null_data.min())
                    max_val = float(non_null_data.max())
                    
                    expectations.append({
                        "expectation_type": "expect_column_values_to_be_between",
                        "kwargs": {
                            "column": col,
                            "min_value": min_val - abs(min_val * 0.1),
                            "max_value": max_val + abs(max_val * 0.1)
                        }
                    })
                
            elif 'bool' in col_dtype:
                # Boolean column expectations
                expectations.append({
                    "expectation_type": "expect_column_values_to_be_in_set",
                    "kwargs": {
                        "column": col,
                        "value_set": [True, False]
                    }
                })
                
            elif 'category' in col_dtype:
                # Category column expectations
                unique_values = col_data.cat.categories.tolist()
                expectations.append({
                    "expectation_type": "expect_column_values_to_be_in_set",
                    "kwargs": {
                        "column": col,
                        "value_set": unique_values
                    }
                })
                
            elif 'datetime' in col_dtype:
                # DateTime column expectations
                non_null_data = col_data.dropna()
                if len(non_null_data) > 0:
                    min_date = non_null_data.min()
                    max_date = non_null_data.max()
                    
                    expectations.append({
                        "expectation_type": "expect_column_values_to_be_between",
                        "kwargs": {
                            "column": col,
                            "min_value": min_date.isoformat(),
                            "max_value": max_date.isoformat()
                        }
                    })
                    
            else:
                # String/Object column expectations
                unique_values = col_data.dropna().unique()
                
                if len(unique_values) <= 100:  # Categorical-like
                    expectations.append({
                        "expectation_type": "expect_column_values_to_be_in_set",
                        "kwargs": {
                            "column": col,
                            "value_set": unique_values.tolist()
                        }
                    })
                else:
                    # String length expectations
                    str_lengths = col_data.dropna().astype(str).str.len()
                    if len(str_lengths) > 0:
                        max_length = str_lengths.max()
                        min_length = str_lengths.min()
                        expectations.append({
                            "expectation_type": "expect_column_value_lengths_to_be_between",
                            "kwargs": {
                                "column": col,
                                "min_value": max(1, min_length),
                                "max_value": max_length * 2  # Allow some buffer
                            }
                        })
            
            # Null value expectations
            null_count = col_data.isnull().sum()
            if null_count == 0:
                expectations.append({
                    "expectation_type": "expect_column_values_to_not_be_null",
                    "kwargs": {
                        "column": col
                    }
                })
            else:
                # Allow current null percentage plus small buffer
                null_percentage = null_count / len(df)
                if null_percentage < 0.8:  # Only if less than 80% null
                    expectations.append({
                        "expectation_type": "expect_column_values_to_not_be_null",
                        "kwargs": {
                            "column": col,
                            "mostly": max(0.1, 1 - null_percentage - 0.05)  # 5% buffer
                        }
                    })
        
        logger.info(f"Created {len(expectations)} typed expectations")
        return expectations
        
    except Exception as e:
        logger.error(f"Error creating typed expectation suite: {str(e)}")
        # Fall back to basic expectations
        return create_basic_expectation_suite(df, suite_name)


def create_basic_expectation_suite(
    df: pd.DataFrame,
    suite_name: str = "basic_validation_suite"
) -> List[Dict[str, Any]]:
    """
    Create a basic set of expectations based on DataFrame analysis.
    
    Args:
        df (pd.DataFrame): DataFrame to analyze and create expectations for
        suite_name (str): Name for the expectation suite
        
    Returns:
        List of expectation configurations
        
    Example:
        >>> expectations = create_basic_expectation_suite(df, "my_data_suite")
    """
    logger.info(f"Creating basic expectation suite: {suite_name}")
    logger.info(f"Analyzing DataFrame with shape: {df.shape}")
    
    expectations = []
    
    try:
        # Basic dataset expectations
        expectations.append({
            "expectation_type": "expect_table_row_count_to_be_between",
            "kwargs": {
                "min_value": 1,
                "max_value": len(df) * 2  # Allow for some growth
            }
        })
        
        expectations.append({
            "expectation_type": "expect_table_column_count_to_equal",
            "kwargs": {
                "value": len(df.columns)
            }
        })
        
        # Column-specific expectations
        for col in df.columns:
            col_data = df[col]
            
            # Basic column existence
            expectations.append({
                "expectation_type": "expect_column_to_exist",
                "kwargs": {
                    "column": col
                }
            })
            
            # Handle missing values
            null_count = col_data.isnull().sum()
            if null_count == 0:
                expectations.append({
                    "expectation_type": "expect_column_values_to_not_be_null",
                    "kwargs": {
                        "column": col
                    }
                })
            else:
                # Allow some null values but not too many
                null_percentage = null_count / len(df)
                if null_percentage < 0.5:  # Less than 50% null
                    expectations.append({
                        "expectation_type": "expect_column_values_to_not_be_null",
                        "kwargs": {
                            "column": col,
                            "mostly": 1 - null_percentage - 0.1  # Allow 10% buffer
                        }
                    })
            
            # Data type specific expectations
            if col_data.dtype in ['int64', 'float64']:
                # Numeric column expectations
                min_val = col_data.min()
                max_val = col_data.max()
                
                if pd.notna(min_val) and pd.notna(max_val):
                    expectations.append({
                        "expectation_type": "expect_column_values_to_be_between",
                        "kwargs": {
                            "column": col,
                            "min_value": float(min_val),
                            "max_value": float(max_val)
                        }
                    })
            
            elif col_data.dtype == 'object':
                # String column expectations
                unique_values = col_data.dropna().unique()
                
                if len(unique_values) <= 50:  # Categorical-like column
                    expectations.append({
                        "expectation_type": "expect_column_values_to_be_in_set",
                        "kwargs": {
                            "column": col,
                            "value_set": unique_values.tolist()
                        }
                    })
                else:
                    # Check for reasonable string lengths
                    str_lengths = col_data.dropna().astype(str).str.len()
                    if len(str_lengths) > 0:
                        max_length = str_lengths.max()
                        expectations.append({
                            "expectation_type": "expect_column_value_lengths_to_be_between",
                            "kwargs": {
                                "column": col,
                                "min_value": 1,
                                "max_value": int(max_length * 1.5)  # Allow some buffer
                            }
                        })
        
        logger.info(f"Created {len(expectations)} expectations")
        return expectations
        
    except Exception as e:
        logger.error(f"Error creating basic expectation suite: {str(e)}")
        raise


def save_expectation_suite(
    expectations: List[Dict[str, Any]],
    suite_name: str,
    output_path: Optional[str] = None
) -> bool:
    """
    Save expectation suite to JSON file.
    
    Args:
        expectations (List[Dict]): List of expectation configurations
        suite_name (str): Name of the expectation suite
        output_path (str, optional): Path to save the suite
        
    Returns:
        bool: True if successful, False otherwise
        
    Example:
        >>> save_expectation_suite(expectations, "my_suite", "expectations/my_suite.json")
    """
    try:
        if output_path is None:
            output_path = f"data/processed/{suite_name}_expectations.json"
        
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        suite_config = {
            "expectation_suite_name": suite_name,
            "data_asset_type": "Dataset",
            "expectations": expectations,
            "meta": {
                "great_expectations_version": "0.18.0",  # Update as needed
                "created_by": "Team A EDA Pipeline"
            }
        }
        
        logger.info(f"Saving expectation suite to: {output_path}")
        
        with open(output_path, 'w') as f:
            json.dump(suite_config, f, indent=2, default=str)
        
        logger.info("Expectation suite saved successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error saving expectation suite: {str(e)}")
        return False


def create_expectation_suite_from_csv(
    input_path: str,
    suite_name: Optional[str] = None,
    output_path: Optional[str] = None
) -> bool:
    """
    Complete pipeline: load CSV, analyze, create expectations, and save suite.
    
    Args:
        input_path (str): Path to input CSV file
        suite_name (str, optional): Name for the expectation suite
        output_path (str, optional): Path to save the suite
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load data
        logger.info(f"Loading data from: {input_path}")
        df = pd.read_csv(input_path)
        logger.info(f"Loaded dataset with shape: {df.shape}")
        
        # Generate suite name if not provided
        if suite_name is None:
            input_name = Path(input_path).stem
            suite_name = f"{input_name}_validation_suite"
        
        # Create expectations
        expectations = create_basic_expectation_suite(df, suite_name)
        
        # Save expectations
        return save_expectation_suite(expectations, suite_name, output_path)
        
    except Exception as e:
        logger.error(f"Error in create_expectation_suite_from_csv: {str(e)}")
        return False


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Setup Great Expectations suite for CSV data"
    )
    parser.add_argument(
        "input_path",
        help="Path to input CSV file"
    )
    parser.add_argument(
        "-n", "--name",
        help="Name for the expectation suite (default: auto-generated)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output path for expectation suite (default: auto-generated)"
    )
    
    args = parser.parse_args()
    
    # Create expectation suite
    success = create_expectation_suite_from_csv(
        input_path=args.input_path,
        suite_name=args.name,
        output_path=args.output
    )
    
    if success:
        print("✅ Expectation suite created successfully")
    else:
        print("❌ Expectation suite creation failed")
        exit(1)


if __name__ == "__main__":
    main() 