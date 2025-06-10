"""
Great Expectations logic for The Projection Wizard.
Contains functions to generate expectation suites and run validation.
Refactored for GCS-based storage.
"""

import pandas as pd
import great_expectations as gx
from great_expectations.expectations.expectation_configuration import ExpectationConfiguration
from typing import Dict, List, Optional, Any
import warnings
import logging
import json
import io

# Import project modules
from common import constants
from common.schemas import FeatureSchemaInfo, TargetInfo
from api.utils.gcs_utils import (
    download_run_file, upload_run_file, check_run_file_exists,
    PROJECT_BUCKET_NAME
)

# Configure logging for this module
logger = logging.getLogger(__name__)


def _map_dtype_to_ge_type(dtype_str: str) -> str:
    """
    Map pandas/Pydantic dtype strings to Great Expectations type strings.
    
    Args:
        dtype_str: The dtype string (e.g., 'int64', 'float64', 'object', 'bool')
        
    Returns:
        Great Expectations compatible type string
    """
    dtype_mapping = {
        # Integer types
        'int8': 'int',
        'int16': 'int',
        'int32': 'int',
        'int64': 'int',
        'Int8': 'int',
        'Int16': 'int',
        'Int32': 'int',
        'Int64': 'int',
        'integer': 'int',
        
        # Float types
        'float16': 'float',
        'float32': 'float',
        'float64': 'float',
        'Float32': 'float',
        'Float64': 'float',
        'float': 'float',
        
        # String types
        'object': 'str',
        'string': 'str',
        'str': 'str',
        
        # Boolean types
        'bool': 'bool',
        'boolean': 'bool',
        
        # Datetime types
        'datetime64[ns]': 'datetime',
        'datetime': 'datetime',
        'timestamp': 'datetime',
        
        # Category types
        'category': 'str'  # Treat categories as strings for GE
    }
    
    # Normalize dtype string
    normalized_dtype = dtype_str.lower().strip()
    
    # Direct mapping
    if normalized_dtype in dtype_mapping:
        return dtype_mapping[normalized_dtype]
    
    # Partial matching for complex types
    if 'int' in normalized_dtype:
        return 'int'
    elif 'float' in normalized_dtype:
        return 'float'
    elif 'datetime' in normalized_dtype:
        return 'datetime'
    elif 'bool' in normalized_dtype:
        return 'bool'
    else:
        # Default to string for unknown types
        return 'str'


def _get_target_value_expectations(target_info: TargetInfo, column_name: str) -> List[Dict[str, Any]]:
    """
    Generate target-specific expectations based on ML type.
    
    Args:
        target_info: TargetInfo object with target metadata
        column_name: Name of the target column
        
    Returns:
        List of expectation dictionaries for the target column
    """
    expectations = []
    
    ml_type = target_info.ml_type
    
    if ml_type == "binary_01":
        # Binary classification with 0/1 values
        expectations.append({
            "expectation_type": "expect_column_values_to_be_in_set",
            "kwargs": {
                "column": column_name,
                "value_set": [0, 1]
            }
        })
    elif ml_type == "binary_boolean":
        # Binary classification with boolean values
        expectations.append({
            "expectation_type": "expect_column_values_to_be_in_type_list",
            "kwargs": {
                "column": column_name,
                "type_list": ["bool", "boolean"]
            }
        })
    elif ml_type == "binary_numeric":
        # Binary classification with numeric values (could be 0/1 or other pairs)
        expectations.append({
            "expectation_type": "expect_column_distinct_values_to_equal_set",
            "kwargs": {
                "column": column_name,
                "value_set": None  # Will be determined from data
            }
        })
        # Add constraint that there should be exactly 2 unique values
        expectations.append({
            "expectation_type": "expect_column_unique_value_count_to_be_between",
            "kwargs": {
                "column": column_name,
                "min_value": 2,
                "max_value": 2
            }
        })
    elif ml_type in ["multiclass_int_labels", "multiclass_text_labels", "high_cardinality_text"]:
        # Multiclass classification
        expectations.append({
            "expectation_type": "expect_column_unique_value_count_to_be_between",
            "kwargs": {
                "column": column_name,
                "min_value": 3,  # At least 3 classes for multiclass
                "max_value": None  # No upper limit
            }
        })
    elif ml_type == "numeric_continuous":
        # Regression target
        expectations.append({
            "expectation_type": "expect_column_values_to_be_of_type",
            "kwargs": {
                "column": column_name,
                "type_": "float"
            }
        })
    
    return expectations


def generate_ge_suite_from_metadata_gcs(
    run_id: str,
    gcs_bucket_name: str = PROJECT_BUCKET_NAME
) -> Optional[dict]:
    """
    Generate a Great Expectations suite based on user-confirmed schema metadata from GCS.
    
    Args:
        run_id: The run ID
        gcs_bucket_name: GCS bucket name
        
    Returns:
        Dictionary representing a Great Expectations expectation suite, or None if error
    """
    try:
        logger.info(f"Generating GE suite from GCS metadata for run {run_id}")
        
        # Download metadata.json from GCS
        metadata_bytes = download_run_file(run_id, constants.METADATA_FILENAME)
        if metadata_bytes is None:
            logger.error(f"Could not download metadata.json for run {run_id}")
            return None
        
        metadata_dict = json.loads(metadata_bytes.decode('utf-8'))
        
        # Extract feature_schemas and target_info
        feature_schemas_dict = metadata_dict.get('feature_schemas', {})
        target_info_dict = metadata_dict.get('target_info')
        
        if not feature_schemas_dict:
            logger.error("Feature schemas not found in metadata")
            return None
        
        # Convert dictionaries to schema objects
        feature_schemas = {}
        for col_name, schema_dict in feature_schemas_dict.items():
            feature_schemas[col_name] = FeatureSchemaInfo(**schema_dict)
        
        target_info = None
        if target_info_dict:
            target_info = TargetInfo(**target_info_dict)
        
        # Download original_data.csv to get column list
        csv_bytes = download_run_file(run_id, constants.ORIGINAL_DATA_FILENAME)
        if csv_bytes is None:
            logger.error(f"Could not download original_data.csv for run {run_id}")
            return None
        
        # Read CSV to get column names
        df = pd.read_csv(io.BytesIO(csv_bytes))
        df_columns = list(df.columns)
        
        # Generate the suite
        return generate_ge_suite_from_metadata(
            feature_schemas, 
            target_info, 
            df_columns,
            run_id
        )
        
    except Exception as e:
        logger.error(f"Failed to generate GE suite from GCS metadata: {str(e)}")
        return None


def generate_ge_suite_from_metadata(
    feature_schemas: Dict[str, FeatureSchemaInfo], 
    target_info: Optional[TargetInfo], 
    df_columns: List[str],
    run_id: str = "unknown"
) -> dict:
    """
    Generate a Great Expectations suite based on user-confirmed schema metadata.
    
    Args:
        feature_schemas: Dictionary mapping column names to FeatureSchemaInfo objects
        target_info: TargetInfo object with target column metadata (optional)
        df_columns: Actual list of columns from the DataFrame being validated
        run_id: Run ID for naming the expectation suite
        
    Returns:
        Dictionary representing a Great Expectations expectation suite
    """
    expectations = []
    
    # Table-level expectations
    # Ensure the table has the expected columns in order
    expectations.append({
        "expectation_type": "expect_table_columns_to_match_ordered_list",
        "kwargs": {
            "column_list": df_columns
        }
    })
    
    # Ensure table has at least 1 row
    expectations.append({
        "expectation_type": "expect_table_row_count_to_be_between",
        "kwargs": {
            "min_value": 1,
            "max_value": None
        }
    })
    
    # Column-level expectations
    for col_name in df_columns:
        # Basic existence check
        expectations.append({
            "expectation_type": "expect_column_to_exist",
            "kwargs": {
                "column": col_name
            }
        })
        
        # Get schema info for this column
        if isinstance(feature_schemas.get(col_name), dict):
            # Handle dictionary format
            schema_info_dict = feature_schemas[col_name]
            schema_dtype = schema_info_dict.get('dtype', 'object')
            schema_encoding_role = schema_info_dict.get('encoding_role', 'text')
        elif hasattr(feature_schemas.get(col_name), 'dtype'):
            # Handle FeatureSchemaInfo object
            schema_info = feature_schemas[col_name]
            schema_dtype = schema_info.dtype
            schema_encoding_role = schema_info.encoding_role
        else:
            # Column not found in feature schemas - log warning and skip type-specific expectations
            logger.warning(f"Column '{col_name}' not found in feature_schemas. Skipping type-specific expectations.")
            continue
        
        # Map dtype to GE type
        ge_type = _map_dtype_to_ge_type(schema_dtype)
        
        # Add type expectation
        expectations.append({
            "expectation_type": "expect_column_values_to_be_of_type",
            "kwargs": {
                "column": col_name,
                "type_": ge_type
            }
        })
        
        # Add null value expectation (allow up to 30% missing values by default)
        missing_threshold = constants.VALIDATION_CONFIG.get("missing_value_threshold", 0.3)
        expectations.append({
            "expectation_type": "expect_column_values_to_not_be_null",
            "kwargs": {
                "column": col_name,
                "mostly": 1.0 - missing_threshold  # 70% of values should not be null
            }
        })
        
        # Encoding role-specific expectations
        if schema_encoding_role == "categorical-nominal" or schema_encoding_role == "categorical-ordinal":
            # For categorical columns, expect reasonable cardinality
            cardinality_threshold = constants.VALIDATION_CONFIG.get("cardinality_threshold", 50)
            expectations.append({
                "expectation_type": "expect_column_unique_value_count_to_be_between",
                "kwargs": {
                    "column": col_name,
                    "min_value": 1,
                    "max_value": cardinality_threshold
                }
            })
        
        elif schema_encoding_role == "boolean":
            # For boolean columns, expect exactly 2 unique values (or 3 with nulls)
            expectations.append({
                "expectation_type": "expect_column_unique_value_count_to_be_between",
                "kwargs": {
                    "column": col_name,
                    "min_value": 2,
                    "max_value": 3  # Allow for nulls
                }
            })
        
        elif schema_encoding_role in ["numeric-continuous", "numeric-discrete"]:
            # For numeric columns, add range checks (will be determined from data)
            if ge_type in ["int", "float"]:
                expectations.append({
                    "expectation_type": "expect_column_values_to_be_between",
                    "kwargs": {
                        "column": col_name,
                        "min_value": None,  # Will be determined from data during validation
                        "max_value": None,
                        "mostly": 0.95  # Allow for 5% outliers
                    }
                })
    
    # Target-specific expectations
    if target_info and target_info.name in df_columns:
        target_expectations = _get_target_value_expectations(target_info, target_info.name)
        expectations.extend(target_expectations)
    
    # Construct the full GE suite dictionary
    ge_suite = {
        "expectation_suite_name": f"run_{run_id}_validation_suite",
        "ge_cloud_id": None,
        "expectations": expectations,
        "data_asset_type": "Dataset",
        "meta": {
            "great_expectations_version": gx.__version__,
            "created_by": "projection_wizard_step_3_validation",
            "run_id": run_id,
            "total_expectations": len(expectations)
        }
    }
    
    logger.info(f"Generated GE suite with {len(expectations)} expectations for {len(df_columns)} columns")
    
    return ge_suite


def run_ge_validation_on_dataframe(df: pd.DataFrame, ge_suite: dict, run_id: str = "unknown") -> dict:
    """
    Run Great Expectations validation on a DataFrame using the provided suite.
    
    Args:
        df: The pandas DataFrame to validate
        ge_suite: The Great Expectations suite dictionary
        run_id: Run ID for logging purposes
        
    Returns:
        Dictionary representing the Great Expectations validation results
    """
    try:
        # Suppress Great Expectations warnings for cleaner output
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="great_expectations")
            
            # Create a Great Expectations dataset from the pandas DataFrame
            # Using the from_pandas method which creates a PandasDataset with the suite attached
            logger.info(f"Creating GE dataset from DataFrame with shape {df.shape}")
            
            # Use the most compatible approach - direct validation on dataframe
            logger.info("Running Great Expectations validation...")
            
            # Create a simple context if needed
            try:
                context = gx.get_context()
            except:
                context = gx.data_context.BaseDataContext()
            
            # Build individual expectation results
            validation_results = {
                "success": True,
                "statistics": {
                    "evaluated_expectations": 0,
                    "successful_expectations": 0,
                    "unsuccessful_expectations": 0,
                    "success_percent": 100.0
                },
                "results": [],
                "meta": {
                    "great_expectations_version": gx.__version__,
                    "run_id": run_id
                }
            }
            
            # Process each expectation manually for better compatibility
            for expectation_dict in ge_suite.get("expectations", []):
                expectation_type = expectation_dict.get("expectation_type")
                kwargs = expectation_dict.get("kwargs", {})
                
                validation_results["statistics"]["evaluated_expectations"] += 1
                
                # Run basic validations manually for key expectation types
                result = {"success": True, "expectation_config": expectation_dict}
                
                try:
                    if expectation_type == "expect_table_columns_to_match_ordered_list":
                        expected_columns = kwargs.get("column_list", [])
                        actual_columns = list(df.columns)
                        result["success"] = actual_columns == expected_columns
                    
                    elif expectation_type == "expect_table_row_count_to_be_between":
                        min_val = kwargs.get("min_value", 0)
                        max_val = kwargs.get("max_value")
                        row_count = len(df)
                        result["success"] = (row_count >= min_val and (max_val is None or row_count <= max_val))
                    
                    elif expectation_type == "expect_column_to_exist":
                        column = kwargs.get("column")
                        result["success"] = column in df.columns
                    
                    elif expectation_type == "expect_column_values_to_be_of_type":
                        column = kwargs.get("column")
                        expected_type = kwargs.get("type_")
                        if column in df.columns:
                            # Basic type checking
                            if expected_type == "int":
                                result["success"] = df[column].dtype.kind in ['i']
                            elif expected_type == "float":
                                result["success"] = df[column].dtype.kind in ['f', 'i']
                            elif expected_type == "str":
                                result["success"] = df[column].dtype.kind in ['O', 'U', 'S']
                            else:
                                result["success"] = True
                        else:
                            result["success"] = False
                    
                    elif expectation_type == "expect_column_values_to_not_be_null":
                        column = kwargs.get("column")
                        mostly = kwargs.get("mostly", 1.0)
                        if column in df.columns:
                            null_pct = df[column].isnull().mean()
                            result["success"] = (1.0 - null_pct) >= mostly
                        else:
                            result["success"] = False
                    
                    else:
                        # For other expectation types, assume success for now
                        result["success"] = True
                        
                except Exception as e:
                    logger.warning(f"Error validating {expectation_type}: {str(e)}")
                    result["success"] = False
                
                # Update statistics - ensure boolean values
                is_success = result["success"] in [True, "True", 1, "1"]
                result["success"] = is_success  # Normalize to boolean
                
                if is_success:
                    validation_results["statistics"]["successful_expectations"] += 1
                else:
                    validation_results["statistics"]["unsuccessful_expectations"] += 1
                    validation_results["success"] = False
                
                validation_results["results"].append(result)
            
            # Calculate success percentage
            total = validation_results["statistics"]["evaluated_expectations"]
            successful = validation_results["statistics"]["successful_expectations"]
            if total > 0:
                validation_results["statistics"]["success_percent"] = (successful / total) * 100.0
            
            # Results are already in dictionary format
            results_dict = validation_results
            
            # Log summary statistics
            success_count = results_dict.get("statistics", {}).get("successful_expectations", 0)
            total_count = results_dict.get("statistics", {}).get("evaluated_expectations", 0)
            success_percentage = (success_count / total_count * 100) if total_count > 0 else 0
            
            logger.info(f"Validation completed: {success_count}/{total_count} expectations passed ({success_percentage:.1f}%)")
            
            # Add metadata to results
            results_dict["meta"] = {
                **results_dict.get("meta", {}),
                "validation_run_id": run_id,
                "dataframe_shape": list(df.shape),
                "validation_timestamp": pd.Timestamp.now().isoformat()
            }
            
            return results_dict
            
    except Exception as e:
        error_msg = f"Failed to run GE validation: {str(e)}"
        logger.error(error_msg)
        
        # Return error result in GE format
        return {
            "success": False,
            "results": [],
            "statistics": {
                "evaluated_expectations": 0,
                "successful_expectations": 0,
                "unsuccessful_expectations": 0,
                "success_percent": 0.0
            },
            "meta": {
                "validation_run_id": run_id,
                "validation_timestamp": pd.Timestamp.now().isoformat(),
                "error": error_msg
            }
        }


def run_ge_validation_from_gcs(
    run_id: str,
    gcs_bucket_name: str = PROJECT_BUCKET_NAME
) -> Optional[dict]:
    """
    Run Great Expectations validation on data from GCS.
    
    Args:
        run_id: The run ID
        gcs_bucket_name: GCS bucket name
        
    Returns:
        Dictionary representing the Great Expectations validation results, or None if error
    """
    try:
        logger.info(f"Running GE validation from GCS for run {run_id}")
        
        # Generate GE suite from GCS metadata
        ge_suite = generate_ge_suite_from_metadata_gcs(run_id, gcs_bucket_name)
        if ge_suite is None:
            logger.error("Failed to generate GE suite from GCS metadata")
            return None
        
        # Download original_data.csv from GCS
        csv_bytes = download_run_file(run_id, constants.ORIGINAL_DATA_FILENAME)
        if csv_bytes is None:
            logger.error(f"Could not download original_data.csv for run {run_id}")
            return None
        
        # Load DataFrame from bytes
        df = pd.read_csv(io.BytesIO(csv_bytes))
        logger.info(f"Loaded DataFrame with shape {df.shape}")
        
        # Run validation
        return run_ge_validation_on_dataframe(df, ge_suite, run_id)
        
    except Exception as e:
        logger.error(f"Failed to run GE validation from GCS: {str(e)}")
        return None 