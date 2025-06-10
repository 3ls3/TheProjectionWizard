"""
Column Mapper for Prediction Input Processing

This module handles the transformation of user input from the original feature format
to the encoded format expected by the trained ML model. It manages one-hot encoding,
feature alignment, and ensures no target columns slip through.
Refactored for GCS-based storage.
"""

import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
import json

from common import storage
from api.utils.gcs_utils import download_run_file, check_run_file_exists, PROJECT_BUCKET_NAME

log = logging.getLogger(__name__)


def load_column_mapping_gcs(run_id: str,
                           gcs_bucket_name: str = PROJECT_BUCKET_NAME) -> Optional[Dict[str, Any]]:
    """
    Load the column mapping information saved during model training from GCS.

    Args:
        run_id: The run identifier
        gcs_bucket_name: GCS bucket name

    Returns:
        Dictionary containing column mapping info, or None if not found
    """
    try:
        # Check if file exists in GCS
        if not check_run_file_exists(run_id, 'column_mapping.json'):
            log.warning(f"Column mapping file not found in GCS for run {run_id}")
            return None

        # Download and parse column mapping from GCS
        mapping_bytes = download_run_file(run_id, 'column_mapping.json')
        if mapping_bytes is None:
            log.warning(f"Failed to download column mapping from GCS for run {run_id}")
            return None

        column_mapping = json.loads(mapping_bytes.decode('utf-8'))
        if column_mapping:
            log.info(f"Loaded column mapping from GCS with {len(column_mapping.get('encoded_columns', []))} encoded columns")
            return column_mapping
        else:
            log.warning("Column mapping file exists but is empty")
            return None

    except Exception as e:
        log.error(f"Failed to load column mapping from GCS: {e}")
        return None


def load_column_mapping(run_id: str) -> Optional[Dict[str, Any]]:
    """
    Legacy compatibility function - redirects to GCS version.

    Args:
        run_id: The run identifier

    Returns:
        Dictionary containing column mapping info, or None if not found
    """
    log.warning("Using legacy load_column_mapping function - redirecting to GCS version")
    return load_column_mapping_gcs(run_id)


def get_original_categorical_columns(df_original: pd.DataFrame, target_column: str) -> Dict[str, List[str]]:
    """
    Identify original categorical columns and their unique values.

    Args:
        df_original: Original dataset
        target_column: Name of target column to exclude

    Returns:
        Dictionary mapping column names to their unique values
    """
    categorical_info = {}

    for col in df_original.columns:
        if col == target_column:
            continue

        col_data = df_original[col].dropna()
        if len(col_data) == 0:
            continue

        # Check if column is categorical (non-numeric)
        if not pd.api.types.is_numeric_dtype(col_data):
            unique_values = sorted(col_data.unique().astype(str))
            categorical_info[col] = unique_values
            log.debug(f"Found categorical column '{col}' with values: {unique_values}")

    return categorical_info


def encode_categorical_to_onehot(user_input: Dict[str, Any],
                                categorical_info: Dict[str, List[str]],
                                encoded_columns: List[str]) -> Dict[str, Any]:
    """
    Convert categorical user inputs to one-hot encoded format.

    Args:
        user_input: Raw user input dictionary
        categorical_info: Information about categorical columns and their values
        encoded_columns: List of columns expected by the model

    Returns:
        Dictionary with one-hot encoded categorical features
    """
    encoded_input = {}

    # Handle each original categorical column
    for col_name, possible_values in categorical_info.items():
        if col_name in user_input:
            selected_value = str(user_input[col_name])

            # Create one-hot encoded columns for this categorical feature
            for value in possible_values:
                encoded_col_name = f"{col_name}_{value}"

                # Only include if the model expects this encoded column
                if encoded_col_name in encoded_columns:
                    encoded_input[encoded_col_name] = 1 if selected_value == value else 0
                    log.debug(f"Encoded {col_name}='{selected_value}' -> {encoded_col_name}={encoded_input[encoded_col_name]}")

    return encoded_input


def add_missing_columns(encoded_input: Dict[str, Any], expected_columns: List[str]) -> Dict[str, Any]:
    """
    Add any missing columns that the model expects with default values.

    Args:
        encoded_input: Current encoded input dictionary
        expected_columns: List of all columns the model expects

    Returns:
        Dictionary with all expected columns present
    """
    complete_input = encoded_input.copy()

    missing_columns = [col for col in expected_columns if col not in complete_input]

    if missing_columns:
        log.info(f"Adding {len(missing_columns)} missing columns with default value 0")
        for col in missing_columns:
            complete_input[col] = 0
            log.debug(f"Added missing column '{col}' = 0")

    return complete_input


def validate_encoded_input(encoded_input: Dict[str, Any],
                          expected_columns: List[str],
                          target_column: str) -> Tuple[bool, List[str]]:
    """
    Validate that the encoded input matches model expectations and contains no target column.

    Args:
        encoded_input: Encoded input dictionary
        expected_columns: List of columns the model expects
        target_column: Name of target column that should NOT be present

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []

    # Check for target column contamination
    if target_column in encoded_input:
        issues.append(f"Target column '{target_column}' found in input - this should not happen!")

    # Check for unexpected columns
    unexpected_columns = [col for col in encoded_input.keys() if col not in expected_columns]
    if unexpected_columns:
        issues.append(f"Unexpected columns in input: {unexpected_columns}")

    # Check for missing required columns
    missing_columns = [col for col in expected_columns if col not in encoded_input]
    if missing_columns:
        issues.append(f"Missing required columns: {missing_columns}")

    # Check column count
    if len(encoded_input) != len(expected_columns):
        issues.append(f"Column count mismatch: got {len(encoded_input)}, expected {len(expected_columns)}")

    return len(issues) == 0, issues


def encode_user_input_gcs(user_input: Dict[str, Any],
                         run_id: str,
                         df_original: pd.DataFrame,
                         target_column: str,
                         gcs_bucket_name: str = PROJECT_BUCKET_NAME) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """
    Main function to encode user input from original format to model-expected format using GCS.

    This function:
    1. Loads the column mapping saved during training from GCS
    2. Converts categorical inputs to one-hot encoded format
    3. Adds missing columns with default values
    4. Validates that no target column is present
    5. Returns a properly formatted DataFrame ready for model prediction

    Args:
        user_input: Dictionary of user inputs (in original column format)
        run_id: Run identifier to load column mapping
        df_original: Original dataset for reference
        target_column: Name of target column to exclude
        gcs_bucket_name: GCS bucket name

    Returns:
        Tuple of (encoded_dataframe, list_of_issues)
        - encoded_dataframe: None if encoding failed, otherwise properly formatted DataFrame
        - list_of_issues: List of validation issues/errors
    """
    issues = []

    try:
        # Step 1: Load column mapping from GCS
        column_mapping = load_column_mapping_gcs(run_id, gcs_bucket_name)
        if not column_mapping:
            issues.append("Could not load column mapping from GCS")
            return None, issues

        encoded_columns = column_mapping.get('encoded_columns', [])
        if not encoded_columns:
            issues.append("No encoded columns found in column mapping")
            return None, issues

        # Filter out target column from encoded_columns if it somehow got included
        original_count = len(encoded_columns)
        encoded_columns = [col for col in encoded_columns if col != target_column]
        filtered_count = len(encoded_columns)

        if original_count != filtered_count:
            log.warning(f"Removed target column '{target_column}' from encoded_columns. Count: {original_count} -> {filtered_count}")
        else:
            log.info(f"Target column '{target_column}' was not present in encoded_columns (good!)")

        log.info(f"Encoding user input for {filtered_count} expected model features (excluding target)")

        # Step 2: Handle numeric columns (direct pass-through)
        encoded_input = {}
        for col in df_original.columns:
            if col == target_column:
                continue  # Skip target column entirely

            if col in user_input and col in encoded_columns:
                # Direct numeric column - pass through as-is
                col_data = df_original[col].dropna()
                if pd.api.types.is_numeric_dtype(col_data):
                    encoded_input[col] = user_input[col]
                    log.debug(f"Direct numeric column: {col} = {user_input[col]}")

        # Step 3: Handle categorical columns (one-hot encoding)
        categorical_info = get_original_categorical_columns(df_original, target_column)
        if categorical_info:
            log.info(f"Processing {len(categorical_info)} categorical columns")
            categorical_encoded = encode_categorical_to_onehot(user_input, categorical_info, encoded_columns)
            encoded_input.update(categorical_encoded)

        # Step 4: Add missing columns with defaults
        encoded_input = add_missing_columns(encoded_input, encoded_columns)

        # Step 5: Validate the encoded input
        is_valid, validation_issues = validate_encoded_input(encoded_input, encoded_columns, target_column)
        if not is_valid:
            issues.extend(validation_issues)
            log.error(f"Validation failed: {validation_issues}")
            return None, issues

        # Step 6: Create DataFrame with proper column order
        # Ensure columns are in the exact order the model expects
        ordered_data = {col: encoded_input[col] for col in encoded_columns}
        encoded_df = pd.DataFrame([ordered_data])

        log.info(f"Successfully encoded user input to DataFrame with shape {encoded_df.shape} (GCS)")
        log.info(f"Final columns: {list(encoded_df.columns)}")

        return encoded_df, issues

    except Exception as e:
        error_msg = f"Failed to encode user input (GCS): {str(e)}"
        log.error(error_msg)
        issues.append(error_msg)
        return None, issues


def encode_user_input(user_input: Dict[str, Any],
                     run_id: str,
                     df_original: pd.DataFrame,
                     target_column: str) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """
    Legacy compatibility function - redirects to GCS version.

    Args:
        user_input: Dictionary of user inputs (in original column format)
        run_id: Run identifier to load column mapping
        df_original: Original dataset for reference
        target_column: Name of target column to exclude

    Returns:
        Tuple of (encoded_dataframe, list_of_issues)
        - encoded_dataframe: None if encoding failed, otherwise properly formatted DataFrame
        - list_of_issues: List of validation issues/errors
    """
    log.warning("Using legacy encode_user_input function - redirecting to GCS version")
    return encode_user_input_gcs(user_input, run_id, df_original, target_column)


def get_input_schema(run_id: str, df_original: pd.DataFrame, target_column: str) -> Dict[str, Any]:
    """
    Get the input schema information for creating the prediction form.

    Args:
        run_id: Run identifier
        df_original: Original dataset
        target_column: Target column name

    Returns:
        Dictionary containing schema information for form creation
    """
    schema = {
        'numeric_columns': {},
        'categorical_columns': {},
        'target_column': target_column
    }

    try:
        for col in df_original.columns:
            if col == target_column:
                continue

            col_data = df_original[col].dropna()
            if len(col_data) == 0:
                continue

            if pd.api.types.is_numeric_dtype(col_data):
                # Numeric column
                schema['numeric_columns'][col] = {
                    'min_value': float(col_data.min()),
                    'max_value': float(col_data.max()),
                    'mean_value': float(col_data.mean()),
                    'std_value': float(col_data.std()) if len(col_data) > 1 else 0.0
                }
            else:
                # Categorical column
                unique_values = sorted(col_data.unique().astype(str))
                schema['categorical_columns'][col] = {
                    'options': unique_values,
                    'default': unique_values[0] if unique_values else None
                }

        log.info(f"Generated input schema: {len(schema['numeric_columns'])} numeric, {len(schema['categorical_columns'])} categorical columns")

    except Exception as e:
        log.error(f"Failed to generate input schema: {e}")

    return schema
