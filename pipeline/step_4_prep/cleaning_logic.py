"""
Data cleaning logic for The Projection Wizard.
Contains functions for data cleaning operations in the prep stage.
Refactored for GCS-based storage.
"""

import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

from common import constants, schemas

# Configure logging for this module
logger = logging.getLogger(__name__)


def clean_data(df_original: pd.DataFrame, 
               feature_schemas: Dict[str, schemas.FeatureSchemaInfo], 
               target_info: schemas.TargetInfo,
               cleaning_config: Optional[dict] = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Clean the original DataFrame based on feature schemas and target information.
    
    Args:
        df_original: The raw DataFrame loaded from original_data.csv
        feature_schemas: Dictionary of FeatureSchemaInfo objects (converted from dicts from metadata.json)
        target_info: TargetInfo object with target column information
        cleaning_config: Optional dictionary for future cleaning strategy configurations
        
    Returns:
        Tuple containing:
        - cleaned_dataframe: The cleaned DataFrame
        - cleaning_steps_performed: List of strings describing cleaning steps taken
    """
    # Create a copy to avoid modifying original data
    df_cleaned = df_original.copy()
    cleaning_steps_performed = []
    
    # Track initial state
    initial_rows = len(df_cleaned)
    initial_cols = len(df_cleaned.columns)
    
    cleaning_steps_performed.append(f"Starting data cleaning with {initial_rows} rows and {initial_cols} columns")
    
    # Step 1: Handle Missing Values
    _handle_missing_values(df_cleaned, feature_schemas, target_info, cleaning_steps_performed)
    
    # Step 2: Remove Duplicates
    _remove_duplicates(df_cleaned, cleaning_steps_performed)
    
    # Log final state
    final_rows = len(df_cleaned)
    final_cols = len(df_cleaned.columns)
    cleaning_steps_performed.append(f"Cleaning completed with {final_rows} rows and {final_cols} columns")
    
    return df_cleaned, cleaning_steps_performed


def _handle_missing_values(df: pd.DataFrame, 
                          feature_schemas: Dict[str, schemas.FeatureSchemaInfo],
                          target_info: schemas.TargetInfo,
                          steps_log: List[str]) -> None:
    """
    Handle missing values based on encoding roles from feature schemas.
    
    Args:
        df: DataFrame to clean (modified in place)
        feature_schemas: Dictionary of FeatureSchemaInfo objects
        target_info: TargetInfo object with target information
        steps_log: List to append cleaning steps to
    """
    steps_log.append("Starting missing value imputation")
    
    # Track columns that have missing values before cleaning
    columns_with_missing = []
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            columns_with_missing.append((col, missing_count))
    
    if not columns_with_missing:
        steps_log.append("No missing values found in dataset")
        return
    
    steps_log.append(f"Found missing values in {len(columns_with_missing)} columns")
    
    # Process each column that has missing values
    for col, missing_count in columns_with_missing:
        # Get the encoding role for this column
        if col in feature_schemas:
            encoding_role = feature_schemas[col].encoding_role
        elif col == target_info.name:
            # For target column, use the target's ML type to determine strategy
            encoding_role = _get_encoding_role_for_target(target_info)
        else:
            # Fallback - shouldn't happen in well-formed data, but handle gracefully
            encoding_role = _infer_encoding_role_fallback(df[col])
            steps_log.append(f"Warning: Column '{col}' not found in feature schemas, inferred role: {encoding_role}")
        
        # Apply appropriate imputation strategy based on encoding role
        if encoding_role in ["numeric-continuous", "numeric-discrete"]:
            # Impute with median for numeric columns
            median_value = df[col].median()
            if pd.isna(median_value):
                # If median is NaN (all values are NaN), fill with 0
                median_value = 0
            df[col].fillna(median_value, inplace=True)
            steps_log.append(f"Imputed {missing_count} NaNs with median ({median_value}) for numeric column: {col}")
            
        elif encoding_role in ["categorical-nominal", "categorical-ordinal", "text"]:
            # Impute with mode for categorical/text columns, or use "_UNKNOWN_" if no mode exists
            mode_values = df[col].mode()
            if len(mode_values) > 0:
                mode_value = mode_values.iloc[0]
                df[col].fillna(mode_value, inplace=True)
                steps_log.append(f"Imputed {missing_count} NaNs with mode ('{mode_value}') for categorical column: {col}")
            else:
                # No mode exists (all values are NaN), use placeholder
                df[col].fillna("_UNKNOWN_", inplace=True)
                steps_log.append(f"Imputed {missing_count} NaNs with '_UNKNOWN_' for categorical column: {col}")
                
        elif encoding_role == "boolean":
            # Impute with mode for boolean columns
            mode_values = df[col].mode()
            if len(mode_values) > 0:
                mode_value = mode_values.iloc[0]
                df[col].fillna(mode_value, inplace=True)
                steps_log.append(f"Imputed {missing_count} NaNs with mode ({mode_value}) for boolean column: {col}")
            else:
                # Default to False if no mode exists
                df[col].fillna(False, inplace=True)
                steps_log.append(f"Imputed {missing_count} NaNs with False for boolean column: {col}")
                
        elif encoding_role == "datetime":
            # For datetime columns, we'll forward fill or use a placeholder date
            # This is a simple strategy - could be enhanced in the future
            if not df[col].dropna().empty:
                # Forward fill if we have some valid dates (using newer pandas syntax)
                df[col] = df[col].ffill().bfill()
                steps_log.append(f"Imputed {missing_count} NaNs with forward/backward fill for datetime column: {col}")
            else:
                # All values are NaN - use a placeholder
                placeholder_date = pd.Timestamp('1900-01-01')
                df[col].fillna(placeholder_date, inplace=True)
                steps_log.append(f"Imputed {missing_count} NaNs with placeholder date for datetime column: {col}")
                
        else:
            # Unknown encoding role - use a conservative approach
            if pd.api.types.is_numeric_dtype(df[col]):
                median_value = df[col].median()
                if pd.isna(median_value):
                    median_value = 0
                df[col].fillna(median_value, inplace=True)
                steps_log.append(f"Imputed {missing_count} NaNs with median for unknown-role numeric column: {col}")
            else:
                df[col].fillna("_UNKNOWN_", inplace=True)
                steps_log.append(f"Imputed {missing_count} NaNs with '_UNKNOWN_' for unknown-role non-numeric column: {col}")


def _remove_duplicates(df: pd.DataFrame, steps_log: List[str]) -> None:
    """
    Remove duplicate rows from the DataFrame.
    
    Args:
        df: DataFrame to clean (modified in place)
        steps_log: List to append cleaning steps to
    """
    initial_rows = len(df)
    
    # Remove duplicates (keep first occurrence)
    df.drop_duplicates(inplace=True)
    
    rows_removed = initial_rows - len(df)
    if rows_removed > 0:
        steps_log.append(f"Removed {rows_removed} duplicate rows")
    else:
        steps_log.append("No duplicate rows found")


def _get_encoding_role_for_target(target_info: schemas.TargetInfo) -> str:
    """
    Get appropriate encoding role for target column based on its ML type.
    
    Args:
        target_info: TargetInfo object
        
    Returns:
        Appropriate encoding role string
    """
    ml_type = target_info.ml_type
    
    if ml_type in ["binary_01", "binary_numeric", "numeric_continuous"]:
        return "numeric-continuous"
    elif ml_type in ["binary_boolean"]:
        return "boolean"
    elif ml_type in ["binary_text_labels", "multiclass_text_labels", "high_cardinality_text"]:
        return "categorical-nominal"
    elif ml_type in ["multiclass_int_labels"]:
        return "categorical-ordinal"
    else:
        # Default fallback
        return "categorical-nominal"


def _infer_encoding_role_fallback(series: pd.Series) -> str:
    """
    Fallback method to infer encoding role when not found in schemas.
    
    Args:
        series: Pandas Series to analyze
        
    Returns:
        Inferred encoding role string
    """
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    elif pd.api.types.is_numeric_dtype(series):
        return "numeric-continuous"
    elif pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    else:
        return "categorical-nominal" 