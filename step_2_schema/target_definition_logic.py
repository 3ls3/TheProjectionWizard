"""
Target definition logic for The Projection Wizard.
Contains functions for suggesting and confirming target column and task type.
"""

import pandas as pd
from datetime import datetime, timezone
from typing import Tuple, Optional
import re

from common import logger, storage, constants


def suggest_target_and_task(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Suggest target column and task type based on DataFrame analysis.
    
    Args:
        df: The pandas DataFrame loaded from original_data.csv
        
    Returns:
        Tuple containing:
        - suggested_target_column_name (str | None): The name of the column most likely to be the target
        - suggested_task_type (str | None): Suggested task, e.g., "classification" or "regression"
        - suggested_target_ml_type (str | None): Suggested ML-ready type/format for the target
    """
    if df.empty or len(df.columns) == 0:
        return None, None, None
    
    # Target column name heuristics
    target_keywords = ['target', 'label', 'class', 'outcome', 'output', 'result', 'y', 'response']
    suggested_target_column = None
    
    # Look for columns with target-like names (case insensitive)
    for col in df.columns:
        col_lower = col.lower()
        for keyword in target_keywords:
            if keyword in col_lower:
                suggested_target_column = col
                break
        if suggested_target_column:
            break
    
    # If no obvious name match, consider the last column
    if suggested_target_column is None:
        suggested_target_column = df.columns[-1]
    
    # If still no suitable candidate (shouldn't happen), return None
    if suggested_target_column is None:
        return None, None, None
    
    # Analyze the suggested target column to determine task type and ML type
    target_series = df[suggested_target_column]
    target_dtype = target_series.dtype
    unique_values = target_series.nunique()
    total_values = len(target_series.dropna())  # Exclude NaN values
    
    suggested_task_type = None
    suggested_target_ml_type = None
    
    # Check for boolean dtype first (pandas treats bool as numeric)
    if pd.api.types.is_bool_dtype(target_dtype):
        suggested_task_type = "classification"
        suggested_target_ml_type = "binary_boolean"
    elif pd.api.types.is_numeric_dtype(target_dtype):
        # Numeric target
        if unique_values <= 10 and target_series.dropna().dtype == int:
            # Few unique integer values - likely classification
            suggested_task_type = "classification"
            if unique_values == 2:
                # Check if binary (0,1) or similar
                unique_vals = sorted(target_series.dropna().unique())
                if len(unique_vals) == 2 and unique_vals[0] == 0 and unique_vals[1] == 1:
                    suggested_target_ml_type = "binary_01"
                else:
                    suggested_target_ml_type = "binary_numeric"
            else:
                suggested_target_ml_type = "multiclass_int_labels"
        else:
            # Many unique values or float - likely regression
            suggested_task_type = "regression"
            suggested_target_ml_type = "numeric_continuous"
    
    elif pd.api.types.is_object_dtype(target_dtype) or pd.api.types.is_categorical_dtype(target_dtype):
        # Categorical/text target
        if unique_values == 2:
            suggested_task_type = "classification"
            suggested_target_ml_type = "binary_text_labels"
        elif 2 < unique_values <= constants.SCHEMA_CONFIG["max_categorical_cardinality"]:
            suggested_task_type = "classification"
            suggested_target_ml_type = "multiclass_text_labels"
        else:
            # Very high cardinality - might not be suitable as target
            # But still suggest classification as it's the only option for text
            suggested_task_type = "classification"
            suggested_target_ml_type = "high_cardinality_text"
    
    else:
        # Other dtypes (datetime, etc.)
        # Default to classification for unknown types
        suggested_task_type = "classification"
        suggested_target_ml_type = "unknown_type"
    
    return suggested_target_column, suggested_task_type, suggested_target_ml_type


def confirm_target_definition(run_id: str, confirmed_target_column: str, 
                            confirmed_task_type: str, confirmed_target_ml_type: str) -> bool:
    """
    Confirm target definition and update metadata.json with target information.
    
    Args:
        run_id: The ID of the current run
        confirmed_target_column: The target column name selected by the user
        confirmed_task_type: The task type (e.g., "classification", "regression") selected by the user
        confirmed_target_ml_type: The ML-ready type/format for the target selected by the user
        
    Returns:
        True if successful, False otherwise
    """
    # Get logger
    run_logger = logger.get_stage_logger(run_id, constants.SCHEMA_STAGE)
    
    # Validate inputs
    if confirmed_task_type not in constants.TASK_TYPES:
        run_logger.error(f"Invalid task type: {confirmed_task_type}. Must be one of {constants.TASK_TYPES}")
        return False
        
    if confirmed_target_ml_type not in constants.TARGET_ML_TYPES:
        run_logger.error(f"Invalid target ML type: {confirmed_target_ml_type}. Must be one of {constants.TARGET_ML_TYPES}")
        return False
    
    try:
        # Read existing metadata.json
        metadata_dict = storage.read_metadata(run_id)
        if metadata_dict is None:
            run_logger.error(f"Could not read metadata.json for run {run_id}")
            return False
        
        # Update metadata dictionary with target information
        metadata_dict['target_info'] = {
            'name': confirmed_target_column,
            'task_type': confirmed_task_type,
            'ml_type': confirmed_target_ml_type,
            'user_confirmed_at': datetime.now(timezone.utc).isoformat()
        }
        
        # Optionally update top-level task_type for convenience
        metadata_dict['task_type'] = confirmed_task_type
        
        # Write updated metadata.json
        storage.write_metadata(run_id, metadata_dict)
        
        # Update status.json
        status_data = {
            'stage': constants.SCHEMA_STAGE,
            'status': 'completed',
            'message': f"Target '{confirmed_target_column}' and task '{confirmed_task_type}' confirmed.",
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'errors': []
        }
        
        storage.write_status(run_id, status_data)
        
        run_logger.info(f"Target definition confirmed: column='{confirmed_target_column}', "
                       f"task='{confirmed_task_type}', ml_type='{confirmed_target_ml_type}'")
        
        return True
        
    except Exception as e:
        run_logger.error(f"Failed to confirm target definition: {str(e)}")
        
        # Update status.json with error
        try:
            status_data = {
                'stage': constants.SCHEMA_STAGE,
                'status': 'failed',
                'message': f"Failed to confirm target definition: {str(e)}",
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'errors': [str(e)]
            }
            storage.write_status(run_id, status_data)
        except Exception as status_error:
            run_logger.error(f"Failed to update status.json with error: {str(status_error)}")
        
        return False 