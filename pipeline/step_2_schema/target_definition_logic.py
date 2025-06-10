"""
Target definition logic for The Projection Wizard.
Contains functions for suggesting and confirming target column and task type.
Refactored for GCS-based storage.
"""

import pandas as pd
from datetime import datetime, timezone
from typing import Tuple, Optional
import re
import json
import io
import logging

from common import constants
from api.utils.gcs_utils import (
    download_run_file, upload_run_file, check_run_file_exists,
    PROJECT_BUCKET_NAME
)

# Configure logging for this module
logger = logging.getLogger(__name__)


def suggest_target_and_task_from_gcs(run_id: str, gcs_bucket_name: str = PROJECT_BUCKET_NAME) -> Tuple[Optional[str], Optional[str], Optional[str], bool]:
    """
    Download CSV from GCS and suggest target column and task type based on DataFrame analysis.
    
    Args:
        run_id: The ID of the current run
        gcs_bucket_name: GCS bucket name (defaults to PROJECT_BUCKET_NAME)
        
    Returns:
        Tuple containing:
        - suggested_target_column_name (str | None): The name of the column most likely to be the target
        - suggested_task_type (str | None): Suggested task, e.g., "classification" or "regression"
        - suggested_target_ml_type (str | None): Suggested ML-ready type/format for the target
        - success (bool): Whether the operation succeeded
    """
    logger.info(f"Starting target suggestion for run_id: {run_id}")
    
    try:
        # Download original_data.csv from GCS
        csv_bytes = download_run_file(run_id, constants.ORIGINAL_DATA_FILE)
        if not csv_bytes:
            logger.error(f"Could not download original_data.csv for run_id: {run_id}")
            return None, None, None, False
        
        # Read CSV into DataFrame
        df = pd.read_csv(io.BytesIO(csv_bytes))
        logger.info(f"Successfully loaded DataFrame: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Analyze the DataFrame
        suggestions = suggest_target_and_task(df)
        logger.info(f"Target suggestions generated: column={suggestions[0]}, task={suggestions[1]}, ml_type={suggestions[2]}")
        
        return suggestions[0], suggestions[1], suggestions[2], True
        
    except Exception as e:
        logger.error(f"Failed to suggest target and task for run_id {run_id}: {str(e)}", exc_info=True)
        return None, None, None, False


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


def confirm_target_definition_gcs(run_id: str, confirmed_target_column: str, 
                                 confirmed_task_type: str, confirmed_target_ml_type: str,
                                 gcs_bucket_name: str = PROJECT_BUCKET_NAME) -> bool:
    """
    Confirm target definition and update metadata.json in GCS with target information.
    
    Args:
        run_id: The ID of the current run
        confirmed_target_column: The target column name selected by the user
        confirmed_task_type: The task type (e.g., "classification", "regression") selected by the user
        confirmed_target_ml_type: The ML-ready type/format for the target selected by the user
        gcs_bucket_name: GCS bucket name (defaults to PROJECT_BUCKET_NAME)
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Starting target definition confirmation for run_id: {run_id}")
    
    try:
        # Step 1: Update status to processing
        if not _update_status_to_processing(run_id, gcs_bucket_name, "target_definition"):
            logger.error(f"Failed to update status to processing for run_id: {run_id}")
            return False
        
        # Step 2: Validate inputs
        if not _validate_target_inputs(confirmed_task_type, confirmed_target_ml_type):
            _update_status_to_failed(run_id, gcs_bucket_name, "Invalid target definition inputs")
            return False
        
        # Step 3: Download and update metadata
        if not _update_metadata_with_target_info(run_id, gcs_bucket_name, confirmed_target_column, 
                                                 confirmed_task_type, confirmed_target_ml_type):
            _update_status_to_failed(run_id, gcs_bucket_name, "Failed to update metadata with target info")
            return False
        
        # Step 4: Update status to completed
        if not _update_status_to_completed(run_id, gcs_bucket_name, "target_definition", 
                                          f"Target '{confirmed_target_column}' and task '{confirmed_task_type}' confirmed."):
            logger.error(f"Failed to update status to completed for run_id: {run_id}")
            return False
        
        logger.info(f"Target definition confirmed successfully: column='{confirmed_target_column}', "
                   f"task='{confirmed_task_type}', ml_type='{confirmed_target_ml_type}'")
        return True
        
    except Exception as e:
        logger.error(f"Critical error during target definition confirmation for run_id {run_id}: {str(e)}", exc_info=True)
        _update_status_to_failed(run_id, gcs_bucket_name, f"Critical error: {str(e)}")
        return False


def _validate_target_inputs(confirmed_task_type: str, confirmed_target_ml_type: str) -> bool:
    """Validate target definition inputs."""
    if confirmed_task_type not in constants.TASK_TYPES:
        logger.error(f"Invalid task type: {confirmed_task_type}. Must be one of {constants.TASK_TYPES}")
        return False
        
    if confirmed_target_ml_type not in constants.TARGET_ML_TYPES:
        logger.error(f"Invalid target ML type: {confirmed_target_ml_type}. Must be one of {constants.TARGET_ML_TYPES}")
        return False
    
    return True


def _update_metadata_with_target_info(run_id: str, gcs_bucket_name: str, confirmed_target_column: str,
                                     confirmed_task_type: str, confirmed_target_ml_type: str) -> bool:
    """Download metadata, update with target info, and upload back to GCS."""
    try:
        # Download metadata.json
        metadata_bytes = download_run_file(run_id, constants.METADATA_FILE)
        if not metadata_bytes:
            logger.error(f"Could not download metadata.json for run_id: {run_id}")
            return False
        
        # Parse metadata
        metadata_dict = json.loads(metadata_bytes.decode('utf-8'))
        
        # Update metadata with target information
        metadata_dict['target_info'] = {
            'name': confirmed_target_column,
            'task_type': confirmed_task_type,
            'ml_type': confirmed_target_ml_type,
            'user_confirmed_at': datetime.now(timezone.utc).isoformat()
        }
        
        # Optionally update top-level task_type for convenience
        metadata_dict['task_type'] = confirmed_task_type
        
        # Upload updated metadata back to GCS
        metadata_json = json.dumps(metadata_dict, indent=2, default=str).encode('utf-8')
        metadata_io = io.BytesIO(metadata_json)
        
        success = upload_run_file(run_id, constants.METADATA_FILE, metadata_io)
        if success:
            logger.info(f"Updated metadata with target info for run_id: {run_id}")
        return success
        
    except Exception as e:
        logger.error(f"Failed to update metadata with target info for run_id {run_id}: {str(e)}")
        return False


def _update_status_to_processing(run_id: str, gcs_bucket_name: str, operation: str) -> bool:
    """Update status.json to indicate processing has started."""
    try:
        # Download current status.json
        status_bytes = download_run_file(run_id, constants.STATUS_FILE)
        if not status_bytes:
            logger.error(f"Could not download status.json for run_id: {run_id}")
            return False
        
        # Parse current status
        current_status = json.loads(status_bytes.decode('utf-8'))
        
        # Update status fields
        current_status.update({
            'current_stage': constants.SCHEMA_STAGE,
            'current_stage_name': constants.STAGE_DISPLAY_NAMES[constants.SCHEMA_STAGE],
            'status': 'processing',
            'message': f'Starting {operation.replace("_", " ")}...',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'last_updated': datetime.now(timezone.utc).isoformat()
        })
        
        # Upload updated status back to GCS
        status_json = json.dumps(current_status, indent=2).encode('utf-8')
        status_io = io.BytesIO(status_json)
        
        success = upload_run_file(run_id, constants.STATUS_FILE, status_io)
        if success:
            logger.info(f"Updated status to processing for {operation} in run_id: {run_id}")
        return success
        
    except Exception as e:
        logger.error(f"Failed to update status to processing for run_id {run_id}: {str(e)}")
        return False


def _update_status_to_completed(run_id: str, gcs_bucket_name: str, operation: str, message: str) -> bool:
    """Update status.json to indicate operation completed successfully."""
    try:
        # Download current status.json
        status_bytes = download_run_file(run_id, constants.STATUS_FILE)
        if not status_bytes:
            logger.error(f"Could not download status.json for run_id: {run_id}")
            return False
        
        # Parse current status
        current_status = json.loads(status_bytes.decode('utf-8'))
        
        # Update status fields
        current_status.update({
            'status': 'completed',
            'message': message,
            'next_stage': constants.VALIDATION_STAGE,
            'next_stage_name': constants.STAGE_DISPLAY_NAMES[constants.VALIDATION_STAGE],
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'last_updated': datetime.now(timezone.utc).isoformat()
        })
        
        # Upload updated status back to GCS
        status_json = json.dumps(current_status, indent=2).encode('utf-8')
        status_io = io.BytesIO(status_json)
        
        success = upload_run_file(run_id, constants.STATUS_FILE, status_io)
        if success:
            logger.info(f"Updated status to completed for {operation} in run_id: {run_id}")
        return success
        
    except Exception as e:
        logger.error(f"Failed to update status to completed for run_id {run_id}: {str(e)}")
        return False


def _update_status_to_failed(run_id: str, gcs_bucket_name: str, error_message: str) -> bool:
    """Update status.json to indicate operation failed."""
    try:
        # Try to download current status.json, create minimal if not available
        status_bytes = download_run_file(run_id, constants.STATUS_FILE)
        if status_bytes:
            current_status = json.loads(status_bytes.decode('utf-8'))
        else:
            logger.warning(f"Could not download status.json for run_id: {run_id}, creating minimal status")
            current_status = {
                'run_id': run_id,
                'current_stage': constants.SCHEMA_STAGE
            }
        
        # Update status fields
        current_status.update({
            'status': 'failed',
            'message': f'Schema stage failed: {error_message}',
            'error_details': error_message,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'last_updated': datetime.now(timezone.utc).isoformat()
        })
        
        # Upload updated status back to GCS
        status_json = json.dumps(current_status, indent=2).encode('utf-8')
        status_io = io.BytesIO(status_json)
        
        success = upload_run_file(run_id, constants.STATUS_FILE, status_io)
        if success:
            logger.info(f"Updated status to failed for run_id: {run_id}")
        return success
        
    except Exception as e:
        logger.error(f"Failed to update status to failed for run_id {run_id}: {str(e)}")
        return False


# Legacy function maintained for backward compatibility
def confirm_target_definition(run_id: str, confirmed_target_column: str, 
                            confirmed_task_type: str, confirmed_target_ml_type: str) -> bool:
    """
    Legacy target definition confirmation function maintained for backward compatibility.
    This function is deprecated in favor of confirm_target_definition_gcs().
    """
    logger.warning("confirm_target_definition() is deprecated. Use confirm_target_definition_gcs() for GCS-based workflows.")
    
    # For now, delegate to the GCS version
    return confirm_target_definition_gcs(run_id, confirmed_target_column, confirmed_task_type, confirmed_target_ml_type) 