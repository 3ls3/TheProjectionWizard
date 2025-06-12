"""
Feature definition logic for The Projection Wizard.
Contains functions for identifying important features, suggesting data types and encoding roles,
and confirming feature schemas for the key features.
Refactored for GCS-based storage.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, f_regression, chi2
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import warnings
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


def identify_key_features_from_gcs(run_id: str, gcs_bucket_name: str = PROJECT_BUCKET_NAME,
                                  num_features_to_surface: int = 5) -> List[str]:
    """
    Download data from GCS and identify potentially important features using basic importance metrics.
    
    Args:
        run_id: The ID of the current run
        gcs_bucket_name: GCS bucket name (defaults to PROJECT_BUCKET_NAME)
        num_features_to_surface: How many top features to suggest (default to ~5)
        
    Returns:
        A list of top N feature column names, empty list if failed
    """
    logger.info(f"Starting feature identification for run_id: {run_id}")
    
    try:
        # Download original_data.csv from GCS
        csv_bytes = download_run_file(run_id, constants.ORIGINAL_DATA_FILE)
        if not csv_bytes:
            logger.error(f"Could not download original_data.csv for run_id: {run_id}")
            return []
        
        # Read CSV into DataFrame
        df_original = pd.read_csv(io.BytesIO(csv_bytes))
        logger.info(f"Successfully loaded DataFrame: {df_original.shape[0]} rows, {df_original.shape[1]} columns")
        
        # Download metadata to get target info
        metadata_bytes = download_run_file(run_id, constants.METADATA_FILE)
        if not metadata_bytes:
            logger.error(f"Could not download metadata.json for run_id: {run_id}")
            return []
        
        metadata = json.loads(metadata_bytes.decode('utf-8'))
        
        # Check if target_info exists in metadata
        if 'target_info' not in metadata:
            logger.error(f"No target_info found in metadata for run_id: {run_id}")
            return []
        
        target_info = metadata['target_info']
        
        # Identify features using the existing logic
        features = identify_key_features(df_original, target_info, num_features_to_surface)
        logger.info(f"Identified {len(features)} key features for run_id: {run_id}")
        
        return features
        
    except Exception as e:
        logger.error(f"Failed to identify key features for run_id {run_id}: {str(e)}", exc_info=True)
        return []


def suggest_initial_feature_schemas_from_gcs(run_id: str, gcs_bucket_name: str = PROJECT_BUCKET_NAME) -> Dict[str, Dict[str, str]]:
    """
    Download data from GCS and suggest initial data types and encoding roles for all columns based on heuristics.
    
    Args:
        run_id: The ID of the current run
        gcs_bucket_name: GCS bucket name (defaults to PROJECT_BUCKET_NAME)
        
    Returns:
        A dictionary where keys are column names, and values are dicts like 
        {'initial_dtype': str(df[col].dtype), 'suggested_encoding_role': 'role_suggestion'}
        Empty dict if failed
    """
    logger.info(f"Starting feature schema suggestions for run_id: {run_id}")
    
    try:
        # Download original_data.csv from GCS
        csv_bytes = download_run_file(run_id, constants.ORIGINAL_DATA_FILE)
        if not csv_bytes:
            logger.error(f"Could not download original_data.csv for run_id: {run_id}")
            return {}
        
        # Read CSV into DataFrame
        df = pd.read_csv(io.BytesIO(csv_bytes))
        logger.info(f"Successfully loaded DataFrame for schema suggestions: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Generate suggestions using existing logic
        suggestions = suggest_initial_feature_schemas(df)
        logger.info(f"Generated schema suggestions for {len(suggestions)} columns")
        
        return suggestions
        
    except Exception as e:
        logger.error(f"Failed to suggest feature schemas for run_id {run_id}: {str(e)}", exc_info=True)
        return {}


def _perform_minimal_stable_cleaning(df: pd.DataFrame, target_column_name: str) -> pd.DataFrame:
    """
    Internal helper function to perform very basic cleaning for stability of importance metric calculations.
    This cleaned state is temporary and NOT persisted for subsequent pipeline stages.
    
    Args:
        df: A copy of the original pandas DataFrame
        target_column_name: The name of the confirmed target column
        
    Returns:
        A pandas DataFrame that has undergone basic cleaning for metric calculation stability
    """
    df_cleaned = df.copy()  
    
    # Clean target column first
    target_series = df_cleaned[target_column_name]
    
    if pd.api.types.is_object_dtype(target_series.dtype) or pd.api.types.is_categorical_dtype(target_series.dtype):
        # Categorical target - encode numerically
        if target_series.isna().sum() > 0:
            # Fill NaN with a placeholder for encoding
            df_cleaned[target_column_name] = target_series.fillna("_MISSING_TARGET_")
        
        # Simple label encoding for target
        le = LabelEncoder()
        df_cleaned[target_column_name] = le.fit_transform(df_cleaned[target_column_name].astype(str))
        
    elif pd.api.types.is_numeric_dtype(target_series.dtype):
        # Numeric target
        if target_series.isna().sum() > 0:
            if pd.api.types.is_integer_dtype(target_series.dtype):
                # Classification likely - use mode
                fill_value = target_series.mode().iloc[0] if not target_series.mode().empty else 0
            else:
                # Regression likely - use median
                fill_value = target_series.median()
            df_cleaned[target_column_name] = target_series.fillna(fill_value)
    
    # Clean feature columns
    feature_columns = [col for col in df_cleaned.columns if col != target_column_name]
    
    for col in feature_columns:
        col_series = df_cleaned[col]
        
        if pd.api.types.is_object_dtype(col_series.dtype):
            # Try to convert to numeric first
            numeric_converted = pd.to_numeric(col_series, errors='coerce')
            if numeric_converted.notna().sum() / len(col_series) >= constants.SCHEMA_CONFIG["min_numeric_threshold"]:
                # Most values are numeric - treat as numeric
                df_cleaned[col] = numeric_converted.fillna(numeric_converted.median())
            else:
                # Treat as categorical
                df_cleaned[col] = col_series.fillna("_MISSING_")
        
        elif pd.api.types.is_numeric_dtype(col_series.dtype):
            # Already numeric - just fill NaN
            fill_value = col_series.median() if col_series.notna().sum() > 0 else 0
            df_cleaned[col] = col_series.fillna(fill_value)
        
        elif pd.api.types.is_bool_dtype(col_series.dtype):
            # Boolean - convert to int and fill NaN
            df_cleaned[col] = col_series.astype(int).fillna(0)
        
        else:
            # Other types (datetime, etc.) - convert to string for now
            df_cleaned[col] = col_series.astype(str).fillna("_MISSING_")
    
    return df_cleaned


def identify_key_features(df_original: pd.DataFrame, target_info: dict, 
                         num_features_to_surface: int = 5) -> List[str]:
    """
    Identify potentially important features using basic importance metrics.
    
    Args:
        df_original: The original pandas DataFrame (loaded from original_data.csv)
        target_info: The dictionary for the target column (name, task_type, ml_type) from metadata.json
        num_features_to_surface: How many top features to suggest (default to ~5)
        
    Returns:
        A list of top N feature column names
    """
    target_column_name = target_info['name']
    task_type = target_info['task_type']
    
    # Get feature columns (exclude target)
    feature_columns = [col for col in df_original.columns if col != target_column_name]
    
    if len(feature_columns) == 0:
        return []
    
    # Limit to available features if requested number is too high
    num_features_to_surface = min(num_features_to_surface, len(feature_columns))
    
    try:
        # Make a copy and perform minimal cleaning
        df_for_analysis = df_original.copy()
        df_cleaned = _perform_minimal_stable_cleaning(df_for_analysis, target_column_name)
        
        # Separate features and target
        X = df_cleaned[feature_columns]
        y = df_cleaned[target_column_name]
        
        # Ensure target is in proper format
        if task_type == "classification":
            y = y.astype(int)
        else:  # regression
            y = y.astype(float)
        
        # Calculate importance scores
        feature_scores = {}
        
        if task_type == "classification":
            # For classification, use mutual information
            try:
                # Prepare features for mutual info
                X_prepared = X.copy()
                
                # Encode categorical features for mutual info
                for col in X_prepared.columns:
                    if pd.api.types.is_object_dtype(X_prepared[col].dtype):
                        le = LabelEncoder()
                        X_prepared[col] = le.fit_transform(X_prepared[col].astype(str))
                
                # Calculate mutual information
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mi_scores = mutual_info_classif(
                        X_prepared, y, 
                        random_state=constants.SCHEMA_CONFIG["mutual_info_random_state"]
                    )
                
                feature_scores = dict(zip(feature_columns, mi_scores))
                
            except Exception as e:
                # Fallback to simpler correlation-based method
                for col in feature_columns:
                    try:
                        if pd.api.types.is_numeric_dtype(X[col].dtype):
                            # For numeric features, use correlation
                            corr = abs(X[col].corr(y))
                            feature_scores[col] = corr if not pd.isna(corr) else 0.0
                        else:
                            # For categorical, use simple association measure
                            feature_scores[col] = 0.1  # Low default score
                    except:
                        feature_scores[col] = 0.0
        
        else:  # regression
            try:
                # Prepare features for f_regression
                X_numeric = X.copy()
                
                # Convert categorical to numeric for f_regression
                for col in X_numeric.columns:
                    if pd.api.types.is_object_dtype(X_numeric[col].dtype):
                        le = LabelEncoder()
                        X_numeric[col] = le.fit_transform(X_numeric[col].astype(str))
                
                # Use f_regression
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    f_scores, _ = f_regression(X_numeric, y)
                
                # Convert to positive scores (handle NaN)
                f_scores = np.nan_to_num(f_scores, nan=0.0)
                feature_scores = dict(zip(feature_columns, f_scores))
                
            except Exception as e:
                # Fallback to correlation
                for col in feature_columns:
                    try:
                        if pd.api.types.is_numeric_dtype(X[col].dtype):
                            corr = abs(X[col].corr(y))
                            feature_scores[col] = corr if not pd.isna(corr) else 0.0
                        else:
                            feature_scores[col] = 0.1  # Low default score
                    except:
                        feature_scores[col] = 0.0
        
        # Sort features by importance score (descending)
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top N feature names
        top_features = [feature for feature, score in sorted_features[:num_features_to_surface]]
        
        return top_features
        
    except Exception as e:
        # If all else fails, return first N features
        return feature_columns[:num_features_to_surface]


def suggest_initial_feature_schemas(df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    """
    Suggest initial data types and encoding roles for all columns based on heuristics.
    
    Args:
        df: The original pandas DataFrame
        
    Returns:
        A dictionary where keys are column names, and values are dicts like 
        {'initial_dtype': str(df[col].dtype), 'suggested_encoding_role': 'role_suggestion'}
    """
    schema_suggestions = {}
    
    for col in df.columns:
        col_series = df[col]
        initial_dtype = str(col_series.dtype)
        nunique = col_series.nunique()
        total_values = len(col_series.dropna())
        
        # Default role
        suggested_encoding_role = "numeric-continuous"
        
        # Apply heuristics
        if pd.api.types.is_bool_dtype(col_series.dtype):
            suggested_encoding_role = "boolean"
            
        elif pd.api.types.is_numeric_dtype(col_series.dtype):
            # Get unique values excluding NaN
            unique_vals = col_series.dropna().unique()
            
            # Check if it's a binary 0/1 column (potential boolean)
            # Handle both int and float representations
            if len(unique_vals) == 2:
                unique_set = set(unique_vals)
                # Check for 0/1 in various forms (int, float)
                if (unique_set == {0, 1} or 
                    unique_set == {0.0, 1.0} or 
                    unique_set == {0, 1.0} or 
                    unique_set == {0.0, 1}):
                    suggested_encoding_role = "boolean"
                else:
                    # Not a boolean, continue with other numeric logic
                    suggested_encoding_role = _determine_numeric_role(col_series, col, unique_vals, nunique)
            else:
                # Not binary, determine numeric role
                suggested_encoding_role = _determine_numeric_role(col_series, col, unique_vals, nunique)
                
        elif pd.api.types.is_datetime64_any_dtype(col_series.dtype):
            suggested_encoding_role = "datetime"
            
        elif pd.api.types.is_object_dtype(col_series.dtype) or pd.api.types.is_categorical_dtype(col_series.dtype):
            # Analyze cardinality for object/categorical columns
            cardinality_ratio = nunique / total_values if total_values > 0 else 0
            
            # Check for identifier patterns
            col_lower = col.lower()
            if any(id_keyword in col_lower for id_keyword in ['id', 'uuid', 'key', 'index']):
                suggested_encoding_role = "text"  # Treat IDs as text to be ignored/hashed
            elif nunique <= 2:
                suggested_encoding_role = "categorical-nominal"
            elif nunique <= constants.SCHEMA_CONFIG["max_categorical_cardinality"]:
                suggested_encoding_role = "categorical-nominal"
            elif cardinality_ratio > 0.8:  # High cardinality
                suggested_encoding_role = "text"
            else:
                # Medium cardinality - could be ordinal
                suggested_encoding_role = "categorical-nominal"
        else:
            # Unknown types
            suggested_encoding_role = "text"
        
        schema_suggestions[col] = {
            'initial_dtype': initial_dtype,
            'suggested_encoding_role': suggested_encoding_role
        }
    
    return schema_suggestions


def _determine_numeric_role(col_series: pd.Series, col: str, unique_vals: np.ndarray, nunique: int) -> str:
    """
    Helper function to determine the encoding role for numeric columns.
    Handles the distinction between integer-like and float-like data even when 
    pandas has upcast due to NaN values.
    """
    col_lower = col.lower()
    
    # These column name patterns suggest numeric features even with few unique values
    numeric_indicators = [
        'age', 'year', 'count', 'number', 'num', 'score', 'rating', 'rank', 
        'bedroom', 'bathroom', 'room', 'floor', 'garage', 'space', 'feet', 
        'size', 'area', 'distance', 'mile', 'meter', 'inch', 'height', 'width',
        'length', 'depth', 'weight', 'price', 'cost', 'value', 'amount', 
        'percent', 'rate', 'level', 'grade', 'quality'
    ]
    
    # Check if column name suggests numeric nature
    is_likely_numeric = any(indicator in col_lower for indicator in numeric_indicators)
    
    # Check if all non-NaN values are actually integers (even if stored as float due to NaN)
    non_nan_values = col_series.dropna()
    if len(non_nan_values) > 0:
        # Check if all values are whole numbers (could be stored as float due to NaN)
        all_whole_numbers = all(
            isinstance(val, (int, np.integer)) or 
            (isinstance(val, (float, np.floating)) and val.is_integer())
            for val in non_nan_values
        )
        
        if all_whole_numbers and nunique <= 10 and not is_likely_numeric:
            # Few unique integers with no numeric indicators - might be categorical
            # But be conservative - only if values look like categories (e.g., 1,2,3 or discrete IDs)
            unique_vals_as_ints = sorted([
                int(val) if isinstance(val, (float, np.floating)) and val.is_integer() else val
                for val in unique_vals
            ])
            
            # If values are consecutive integers starting from 0 or 1, likely numeric
            if len(unique_vals_as_ints) > 1:
                min_val, max_val = min(unique_vals_as_ints), max(unique_vals_as_ints)
                if unique_vals_as_ints == list(range(min_val, max_val + 1)):
                    # Consecutive integers - likely numeric/ordinal
                    return "numeric-discrete"
                else:
                    # Non-consecutive integers - might be categorical
                    return "categorical-nominal"
            else:
                return "numeric-discrete"
        elif all_whole_numbers:
            # Integer-like values
            return "numeric-discrete"
        else:
            # Has decimal values
            return "numeric-continuous"
    else:
        # All values are NaN - default to continuous
        return "numeric-continuous"


def confirm_feature_schemas_gcs(run_id: str, user_confirmed_schemas: Dict[str, Dict[str, str]], 
                               all_initial_schemas: Dict[str, Dict[str, str]],
                               gcs_bucket_name: str = PROJECT_BUCKET_NAME) -> bool:
    """
    Confirm feature schemas and update metadata.json in GCS with feature schema information.
    
    Args:
        run_id: The ID of the current run
        user_confirmed_schemas: Dictionary where keys are column names (only for columns the user 
                               explicitly reviewed/changed) and values are dicts: 
                               {'final_dtype': str, 'final_encoding_role': str}
        all_initial_schemas: The full dictionary of initial suggestions for all columns 
                            (from suggest_initial_feature_schemas)
        gcs_bucket_name: GCS bucket name (defaults to PROJECT_BUCKET_NAME)
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Starting feature schema confirmation for run_id: {run_id}")
    
    try:
        # Step 1: Update status to processing
        if not _update_status_to_processing(run_id, gcs_bucket_name, "feature_schema_confirmation"):
            logger.error(f"Failed to update status to processing for run_id: {run_id}")
            return False
        
        # Step 2: Download and update metadata
        if not _update_metadata_with_feature_schemas(run_id, gcs_bucket_name, user_confirmed_schemas, all_initial_schemas):
            _update_status_to_failed(run_id, gcs_bucket_name, "Failed to update metadata with feature schemas")
            return False
        
        # Step 3: Update status to completed
        message = f"Feature schemas confirmed for {len(all_initial_schemas)} columns."
        if not _update_status_to_completed(run_id, gcs_bucket_name, "feature_schema_confirmation", message):
            logger.error(f"Failed to update status to completed for run_id: {run_id}")
            return False
        
        logger.info(f"Feature schemas confirmed successfully for {len(all_initial_schemas)} columns. "
                   f"User confirmed: {len(user_confirmed_schemas)}, "
                   f"System defaulted: {len(all_initial_schemas) - len(user_confirmed_schemas)}")
        return True
        
    except Exception as e:
        logger.error(f"Critical error during feature schema confirmation for run_id {run_id}: {str(e)}", exc_info=True)
        _update_status_to_failed(run_id, gcs_bucket_name, f"Critical error: {str(e)}")
        return False


def _update_metadata_with_feature_schemas(run_id: str, gcs_bucket_name: str,
                                         user_confirmed_schemas: Dict[str, Dict[str, str]],
                                         all_initial_schemas: Dict[str, Dict[str, str]]) -> bool:
    """Download metadata, update with feature schemas, and upload back to GCS."""
    try:
        # Download metadata.json
        metadata_bytes = download_run_file(run_id, constants.METADATA_FILE)
        if not metadata_bytes:
            logger.error(f"Could not download metadata.json for run_id: {run_id}")
            return False
        
        # Parse metadata
        metadata_dict = json.loads(metadata_bytes.decode('utf-8'))
        
        # Construct final feature schemas
        final_feature_schemas = {}
        
        for column_name in all_initial_schemas.keys():
            if column_name in user_confirmed_schemas:
                # User explicitly confirmed/changed this column
                final_dtype = user_confirmed_schemas[column_name]['final_dtype']
                final_encoding_role = user_confirmed_schemas[column_name]['final_encoding_role']
                source = 'user_confirmed'
            else:
                # Use system defaults for non-reviewed columns
                final_dtype = all_initial_schemas[column_name]['initial_dtype']
                final_encoding_role = all_initial_schemas[column_name]['suggested_encoding_role']
                source = 'system_defaulted'
            
            final_feature_schemas[column_name] = {
                'dtype': final_dtype,
                'encoding_role': final_encoding_role,
                'source': source
            }
        
        # Update metadata with feature schemas
        metadata_dict['feature_schemas'] = final_feature_schemas
        metadata_dict['feature_schemas_confirmed_at'] = datetime.now(timezone.utc).isoformat()
        
        # Upload updated metadata back to GCS
        metadata_json = json.dumps(metadata_dict, indent=2, default=str).encode('utf-8')
        metadata_io = io.BytesIO(metadata_json)
        
        success = upload_run_file(run_id, constants.METADATA_FILE, metadata_io)
        if success:
            logger.info(f"Updated metadata with feature schemas for run_id: {run_id}")
        return success
        
    except Exception as e:
        logger.error(f"Failed to update metadata with feature schemas for run_id {run_id}: {str(e)}")
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
def confirm_feature_schemas(run_id: str, user_confirmed_schemas: Dict[str, Dict[str, str]], 
                           all_initial_schemas: Dict[str, Dict[str, str]]) -> bool:
    """
    Legacy feature schema confirmation function maintained for backward compatibility.
    This function is deprecated in favor of confirm_feature_schemas_gcs().
    """
    logger.warning("confirm_feature_schemas() is deprecated. Use confirm_feature_schemas_gcs() for GCS-based workflows.")
    
    # For now, delegate to the GCS version
    return confirm_feature_schemas_gcs(run_id, user_confirmed_schemas, all_initial_schemas) 