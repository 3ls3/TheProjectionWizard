"""
Feature definition logic for The Projection Wizard.
Contains functions for identifying important features, suggesting data types and encoding roles,
and confirming feature schemas for the key features.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, f_regression, chi2
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import warnings

from common import logger, storage, constants


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
            if pd.api.types.is_integer_dtype(col_series.dtype) and nunique <= 10:
                # Few unique integers - might be categorical
                suggested_encoding_role = "categorical-nominal"
            else:
                suggested_encoding_role = "numeric-continuous"
                
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


def confirm_feature_schemas(run_id: str, user_confirmed_schemas: Dict[str, Dict[str, str]], 
                           all_initial_schemas: Dict[str, Dict[str, str]]) -> bool:
    """
    Confirm feature schemas and update metadata.json with feature schema information.
    
    Args:
        run_id: The ID of the current run
        user_confirmed_schemas: Dictionary where keys are column names (only for columns the user 
                               explicitly reviewed/changed) and values are dicts: 
                               {'final_dtype': str, 'final_encoding_role': str}
        all_initial_schemas: The full dictionary of initial suggestions for all columns 
                            (from suggest_initial_feature_schemas)
        
    Returns:
        True if successful, False otherwise
    """
    # Get loggers
    run_logger = logger.get_stage_logger(run_id, constants.SCHEMA_STAGE)
    structured_log = logger.get_stage_structured_logger(run_id, constants.SCHEMA_STAGE)
    
    try:
        # Structured log: Feature schema confirmation started
        logger.log_structured_event(
            structured_log,
            "feature_schema_confirmation_started",
            {
                "total_columns": len(all_initial_schemas),
                "user_confirmed_count": len(user_confirmed_schemas),
                "system_default_count": len(all_initial_schemas) - len(user_confirmed_schemas)
            },
            f"Feature schema confirmation started for {len(all_initial_schemas)} columns"
        )
        
        # Read existing metadata.json
        metadata_dict = storage.read_metadata(run_id)
        if metadata_dict is None:
            error_msg = f"Could not read metadata.json for run {run_id}"
            run_logger.error(error_msg)
            logger.log_structured_error(
                structured_log,
                "metadata_load_failed",
                error_msg,
                {"stage": constants.SCHEMA_STAGE}
            )
            return False
        
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
        
        # Analyze feature schema distribution for structured logging
        encoding_role_counts = {}
        source_counts = {"user_confirmed": 0, "system_defaulted": 0}
        
        for schema_info in final_feature_schemas.values():
            role = schema_info['encoding_role']
            source = schema_info['source']
            
            encoding_role_counts[role] = encoding_role_counts.get(role, 0) + 1
            source_counts[source] += 1
        
        # Structured log: Feature schemas processed
        logger.log_structured_event(
            structured_log,
            "feature_schemas_processed",
            {
                "total_features": len(final_feature_schemas),
                "encoding_role_distribution": encoding_role_counts,
                "source_distribution": source_counts,
                "user_confirmed_count": source_counts["user_confirmed"],
                "system_defaulted_count": source_counts["system_defaulted"]
            },
            f"Feature schemas processed: {len(final_feature_schemas)} columns configured"
        )
        
        # Update metadata with feature schemas
        metadata_dict['feature_schemas'] = final_feature_schemas
        metadata_dict['feature_schemas_confirmed_at'] = datetime.now(timezone.utc).isoformat()
        
        # Write updated metadata.json
        storage.write_metadata(run_id, metadata_dict)
        
        # Structured log: Metadata updated
        logger.log_structured_event(
            structured_log,
            "metadata_updated",
            {
                "feature_schemas_count": len(final_feature_schemas),
                "metadata_keys_added": ["feature_schemas", "feature_schemas_confirmed_at"]
            },
            f"Metadata updated with {len(final_feature_schemas)} feature schemas"
        )
        
        # Update status.json
        status_data = {
            'stage': constants.SCHEMA_STAGE,
            'status': 'completed',
            'message': f"Feature schemas confirmed for {len(final_feature_schemas)} columns.",
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'errors': []
        }
        
        storage.write_status(run_id, status_data)
        
        # Structured log: Status updated
        logger.log_structured_event(
            structured_log,
            "status_updated",
            {
                "status": "completed",
                "stage": constants.SCHEMA_STAGE,
                "feature_count": len(final_feature_schemas)
            },
            "Feature schema confirmation status updated to completed"
        )
        
        run_logger.info(f"Feature schemas confirmed for {len(final_feature_schemas)} columns. "
                       f"User confirmed: {len(user_confirmed_schemas)}, "
                       f"System defaulted: {len(final_feature_schemas) - len(user_confirmed_schemas)}")
        
        # Structured log: Feature schema confirmation completed
        logger.log_structured_event(
            structured_log,
            "feature_schema_confirmation_completed",
            {
                "success": True,
                "total_columns": len(final_feature_schemas),
                "user_confirmed": len(user_confirmed_schemas),
                "system_defaulted": len(final_feature_schemas) - len(user_confirmed_schemas),
                "encoding_roles": list(encoding_role_counts.keys())
            },
            f"Feature schema confirmation completed successfully for {len(final_feature_schemas)} columns"
        )
        
        return True
        
    except Exception as e:
        error_msg = f"Failed to confirm feature schemas: {str(e)}"
        run_logger.error(error_msg)
        
        # Structured log: Feature schema confirmation failed
        logger.log_structured_error(
            structured_log,
            "feature_schema_confirmation_failed",
            error_msg,
            {
                "stage": constants.SCHEMA_STAGE,
                "total_columns": len(all_initial_schemas),
                "user_confirmed_count": len(user_confirmed_schemas)
            }
        )
        
        # Update status.json with error
        try:
            status_data = {
                'stage': constants.SCHEMA_STAGE,
                'status': 'failed',
                'message': f"Failed to confirm feature schemas: {str(e)}",
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'errors': [str(e)]
            }
            storage.write_status(run_id, status_data)
        except Exception as status_error:
            run_logger.error(f"Failed to update status.json with error: {str(status_error)}")
        
        return False 