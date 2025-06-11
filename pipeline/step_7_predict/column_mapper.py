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
import numpy as np

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


def get_enhanced_prediction_schema_gcs(run_id: str,
                                      df_original: pd.DataFrame,
                                      target_column: str,
                                      gcs_bucket_name: str = PROJECT_BUCKET_NAME) -> Dict[str, Any]:
    """
    Generate enhanced prediction schema with rich UI metadata for Option 2 API.

    Args:
        run_id: Run identifier
        df_original: Original dataset
        target_column: Target column name
        gcs_bucket_name: GCS bucket name

    Returns:
        Dictionary containing enhanced schema information for rich UI
    """
    try:
        # Load metadata from GCS
        metadata_bytes = download_run_file(run_id, 'metadata.json')
        if metadata_bytes is None:
            raise Exception("Could not load metadata from GCS")
        
        metadata = json.loads(metadata_bytes.decode('utf-8'))
        target_info = metadata.get('target_info', {})
        automl_info = metadata.get('automl_info', {})
        
        # Get basic schema
        basic_schema = get_input_schema(run_id, df_original, target_column)
        
        # Enhanced schema structure
        enhanced_schema = {
            'numeric_columns': {},
            'categorical_columns': {},
            'feature_metadata': {},
            'target_info': target_info,
            'model_info': {
                'model_name': automl_info.get('best_model_name', 'Unknown'),
                'task_type': target_info.get('task_type', 'unknown'),
                'performance_metrics': automl_info.get('performance_metrics', {})
            },
            'validation_rules': {}
        }
        
        # Try to load feature importance if available
        feature_importance = {}
        try:
            importance_bytes = download_run_file(run_id, 'feature_importance.json')
            if importance_bytes:
                importance_data = json.loads(importance_bytes.decode('utf-8'))
                feature_importance = importance_data.get('feature_importance', {})
        except:
            log.warning(f"Could not load feature importance for run {run_id}")
        
        # Calculate correlations with target
        target_correlations = {}
        if target_column in df_original.columns:
            try:
                for col in df_original.columns:
                    if col != target_column and pd.api.types.is_numeric_dtype(df_original[col]):
                        correlation = df_original[col].corr(df_original[target_column])
                        if pd.notna(correlation):
                            target_correlations[col] = float(correlation)
            except Exception as e:
                log.warning(f"Could not calculate correlations: {e}")
        
        # Process numeric columns
        for col, col_info in basic_schema['numeric_columns'].items():
            col_data = df_original[col].dropna()
            
            # Calculate enhanced statistics
            q25 = float(col_data.quantile(0.25))
            q75 = float(col_data.quantile(0.75))
            median = float(col_data.median())
            
            # Calculate appropriate step size
            data_range = col_info['max_value'] - col_info['min_value']
            step_size = data_range / 100 if data_range > 0 else 1.0
            
            # Round step size to reasonable precision
            if step_size < 1:
                step_size = round(step_size, max(0, -int(np.floor(np.log10(abs(step_size))))))
            else:
                step_size = max(1, round(step_size))
            
            # Suggest default value (prefer median over mean for robustness)
            suggested_value = median
            
            enhanced_schema['numeric_columns'][col] = {
                'min_value': col_info['min_value'],
                'max_value': col_info['max_value'],
                'default_value': col_info['mean_value'],
                'step_size': step_size,
                'suggested_value': suggested_value,
                'display_format': '.3f' if step_size < 1 else '.0f'
            }
            
            # Add feature metadata
            importance_rank = None
            importance_score = None
            if col in feature_importance:
                importance_score = float(feature_importance[col])
                # Calculate rank based on importance
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                importance_rank = next((i+1 for i, (feat, _) in enumerate(sorted_features) if feat == col), None)
            
            enhanced_schema['feature_metadata'][col] = {
                'importance_rank': importance_rank,
                'importance_score': importance_score,
                'correlation_with_target': target_correlations.get(col),
                'data_type': 'numeric',
                'quartiles': {'q25': q25, 'median': median, 'q75': q75},
                'category': _infer_feature_category(col)
            }
            
            # Add validation rules
            enhanced_schema['validation_rules'][col] = {
                'required': True,
                'min_value': col_info['min_value'],
                'max_value': col_info['max_value'],
                'type': 'numeric'
            }
        
        # Process categorical columns
        for col, col_info in basic_schema['categorical_columns'].items():
            col_data = df_original[col].dropna()
            
            # Calculate option frequencies for better UX
            value_counts = col_data.value_counts()
            option_frequencies = {str(val): int(count) for val, count in value_counts.items()}
            
            # Create display names (could be enhanced with domain knowledge)
            display_names = {opt: _create_display_name(opt) for opt in col_info['options']}
            
            # Find most common option as suggested default
            most_common = value_counts.index[0] if len(value_counts) > 0 else col_info['options'][0]
            
            enhanced_schema['categorical_columns'][col] = {
                'options': col_info['options'],
                'default_option': str(most_common),
                'display_names': display_names,
                'option_frequencies': option_frequencies
            }
            
            # Add feature metadata
            importance_rank = None
            importance_score = None
            if col in feature_importance:
                importance_score = float(feature_importance[col])
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                importance_rank = next((i+1 for i, (feat, _) in enumerate(sorted_features) if feat == col), None)
            
            enhanced_schema['feature_metadata'][col] = {
                'importance_rank': importance_rank,
                'importance_score': importance_score,
                'correlation_with_target': None,  # Categorical correlation more complex
                'data_type': 'categorical',
                'cardinality': len(col_info['options']),
                'category': _infer_feature_category(col)
            }
            
            # Add validation rules
            enhanced_schema['validation_rules'][col] = {
                'required': True,
                'allowed_values': col_info['options'],
                'type': 'categorical'
            }
        
        log.info(f"Generated enhanced schema for {len(enhanced_schema['numeric_columns'])} numeric and {len(enhanced_schema['categorical_columns'])} categorical columns")
        return enhanced_schema
        
    except Exception as e:
        log.error(f"Failed to generate enhanced prediction schema: {e}")
        # Fallback to basic schema
        return get_input_schema(run_id, df_original, target_column)


def _infer_feature_category(column_name: str) -> str:
    """
    Infer feature category from column name for better UI organization.
    
    Args:
        column_name: Name of the column
        
    Returns:
        Inferred category string
    """
    col_lower = column_name.lower()
    
    # Demographic features
    if any(term in col_lower for term in ['age', 'gender', 'sex', 'race', 'ethnicity', 'marital', 'education']):
        return 'demographic'
    
    # Financial features
    if any(term in col_lower for term in ['income', 'salary', 'price', 'cost', 'amount', 'balance', 'payment']):
        return 'financial'
    
    # Behavioral features
    if any(term in col_lower for term in ['frequency', 'count', 'usage', 'activity', 'behavior', 'visits']):
        return 'behavioral'
    
    # Geographic features
    if any(term in col_lower for term in ['location', 'city', 'state', 'country', 'region', 'zip', 'postal']):
        return 'geographic'
    
    # Temporal features
    if any(term in col_lower for term in ['date', 'time', 'year', 'month', 'day', 'duration', 'since']):
        return 'temporal'
    
    # Product/Service features
    if any(term in col_lower for term in ['product', 'service', 'category', 'type', 'brand', 'model']):
        return 'product'
    
    return 'general'


def _create_display_name(option_value: str) -> str:
    """
    Create user-friendly display name for categorical options.
    
    Args:
        option_value: Raw option value
        
    Returns:
        User-friendly display name
    """
    # Convert to string and clean up
    display = str(option_value).strip()
    
    # Replace underscores and hyphens with spaces
    display = display.replace('_', ' ').replace('-', ' ')
    
    # Capitalize words
    display = ' '.join(word.capitalize() for word in display.split())
    
    return display


def get_enhanced_prediction_schema(run_id: str, df_original: pd.DataFrame, target_column: str) -> Dict[str, Any]:
    """
    Legacy compatibility function - redirects to GCS version.

    Args:
        run_id: Run identifier
        df_original: Original dataset
        target_column: Target column name

    Returns:
        Dictionary containing enhanced schema information
    """
    log.warning("Using legacy get_enhanced_prediction_schema function - redirecting to GCS version")
    return get_enhanced_prediction_schema_gcs(run_id, df_original, target_column)


def generate_enhanced_prediction_schema(df: pd.DataFrame, target_column: str = None, metadata: dict = None) -> dict:
    """
    Generate enhanced prediction schema with rich UI metadata and slider configurations.
    Now integrates real SHAP-based feature importance when available.
    
    Args:
        df: Original DataFrame
        target_column: Name of target column to exclude
        metadata: Additional metadata (can include model and task_type for SHAP)
        
    Returns:
        Dictionary with enhanced schema information for UI
    """
    from datetime import datetime
    
    log = logging.getLogger(__name__)
    
    # Remove target column if present  
    if target_column and target_column in df.columns:
        df = df.drop(columns=[target_column])
    
    features_config = []
    feature_categories = categorize_features_by_type(df.columns.tolist())
    
    # Try to get feature importance from SHAP if model is available
    feature_importance = {}
    if metadata and 'model' in metadata and 'task_type' in metadata:
        try:
            from .predict_logic import get_global_feature_importance_from_shap
            
            model = metadata['model']
            task_type = metadata['task_type']
            sample_data = df.sample(n=min(100, len(df)), random_state=42) if len(df) > 100 else df
            
            feature_importance = get_global_feature_importance_from_shap(
                model, sample_data, target_column, task_type
            )
            
            if feature_importance:
                log.info(f"Successfully retrieved SHAP-based feature importance for {len(feature_importance)} features")
            else:
                log.info("SHAP-based feature importance not available, using column ordering")
                
        except Exception as e:
            log.warning(f"Could not get SHAP-based feature importance: {e}")
    
    # If no SHAP importance available, create basic importance ranking
    if not feature_importance:
        # Basic importance based on variance for numeric columns
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                try:
                    # Use coefficient of variation as basic importance measure
                    mean_val = df[col].mean()
                    if mean_val != 0:
                        cv = df[col].std() / abs(mean_val)
                        feature_importance[col] = float(cv)
                    else:
                        feature_importance[col] = float(df[col].std())
                except:
                    feature_importance[col] = 1.0
            else:
                # For categorical columns, use number of unique values
                feature_importance[col] = float(len(df[col].unique()))
    
    # Normalize importance scores to 0-1 range
    if feature_importance:
        max_importance = max(feature_importance.values())
        if max_importance > 0:
            feature_importance = {k: v/max_importance for k, v in feature_importance.items()}
    
    # Sort features by importance (highest first)
    sorted_features = sorted(df.columns, key=lambda x: feature_importance.get(x, 0), reverse=True)
    
    for i, column in enumerate(sorted_features):
        col_data = df[column]
        feature_info = generate_feature_slider_config(col_data, column)
        
        # Add importance and ranking information
        importance_score = feature_importance.get(column, 0.0)
        feature_info['importance_score'] = importance_score
        feature_info['importance_rank'] = i + 1
        feature_info['category'] = feature_categories.get(column, 'other')
        feature_info['shap_available'] = bool(metadata and 'model' in metadata)
        
        # Add guidance based on importance
        if importance_score >= 0.7:
            feature_info['ui_guidance'] = {
                'priority': 'high',
                'recommendation': 'Key feature - small changes may significantly impact predictions',
                'highlight': True
            }
        elif importance_score >= 0.3:
            feature_info['ui_guidance'] = {
                'priority': 'medium', 
                'recommendation': 'Moderate impact on predictions',
                'highlight': False
            }
        else:
            feature_info['ui_guidance'] = {
                'priority': 'low',
                'recommendation': 'Lower impact on predictions',
                'highlight': False
            }
        
        features_config.append(feature_info)
    
    # Calculate schema-level statistics
    total_features = len(features_config)
    high_importance_count = sum(1 for f in features_config if f['importance_score'] >= 0.7)
    medium_importance_count = sum(1 for f in features_config if 0.3 <= f['importance_score'] < 0.7)
    
    return {
        'features': features_config,
        'schema_metadata': {
            'total_features': total_features,
            'high_importance_features': high_importance_count,
            'medium_importance_features': medium_importance_count,
            'low_importance_features': total_features - high_importance_count - medium_importance_count,
            'feature_categories': {category: len(features) for category, features in feature_categories.items()},
            'importance_method': 'shap' if (metadata and 'model' in metadata and feature_importance) else 'statistical',
            'schema_version': '2.0',
            'generated_timestamp': datetime.now().isoformat(),
            'target_column': target_column,
            'supports_shap_explanations': bool(metadata and 'model' in metadata)
        },
        'ui_recommendations': {
            'primary_features': [f['column_name'] for f in features_config if f['importance_score'] >= 0.7][:5],
            'suggested_focus_order': [f['column_name'] for f in features_config[:10]],  # Top 10 by importance
            'feature_grouping_suggested': True if total_features > 15 else False,
            'show_importance_indicators': True,
            'enable_real_time_shap': bool(metadata and 'model' in metadata)
        }
    }
