"""
Feature encoding logic for The Projection Wizard.
Contains functions for converting cleaned data into ML-ready formats.
Refactored for GCS-based storage.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import joblib
import warnings
import io
import logging
import tempfile

# sklearn imports
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from common import constants, schemas
from api.utils.gcs_utils import upload_run_file, PROJECT_BUCKET_NAME

# Configure logging for this module
logger = logging.getLogger(__name__)


def encode_features_gcs(df_cleaned: pd.DataFrame, 
                       feature_schemas: Dict[str, schemas.FeatureSchemaInfo], 
                       target_info: schemas.TargetInfo,
                       run_id: str,
                       gcs_bucket_name: str = PROJECT_BUCKET_NAME) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Encode features for ML based on feature schemas and target information.
    Saves encoders/scalers to GCS instead of local filesystem.
    
    Args:
        df_cleaned: The DataFrame after cleaning from cleaning_logic.py
        feature_schemas: Dictionary of FeatureSchemaInfo objects
        target_info: TargetInfo object with target information
        run_id: Current run ID, used for saving encoders/scalers
        gcs_bucket_name: GCS bucket name
        
    Returns:
        Tuple containing:
        - encoded_dataframe: The DataFrame with ML-ready encoded features
        - encoders_scalers_info: Dictionary with paths and info about saved encoders/scalers
    """
    # Create a copy to avoid modifying the input
    df_encoded = df_cleaned.copy()
    encoders_scalers_info = {}
    
    logger.info(f"Starting feature encoding for {len(df_encoded.columns)} columns")
    
    # Step 1: Handle Target Variable Encoding
    df_encoded, target_encoder_info = _encode_target_variable_gcs(
        df_encoded, target_info, run_id, gcs_bucket_name
    )
    if target_encoder_info:
        encoders_scalers_info.update(target_encoder_info)
    
    # Step 2: Handle Feature Encoding (non-target columns)
    df_encoded, feature_encoder_info = _encode_features_gcs(
        df_encoded, feature_schemas, target_info, run_id, gcs_bucket_name
    )
    encoders_scalers_info.update(feature_encoder_info)
    
    logger.info(f"Encoding completed. Final shape: {df_encoded.shape}")
    logger.info(f"Saved {len(encoders_scalers_info)} encoders/scalers to GCS")
    
    return df_encoded, encoders_scalers_info


def _encode_target_variable_gcs(df: pd.DataFrame, 
                               target_info: schemas.TargetInfo,
                               run_id: str,
                               gcs_bucket_name: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Encode the target variable based on its ML type.
    Saves encoders to GCS.
    
    Args:
        df: DataFrame to modify (modified in place)
        target_info: TargetInfo object
        run_id: Run ID for GCS paths
        gcs_bucket_name: GCS bucket name
        
    Returns:
        Tuple of (modified DataFrame, encoder info dict)
    """
    target_col = target_info.name
    ml_type = target_info.ml_type
    encoder_info = {}
    
    if target_col not in df.columns:
        logger.warning(f"Target column '{target_col}' not found in DataFrame")
        return df, encoder_info
        
    logger.info(f"Encoding target column '{target_col}' with ML type '{ml_type}'")
    
    if ml_type == "binary_01":
        # Ensure column is int 0/1
        unique_vals = df[target_col].unique()
        if set(unique_vals) == {0, 1} or set(unique_vals) == {0.0, 1.0}:
            df[target_col] = df[target_col].astype(int)
            logger.info(f"Target '{target_col}' already in 0/1 format, converted to int")
        elif set(unique_vals) == {True, False}:
            df[target_col] = df[target_col].astype(int)
            logger.info(f"Target '{target_col}' converted from boolean to 0/1")
        else:
            # Map other values to 0/1 (take first unique as 0, second as 1)
            sorted_vals = sorted(unique_vals)
            mapping = {sorted_vals[0]: 0, sorted_vals[1]: 1}
            df[target_col] = df[target_col].map(mapping)
            logger.info(f"Target '{target_col}' mapped to 0/1: {mapping}")
            
    elif ml_type == "multiclass_int_labels":
        # Ensure column is int
        df[target_col] = df[target_col].astype(int)
        logger.info(f"Target '{target_col}' converted to int labels")
        
    elif ml_type in ["binary_text_labels", "multiclass_text_labels"]:
        # Use LabelEncoder
        le = LabelEncoder()
        df[target_col] = le.fit_transform(df[target_col])
        
        # Save the encoder to GCS
        encoder_filename = f"{target_col}_label_encoder.joblib"
        gcs_path = f"models/{encoder_filename}"
        
        # Serialize encoder to bytes and upload to GCS
        with tempfile.NamedTemporaryFile() as tmp_file:
            joblib.dump(le, tmp_file.name)
            with open(tmp_file.name, 'rb') as f:
                encoder_bytes = f.read()
        
        upload_success = upload_run_file(run_id, gcs_path, io.BytesIO(encoder_bytes))
        
        if upload_success:
            encoder_info[f"{target_col}_label_encoder"] = {
                "type": "LabelEncoder",
                "gcs_path": gcs_path,
                "classes": le.classes_.tolist(),
                "column": target_col
            }
            logger.info(f"Target '{target_col}' encoded with LabelEncoder, saved to GCS: {gcs_path}")
        else:
            logger.error(f"Failed to save LabelEncoder to GCS: {gcs_path}")
        
    elif ml_type == "numeric_continuous":
        # Ensure it's float
        df[target_col] = df[target_col].astype(float)
        logger.info(f"Target '{target_col}' converted to float")
        
    else:
        logger.warning(f"Unknown target ML type '{ml_type}', leaving as-is")
    
    return df, encoder_info


def _encode_features_gcs(df: pd.DataFrame,
                        feature_schemas: Dict[str, schemas.FeatureSchemaInfo],
                        target_info: schemas.TargetInfo,
                        run_id: str,
                        gcs_bucket_name: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Encode feature columns based on their encoding roles.
    Saves encoders to GCS.
    
    Args:
        df: DataFrame to modify
        feature_schemas: Dictionary of feature schema info
        target_info: Target information
        run_id: Run ID for GCS paths
        gcs_bucket_name: GCS bucket name
        
    Returns:
        Tuple of (modified DataFrame, encoders info dict)
    """
    encoders_info = {}
    columns_to_drop = []
    
    # Process each feature column (excluding target)
    for col in df.columns:
        if col == target_info.name:
            continue  # Skip target column
            
        if col not in feature_schemas:
            logger.warning(f"Column '{col}' not found in feature schemas, skipping encoding")
            continue
            
        encoding_role = feature_schemas[col].encoding_role
        logger.info(f"Encoding column '{col}' with role '{encoding_role}'")
        
        if encoding_role in ["numeric-continuous", "numeric-discrete"]:
            # Apply StandardScaler
            scaler = StandardScaler()
            df[col] = scaler.fit_transform(df[[col]]).flatten()
            
            # Save scaler to GCS
            scaler_filename = f"{col}_scaler.joblib"
            gcs_path = f"models/{scaler_filename}"
            
            # Serialize scaler to bytes and upload to GCS
            with tempfile.NamedTemporaryFile() as tmp_file:
                joblib.dump(scaler, tmp_file.name)
                with open(tmp_file.name, 'rb') as f:
                    scaler_bytes = f.read()
            
            upload_success = upload_run_file(run_id, gcs_path, io.BytesIO(scaler_bytes))
            
            if upload_success:
                encoders_info[f"{col}_scaler"] = {
                    "type": "StandardScaler",
                    "gcs_path": gcs_path,
                    "column": col,
                    "mean": float(scaler.mean_[0]),
                    "scale": float(scaler.scale_[0])
                }
                logger.info(f"Applied StandardScaler to '{col}', saved to GCS: {gcs_path}")
            else:
                logger.error(f"Failed to save StandardScaler to GCS: {gcs_path}")
            
        elif encoding_role == "categorical-nominal":
            # Use pd.get_dummies for one-hot encoding
            original_cols = set(df.columns)
            df = pd.get_dummies(df, columns=[col], prefix=col, dummy_na=False)
            new_cols = list(set(df.columns) - original_cols)
            
            encoders_info[f"{col}_onehot"] = {
                "type": "OneHotEncoder",
                "original_column": col,
                "new_columns": new_cols,
                "n_categories": len(new_cols)
            }
            
            logger.info(f"Applied one-hot encoding to '{col}', created {len(new_cols)} new columns")
            
        elif encoding_role == "categorical-ordinal":
            # Handle ordinal encoding - for MVP, use simple integer codes
            # In future, could add ordinal_order to FeatureSchemaInfo
            if pd.api.types.is_categorical_dtype(df[col]):
                df[col] = df[col].cat.codes
            else:
                # Convert to categorical first, then to codes
                df[col] = pd.Categorical(df[col]).codes
                
            encoders_info[f"{col}_ordinal"] = {
                "type": "OrdinalEncoder_Simple",
                "column": col,
                "method": "categorical_codes"
            }
            
            logger.info(f"Applied simple ordinal encoding to '{col}' using categorical codes")
            
        elif encoding_role == "boolean":
            # Ensure 0/1 or True/False
            unique_vals = df[col].unique()
            if set(unique_vals) <= {True, False}:
                df[col] = df[col].astype(int)
                logger.info(f"Boolean column '{col}' converted to 0/1")
            elif set(unique_vals) <= {0, 1, 0.0, 1.0}:
                df[col] = df[col].astype(int)
                logger.info(f"Boolean column '{col}' ensured as int 0/1")
            else:
                # Handle text boolean values
                df[col] = _encode_text_boolean(df[col])
                logger.info(f"Text boolean column '{col}' converted to 0/1")
                
        elif encoding_role == "datetime":
            # Extract datetime features
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[f'{col}_year'] = df[col].dt.year
                df[f'{col}_month'] = df[col].dt.month
                df[f'{col}_day'] = df[col].dt.day
                df[f'{col}_dayofweek'] = df[col].dt.dayofweek
                
                new_features = [f'{col}_year', f'{col}_month', f'{col}_day', f'{col}_dayofweek']
                columns_to_drop.append(col)
                
                encoders_info[f"{col}_datetime"] = {
                    "type": "DatetimeFeatureExtractor",
                    "original_column": col,
                    "new_features": new_features
                }
                
                logger.info(f"Extracted datetime features from '{col}': {new_features}")
            else:
                logger.warning(f"Column '{col}' marked as datetime but not datetime type")
                
        elif encoding_role == "text":
            # Apply TF-IDF vectorization
            vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
            
            # Ensure text data is string type and handle NaN
            text_data = df[col].fillna('').astype(str)
            
            try:
                tfidf_matrix = vectorizer.fit_transform(text_data)
                
                # Create new columns for TF-IDF features
                feature_names = [f"{col}_tfidf_{i}" for i in range(tfidf_matrix.shape[1])]
                tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names, index=df.index)
                
                # Add TF-IDF columns to main dataframe
                df = pd.concat([df, tfidf_df], axis=1)
                columns_to_drop.append(col)
                
                # Save vectorizer to GCS
                vectorizer_filename = f"{col}_tfidf_vectorizer.joblib"
                gcs_path = f"models/{vectorizer_filename}"
                
                # Serialize vectorizer to bytes and upload to GCS
                with tempfile.NamedTemporaryFile() as tmp_file:
                    joblib.dump(vectorizer, tmp_file.name)
                    with open(tmp_file.name, 'rb') as f:
                        vectorizer_bytes = f.read()
                
                upload_success = upload_run_file(run_id, gcs_path, io.BytesIO(vectorizer_bytes))
                
                if upload_success:
                    encoders_info[f"{col}_tfidf"] = {
                        "type": "TfidfVectorizer",
                        "gcs_path": gcs_path,
                        "original_column": col,
                        "new_features": feature_names,
                        "n_features": len(feature_names),
                        "max_features": 50
                    }
                    logger.info(f"Applied TF-IDF to '{col}', saved vectorizer to GCS: {gcs_path}")
                else:
                    logger.error(f"Failed to save TfidfVectorizer to GCS: {gcs_path}")
                
            except Exception as e:
                logger.error(f"Failed to apply TF-IDF to '{col}': {str(e)}")
                # Keep original column if TF-IDF fails
                
        elif encoding_role == "identifier_ignore":
            # Drop column
            columns_to_drop.append(col)
            logger.info(f"Dropping identifier column '{col}'")
            
        else:
            logger.warning(f"Unknown encoding role '{encoding_role}' for column '{col}', leaving as-is")
    
    # Drop columns that were replaced
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
        logger.info(f"Dropped {len(columns_to_drop)} original columns: {columns_to_drop}")
    
    return df, encoders_info


def _encode_text_boolean(series: pd.Series) -> pd.Series:
    """
    Convert text boolean values to 0/1.
    
    Args:
        series: Pandas Series with text boolean values
        
    Returns:
        Series with 0/1 values
    """
    # Common boolean text patterns
    true_values = {'yes', 'y', 'true', '1', 'on', 'enabled'}
    false_values = {'no', 'n', 'false', '0', 'off', 'disabled'}
    
    # Convert to lowercase string for comparison
    lower_series = series.astype(str).str.lower()
    
    result = pd.Series(index=series.index, dtype=int)
    result[lower_series.isin(true_values)] = 1
    result[lower_series.isin(false_values)] = 0
    
    # For any values not matching patterns, use the original logic
    # (map first unique to 0, second unique to 1)
    unmapped = result.isna()
    if unmapped.any():
        unique_vals = series[unmapped].unique()
        if len(unique_vals) >= 2:
            mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
            result[unmapped] = series[unmapped].map(mapping)
        else:
            result[unmapped] = 0  # Default to 0
    
    return result


# Legacy compatibility function (redirects to GCS version)
def encode_features(df_cleaned: pd.DataFrame, 
                   feature_schemas: Dict[str, schemas.FeatureSchemaInfo], 
                   target_info: schemas.TargetInfo,
                   run_id: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Legacy compatibility function - redirects to GCS version.
    """
    logger.warning("Using legacy encode_features function - redirecting to GCS version")
    return encode_features_gcs(df_cleaned, feature_schemas, target_info, run_id) 