"""
Prediction logic for The Projection Wizard.
Handles loading trained models and generating predictions with proper feature alignment.
Refactored for GCS-based storage.
"""

import pandas as pd
import joblib
from pathlib import Path
from typing import Any, Optional, Union
import numpy as np
import tempfile
import logging

from common import logger, constants
from api.utils.gcs_utils import download_run_file, PROJECT_BUCKET_NAME


def load_pipeline_gcs(run_id: str, gcs_bucket_name: str = PROJECT_BUCKET_NAME) -> Optional[Any]:
    """
    Load the saved PyCaret/scikit-learn pipeline from GCS.

    Args:
        run_id: Unique run identifier
        gcs_bucket_name: GCS bucket name for storage

    Returns:
        Loaded pipeline object, or None if loading fails

    Raises:
        FileNotFoundError: If the pipeline file doesn't exist in GCS
        Exception: If loading fails for other reasons
    """
    log = logger.get_logger(run_id, "predict_logic")
    
    try:
        # Construct GCS path to the pipeline file
        pipeline_gcs_path = f"{constants.MODEL_DIR}/pycaret_pipeline.pkl"
        
        log.info(f"Loading pipeline from GCS: {pipeline_gcs_path}")
        
        # Download pipeline from GCS
        pipeline_bytes = download_run_file(run_id, pipeline_gcs_path)
        if pipeline_bytes is None:
            raise FileNotFoundError(f"Pipeline file not found in GCS: {pipeline_gcs_path}")

        # Load pipeline from bytes using temporary file
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            tmp_file.write(pipeline_bytes)
            tmp_file.flush()
            pipeline = joblib.load(tmp_file.name)
        
        # Clean up temporary file
        try:
            Path(tmp_file.name).unlink()
        except Exception as cleanup_error:
            log.warning(f"Could not clean up temporary file {tmp_file.name}: {cleanup_error}")

        # Basic validation that it's a model-like object
        if not hasattr(pipeline, 'predict'):
            raise ValueError("Loaded object does not have a 'predict' method")

        log.info("Pipeline loaded successfully from GCS")
        return pipeline

    except Exception as e:
        log.error(f"Failed to load pipeline from GCS {pipeline_gcs_path}: {str(e)}")
        raise Exception(f"Failed to load pipeline from GCS {pipeline_gcs_path}: {str(e)}")


def load_pipeline(run_dir: Union[str, Path]) -> Optional[Any]:
    """
    Legacy compatibility function - warns about local filesystem usage.
    
    Args:
        run_dir: Path to the run directory (can be string or Path object)

    Returns:
        Loaded pipeline object, or None if loading fails

    Raises:
        FileNotFoundError: If the pipeline file doesn't exist
        Exception: If loading fails for other reasons
    """
    # Convert to Path object if string
    if isinstance(run_dir, str):
        run_dir = Path(run_dir)

    # Construct path to the pipeline file
    pipeline_path = run_dir / constants.MODEL_DIR / "pycaret_pipeline.pkl"

    log = logger.get_logger("legacy", "predict_logic")
    log.warning("Using legacy load_pipeline function for local filesystem")
    log.warning("Consider using load_pipeline_gcs for GCS storage")

    try:
        # Check if file exists
        if not pipeline_path.exists():
            raise FileNotFoundError(f"Pipeline file not found: {pipeline_path}")

        # Load the pipeline
        pipeline = joblib.load(pipeline_path)

        # Basic validation that it's a model-like object
        if not hasattr(pipeline, 'predict'):
            raise ValueError("Loaded object does not have a 'predict' method")

        return pipeline

    except Exception as e:
        raise Exception(f"Failed to load pipeline from {pipeline_path}: {str(e)}")


def generate_predictions(model: Any, input_df: pd.DataFrame, target_column: str = None) -> pd.DataFrame:
    """
    Generate predictions using the trained model with proper feature alignment.

    This function ensures that the input DataFrame exactly matches the features
    the model expects by:
    1. Dropping extra columns not needed by the model
    2. Adding missing columns with default values (0)
    3. Reordering columns to match model expectations
    4. Explicitly filtering out target column to prevent contamination
    5. Making predictions
    6. Returning aligned features + predictions

    Args:
        model: Trained model/pipeline object with predict() method
        input_df: Input DataFrame to make predictions on
        target_column: Name of target column to explicitly exclude (optional)

    Returns:
        DataFrame containing aligned features plus a "prediction" column

    Raises:
        ValueError: If model doesn't have required attributes or input is invalid
        Exception: If prediction fails
    """
    # Create a logger for this function
    log = logging.getLogger(__name__)
    
    if input_df is None or input_df.empty:
        raise ValueError("Input DataFrame is None or empty")

    if not hasattr(model, 'predict'):
        raise ValueError("Model does not have a 'predict' method")

    # DEBUG: Log input details
    log.info(f"DEBUG: Input DataFrame shape: {input_df.shape}")
    log.info(f"DEBUG: Input DataFrame columns: {list(input_df.columns)}")

    # Get the feature names the model expects
    expected_features = None

    # Try different ways to get expected feature names
    if hasattr(model, 'feature_names_in_'):
        expected_features = list(model.feature_names_in_)
        log.info(f"DEBUG: Found model.feature_names_in_: {expected_features}")
    elif hasattr(model, 'named_steps'):
        # For sklearn pipelines, check the final estimator
        log.info("DEBUG: Checking named_steps for feature names...")
        for step_name, step_model in reversed(list(model.named_steps.items())):
            log.info(f"DEBUG: Checking step '{step_name}' of type {type(step_model)}")
            if hasattr(step_model, 'feature_names_in_'):
                expected_features = list(step_model.feature_names_in_)
                log.info(f"DEBUG: Found feature_names_in_ in step '{step_name}': {expected_features}")
                break
    elif hasattr(model, '_final_estimator'):
        # Another way to access final estimator
        log.info("DEBUG: Checking _final_estimator for feature names...")
        if hasattr(model._final_estimator, 'feature_names_in_'):
            expected_features = list(model._final_estimator.feature_names_in_)
            log.info(f"DEBUG: Found feature_names_in_ in _final_estimator: {expected_features}")

    if expected_features is None:
        log.error("DEBUG: Could not determine expected feature names from model")
        raise ValueError("Could not determine expected feature names from model")

        # CRITICAL: Force perfect column alignment to model.feature_names_in_
    # Step 1: Get the exact feature list from model.feature_names_in_
    model_trained_features = None

    # Get the authoritative feature list from the trained model
    if hasattr(model, 'feature_names_in_'):
        model_trained_features = list(model.feature_names_in_)
        log.info(f"DEBUG: Using model.feature_names_in_: {model_trained_features}")
    elif hasattr(model, 'named_steps'):
        # For sklearn pipelines, get from final estimator
        for step_name, step_model in reversed(list(model.named_steps.items())):
            if hasattr(step_model, 'feature_names_in_'):
                model_trained_features = list(step_model.feature_names_in_)
                log.info(f"DEBUG: Using feature_names_in_ from step '{step_name}': {model_trained_features}")
                break
    elif hasattr(model, '_final_estimator') and hasattr(model._final_estimator, 'feature_names_in_'):
        model_trained_features = list(model._final_estimator.feature_names_in_)
        log.info(f"DEBUG: Using feature_names_in_ from _final_estimator: {model_trained_features}")

    if model_trained_features is None:
        log.error("DEBUG: Cannot retrieve model.feature_names_in_ - model alignment impossible")
        raise ValueError("Cannot retrieve model.feature_names_in_ - model alignment impossible")

    # DEBUG: Check for suspicious features
    log.info(f"DEBUG: Model expects {len(model_trained_features)} features")
    suspicious_features = [f for f in model_trained_features if 'purchased' in f.lower() or 'target' in f.lower()]
    if suspicious_features:
        log.warning(f"DEBUG: SUSPICIOUS FEATURES that might be target-related: {suspicious_features}")

    # CRITICAL FIX: Remove target column from model_trained_features if present
    # This prevents target column contamination in prediction input
    original_feature_count = len(model_trained_features)
    if target_column:
        model_trained_features = [f for f in model_trained_features if f != target_column]
        filtered_feature_count = len(model_trained_features)
        if original_feature_count != filtered_feature_count:
            log.warning(f"DEBUG: FILTERED OUT target column '{target_column}' from model features. Count: {original_feature_count} -> {filtered_feature_count}")
        else:
            log.info(f"DEBUG: Target column '{target_column}' was not present in model features (good!)")
    else:
        log.warning("DEBUG: No target_column provided - cannot filter target from model features")

    # Step 2: Build DataFrame with ONLY the features model was trained on (excluding target)
    # This guarantees no target column or extra features can slip through
    perfectly_aligned_data = {}

    for required_feature in model_trained_features:
        if required_feature in input_df.columns:
            # Use the input value for this feature
            perfectly_aligned_data[required_feature] = input_df[required_feature].iloc[0]
            log.debug(f"DEBUG: Using input value for '{required_feature}': {perfectly_aligned_data[required_feature]}")
        else:
            # Missing feature - use neutral default
            perfectly_aligned_data[required_feature] = 0
            log.warning(f"DEBUG: Missing feature '{required_feature}', using default value 0")

    # Create DataFrame with features in EXACT order of model.feature_names_in_
    aligned_df = pd.DataFrame([perfectly_aligned_data], columns=model_trained_features)
    
    log.info(f"DEBUG: Created aligned DataFrame with shape {aligned_df.shape}")
    log.info(f"DEBUG: Aligned DataFrame columns: {list(aligned_df.columns)}")

    # Step 3: GUARANTEE perfect alignment
    # Verify columns match model.feature_names_in_ exactly
    if list(aligned_df.columns) != model_trained_features:
        log.error(f"DEBUG: ALIGNMENT FAILURE: DataFrame columns {list(aligned_df.columns)} != model.feature_names_in_ {model_trained_features}")
        raise ValueError(f"ALIGNMENT FAILURE: DataFrame columns {list(aligned_df.columns)} != model.feature_names_in_ {model_trained_features}")

    # Verify no extra or missing columns
    if aligned_df.shape[1] != len(model_trained_features):
        log.error(f"DEBUG: COLUMN COUNT MISMATCH: DataFrame has {aligned_df.shape[1]} columns, model expects {len(model_trained_features)}")
        raise ValueError(f"COLUMN COUNT MISMATCH: DataFrame has {aligned_df.shape[1]} columns, model expects {len(model_trained_features)}")

    # Verify DataFrame uses exact same column index as model expects
    if not aligned_df.columns.equals(pd.Index(model_trained_features)):
        log.error(f"DEBUG: COLUMN INDEX MISMATCH: DataFrame index != model.feature_names_in_ index")
        raise ValueError(f"COLUMN INDEX MISMATCH: DataFrame index != model.feature_names_in_ index")

    # Step 5: Handle data type cleaning
    try:
        # Ensure all columns are properly typed
        for col in aligned_df.columns:
            if aligned_df[col].dtype == 'object':
                # Try to convert to numeric, fill NaN with 0
                aligned_df[col] = pd.to_numeric(aligned_df[col], errors='coerce').fillna(0)
            else:
                # Fill any NaN values with 0
                aligned_df[col] = aligned_df[col].fillna(0)
    except Exception as e:
        log.error(f"DEBUG: Failed to clean aligned DataFrame: {str(e)}")
        raise ValueError(f"Failed to clean aligned DataFrame: {str(e)}")

    # Step 6: Make predictions with properly aligned DataFrame
    try:
        log.info("DEBUG: About to call model.predict() with aligned DataFrame")
        log.info(f"DEBUG: Final DataFrame for prediction: shape={aligned_df.shape}, columns={list(aligned_df.columns)}")
        
        predictions = model.predict(aligned_df)
        
        log.info(f"DEBUG: Model prediction successful, got {len(predictions)} predictions")

        # Convert predictions to a pandas Series with proper index
        if isinstance(predictions, np.ndarray):
            predictions = pd.Series(predictions, index=aligned_df.index, name='prediction')
        elif not isinstance(predictions, pd.Series):
            predictions = pd.Series(predictions, index=aligned_df.index, name='prediction')

    except Exception as e:
        log.error(f"DEBUG: Model prediction failed with error: {str(e)}")
        log.error(f"DEBUG: Model type: {type(model)}")
        log.error(f"DEBUG: Model attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}")
        raise Exception(f"Model prediction failed: {str(e)}")

    # Step 6: Combine aligned features with predictions
    result_df = aligned_df.copy()
    result_df['prediction'] = predictions

    log.info(f"DEBUG: Final result DataFrame shape: {result_df.shape}")
    log.info(f"DEBUG: Final result columns: {list(result_df.columns)}")

    return result_df


def validate_prediction_inputs(model: Any, input_df: pd.DataFrame) -> tuple[bool, list[str]]:
    """
    Validate inputs for prediction generation.

    Args:
        model: Model object to validate
        input_df: Input DataFrame to validate

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []

    # Validate model
    if model is None:
        issues.append("Model is None")
    elif not hasattr(model, 'predict'):
        issues.append("Model does not have 'predict' method")

    # Validate input DataFrame
    if input_df is None:
        issues.append("Input DataFrame is None")
    elif input_df.empty:
        issues.append("Input DataFrame is empty")
    elif len(input_df.columns) == 0:
        issues.append("Input DataFrame has no columns")

    # Check for expected feature names
    if model is not None and input_df is not None:
        try:
            expected_features = None
            if hasattr(model, 'feature_names_in_'):
                expected_features = list(model.feature_names_in_)

            if expected_features is None:
                issues.append("Could not determine expected features from model")
            elif len(expected_features) == 0:
                issues.append("Model expects zero features")

        except Exception as e:
            issues.append(f"Error checking model features: {str(e)}")

    return len(issues) == 0, issues


def get_prediction_summary(result_df: pd.DataFrame, task_type: str = "unknown") -> dict:
    """
    Generate a summary of prediction results.

    Args:
        result_df: DataFrame with predictions
        task_type: Type of task ("classification" or "regression")

    Returns:
        Dictionary with prediction summary statistics
    """
    if result_df is None or result_df.empty or 'prediction' not in result_df.columns:
        return {"error": "Invalid prediction results"}

    predictions = result_df['prediction']

    summary = {
        "total_predictions": len(predictions),
        "feature_count": len(result_df.columns) - 1,  # Exclude prediction column
        "has_missing_predictions": predictions.isnull().sum() > 0,
        "missing_prediction_count": int(predictions.isnull().sum())
    }

    if task_type.lower() == "classification":
        summary.update({
            "unique_predictions": sorted(predictions.dropna().unique().tolist()),
            "prediction_counts": predictions.value_counts().to_dict()
        })
    elif task_type.lower() == "regression":
        summary.update({
            "min_prediction": float(predictions.min()) if not predictions.isnull().all() else None,
            "max_prediction": float(predictions.max()) if not predictions.isnull().all() else None,
            "mean_prediction": float(predictions.mean()) if not predictions.isnull().all() else None,
            "std_prediction": float(predictions.std()) if not predictions.isnull().all() else None
        })

    return summary
