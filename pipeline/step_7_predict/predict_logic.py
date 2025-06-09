"""
Prediction logic for The Projection Wizard.
Handles loading trained models and generating predictions with proper feature alignment.
"""

import pandas as pd
import joblib
from pathlib import Path
from typing import Any, Optional, Union
import numpy as np

from common import logger, constants


def load_pipeline(run_dir: Union[str, Path]) -> Optional[Any]:
    """
    Load the saved PyCaret/scikit-learn pipeline from the model directory.

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


def generate_predictions(model: Any, input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate predictions using the trained model with proper feature alignment.

    This function ensures that the input DataFrame exactly matches the features
    the model expects by:
    1. Dropping extra columns not needed by the model
    2. Adding missing columns with default values (0)
    3. Reordering columns to match model expectations
    4. Making predictions
    5. Returning aligned features + predictions

    Args:
        model: Trained model/pipeline object with predict() method
        input_df: Input DataFrame to make predictions on

    Returns:
        DataFrame containing aligned features plus a "prediction" column

    Raises:
        ValueError: If model doesn't have required attributes or input is invalid
        Exception: If prediction fails
    """
    if input_df is None or input_df.empty:
        raise ValueError("Input DataFrame is None or empty")

    if not hasattr(model, 'predict'):
        raise ValueError("Model does not have a 'predict' method")

    # Get the feature names the model expects
    expected_features = None

    # Try different ways to get expected feature names
    if hasattr(model, 'feature_names_in_'):
        expected_features = list(model.feature_names_in_)
    elif hasattr(model, 'named_steps'):
        # For sklearn pipelines, check the final estimator
        for step_name, step_model in reversed(list(model.named_steps.items())):
            if hasattr(step_model, 'feature_names_in_'):
                expected_features = list(step_model.feature_names_in_)
                break
    elif hasattr(model, '_final_estimator'):
        # Another way to access final estimator
        if hasattr(model._final_estimator, 'feature_names_in_'):
            expected_features = list(model._final_estimator.feature_names_in_)

    if expected_features is None:
        raise ValueError("Could not determine expected feature names from model")

        # CRITICAL: Force perfect column alignment to model.feature_names_in_
    # Step 1: Get the exact feature list from model.feature_names_in_
    model_trained_features = None

    # Get the authoritative feature list from the trained model
    if hasattr(model, 'feature_names_in_'):
        model_trained_features = list(model.feature_names_in_)
    elif hasattr(model, 'named_steps'):
        # For sklearn pipelines, get from final estimator
        for step_name, step_model in reversed(list(model.named_steps.items())):
            if hasattr(step_model, 'feature_names_in_'):
                model_trained_features = list(step_model.feature_names_in_)
                break
    elif hasattr(model, '_final_estimator') and hasattr(model._final_estimator, 'feature_names_in_'):
        model_trained_features = list(model._final_estimator.feature_names_in_)

    if model_trained_features is None:
        raise ValueError("Cannot retrieve model.feature_names_in_ - model alignment impossible")

    # Step 2: Build DataFrame with ONLY the features model was trained on
    # This guarantees no target column or extra features can slip through
    perfectly_aligned_data = {}

    for required_feature in model_trained_features:
        if required_feature in input_df.columns:
            # Use the input value for this feature
            perfectly_aligned_data[required_feature] = input_df[required_feature].iloc[0]
        else:
            # Missing feature - use neutral default
            perfectly_aligned_data[required_feature] = 0

    # Create DataFrame with features in EXACT order of model.feature_names_in_
    aligned_df = pd.DataFrame([perfectly_aligned_data], columns=model_trained_features)

    # Step 3: GUARANTEE perfect alignment
    # Verify columns match model.feature_names_in_ exactly
    if list(aligned_df.columns) != model_trained_features:
        raise ValueError(f"ALIGNMENT FAILURE: DataFrame columns {list(aligned_df.columns)} != model.feature_names_in_ {model_trained_features}")

    # Verify no extra or missing columns
    if aligned_df.shape[1] != len(model_trained_features):
        raise ValueError(f"COLUMN COUNT MISMATCH: DataFrame has {aligned_df.shape[1]} columns, model expects {len(model_trained_features)}")

    # Verify DataFrame uses exact same column index as model expects
    if not aligned_df.columns.equals(pd.Index(model_trained_features)):
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
        raise ValueError(f"Failed to clean aligned DataFrame: {str(e)}")

    # Step 6: Make predictions with properly aligned DataFrame
    try:
        predictions = model.predict(aligned_df)

        # Convert predictions to a pandas Series with proper index
        if isinstance(predictions, np.ndarray):
            predictions = pd.Series(predictions, index=aligned_df.index, name='prediction')
        elif not isinstance(predictions, pd.Series):
            predictions = pd.Series(predictions, index=aligned_df.index, name='prediction')

    except Exception as e:
        raise Exception(f"Model prediction failed: {str(e)}")

    # Step 6: Combine aligned features with predictions
    result_df = aligned_df.copy()
    result_df['prediction'] = predictions

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
