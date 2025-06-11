"""
Prediction logic for The Projection Wizard.
Handles loading trained models and generating predictions with proper feature alignment.
Refactored for GCS-based storage.
"""

import pandas as pd
import joblib
from pathlib import Path
from typing import Any, Optional, Union, List
import numpy as np
import tempfile
import logging
import pickle
from datetime import datetime
import uuid

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


# Enhanced prediction functions for Option 2 API

def generate_predictions_with_probabilities(model: Any, input_df: pd.DataFrame, target_column: str = None) -> tuple:
    """
    Generate predictions with class probabilities for classification tasks.

    Args:
        model: Trained model/pipeline object
        input_df: Input DataFrame to make predictions on
        target_column: Name of target column to explicitly exclude

    Returns:
        Tuple of (result_df, probabilities_dict)
        - result_df: DataFrame with aligned features and predictions
        - probabilities_dict: Dict with class probabilities (None for regression)
    """
    # Generate standard predictions
    result_df = generate_predictions(model, input_df, target_column)
    
    probabilities_dict = None
    
    # Try to get prediction probabilities for classification
    try:
        if hasattr(model, 'predict_proba'):
            # Get the aligned input (without prediction column)
            aligned_input = result_df.drop(columns=['prediction'])
            probabilities = model.predict_proba(aligned_input)
            
            # Get class labels
            if hasattr(model, 'classes_'):
                class_labels = model.classes_
                # Convert to dict format
                probabilities_dict = {
                    str(class_labels[i]): float(probabilities[0][i]) 
                    for i in range(len(class_labels))
                }
                
                # Find predicted class and confidence
                predicted_class = str(result_df['prediction'].iloc[0])
                confidence = float(max(probabilities[0]))
                
                probabilities_dict['_predicted_class'] = predicted_class
                probabilities_dict['_confidence'] = confidence
                
    except Exception as e:
        log = logging.getLogger(__name__)
        log.warning(f"Could not get prediction probabilities: {e}")
    
    return result_df, probabilities_dict


def calculate_feature_contributions(model: Any, input_df: pd.DataFrame, target_column: str = None) -> List[dict]:
    """
    Calculate feature contributions to the prediction using various methods.

    Args:
        model: Trained model/pipeline object
        input_df: Input DataFrame
        target_column: Name of target column to exclude

    Returns:
        List of feature contribution dictionaries
    """
    log = logging.getLogger(__name__)
    contributions = []
    
    try:
        # Generate prediction to get aligned features
        result_df = generate_predictions(model, input_df, target_column)
        aligned_input = result_df.drop(columns=['prediction'])
        
        # Method 1: Try to get feature importances from the model
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # For linear models
            feature_importance = np.abs(model.coef_).flatten() if model.coef_.ndim > 1 else np.abs(model.coef_)
        elif hasattr(model, 'named_steps'):
            # For pipelines, try to get from final estimator
            for step_name, step_model in reversed(list(model.named_steps.items())):
                if hasattr(step_model, 'feature_importances_'):
                    feature_importance = step_model.feature_importances_
                    break
                elif hasattr(step_model, 'coef_'):
                    feature_importance = np.abs(step_model.coef_).flatten() if step_model.coef_.ndim > 1 else np.abs(step_model.coef_)
                    break
        
        if feature_importance is not None and len(feature_importance) == len(aligned_input.columns):
            # Create contributions based on feature importance * feature value
            feature_values = aligned_input.iloc[0]
            
            for i, feature_name in enumerate(aligned_input.columns):
                feature_value = feature_values[feature_name]
                importance = float(feature_importance[i])
                
                # Calculate contribution (importance * normalized feature value)
                contribution_value = importance * float(feature_value) if pd.notna(feature_value) else 0.0
                
                contributions.append({
                    'feature_name': feature_name,
                    'contribution_value': contribution_value,
                    'feature_value': feature_value,
                    'contribution_direction': 'positive' if contribution_value > 0 else ('negative' if contribution_value < 0 else 'neutral'),
                    'importance_score': importance
                })
        else:
            log.warning("Could not extract feature importances from model")
            
            # Fallback: Create basic contributions based on feature values
            feature_values = aligned_input.iloc[0]
            for feature_name in aligned_input.columns:
                feature_value = feature_values[feature_name]
                contributions.append({
                    'feature_name': feature_name,
                    'contribution_value': 0.0,
                    'feature_value': feature_value,
                    'contribution_direction': 'neutral',
                    'importance_score': 0.0
                })
                
    except Exception as e:
        log.error(f"Failed to calculate feature contributions: {e}")
    
    # Sort by absolute contribution value
    contributions.sort(key=lambda x: abs(x['contribution_value']), reverse=True)
    return contributions


def calculate_confidence_interval(model: Any, input_df: pd.DataFrame, target_column: str = None, confidence_level: float = 0.95) -> dict:
    """
    Calculate confidence interval for regression predictions.

    Args:
        model: Trained model/pipeline object
        input_df: Input DataFrame
        target_column: Name of target column to exclude
        confidence_level: Confidence level (default 0.95)

    Returns:
        Dictionary with lower_bound, upper_bound, confidence_level
    """
    log = logging.getLogger(__name__)
    
    try:
        # Generate standard prediction
        result_df = generate_predictions(model, input_df, target_column)
        prediction_value = result_df['prediction'].iloc[0]
        
        # Try to get prediction standard error
        std_error = None
        
        # For linear models, try to estimate standard error
        if hasattr(model, 'predict'):
            # This is a simplified approach - in practice, you'd want to use
            # the model's training data or cross-validation to estimate uncertainty
            
            # As a simple heuristic, we'll use a percentage of the prediction value
            # In a real implementation, you'd use more sophisticated methods
            uncertainty_factor = 0.1  # 10% uncertainty as default
            std_error = abs(float(prediction_value)) * uncertainty_factor
        
        if std_error is not None:
            # Calculate confidence interval using normal approximation
            from scipy import stats
            z_score = stats.norm.ppf((1 + confidence_level) / 2)
            margin_of_error = z_score * std_error
            
            return {
                'lower_bound': float(prediction_value - margin_of_error),
                'upper_bound': float(prediction_value + margin_of_error),
                'confidence_level': confidence_level,
                'std_error': std_error
            }
        else:
            log.warning("Could not calculate confidence interval - using prediction value ±10%")
            margin = abs(float(prediction_value)) * 0.1
            return {
                'lower_bound': float(prediction_value - margin),
                'upper_bound': float(prediction_value + margin),
                'confidence_level': confidence_level,
                'std_error': margin
            }
            
    except Exception as e:
        log.error(f"Failed to calculate confidence interval: {e}")
        return {
            'lower_bound': None,
            'upper_bound': None,
            'confidence_level': confidence_level,
            'std_error': None
        }


def generate_enhanced_prediction(model: Any, input_df: pd.DataFrame, target_column: str = None, task_type: str = "unknown") -> dict:
    """
    Generate enhanced prediction with probabilities, contributions, and confidence intervals.

    Args:
        model: Trained model/pipeline object
        input_df: Input DataFrame
        target_column: Name of target column to exclude
        task_type: Type of task ("classification" or "regression")

    Returns:
        Dictionary with enhanced prediction information
    """
    import uuid
    from datetime import datetime
    
    # Generate predictions with probabilities
    result_df, probabilities_dict = generate_predictions_with_probabilities(model, input_df, target_column)
    prediction_value = result_df['prediction'].iloc[0]
    
    # Convert numpy types to Python types for JSON serialization
    if hasattr(prediction_value, 'item'):
        prediction_value = prediction_value.item()
    
    # Calculate feature contributions
    contributions = calculate_feature_contributions(model, input_df, target_column)
    
    # Get top contributing features
    top_features = [contrib['feature_name'] for contrib in contributions[:5]]
    
    # Prepare input and processed features
    input_features = input_df.iloc[0].to_dict()
    processed_features = result_df.drop(columns=['prediction']).iloc[0].to_dict()
    
    # Generate unique prediction ID
    prediction_id = str(uuid.uuid4())[:8]
    
    enhanced_prediction = {
        'prediction_value': prediction_value,
        'prediction_id': prediction_id,
        'feature_contributions': contributions,
        'top_contributing_features': top_features,
        'input_features': input_features,
        'processed_features': processed_features,
        'prediction_timestamp': datetime.now().isoformat(),
        'task_type': task_type,
        'target_column': target_column or "unknown"
    }
    
    # Add probabilities for classification
    if probabilities_dict and task_type.lower() == "classification":
        enhanced_prediction['probabilities'] = probabilities_dict
    
    # Add confidence interval for regression
    if task_type.lower() == "regression":
        confidence_interval = calculate_confidence_interval(model, input_df, target_column)
        enhanced_prediction['confidence_interval'] = confidence_interval
    
    return enhanced_prediction


def generate_batch_predictions(model: Any, inputs_list: List[pd.DataFrame], target_column: str = None, task_type: str = "unknown") -> dict:
    """
    Generate batch predictions with summary statistics.

    Args:
        model: Trained model/pipeline object
        inputs_list: List of input DataFrames
        target_column: Name of target column to exclude
        task_type: Type of task ("classification" or "regression")

    Returns:
        Dictionary with batch prediction results and summary
    """
    import time
    from datetime import datetime
    
    start_time = time.time()
    
    predictions = []
    all_prediction_values = []
    
    for i, input_df in enumerate(inputs_list):
        try:
            enhanced_pred = generate_enhanced_prediction(model, input_df, target_column, task_type)
            
            prediction_item = {
                'input_index': i,
                'prediction_value': enhanced_pred['prediction_value'],
                'prediction_id': enhanced_pred['prediction_id']
            }
            
            # Add probabilities if available
            if 'probabilities' in enhanced_pred:
                prediction_item['probabilities'] = enhanced_pred['probabilities']
            
            # Add confidence interval if available
            if 'confidence_interval' in enhanced_pred:
                prediction_item['confidence_interval'] = enhanced_pred['confidence_interval']
            
            predictions.append(prediction_item)
            all_prediction_values.append(enhanced_pred['prediction_value'])
            
        except Exception as e:
            log = logging.getLogger(__name__)
            log.error(f"Failed to generate prediction for input {i}: {e}")
            # Add failed prediction placeholder
            predictions.append({
                'input_index': i,
                'prediction_value': None,
                'prediction_id': f"failed_{i}",
                'error': str(e)
            })
    
    processing_time = time.time() - start_time
    
    # Generate summary
    valid_predictions = [p for p in all_prediction_values if p is not None]
    summary = {
        'total_predictions': len(predictions),
        'successful_predictions': len(valid_predictions),
        'failed_predictions': len(predictions) - len(valid_predictions),
        'processing_time_seconds': processing_time
    }
    
    if task_type.lower() == "classification" and valid_predictions:
        from collections import Counter
        prediction_counts = Counter(valid_predictions)
        summary['prediction_distribution'] = dict(prediction_counts)
    elif task_type.lower() == "regression" and valid_predictions:
        import numpy as np
        summary['prediction_range'] = {
            'min': float(np.min(valid_predictions)),
            'max': float(np.max(valid_predictions)),
            'mean': float(np.mean(valid_predictions)),
            'std': float(np.std(valid_predictions)) if len(valid_predictions) > 1 else 0.0
        }
    
    return {
        'predictions': predictions,
        'summary': summary,
        'batch_timestamp': datetime.now().isoformat(),
        'task_type': task_type,
        'target_column': target_column or "unknown"
    }


def calculate_shap_values_for_prediction(model: Any, input_df: pd.DataFrame, target_column: str = None, task_type: str = "unknown") -> dict:
    """
    Calculate real SHAP values for a single prediction using the existing SHAP logic.

    Args:
        model: Trained model/pipeline object
        input_df: Input DataFrame for prediction
        target_column: Name of target column to exclude
        task_type: Type of task ("classification" or "regression")

    Returns:
        Dictionary with SHAP values and feature contributions
    """
    log = logging.getLogger(__name__)
    
    try:
        # Import SHAP logic
        from pipeline.step_6_explain.shap_logic import _create_prediction_function
        import shap
        
        # Generate prediction to get aligned features
        result_df = generate_predictions(model, input_df, target_column)
        aligned_input = result_df.drop(columns=['prediction'])
        
        # Create prediction function for SHAP
        prediction_function = _create_prediction_function(model, task_type, log)
        
        # Create explainer with small background sample (just the input for single prediction)
        background_sample = aligned_input.copy()
        
        # For single prediction, we can use TreeExplainer if available, otherwise Explainer
        try:
            # Try TreeExplainer first (faster for tree-based models)
            explainer = shap.TreeExplainer(model)
            log.info("Using TreeExplainer for SHAP values")
        except:
            try:
                # Fall back to general Explainer
                explainer = shap.Explainer(prediction_function, background_sample)
                log.info("Using general Explainer for SHAP values")
            except:
                # Final fallback to KernelExplainer with minimal background
                explainer = shap.KernelExplainer(prediction_function, background_sample.iloc[:1])
                log.info("Using KernelExplainer for SHAP values")
        
        # Calculate SHAP values
        shap_values = explainer(aligned_input)
        
        # Extract values based on task type and structure
        feature_names = list(aligned_input.columns)
        shap_dict = {}
        
        if hasattr(shap_values, 'values'):
            values = shap_values.values
            
            # Handle different shapes
            if len(values.shape) == 3:  # Multi-class classification
                if task_type == "classification" and values.shape[2] == 2:
                    # Binary classification - use positive class
                    values = values[0, :, 1]
                else:
                    # Multi-class - use mean absolute values
                    values = np.mean(np.abs(values), axis=2)[0]
            elif len(values.shape) == 2:
                # Standard case
                values = values[0]
            else:
                # Single dimension
                values = values
            
            # Create SHAP dictionary
            for i, feature_name in enumerate(feature_names):
                shap_dict[feature_name] = float(values[i])
        
        # Calculate feature contributions using SHAP values
        contributions = []
        feature_values = aligned_input.iloc[0]
        
        for feature_name, shap_value in shap_dict.items():
            feature_value = feature_values[feature_name]
            
            contributions.append({
                'feature_name': feature_name,
                'contribution_value': float(shap_value),
                'feature_value': feature_value,
                'contribution_direction': 'positive' if shap_value > 0 else ('negative' if shap_value < 0 else 'neutral'),
                'shap_value': float(shap_value)
            })
        
        # Sort by absolute SHAP value
        contributions.sort(key=lambda x: abs(x['shap_value']), reverse=True)
        
        return {
            'shap_values': shap_dict,
            'feature_contributions': contributions,
            'base_value': float(shap_values.base_values[0]) if hasattr(shap_values, 'base_values') and shap_values.base_values is not None else 0.0
        }
        
    except Exception as e:
        log.warning(f"Failed to calculate SHAP values, falling back to model importance: {e}")
        # Fall back to existing feature contribution method
        contributions = calculate_feature_contributions(model, input_df, target_column)
        
        # Extract SHAP-like values from contributions
        shap_dict = {contrib['feature_name']: contrib['contribution_value'] for contrib in contributions}
        
        return {
            'shap_values': shap_dict,
            'feature_contributions': contributions,
            'base_value': 0.0,
            'fallback_used': True
        }


def get_global_feature_importance_from_shap(model: Any, sample_data: pd.DataFrame, target_column: str = None, task_type: str = "unknown") -> dict:
    """
    Calculate global feature importance using SHAP on a sample of data.

    Args:
        model: Trained model/pipeline object
        sample_data: Sample DataFrame for importance calculation
        target_column: Name of target column to exclude
        task_type: Type of task ("classification" or "regression")

    Returns:
        Dictionary with feature importance scores
    """
    log = logging.getLogger(__name__)
    
    try:
        # Import SHAP logic
        from pipeline.step_6_explain.shap_logic import _create_prediction_function
        import shap
        
        # Prepare sample data (limit size for performance)
        max_sample_size = 100
        if len(sample_data) > max_sample_size:
            sample_data = sample_data.sample(n=max_sample_size, random_state=42)
        
        # Remove target column if present
        if target_column and target_column in sample_data.columns:
            sample_data = sample_data.drop(columns=[target_column])
        
        # Create prediction function for SHAP
        prediction_function = _create_prediction_function(model, task_type, log)
        
        # Create explainer
        try:
            explainer = shap.TreeExplainer(model)
            log.info("Using TreeExplainer for global feature importance")
        except:
            try:
                background_sample = sample_data.sample(n=min(50, len(sample_data)), random_state=42)
                explainer = shap.Explainer(prediction_function, background_sample)
                log.info("Using general Explainer for global feature importance")
            except:
                background_sample = sample_data.sample(n=min(10, len(sample_data)), random_state=42)
                explainer = shap.KernelExplainer(prediction_function, background_sample)
                log.info("Using KernelExplainer for global feature importance")
        
        # Calculate SHAP values for sample
        shap_values = explainer(sample_data)
        
        # Calculate mean absolute SHAP values as feature importance
        feature_names = list(sample_data.columns)
        importance_dict = {}
        
        if hasattr(shap_values, 'values'):
            values = shap_values.values
            
            # Handle different shapes
            if len(values.shape) == 3:  # Multi-class classification
                if task_type == "classification" and values.shape[2] == 2:
                    # Binary classification - use positive class
                    values = values[:, :, 1]
                else:
                    # Multi-class - use mean absolute values across classes
                    values = np.mean(np.abs(values), axis=2)
            
            # Calculate mean absolute SHAP values
            mean_abs_shap = np.mean(np.abs(values), axis=0)
            
            for i, feature_name in enumerate(feature_names):
                importance_dict[feature_name] = float(mean_abs_shap[i])
        
        log.info(f"Calculated SHAP-based feature importance for {len(importance_dict)} features")
        return importance_dict
        
    except Exception as e:
        log.warning(f"Failed to calculate SHAP-based feature importance: {e}")
        
        # Fall back to model-based feature importance
        try:
            feature_importance = None
            feature_names = list(sample_data.columns)
            
            if hasattr(model, 'feature_importances_'):
                feature_importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                feature_importance = np.abs(model.coef_).flatten() if model.coef_.ndim > 1 else np.abs(model.coef_)
            elif hasattr(model, 'named_steps'):
                # For pipelines, try to get from final estimator
                for step_name, step_model in reversed(list(model.named_steps.items())):
                    if hasattr(step_model, 'feature_importances_'):
                        feature_importance = step_model.feature_importances_
                        break
                    elif hasattr(step_model, 'coef_'):
                        feature_importance = np.abs(step_model.coef_).flatten() if step_model.coef_.ndim > 1 else np.abs(step_model.coef_)
                        break
            
            if feature_importance is not None and len(feature_importance) == len(feature_names):
                importance_dict = {feature_names[i]: float(feature_importance[i]) for i in range(len(feature_names))}
                log.info("Using model-based feature importance as fallback")
                return importance_dict
        except Exception as fallback_error:
            log.warning(f"Fallback feature importance also failed: {fallback_error}")
        
        # Return empty dict if all methods fail
        return {}


def generate_enhanced_prediction_with_shap(model: Any, input_df: pd.DataFrame, target_column: str = None, task_type: str = "unknown") -> dict:
    """
    Generate enhanced prediction with real SHAP values and explanations.

    Args:
        model: Trained model/pipeline object
        input_df: Input DataFrame
        target_column: Name of target column to exclude
        task_type: Type of task ("classification" or "regression")

    Returns:
        Dictionary with enhanced prediction information including real SHAP values
    """
    import uuid
    from datetime import datetime
    
    # Start with the existing enhanced prediction
    enhanced_pred = generate_enhanced_prediction(model, input_df, target_column, task_type)
    
    # Add real SHAP values and explanations
    try:
        shap_data = calculate_shap_values_for_prediction(model, input_df, target_column, task_type)
        
        # Update with real SHAP data
        enhanced_pred['shap_values'] = shap_data['shap_values']
        enhanced_pred['shap_base_value'] = shap_data['base_value']
        
        # CRITICAL FIX: Update feature contributions with proper original input values
        shap_contributions = []
        for contrib in shap_data['feature_contributions']:
            feature_name = contrib['feature_name']
            
            # Handle different types of features
            feature_value = contrib['feature_value']  # Default to processed value
            
            if original_input:
                # Direct match for numeric features
                if feature_name in original_input:
                    feature_value = original_input[feature_name]
                # Handle one-hot encoded categorical features (e.g., "property_type_Single Family")
                elif '_' in feature_name:
                    parts = feature_name.split('_', 1)
                    if len(parts) == 2:
                        original_col, encoded_value = parts
                        if original_col in original_input:
                            # If this is the active category, use value 1, else 0
                            if str(original_input[original_col]) == encoded_value:
                                feature_value = 1
                            else:
                                feature_value = 0
            
            shap_contributions.append({
                'feature_name': feature_name,
                'contribution_value': contrib['contribution_value'],
                'feature_value': feature_value,
                'contribution_direction': contrib['contribution_direction'],
                'shap_value': contrib['shap_value']
            })
        
        enhanced_pred['feature_contributions'] = shap_contributions
        
        # Update top contributing features based on SHAP
        enhanced_pred['top_contributing_features'] = [
            contrib['feature_name'] for contrib in shap_contributions[:5]
        ]
        
        # Add SHAP metadata
        enhanced_pred['shap_available'] = True
        enhanced_pred['shap_fallback_used'] = shap_data.get('fallback_used', False)
        
    except Exception as e:
        log = logging.getLogger(__name__)
        log.warning(f"Failed to add SHAP values to prediction: {e}")
        # Keep the existing feature contributions from model importance
        enhanced_pred['shap_available'] = False
        enhanced_pred['shap_fallback_used'] = True
    
    return enhanced_pred


def generate_enhanced_prediction_with_shap_fixed(model: Any, input_df: pd.DataFrame, target_column: str = None, task_type: str = "unknown", original_input: dict = None) -> dict:
    """
    FIXED VERSION: Generate enhanced prediction with real SHAP values and preserve original input.

    Args:
        model: Trained model/pipeline object
        input_df: Encoded/processed input DataFrame
        target_column: Name of target column to exclude
        task_type: Type of task ("classification" or "regression")
        original_input: Original user input dictionary (before encoding)

    Returns:
        Dictionary with enhanced prediction information including real SHAP values and preserved original input
    """
    import uuid
    from datetime import datetime
    
    # Generate predictions with probabilities using the encoded input
    result_df, probabilities_dict = generate_predictions_with_probabilities(model, input_df, target_column)
    prediction_value = result_df['prediction'].iloc[0]
    
    # Convert numpy types to Python types for JSON serialization
    if hasattr(prediction_value, 'item'):
        prediction_value = prediction_value.item()
    
    # Calculate feature contributions using the encoded input
    contributions = calculate_feature_contributions(model, input_df, target_column)
    
    # Get top contributing features
    top_features = [contrib['feature_name'] for contrib in contributions[:5]]
    
    # CRITICAL FIX: Preserve original user input values
    if original_input is not None:
        # Use original user input for input_features
        input_features = original_input.copy()
    else:
        # Fallback to encoded values if original not available
        input_features = input_df.iloc[0].to_dict()
    
    # Use processed/encoded values for processed_features
    processed_features = result_df.drop(columns=['prediction']).iloc[0].to_dict()
    
    # Generate unique prediction ID
    prediction_id = str(uuid.uuid4())[:8]
    
    enhanced_prediction = {
        'api_version': 'v1',
        'prediction_value': prediction_value,
        'prediction_id': prediction_id,
        'feature_contributions': contributions,
        'top_contributing_features': top_features,
        'input_features': input_features,  # Original user input
        'processed_features': processed_features,  # Encoded/processed values
        'prediction_timestamp': datetime.now().isoformat(),
        'task_type': task_type,
        'target_column': target_column or "unknown"
    }
    
    # Add probabilities for classification
    if probabilities_dict and task_type.lower() == "classification":
        enhanced_prediction['probabilities'] = probabilities_dict
    
    # Add confidence interval for regression
    if task_type.lower() == "regression":
        confidence_interval = calculate_confidence_interval(model, input_df, target_column)
        enhanced_prediction['confidence_interval'] = confidence_interval
    
    # Add real SHAP values and explanations
    try:
        shap_data = calculate_shap_values_for_prediction(model, input_df, target_column, task_type)
        
        # CRITICAL FIX: Update feature contributions with proper original input values
        shap_contributions = []
        for contrib in shap_data['feature_contributions']:
            feature_name = contrib['feature_name']
            
            # Handle different types of features
            feature_value = contrib['feature_value']  # Default to processed value
            
            if original_input:
                # Direct match for numeric features
                if feature_name in original_input:
                    feature_value = original_input[feature_name]
                # Handle one-hot encoded categorical features (e.g., "property_type_Single Family")
                elif '_' in feature_name:
                    parts = feature_name.split('_', 1)
                    if len(parts) == 2:
                        original_col, encoded_value = parts
                        if original_col in original_input:
                            # If this is the active category, use value 1, else 0
                            if str(original_input[original_col]) == encoded_value:
                                feature_value = 1
                            else:
                                feature_value = 0
            
            shap_contributions.append({
                'feature_name': feature_name,
                'contribution_value': contrib['contribution_value'],
                'feature_value': feature_value,
                'contribution_direction': contrib['contribution_direction'],
                'shap_value': contrib['shap_value']
            })
        
        enhanced_prediction['feature_contributions'] = shap_contributions
        
        # Update top contributing features based on SHAP
        enhanced_prediction['top_contributing_features'] = [
            contrib['feature_name'] for contrib in shap_contributions[:5]
        ]
        
        # Add SHAP metadata
        enhanced_prediction['shap_available'] = True
        enhanced_prediction['shap_fallback_used'] = shap_data.get('fallback_used', False)
        
    except Exception as e:
        log = logging.getLogger(__name__)
        log.warning(f"Failed to add SHAP values to prediction: {e}")
        
        # CRITICAL FIX: Even without SHAP, fix the feature contributions to show original values
        fixed_contributions = []
        for contrib in contributions:
            feature_name = contrib['feature_name']
            
            # Handle different types of features
            feature_value = contrib['feature_value']  # Default to processed value
            
            if original_input:
                # Direct match for numeric features
                if feature_name in original_input:
                    feature_value = original_input[feature_name]
                # Handle one-hot encoded categorical features (e.g., "property_type_Single Family")
                elif '_' in feature_name:
                    parts = feature_name.split('_', 1)
                    if len(parts) == 2:
                        original_col, encoded_value = parts
                        if original_col in original_input:
                            # If this is the active category, use value 1, else 0
                            if str(original_input[original_col]) == encoded_value:
                                feature_value = 1
                            else:
                                feature_value = 0
            
            fixed_contributions.append({
                'feature_name': feature_name,
                'contribution_value': contrib['contribution_value'],
                'feature_value': feature_value,
                'contribution_direction': contrib['contribution_direction'],
                'shap_value': contrib.get('shap_value', 0.0)
            })
        
        enhanced_prediction['feature_contributions'] = fixed_contributions
        enhanced_prediction['shap_available'] = False
        enhanced_prediction['shap_fallback_used'] = True
    
    return enhanced_prediction
