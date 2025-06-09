"""
SHAP Logic for The Projection Wizard.
Core logic for generating SHAP explanations and summary plots.
"""

import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Any, Optional, Union, Callable
import warnings

# Suppress some warnings that might come from SHAP/matplotlib
warnings.filterwarnings('ignore', category=FutureWarning, module='shap')
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


def _create_prediction_function(pipeline: Any, task_type: str, logger: Optional[logging.Logger] = None) -> Callable:
    """
    Create an appropriate prediction function for SHAP based on available methods.
    
    Args:
        pipeline: The model pipeline
        task_type: "classification" or "regression"
        logger: Optional logger
        
    Returns:
        A callable function for SHAP to use for predictions
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if task_type == "classification":
        if hasattr(pipeline, 'predict_proba'):
            logger.info("Using predict_proba for SHAP classification")
            return pipeline.predict_proba
        elif hasattr(pipeline, 'decision_function'):
            logger.info("Using decision_function wrapper for SHAP classification")
            
            def decision_function_wrapper(X):
                """Convert decision function output to probability-like format"""
                decision_scores = pipeline.decision_function(X)
                
                # Handle different output shapes
                if decision_scores.ndim == 1:
                    # Binary classification: convert to 2D probabilities
                    # Use sigmoid to convert to [0, 1] range
                    proba_positive = 1 / (1 + np.exp(-decision_scores))
                    proba_negative = 1 - proba_positive
                    return np.column_stack([proba_negative, proba_positive])
                else:
                    # Multi-class: softmax transformation
                    exp_scores = np.exp(decision_scores - np.max(decision_scores, axis=1, keepdims=True))
                    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            
            return decision_function_wrapper
        else:
            # Fall back to predict and create dummy probabilities
            logger.warning("Using predict with dummy probabilities for SHAP classification")
            
            def predict_wrapper(X):
                """Convert class predictions to dummy probabilities"""
                predictions = pipeline.predict(X)
                # Create dummy probabilities: 0.9 for predicted class, 0.1 for others
                unique_classes = np.unique(predictions)
                n_classes = len(unique_classes)
                n_samples = len(predictions)
                
                # Create probability matrix
                probabilities = np.full((n_samples, n_classes), 0.1 / (n_classes - 1) if n_classes > 1 else 0.5)
                
                for i, pred in enumerate(predictions):
                    class_idx = np.where(unique_classes == pred)[0][0]
                    probabilities[i, class_idx] = 0.9
                
                return probabilities
            
            return predict_wrapper
    
    else:  # regression
        if hasattr(pipeline, 'predict'):
            logger.info("Using predict for SHAP regression")
            return pipeline.predict
        else:
            raise ValueError("Pipeline has no suitable prediction method for regression")


def generate_shap_summary_plot(
    pycaret_pipeline: Any, 
    X_data_sample: pd.DataFrame, 
    plot_save_path: Path, 
    task_type: str,
    logger: Optional[logging.Logger] = None
) -> bool:
    """
    Generate a SHAP summary plot for the given PyCaret pipeline and save it as an image.
    
    Args:
        pycaret_pipeline: The loaded PyCaret pipeline object (from pycaret_pipeline.pkl)
        X_data_sample: A pandas DataFrame sample of the features (without target column)
        plot_save_path: Full Path object where the SHAP summary plot image will be saved
        task_type: "classification" or "regression" to guide SHAP explainer type
        logger: Optional logger for detailed logging
        
    Returns:
        True if plot generation and saving were successful, False otherwise
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Starting SHAP summary plot generation for {task_type} task")
        logger.info(f"Input data sample shape: {X_data_sample.shape}")
        logger.info(f"Plot save path: {plot_save_path}")
        
        # Ensure the output directory exists
        plot_save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # =====================================
        # 1. PREPARE DATA SAMPLE FOR SHAP
        # =====================================
        
        # Limit sample size for performance if needed
        max_sample_size = 500
        if len(X_data_sample) > max_sample_size:
            logger.info(f"Sampling {max_sample_size} rows from {len(X_data_sample)} for SHAP performance")
            X_sample = X_data_sample.sample(n=max_sample_size, random_state=42)
        else:
            X_sample = X_data_sample.copy()
        
        logger.info(f"Using {len(X_sample)} samples for SHAP explanation")
        
        # Ensure data is numeric (should be after prep stage, but validate)
        if not all(X_sample.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
            logger.warning("Non-numeric columns detected in data sample")
            # Log column types for debugging
            for col, dtype in X_sample.dtypes.items():
                if not np.issubdtype(dtype, np.number):
                    logger.warning(f"  Non-numeric column '{col}': {dtype}")
        
        # Convert boolean columns to integers to prevent SHAP isfinite errors
        bool_columns = X_sample.select_dtypes(include=['bool']).columns
        if len(bool_columns) > 0:
            logger.info(f"Converting {len(bool_columns)} boolean columns to integers for SHAP compatibility")
            X_sample = X_sample.copy()
            X_sample[bool_columns] = X_sample[bool_columns].astype(int)
            logger.info("Boolean to integer conversion completed")
        
        # Final validation that all columns are numeric
        non_numeric_cols = []
        for col in X_sample.columns:
            if not np.issubdtype(X_sample[col].dtype, np.number):
                non_numeric_cols.append(f"{col} ({X_sample[col].dtype})")
        
        if non_numeric_cols:
            logger.error(f"Still have non-numeric columns after conversion: {non_numeric_cols}")
            return False
        
        # =====================================
        # 2. CREATE SHAP EXPLAINER
        # =====================================
        
        logger.info("Creating SHAP explainer...")
        
        try:
            # Get appropriate prediction function
            prediction_function = _create_prediction_function(pycaret_pipeline, task_type, logger)
            
            # Create SHAP explainer
            logger.info("Creating SHAP explainer with adaptive prediction function")
            explainer = shap.Explainer(prediction_function, X_sample)
            
            logger.info("SHAP explainer created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create SHAP explainer: {e}")
            # Try fallback approach with Kernel explainer
            try:
                logger.info("Attempting fallback with KernelExplainer...")
                prediction_function = _create_prediction_function(pycaret_pipeline, task_type, logger)
                explainer = shap.KernelExplainer(
                    prediction_function, 
                    shap.sample(X_sample, min(50, len(X_sample)))
                )
                logger.info("Fallback KernelExplainer created successfully")
            except Exception as e2:
                logger.error(f"Fallback explainer also failed: {e2}")
                return False
        
        # =====================================
        # 3. CALCULATE SHAP VALUES
        # =====================================
        
        logger.info("Calculating SHAP values...")
        
        try:
            # Calculate SHAP values for the sample
            shap_values = explainer(X_sample)
            logger.info("SHAP values calculated successfully")
            
            # Log shape information for debugging
            if hasattr(shap_values, 'values'):
                logger.info(f"SHAP values shape: {shap_values.values.shape}")
            else:
                logger.info(f"SHAP values type: {type(shap_values)}")
            
        except Exception as e:
            logger.error(f"Failed to calculate SHAP values: {e}")
            return False
        
        # =====================================
        # 4. HANDLE MULTI-CLASS CLASSIFICATION
        # =====================================
        
        # For classification with multiple outputs (multi-class), select appropriate values
        shap_values_for_plot = shap_values
        
        if task_type == "classification" and hasattr(shap_values, 'values'):
            if len(shap_values.values.shape) == 3:  # Multi-class case
                logger.info(f"Multi-class classification detected, shape: {shap_values.values.shape}")
                # Use the positive class (index 1) for binary, or sum for multi-class
                if shap_values.values.shape[2] == 2:
                    # Binary classification - use positive class
                    logger.info("Using positive class (index 1) for binary classification")
                    shap_values_for_plot = shap.Explanation(
                        values=shap_values.values[:, :, 1],
                        base_values=shap_values.base_values[:, 1] if hasattr(shap_values, 'base_values') else None,
                        data=shap_values.data if hasattr(shap_values, 'data') else X_sample.values,
                        feature_names=list(X_sample.columns)
                    )
                else:
                    # Multi-class - use mean absolute values across classes
                    logger.info("Using mean absolute values across classes for multi-class")
                    mean_abs_values = np.mean(np.abs(shap_values.values), axis=2)
                    shap_values_for_plot = shap.Explanation(
                        values=mean_abs_values,
                        base_values=None,
                        data=shap_values.data if hasattr(shap_values, 'data') else X_sample.values,
                        feature_names=list(X_sample.columns)
                    )
        
        # =====================================
        # 5. GENERATE AND SAVE PLOT
        # =====================================
        
        logger.info("Generating SHAP summary plot...")
        
        try:
            # Create a new figure to ensure clean state
            plt.figure(figsize=(10, 8))
            
            # Generate SHAP summary plot
            # Use bar plot for cleaner visualization in MVP
            shap.summary_plot(
                shap_values_for_plot, 
                X_sample, 
                show=False,  # Don't show, just prepare for saving
                plot_type="bar",  # Bar plot is cleaner for MVP
                max_display=20  # Limit to top 20 features for readability
            )
            
            # Improve plot appearance
            plt.title(f"SHAP Feature Importance - {task_type.title()}", fontsize=14, fontweight='bold')
            plt.xlabel("Mean |SHAP Value|", fontsize=12)
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(
                plot_save_path, 
                bbox_inches='tight', 
                dpi=150,  # Good quality for reports
                facecolor='white',
                edgecolor='none'
            )
            
            # Close the figure to free memory
            plt.close()
            
            logger.info(f"SHAP summary plot saved successfully to: {plot_save_path}")
            
            # Verify file was created and has reasonable size
            if plot_save_path.exists():
                file_size_kb = plot_save_path.stat().st_size / 1024
                logger.info(f"Plot file size: {file_size_kb:.1f} KB")
                
                if file_size_kb < 1:  # Less than 1KB probably indicates an issue
                    logger.warning("Plot file size is very small, may indicate an issue")
                    return False
            else:
                logger.error("Plot file was not created")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate or save SHAP plot: {e}")
            # Ensure we close any open figures
            plt.close('all')
            return False
    
    except Exception as e:
        logger.error(f"Unexpected error in SHAP plot generation: {e}")
        # Ensure we close any open figures in case of error
        plt.close('all')
        return False


def validate_shap_inputs(
    pycaret_pipeline: Any,
    X_data_sample: pd.DataFrame,
    task_type: str,
    logger: Optional[logging.Logger] = None
) -> tuple[bool, list[str]]:
    """
    Validate inputs for SHAP analysis.
    
    Args:
        pycaret_pipeline: PyCaret pipeline object
        X_data_sample: Feature data sample
        task_type: "classification" or "regression"
        logger: Optional logger
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    issues = []
    
    # Check pipeline
    if pycaret_pipeline is None:
        issues.append("PyCaret pipeline is None")
    
    # Check data sample
    if X_data_sample is None:
        issues.append("Data sample is None")
    elif X_data_sample.empty:
        issues.append("Data sample is empty")
    elif len(X_data_sample) < 2:
        issues.append(f"Data sample too small: {len(X_data_sample)} rows (minimum 2 required)")
    
    # Check if data has features
    if X_data_sample is not None and len(X_data_sample.columns) == 0:
        issues.append("Data sample has no features")
    
    # Check task type
    if task_type not in ["classification", "regression"]:
        issues.append(f"Invalid task type: {task_type} (must be 'classification' or 'regression')")
    
    # Check if pipeline has required methods
    if pycaret_pipeline is not None:
        if task_type == "classification":
            # For classification, check for any usable prediction method
            has_predict_proba = hasattr(pycaret_pipeline, 'predict_proba')
            has_decision_function = hasattr(pycaret_pipeline, 'decision_function')
            has_predict = hasattr(pycaret_pipeline, 'predict')
            
            if not (has_predict_proba or has_decision_function or has_predict):
                issues.append("Pipeline missing prediction methods (predict_proba, decision_function, or predict) required for classification")
        elif task_type == "regression" and not hasattr(pycaret_pipeline, 'predict'):
            issues.append("Pipeline missing predict method required for regression")
    
    is_valid = len(issues) == 0
    
    if logger and not is_valid:
        logger.warning("SHAP input validation failed:")
        for issue in issues:
            logger.warning(f"  - {issue}")
    
    return is_valid, issues


def test_pipeline_prediction(
    pycaret_pipeline: Any,
    X_data_sample: pd.DataFrame,
    task_type: str,
    logger: Optional[logging.Logger] = None
) -> bool:
    """
    Test if the pipeline can make predictions on the sample data.
    
    Args:
        pycaret_pipeline: PyCaret pipeline object
        X_data_sample: Feature data sample
        task_type: "classification" or "regression"
        logger: Optional logger
        
    Returns:
        True if prediction test passes, False otherwise
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    try:
        # Take a small sample for testing
        test_sample = X_data_sample.head(min(5, len(X_data_sample)))
        
        if task_type == "classification":
            # Test the prediction function that will be used by SHAP
            try:
                prediction_function = _create_prediction_function(pycaret_pipeline, task_type, logger)
                result = prediction_function(test_sample)
                logger.info(f"Prediction test passed - output shape: {result.shape}")
                return True
            except Exception as e:
                logger.error(f"Classification prediction function test failed: {e}")
                return False
        else:
            # Test predict for regression
            pred_result = pycaret_pipeline.predict(test_sample)
            logger.info(f"Prediction test passed - predict shape: {pred_result.shape}")
            return True
        
    except Exception as e:
        logger.error(f"Pipeline prediction test failed: {e}")
        return False 