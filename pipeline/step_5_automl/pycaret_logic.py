"""
PyCaret interaction logic for The Projection Wizard.
Handles AutoML training using PyCaret for classification and regression tasks.
"""

import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import traceback

from common import logger, constants


def run_pycaret_experiment(
    df_ml_ready: pd.DataFrame,
    target_column_name: str,
    task_type: str,
    run_id: str,
    pycaret_model_dir: Path,
    session_id: int = 123,
    top_n_models_to_compare: int = 3,
    allow_lightgbm_and_xgboost: bool = True,
    test_mode: bool = False
) -> Tuple[Optional[Any], Optional[dict], Optional[str]]:
    """
    Run PyCaret AutoML experiment for classification or regression.
    
    Args:
        df_ml_ready: The ML-ready DataFrame from cleaned_data.csv
        target_column_name: Name of the target column
        task_type: From target_info.task_type ("classification" or "regression")
        run_id: For logging and unique experiment naming
        pycaret_model_dir: Path object to data/runs/<run_id>/model/ where PyCaret will save its pipeline
        session_id: Integer for PyCaret reproducibility
        top_n_models_to_compare: How many top models from compare_models to consider
        allow_lightgbm_and_xgboost: Flag to include/exclude potentially problematic models
        test_mode: Flag to allow AutoML to work with very small datasets for testing purposes
        
    Returns:
        Tuple of (final_pycaret_pipeline, performance_metrics, best_model_name_str)
        Returns (None, None, None) if any step fails
    """
    # Get logger for this run
    log = logger.get_stage_logger(run_id, constants.AUTOML_STAGE)
    
    try:
        log.info(f"Starting PyCaret experiment for {task_type} task")
        log.info(f"Target column: {target_column_name}")
        log.info(f"Dataset shape: {df_ml_ready.shape}")
        log.info(f"Session ID: {session_id}")
        log.info(f"Top N models to compare: {top_n_models_to_compare}")
        log.info(f"Allow LightGBM/XGBoost: {allow_lightgbm_and_xgboost}")
        if test_mode:
            log.warning("TEST MODE ENABLED - allowing very small datasets")
        
        # Basic validation (always performed)
        if df_ml_ready.empty:
            log.error("Input DataFrame is empty")
            return None, None, None
            
        if target_column_name not in df_ml_ready.columns:
            log.error(f"Target column '{target_column_name}' not found in DataFrame")
            log.error(f"Available columns: {list(df_ml_ready.columns)}")
            return None, None, None
            
        if task_type not in ["classification", "regression"]:
            log.error(f"Invalid task_type: {task_type}. Must be 'classification' or 'regression'")
            return None, None, None
        
        # Full validation (unless in test mode)
        if not test_mode:
            is_valid, validation_issues = validate_pycaret_inputs(
                df_ml_ready=df_ml_ready,
                target_column_name=target_column_name,
                task_type=task_type
            )
            
            if not is_valid:
                log.error("Input validation failed for PyCaret:")
                for issue in validation_issues:
                    log.error(f"  - {issue}")
                return None, None, None
        else:
            log.warning("Skipping dataset size validation due to test mode")
        
        # Check for missing values in target
        target_nulls = df_ml_ready[target_column_name].isnull().sum()
        if target_nulls > 0:
            log.warning(f"Target column has {target_nulls} missing values")
            # Drop rows with missing targets for PyCaret
            df_ml_ready = df_ml_ready.dropna(subset=[target_column_name])
            log.info(f"Dropped rows with missing targets, new shape: {df_ml_ready.shape}")
        
        # Import appropriate PyCaret module based on task type
        if task_type == "classification":
            try:
                from pycaret.classification import (
                    setup, compare_models, finalize_model, save_model, 
                    predict_model, pull, get_config
                )
                log.info("Successfully imported PyCaret classification modules")
            except ImportError as e:
                log.error(f"Failed to import PyCaret classification modules: {e}")
                return None, None, None
        else:  # regression
            try:
                from pycaret.regression import (
                    setup, compare_models, finalize_model, save_model,
                    predict_model, pull, get_config
                )
                log.info("Successfully imported PyCaret regression modules")
            except ImportError as e:
                log.error(f"Failed to import PyCaret regression modules: {e}")
                return None, None, None
        
        # =============================
        # 1. PYCARET SETUP
        # =============================
        log.info("Setting up PyCaret environment...")
        
        try:
            pc_setup = setup(
                data=df_ml_ready,
                target=target_column_name,
                session_id=session_id,
                log_experiment=False,
                verbose=False,
                html=False,
                use_gpu=False,
                train_size=0.8  # 80% for training, 20% for test
            )
            log.info("PyCaret setup completed successfully")
            log.info(f"Setup configuration: {type(pc_setup)}")
        except Exception as e:
            log.error(f"PyCaret setup failed: {e}")
            log.error(f"Traceback: {traceback.format_exc()}")
            return None, None, None
        
        # =============================
        # 2. COMPARE MODELS
        # =============================
        log.info("Comparing models...")
        
        try:
            # Determine which models to include/exclude
            include_models = None
            exclude_models = []
            
            if not allow_lightgbm_and_xgboost:
                exclude_models = ['lightgbm', 'xgboost']
                log.info(f"Excluding models: {exclude_models}")
            
            # Adjust cross-validation folds based on dataset size
            n_rows = len(df_ml_ready)
            if test_mode and n_rows < 30:
                cv_folds = 2  # 2-fold for very small test datasets
                log.warning(f"Using minimal {cv_folds}-fold CV in test mode for {n_rows} rows")
            elif n_rows < 50:
                cv_folds = 3  # 3-fold for small datasets
            elif n_rows < 100:
                cv_folds = 5  # Default 5-fold for medium datasets
            else:
                cv_folds = 5  # 5-fold for larger datasets
            
            log.info(f"Using {cv_folds}-fold cross-validation for dataset with {n_rows} rows")
            
            # Compare models to find the best ones
            if exclude_models:
                top_models = compare_models(
                    n_select=top_n_models_to_compare,
                    exclude=exclude_models,
                    verbose=False,
                    fold=cv_folds
                )
            else:
                top_models = compare_models(
                    n_select=top_n_models_to_compare,
                    verbose=False,
                    fold=cv_folds
                )
            
            log.info(f"Model comparison completed, got {len(top_models) if isinstance(top_models, list) else 1} model(s)")
            
            # Handle single model vs list of models
            if not isinstance(top_models, list):
                if top_models is None:
                    log.error("No models returned from compare_models - creating fallback model")
                    # Create a simple fallback model
                    best_initial_model = _create_fallback_model(task_type, log)
                    if best_initial_model is None:
                        return None, None, None
                else:
                    best_initial_model = top_models
                    log.info("Single model returned from compare_models")
            else:
                if len(top_models) == 0:
                    log.error("No models returned from compare_models - creating fallback model")
                    # Create a simple fallback model
                    best_initial_model = _create_fallback_model(task_type, log)
                    if best_initial_model is None:
                        return None, None, None
                else:
                    best_initial_model = top_models[0]
                    log.info(f"Using first model from {len(top_models)} compared models")
            
            # Get model name
            try:
                model_name = str(type(best_initial_model).__name__)
                log.info(f"Best initial model: {model_name}")
            except:
                model_name = "Unknown Model"
                log.warning("Could not determine model name")
            
        except Exception as e:
            log.error(f"Model comparison failed: {e}")
            log.error(f"Traceback: {traceback.format_exc()}")
            return None, None, None
        
        # =============================
        # 3. FINALIZE MODEL (SKIP TUNING FOR MVP)
        # =============================
        log.info("Finalizing model (training on full dataset)...")
        
        try:
            # For MVP, skip tuning and directly finalize
            tuned_model = best_initial_model
            
            # Finalize trains on the full dataset (including hold-out test set)
            final_pipeline = finalize_model(tuned_model)
            log.info("Model finalization completed successfully")
            
        except Exception as e:
            log.error(f"Model finalization failed: {e}")
            log.error(f"Traceback: {traceback.format_exc()}")
            return None, None, None
        
        # =============================
        # 4. SAVE MODEL
        # =============================
        log.info("Saving PyCaret pipeline...")
        
        try:
            # Ensure the model directory exists
            pycaret_model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save the finalized pipeline
            model_save_path = str(pycaret_model_dir / 'pycaret_pipeline')
            save_model(final_pipeline, model_save_path)
            
            # PyCaret automatically adds .pkl extension
            actual_model_file = pycaret_model_dir / 'pycaret_pipeline.pkl'
            if actual_model_file.exists():
                log.info(f"Model saved successfully: {actual_model_file}")
            else:
                log.warning(f"Model file not found at expected location: {actual_model_file}")
            
        except Exception as e:
            log.error(f"Model saving failed: {e}")
            log.error(f"Traceback: {traceback.format_exc()}")
            return None, None, None
        
        # =============================
        # 5. EXTRACT PERFORMANCE METRICS
        # =============================
        log.info("Extracting performance metrics...")
        
        try:
            # Get the last metrics from PyCaret
            metrics_df = pull()
            
            if metrics_df is not None and not metrics_df.empty:
                # Convert metrics DataFrame to dictionary
                performance_metrics = {}
                
                if task_type == "classification":
                    # Extract key classification metrics
                    metric_columns = ['AUC', 'Accuracy', 'F1', 'Precision', 'Recall']
                    for metric in metric_columns:
                        if metric in metrics_df.columns:
                            # Get the value (usually in the first row)
                            value = metrics_df[metric].iloc[0] if len(metrics_df) > 0 else None
                            if value is not None:
                                performance_metrics[metric] = float(value)
                                log.info(f"  {metric}: {value:.4f}")
                
                else:  # regression
                    # Extract key regression metrics
                    metric_columns = ['R2', 'RMSE', 'MAE', 'MAPE']
                    for metric in metric_columns:
                        if metric in metrics_df.columns:
                            value = metrics_df[metric].iloc[0] if len(metrics_df) > 0 else None
                            if value is not None:
                                performance_metrics[metric] = float(value)
                                log.info(f"  {metric}: {value:.4f}")
                
                # If no standard metrics found, try to extract whatever is available
                if not performance_metrics:
                    log.warning("No standard metrics found, extracting available metrics")
                    for col in metrics_df.columns:
                        try:
                            value = metrics_df[col].iloc[0]
                            if isinstance(value, (int, float)):
                                performance_metrics[col] = float(value)
                        except:
                            continue
                
                log.info(f"Extracted {len(performance_metrics)} performance metrics")
                
            else:
                log.warning("Could not extract metrics from PyCaret, creating basic metrics")
                performance_metrics = {"status": "completed", "metrics_available": False}
        
        except Exception as e:
            log.warning(f"Failed to extract performance metrics: {e}")
            performance_metrics = {"status": "completed", "metrics_extraction_failed": True}
        
        # =============================
        # 6. GET BEST MODEL NAME
        # =============================
        try:
            # Try to get a more descriptive model name
            if hasattr(final_pipeline, 'named_steps'):
                # If it's a pipeline, get the last step (the actual model)
                estimator_step = list(final_pipeline.named_steps.values())[-1]
                best_model_name_str = str(type(estimator_step).__name__)
            else:
                # Direct model
                best_model_name_str = str(type(final_pipeline).__name__)
            
            # Clean up the name
            if best_model_name_str.endswith('Classifier') or best_model_name_str.endswith('Regressor'):
                # Remove common suffixes for cleaner display
                pass
            
            log.info(f"Final model name: {best_model_name_str}")
            
        except Exception as e:
            log.warning(f"Could not determine final model name: {e}")
            best_model_name_str = model_name if 'model_name' in locals() else "Unknown Model"
        
        # =============================
        # SUCCESS
        # =============================
        log.info("PyCaret experiment completed successfully")
        log.info(f"Final pipeline type: {type(final_pipeline)}")
        log.info(f"Model name: {best_model_name_str}")
        log.info(f"Performance metrics: {len(performance_metrics)} metrics extracted")
        
        return final_pipeline, performance_metrics, best_model_name_str
        
    except Exception as e:
        log.error(f"Unexpected error in PyCaret experiment: {e}")
        log.error(f"Traceback: {traceback.format_exc()}")
        return None, None, None


def validate_pycaret_inputs(
    df_ml_ready: pd.DataFrame,
    target_column_name: str,
    task_type: str
) -> Tuple[bool, List[str]]:
    """
    Validate inputs for PyCaret experiment.
    
    Args:
        df_ml_ready: The ML-ready DataFrame
        target_column_name: Name of the target column
        task_type: Task type ("classification" or "regression")
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    # Check DataFrame
    if df_ml_ready is None or df_ml_ready.empty:
        issues.append("DataFrame is None or empty")
        return False, issues
    
    # Check target column
    if not target_column_name or target_column_name not in df_ml_ready.columns:
        issues.append(f"Target column '{target_column_name}' not found in DataFrame")
    
    # Check task type
    if task_type not in ["classification", "regression"]:
        issues.append(f"Invalid task_type: {task_type}")
    
    # Check for minimum rows
    MIN_ROWS_FOR_ML = 30  # Increased from 10 to 30 for reliable cross-validation
    if len(df_ml_ready) < MIN_ROWS_FOR_ML:
        issues.append(f"Dataset too small: {len(df_ml_ready)} rows (minimum {MIN_ROWS_FOR_ML} required for reliable AutoML)")
    
    # Check for minimum features
    if len(df_ml_ready.columns) < 2:
        issues.append(f"Too few columns: {len(df_ml_ready.columns)} (minimum 2 required)")
    
    # Check target column has valid values
    if target_column_name in df_ml_ready.columns:
        target_nulls = df_ml_ready[target_column_name].isnull().sum()
        total_rows = len(df_ml_ready)
        
        if target_nulls == total_rows:
            issues.append("Target column is entirely null")
        elif target_nulls > total_rows * 0.5:
            issues.append(f"Target column has too many nulls: {target_nulls}/{total_rows} ({target_nulls/total_rows*100:.1f}%)")
        
        # Check for target variability
        unique_targets = df_ml_ready[target_column_name].nunique()
        if unique_targets < 2:
            issues.append(f"Target column has insufficient variability: {unique_targets} unique values")
    
    return len(issues) == 0, issues 


def _create_fallback_model(task_type: str, log):
    """
    Create a simple fallback model when PyCaret's compare_models fails.
    
    Args:
        task_type: "classification" or "regression"
        log: Logger instance
        
    Returns:
        A simple model or None if creation fails
    """
    try:
        log.info(f"Creating fallback model for {task_type}")
        
        if task_type == "classification":
            from pycaret.classification import create_model
            # Use a simple, robust classifier
            fallback_model = create_model('lr')  # Logistic Regression
            log.info("Created fallback Logistic Regression model")
        else:  # regression
            from pycaret.regression import create_model
            # Use a simple, robust regressor
            fallback_model = create_model('lr')  # Linear Regression
            log.info("Created fallback Linear Regression model")
        
        return fallback_model
        
    except Exception as e:
        log.error(f"Failed to create fallback model: {e}")
        return None