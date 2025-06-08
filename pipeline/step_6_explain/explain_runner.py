"""
Explainability Stage Runner for The Projection Wizard.
Orchestrates the complete model explainability stage using SHAP for global explanations.
"""

import pandas as pd
import joblib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple
import warnings

from common import storage, constants, schemas, logger
from . import shap_logic

# Suppress warnings during SHAP processing
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)


def run_explainability_stage(run_id: str) -> bool:
    """
    Execute the explainability stage to generate SHAP global feature importance analysis.
    
    This stage:
    1. Loads the trained PyCaret pipeline and metadata
    2. Loads the cleaned ML-ready data
    3. Validates inputs for SHAP analysis
    4. Generates and saves a SHAP summary plot
    5. Updates metadata with explainability results
    
    Args:
        run_id: Unique run identifier
        
    Returns:
        True if stage completes successfully, False otherwise
    """
    # Get stage-specific loggers for this run
    log = logger.get_stage_logger(run_id, constants.EXPLAIN_STAGE)
    structured_log = logger.get_stage_structured_logger(run_id, constants.EXPLAIN_STAGE)
    
    # Track timing
    start_time = datetime.now()
    
    try:
        # Log stage start
        log.info(f"Starting explainability stage for run {run_id}")
        log.info("="*50)
        log.info("EXPLAINABILITY STAGE - SHAP GLOBAL ANALYSIS")
        log.info("="*50)
        
        # Structured log: Stage started
        logger.log_structured_event(
            structured_log,
            "stage_started",
            {"stage": constants.EXPLAIN_STAGE},
            "Explainability stage started"
        )
        
        # Check if validation stage failed - prevent explainability from running
        try:
            status_data = storage.read_json(run_id, constants.STATUS_FILENAME)
            if status_data and status_data.get('stage') == constants.VALIDATION_STAGE:
                if status_data.get('status') == 'failed':
                    error_msg = "Cannot run explainability: validation stage failed"
                    validation_errors = status_data.get('errors', [])
                    if validation_errors:
                        error_msg += f" - Reasons: {'; '.join(validation_errors[:3])}"
                    
                    log.error(error_msg)
                    logger.log_structured_error(
                        structured_log,
                        "validation_failed_prerequisite",
                        error_msg,
                        {"stage": constants.EXPLAIN_STAGE, "validation_errors": validation_errors}
                    )
                    _update_status_failed(run_id, error_msg)
                    return False
        except Exception as e:
            log.warning(f"Could not check validation status: {e}")
            # Continue execution - don't fail on status check errors
        
        # Update status to running
        try:
            status_data = {
                "stage": constants.EXPLAIN_STAGE,
                "status": "running",
                "message": "Model explainability analysis in progress..."
            }
            storage.write_json_atomic(run_id, constants.STATUS_FILENAME, status_data)
            
            # Structured log: Status updated
            logger.log_structured_event(
                structured_log,
                "status_updated",
                {"status": "running", "stage": constants.EXPLAIN_STAGE},
                "Status updated to running"
            )
            
        except Exception as e:
            log.warning(f"Could not update status to running: {e}")
            logger.log_structured_error(
                structured_log,
                "status_update_failed",
                f"Could not update status to running: {e}",
                {"stage": constants.EXPLAIN_STAGE}
            )
        
        # =============================
        # 1. LOAD AND VALIDATE INPUTS
        # =============================
        log.info("Loading inputs: metadata, model pipeline, and cleaned data")
        
        # Load metadata.json
        try:
            metadata_dict = storage.read_json(run_id, constants.METADATA_FILENAME)
        except Exception as e:
            log.error(f"Failed to load metadata.json: {e}")
            logger.log_structured_error(
                structured_log,
                "metadata_load_failed",
                f"Failed to load metadata.json: {e}",
                {"stage": constants.EXPLAIN_STAGE}
            )
            _update_status_failed(run_id, f"Failed to load metadata: {str(e)}")
            return False
        
        if not metadata_dict:
            log.error("metadata.json is empty or invalid")
            logger.log_structured_error(
                structured_log,
                "metadata_invalid",
                "metadata.json is empty or invalid",
                {"stage": constants.EXPLAIN_STAGE}
            )
            _update_status_failed(run_id, "Empty or invalid metadata")
            return False
        
        # Get run directory
        run_dir = storage.get_run_dir(run_id)
        
        # Structured log: Metadata loaded
        logger.log_structured_event(
            structured_log,
            "metadata_loaded",
            {
                "file": constants.METADATA_FILENAME,
                "metadata_keys": list(metadata_dict.keys()),
                "run_directory": str(run_dir)
            },
            "Metadata loaded successfully"
        )
        
        # Validate required metadata components
        validation_success, metadata_components = _validate_metadata_for_explainability(
            metadata_dict, log
        )
        if not validation_success:
            logger.log_structured_error(
                structured_log,
                "metadata_validation_failed",
                "Invalid metadata for explainability stage",
                {"stage": constants.EXPLAIN_STAGE}
            )
            _update_status_failed(run_id, "Invalid metadata for explainability stage")
            return False
        
        target_info, automl_info = metadata_components
        
        # Structured log: Metadata validated
        logger.log_structured_event(
            structured_log,
            "metadata_validated",
            {
                "target_column": target_info.name,
                "task_type": target_info.task_type,
                "model_name": automl_info.best_model_name,
                "pipeline_path": automl_info.pycaret_pipeline_path
            },
            f"Metadata validation passed: {target_info.name} ({target_info.task_type})"
        )
        
        # Load PyCaret pipeline
        try:
            pycaret_pipeline_path = run_dir / automl_info.pycaret_pipeline_path
            
            if not pycaret_pipeline_path.exists():
                log.error(f"PyCaret pipeline not found: {pycaret_pipeline_path}")
                logger.log_structured_error(
                    structured_log,
                    "pipeline_file_not_found",
                    f"PyCaret pipeline not found: {pycaret_pipeline_path}",
                    {"stage": constants.EXPLAIN_STAGE, "pipeline_path": str(pycaret_pipeline_path)}
                )
                _update_status_failed(run_id, "PyCaret pipeline file not found")
                return False
            
            log.info(f"Loading PyCaret pipeline from: {pycaret_pipeline_path}")
            pycaret_pipeline = joblib.load(pycaret_pipeline_path)
            log.info("PyCaret pipeline loaded successfully")
            
            # Structured log: Pipeline loaded
            logger.log_structured_event(
                structured_log,
                "pipeline_loaded",
                {
                    "pipeline_path": str(pycaret_pipeline_path),
                    "model_name": automl_info.best_model_name
                },
                f"PyCaret pipeline loaded: {automl_info.best_model_name}"
            )
            
        except Exception as e:
            log.error(f"Failed to load PyCaret pipeline: {e}")
            logger.log_structured_error(
                structured_log,
                "pipeline_load_failed",
                f"Failed to load PyCaret pipeline: {e}",
                {"stage": constants.EXPLAIN_STAGE, "pipeline_path": str(pycaret_pipeline_path)}
            )
            _update_status_failed(run_id, f"Failed to load model pipeline: {str(e)}")
            return False
        
        # Load cleaned data
        try:
            cleaned_data_path = run_dir / constants.CLEANED_DATA_FILE
            
            if not cleaned_data_path.exists():
                log.error(f"Cleaned data file not found: {cleaned_data_path}")
                logger.log_structured_error(
                    structured_log,
                    "cleaned_data_not_found",
                    f"Cleaned data file not found: {cleaned_data_path}",
                    {"stage": constants.EXPLAIN_STAGE, "data_path": str(cleaned_data_path)}
                )
                _update_status_failed(run_id, "Cleaned data file not found")
                return False
            
            df_ml_ready = pd.read_csv(cleaned_data_path)
            log.info(f"Loaded cleaned data: shape {df_ml_ready.shape}")
            
            if df_ml_ready.empty:
                log.error("Cleaned data is empty")
                logger.log_structured_error(
                    structured_log,
                    "cleaned_data_empty",
                    "Cleaned data is empty",
                    {"stage": constants.EXPLAIN_STAGE}
                )
                _update_status_failed(run_id, "Cleaned data file is empty")
                return False
            
            # Structured log: Data loaded
            logger.log_structured_event(
                structured_log,
                "data_loaded",
                {
                    "data_shape": {"rows": df_ml_ready.shape[0], "columns": df_ml_ready.shape[1]},
                    "data_file": constants.CLEANED_DATA_FILE
                },
                f"Cleaned data loaded: {df_ml_ready.shape}"
            )
            
        except Exception as e:
            log.error(f"Failed to load cleaned data: {e}")
            logger.log_structured_error(
                structured_log,
                "data_load_failed",
                f"Failed to load cleaned data: {e}",
                {"stage": constants.EXPLAIN_STAGE, "data_path": str(cleaned_data_path)}
            )
            _update_status_failed(run_id, f"Failed to read cleaned data: {str(e)}")
            return False
        
        # Prepare feature data (remove target column)
        target_column = target_info.name
        if target_column not in df_ml_ready.columns:
            log.error(f"Target column '{target_column}' not found in cleaned data")
            logger.log_structured_error(
                structured_log,
                "target_column_missing",
                f"Target column '{target_column}' not found in cleaned data",
                {"stage": constants.EXPLAIN_STAGE, "target_column": target_column, "available_columns": list(df_ml_ready.columns)}
            )
            _update_status_failed(run_id, f"Target column '{target_column}' not found")
            return False
        
        X_data = df_ml_ready.drop(columns=[target_column])
        log.info(f"Feature data prepared: {X_data.shape} (removed target column '{target_column}')")
        
        # Structured log: Feature data prepared
        logger.log_structured_event(
            structured_log,
            "feature_data_prepared",
            {
                "feature_shape": {"rows": X_data.shape[0], "columns": X_data.shape[1]},
                "target_column": target_column,
                "feature_columns": list(X_data.columns)
            },
            f"Feature data prepared: {X_data.shape} features"
        )
        
        # =============================
        # 2. VALIDATE INPUTS FOR SHAP
        # =============================
        log.info("Validating inputs for SHAP analysis...")
        
        try:
            is_valid, validation_issues = shap_logic.validate_shap_inputs(
                pycaret_pipeline=pycaret_pipeline,
                X_data_sample=X_data,
                task_type=target_info.task_type,
                logger=log
            )
            
            if not is_valid:
                log.error("SHAP input validation failed:")
                for issue in validation_issues:
                    log.error(f"  - {issue}")
                logger.log_structured_error(
                    structured_log,
                    "shap_validation_failed",
                    f"SHAP validation failed: {'; '.join(validation_issues)}",
                    {"stage": constants.EXPLAIN_STAGE, "validation_issues": validation_issues}
                )
                _update_status_failed(run_id, f"SHAP validation failed: {'; '.join(validation_issues)}")
                return False
            
            log.info("SHAP input validation passed")
            
            # Structured log: SHAP validation passed
            logger.log_structured_event(
                structured_log,
                "shap_validation_passed",
                {"validation_issues_count": 0},
                "SHAP input validation passed"
            )
            
        except Exception as e:
            log.error(f"SHAP input validation error: {e}")
            logger.log_structured_error(
                structured_log,
                "shap_validation_error",
                f"SHAP input validation error: {e}",
                {"stage": constants.EXPLAIN_STAGE}
            )
            _update_status_failed(run_id, f"SHAP validation error: {str(e)}")
            return False
        
        # Test pipeline prediction capability
        log.info("Testing pipeline prediction capability...")
        try:
            if not shap_logic.test_pipeline_prediction(
                pycaret_pipeline, X_data, target_info.task_type, log
            ):
                log.error("Pipeline prediction test failed")
                logger.log_structured_error(
                    structured_log,
                    "pipeline_prediction_test_failed",
                    "Model pipeline prediction test failed",
                    {"stage": constants.EXPLAIN_STAGE}
                )
                _update_status_failed(run_id, "Model pipeline prediction test failed")
                return False
            
            log.info("Pipeline prediction test passed")
            
            # Structured log: Pipeline test passed
            logger.log_structured_event(
                structured_log,
                "pipeline_test_passed",
                {"test_type": "prediction_capability"},
                "Pipeline prediction test passed"
            )
            
        except Exception as e:
            log.error(f"Pipeline prediction test error: {e}")
            logger.log_structured_error(
                structured_log,
                "pipeline_test_error",
                f"Pipeline prediction test error: {e}",
                {"stage": constants.EXPLAIN_STAGE}
            )
            _update_status_failed(run_id, f"Pipeline prediction test error: {str(e)}")
            return False
        
        # =============================
        # 3. GENERATE SHAP SUMMARY PLOT
        # =============================
        log.info("Generating SHAP summary plot...")
        
        # Create plots directory and define plot path
        plots_dir = run_dir / constants.PLOTS_DIR
        plots_dir.mkdir(exist_ok=True)
        plot_save_path = plots_dir / constants.SHAP_SUMMARY_PLOT
        
        log.info(f"SHAP plot will be saved to: {plot_save_path}")
        
        # Structured log: Plot generation started
        logger.log_structured_event(
            structured_log,
            "plot_generation_started",
            {
                "plot_type": "shap_summary",
                "plots_directory": str(plots_dir),
                "plot_file": constants.SHAP_SUMMARY_PLOT
            },
            "SHAP summary plot generation started"
        )
        
        try:
            # Generate SHAP summary plot
            plot_success = shap_logic.generate_shap_summary_plot(
                pycaret_pipeline=pycaret_pipeline,
                X_data_sample=X_data,
                plot_save_path=plot_save_path,
                task_type=target_info.task_type,
                logger=log
            )
            
            if not plot_success:
                log.error("SHAP summary plot generation failed")
                logger.log_structured_error(
                    structured_log,
                    "plot_generation_failed",
                    "SHAP summary plot generation failed",
                    {"stage": constants.EXPLAIN_STAGE}
                )
                _update_status_failed(run_id, "SHAP plot generation failed")
                return False
            
            log.info("SHAP summary plot generated successfully")
            
            # Structured log: Plot generated
            logger.log_structured_event(
                structured_log,
                "plot_generated",
                {
                    "plot_type": "shap_summary",
                    "plots_directory": str(plots_dir),
                    "plot_file": constants.SHAP_SUMMARY_PLOT
                },
                "SHAP summary plot generated successfully"
            )
            
        except Exception as e:
            log.error(f"SHAP plot generation error: {e}")
            logger.log_structured_error(
                structured_log,
                "plot_generation_error",
                f"SHAP plot generation error: {e}",
                {"stage": constants.EXPLAIN_STAGE}
            )
            _update_status_failed(run_id, f"SHAP plot generation error: {str(e)}")
            return False
        
        # =============================
        # 4. UPDATE METADATA WITH EXPLAINABILITY INFO
        # =============================
        log.info("Updating metadata with explainability results...")
        
        try:
            # Create explainability info
            explain_info = {
                'tool_used': 'SHAP',
                'explanation_type': 'global_summary',
                'shap_summary_plot_path': str(Path(constants.PLOTS_DIR) / constants.SHAP_SUMMARY_PLOT),
                'explain_completed_at': datetime.utcnow().isoformat(),
                'target_column': target_info.name,
                'task_type': target_info.task_type,
                'features_explained': len(X_data.columns),
                'samples_used_for_explanation': len(X_data)
            }
            
            # Update metadata with explainability info
            metadata_dict['explain_info'] = explain_info
            storage.write_json_atomic(run_id, constants.METADATA_FILENAME, metadata_dict)
            
            log.info("Metadata updated with explainability information")
            
            # Structured log: Metadata updated
            logger.log_structured_event(
                structured_log,
                "metadata_updated",
                {
                    "file": constants.METADATA_FILENAME,
                    "metadata_keys": list(metadata_dict.keys()),
                    "run_directory": str(run_dir)
                },
                "Metadata updated with explainability information"
            )
            
        except Exception as e:
            log.error(f"Failed to update metadata: {e}")
            logger.log_structured_error(
                structured_log,
                "metadata_update_failed",
                f"Failed to update metadata: {e}",
                {"stage": constants.EXPLAIN_STAGE}
            )
            _update_status_failed(run_id, f"Failed to update metadata: {str(e)}")
            return False
        
        # =============================
        # 5. UPDATE PIPELINE STATUS
        # =============================
        log.info("Updating pipeline status to completed...")
        
        try:
            status_data = {
                "stage": constants.EXPLAIN_STAGE,
                "status": "completed",
                "message": "Model explainability analysis completed successfully"
            }
            storage.write_json_atomic(run_id, constants.STATUS_FILENAME, status_data)
            
            # Structured log: Status updated
            logger.log_structured_event(
                structured_log,
                "status_updated",
                {"status": "completed", "stage": constants.EXPLAIN_STAGE},
                "Status updated to completed"
            )
            
        except Exception as e:
            log.warning(f"Could not update final status: {e}")
            logger.log_structured_error(
                structured_log,
                "final_status_update_failed",
                f"Could not update final status: {e}",
                {"stage": constants.EXPLAIN_STAGE}
            )
        
        # Calculate stage duration
        end_time = datetime.now()
        stage_duration = (end_time - start_time).total_seconds()
        
        log.info("="*50)
        log.info("EXPLAINABILITY STAGE COMPLETED SUCCESSFULLY")
        log.info("="*50)
        log.info(f"SHAP summary plot saved to: {plot_save_path}")
        log.info(f"Features explained: {len(X_data.columns)}")
        log.info(f"Samples used: {len(X_data)}")
        
        # Structured log: Stage completed
        logger.log_structured_event(
            structured_log,
            "stage_completed",
            {
                "stage": constants.EXPLAIN_STAGE,
                "success": True,
                "duration_seconds": stage_duration,
                "completed_at": end_time.isoformat(),
                "plots_generated": [constants.SHAP_SUMMARY_PLOT],
                "features_explained": len(X_data.columns),
                "samples_used": len(X_data)
            },
            f"Explainability stage completed successfully in {stage_duration:.1f}s"
        )
        
        return True
        
    except Exception as e:
        log.error(f"Unexpected error in explainability stage: {e}")
        logger.log_structured_error(
            structured_log,
            "unexpected_error",
            f"Unexpected error in explainability stage: {e}",
            {"stage": constants.EXPLAIN_STAGE}
        )
        _update_status_failed(run_id, f"Unexpected error: {str(e)}")
        return False


def _validate_metadata_for_explainability(
    metadata_dict: dict, 
    log: logger.logging.Logger
) -> Tuple[bool, Optional[Tuple[schemas.TargetInfo, schemas.AutoMLInfo]]]:
    """
    Validate that metadata contains required information for explainability stage.
    
    Args:
        metadata_dict: Loaded metadata dictionary
        log: Logger instance
        
    Returns:
        Tuple of (is_valid, (target_info, automl_info) or None)
    """
    try:
        # Check for target_info
        target_info_dict = metadata_dict.get('target_info')
        if not target_info_dict:
            log.error("No target_info found in metadata")
            return False, None
        
        try:
            target_info = schemas.TargetInfo(**target_info_dict)
            log.info(f"Target info loaded: column='{target_info.name}', task_type='{target_info.task_type}'")
        except Exception as e:
            log.error(f"Failed to parse target_info: {e}")
            return False, None
        
        # Check for automl_info
        automl_info_dict = metadata_dict.get('automl_info')
        if not automl_info_dict:
            log.error("No automl_info found in metadata - AutoML stage must be completed first")
            return False, None
        
        try:
            automl_info = schemas.AutoMLInfo(**automl_info_dict)
            log.info(f"AutoML info loaded: tool='{automl_info.tool_used}', model='{automl_info.best_model_name}'")
        except Exception as e:
            log.error(f"Failed to parse automl_info: {e}")
            return False, None
        
        # Validate pipeline path exists in automl_info
        if not automl_info.pycaret_pipeline_path:
            log.error("No pycaret_pipeline_path found in automl_info")
            return False, None
        
        log.info("Metadata validation passed for explainability stage")
        return True, (target_info, automl_info)
        
    except Exception as e:
        log.error(f"Metadata validation error: {e}")
        return False, None


def _update_status_failed(run_id: str, error_message: str) -> None:
    """
    Update pipeline status to failed with error message.
    
    Args:
        run_id: Unique run identifier
        error_message: Error message to include in status
    """
    try:
        status_data = {
            "stage": constants.EXPLAIN_STAGE,
            "status": "failed",
            "message": error_message
        }
        storage.write_json_atomic(run_id, constants.STATUS_FILENAME, status_data)
    except Exception as e:
        # If we can't even update status, log it but don't fail further
        print(f"Warning: Could not update failed status for run {run_id}: {e}")


def validate_explainability_stage_inputs(run_id: str) -> bool:
    """
    Validate that all required inputs exist for the explainability stage.
    
    Args:
        run_id: Unique run identifier
        
    Returns:
        True if all inputs are valid, False otherwise
    """
    try:
        # Check metadata
        metadata_dict = storage.read_json(run_id, constants.METADATA_FILENAME)
        if not metadata_dict:
            return False
        
        # Validate metadata components
        validation_success, _ = _validate_metadata_for_explainability(
            metadata_dict, logger.get_logger(run_id, "validation")
        )
        if not validation_success:
            return False
        
        # Check files exist
        run_dir = storage.get_run_dir(run_id)
        
        # Check cleaned data
        cleaned_data_path = run_dir / constants.CLEANED_DATA_FILE
        if not cleaned_data_path.exists():
            return False
        
        # Check PyCaret pipeline
        automl_info_dict = metadata_dict.get('automl_info', {})
        pipeline_path = automl_info_dict.get('pycaret_pipeline_path')
        if not pipeline_path:
            return False
        
        pycaret_pipeline_path = run_dir / pipeline_path
        if not pycaret_pipeline_path.exists():
            return False
        
        return True
        
    except Exception:
        return False


def get_explainability_stage_summary(run_id: str) -> Optional[dict]:
    """
    Get summary information about the explainability stage results.
    
    Args:
        run_id: Unique run identifier
        
    Returns:
        Dictionary with explainability stage summary, or None if not completed
    """
    try:
        metadata_dict = storage.read_json(run_id, constants.METADATA_FILENAME)
        if not metadata_dict:
            return None
        
        explain_info = metadata_dict.get('explain_info')
        if not explain_info:
            return None
        
        # Get additional context
        target_info_dict = metadata_dict.get('target_info', {})
        automl_info_dict = metadata_dict.get('automl_info', {})
        
        run_dir = storage.get_run_dir(run_id)
        plot_path = run_dir / explain_info.get('shap_summary_plot_path', '')
        
        summary = {
            'explain_completed': True,
            'tool_used': explain_info.get('tool_used', 'SHAP'),
            'explanation_type': explain_info.get('explanation_type', 'global_summary'),
            'target_column': explain_info.get('target_column'),
            'task_type': explain_info.get('task_type'),
            'features_explained': explain_info.get('features_explained'),
            'samples_used': explain_info.get('samples_used_for_explanation'),
            'completed_at': explain_info.get('explain_completed_at'),
            'plot_file_exists': plot_path.exists() if plot_path else False,
            'plot_path': str(plot_path) if plot_path and plot_path.exists() else None,
            'best_model_name': automl_info_dict.get('best_model_name'),
            'performance_metrics': automl_info_dict.get('performance_metrics', {})
        }
        
        return summary
        
    except Exception:
        return None 