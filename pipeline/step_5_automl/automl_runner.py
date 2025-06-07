"""
AutoML Stage Runner for The Projection Wizard.
Orchestrates the complete AutoML stage using PyCaret for model training.
"""

import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from common import logger, storage, constants, schemas
from . import pycaret_logic


def run_automl_stage(run_id: str, test_mode: bool = False) -> bool:
    """
    Execute the complete AutoML stage for a given run.
    
    This function orchestrates:
    1. Loading inputs (metadata.json, cleaned_data.csv)
    2. Running PyCaret AutoML experiment
    3. Saving model and updating metadata
    4. Updating pipeline status
    
    Args:
        run_id: Unique run identifier
        test_mode: Flag to allow AutoML to work with very small datasets for testing purposes
        
    Returns:
        True if stage completes successfully, False otherwise
    """
    # Get loggers for this run and stage
    log = logger.get_stage_logger(run_id, constants.AUTOML_STAGE)
    structured_log = logger.get_stage_structured_logger(run_id, constants.AUTOML_STAGE)
    
    # Get pipeline summary logger
    summary_logger = logger.get_pipeline_summary_logger(run_id)
    
    # Track timing for summary
    start_time = datetime.now()
    
    try:
        # =============================
        # 1. VALIDATE INPUTS
        # =============================
        log.info("Starting AutoML stage validation...")
        
        # Structured log: Stage started
        logger.log_structured_event(
            structured_log,
            "stage_started",
            {"stage": constants.AUTOML_STAGE, "test_mode": test_mode},
            "AutoML stage validation started"
        )
        
        if not validate_automl_stage_inputs(run_id):
            log.error("AutoML stage input validation failed")
            logger.log_structured_error(
                structured_log,
                "input_validation_failed",
                "AutoML stage input validation failed",
                {"stage": constants.AUTOML_STAGE}
            )
            _update_status_failed(run_id, "Input validation failed")
            return False
        
        log.info("AutoML stage inputs validated successfully")
        
        # Structured log: Validation passed
        logger.log_structured_event(
            structured_log,
            "input_validation_passed",
            {"stage": constants.AUTOML_STAGE},
            "AutoML stage input validation passed"
        )
        
        # =============================
        # 2. LOAD METADATA AND DATA
        # =============================
        log.info("Loading metadata and cleaned data...")
        
        try:
            # Load metadata
            metadata_dict = storage.read_json(run_id, constants.METADATA_FILENAME)
            target_info_dict = metadata_dict['target_info']
            target_info = schemas.TargetInfo(**target_info_dict)
            
            log.info(f"Target info loaded: column='{target_info.name}', task_type='{target_info.task_type}'")
            
            # Load cleaned data
            df_ml_ready = storage.read_cleaned_data(run_id)
            if df_ml_ready is None:
                raise Exception("Could not load cleaned data")
            
            log.info(f"Cleaned data loaded: shape={df_ml_ready.shape}")
            
            # Structured log: Data loaded
            logger.log_structured_event(
                structured_log,
                "data_loaded",
                {
                    "target_column": target_info.name,
                    "task_type": target_info.task_type,
                    "data_shape": {"rows": df_ml_ready.shape[0], "columns": df_ml_ready.shape[1]}
                },
                f"Training data loaded: {df_ml_ready.shape}"
            )
            
        except Exception as e:
            log.error(f"Failed to load inputs: {e}")
            logger.log_structured_error(
                structured_log,
                "data_loading_failed",
                f"Failed to load inputs: {str(e)}",
                {"stage": constants.AUTOML_STAGE}
            )
            _update_status_failed(run_id, f"Failed to load inputs: {str(e)}")
            return False
        
        # =============================
        # 3. RUN AUTOML
        # =============================
        log.info("Starting PyCaret AutoML experiment...")
        
        # Structured log: Training started
        logger.log_structured_event(
            structured_log,
            "training_started",
            {
                "tool": "PyCaret",
                "dataset_shape": {"rows": df_ml_ready.shape[0], "columns": df_ml_ready.shape[1]},
                "target_column": target_info.name,
                "task_type": target_info.task_type
            },
            "AutoML training started with PyCaret"
        )
        
        try:
            # Run PyCaret AutoML
            best_model, metrics, model_name = pycaret_logic.run_pycaret_experiment(
                df_ml_ready=df_ml_ready,
                target_column_name=target_info.name,
                task_type=target_info.task_type,
                run_id=run_id,
                pycaret_model_dir=storage.get_run_dir(run_id) / constants.MODEL_DIR,
                test_mode=test_mode
            )
            
            log.info(f"AutoML experiment completed. Best model: {model_name}")
            log.info(f"Performance metrics: {metrics}")
            
            # Structured log: Training completed with metrics
            logger.log_structured_event(
                structured_log,
                "training_completed",
                {
                    "model_name": model_name,
                    "metrics": metrics,
                    "tool": "PyCaret"
                },
                f"AutoML training completed: {model_name}"
            )
            
            # Log individual metrics as structured events
            for metric_name, metric_value in metrics.items():
                logger.log_structured_metric(
                    structured_log,
                    metric_name,
                    metric_value,
                    "performance",
                    {"model_name": model_name, "task_type": target_info.task_type}
                )
            
        except Exception as e:
            log.error(f"PyCaret AutoML failed: {e}")
            logger.log_structured_error(
                structured_log,
                "training_failed",
                f"PyCaret AutoML failed: {str(e)}",
                {"tool": "PyCaret", "stage": constants.AUTOML_STAGE}
            )
            _update_status_failed(run_id, f"AutoML training failed: {str(e)}")
            return False
        
        # =============================
        # 4. SAVE MODEL
        # =============================
        log.info("Saving trained model...")
        
        try:
            # Verify model was saved by the pycaret_experiment function
            run_dir = storage.get_run_dir(run_id)
            model_dir = run_dir / constants.MODEL_DIR
            model_path = model_dir / "pycaret_pipeline.pkl"
            
            if model_path.exists():
                log.info(f"Model verified saved at: {model_path}")
                
                # Structured log: Model saved
                logger.log_structured_event(
                    structured_log,
                    "model_saved",
                    {
                        "model_path": str(model_path.relative_to(run_dir)),
                        "model_name": model_name,
                        "file_size_bytes": model_path.stat().st_size
                    },
                    f"Model saved: {model_path.name}"
                )
            else:
                raise FileNotFoundError(f"Model was not saved at expected location: {model_path}")
            
        except Exception as e:
            log.error(f"Failed to save model: {e}")
            logger.log_structured_error(
                structured_log,
                "model_save_failed",
                f"Failed to save model: {str(e)}",
                {"stage": constants.AUTOML_STAGE}
            )
            _update_status_failed(run_id, f"Failed to save model: {str(e)}")
            return False
        
        # =============================
        # 5. UPDATE METADATA WITH AUTOML INFO
        # =============================
        log.info("Updating metadata with AutoML results...")
        
        try:
            # Create automl_info dictionary
            automl_info = {
                'tool_used': 'PyCaret',
                'best_model_name': model_name,
                'pycaret_pipeline_path': str(Path(constants.MODEL_DIR) / 'pycaret_pipeline.pkl'),  # Relative to run_dir
                'performance_metrics': metrics,
                'automl_completed_at': datetime.now(timezone.utc).isoformat(),
                'target_column': target_info.name,
                'task_type': target_info.task_type,
                'dataset_shape_for_training': list(df_ml_ready.shape)
            }
            
            # Add automl_info to existing metadata
            metadata_dict['automl_info'] = automl_info
            
            # Save updated metadata
            storage.write_json_atomic(run_id, constants.METADATA_FILENAME, metadata_dict)
            log.info("Metadata updated with AutoML results")
            
            # Structured log: Metadata updated
            logger.log_structured_event(
                structured_log,
                "metadata_updated",
                {"automl_info_keys": list(automl_info.keys())},
                "Metadata updated with AutoML results"
            )
            
        except Exception as e:
            log.error(f"Failed to update metadata: {e}")
            logger.log_structured_error(
                structured_log,
                "metadata_update_failed",
                f"Failed to update metadata: {str(e)}",
                {"stage": constants.AUTOML_STAGE}
            )
            _update_status_failed(run_id, f"Failed to update metadata: {str(e)}")
            return False
        
        # =============================
        # 6. UPDATE STATUS TO COMPLETED
        # =============================
        try:
            status_data = {
                "stage": constants.AUTOML_STAGE,
                "status": "completed",
                "message": f"AutoML completed. Best model: {model_name}"
            }
            storage.write_json_atomic(run_id, constants.STATUS_FILENAME, status_data)
            log.info("Status updated to completed")
            
            # Structured log: Status updated
            logger.log_structured_event(
                structured_log,
                "status_updated",
                {"status": "completed", "stage": constants.AUTOML_STAGE},
                "AutoML stage status updated to completed"
            )
            
        except Exception as e:
            log.warning(f"Could not update final status: {e}")
        
        # =============================
        # 7. LOG COMPLETION SUMMARY
        # =============================
        end_time = datetime.now()
        training_duration = (end_time - start_time).total_seconds()
        
        # Log detailed completion for technical log
        log.info("="*50)
        log.info("AUTOML STAGE COMPLETED SUCCESSFULLY")
        log.info("="*50)
        log.info(f"Dataset shape: {df_ml_ready.shape}")
        log.info(f"Target column: {target_info.name}")
        log.info(f"Task type: {target_info.task_type}")
        log.info(f"Best model: {model_name}")
        log.info(f"Performance metrics:")
        for metric_name, metric_value in metrics.items():
            log.info(f"  - {metric_name}: {metric_value}")
        log.info(f"Output files:")
        log.info(f"  - {constants.MODEL_DIR}/pycaret_pipeline.pkl")
        log.info(f"  - {constants.METADATA_FILENAME} (updated)")
        log.info(f"  - {constants.STATUS_FILENAME} (updated)")
        log.info("="*50)
        
        # Structured log: Stage completed
        logger.log_structured_event(
            structured_log,
            "stage_completed",
            {
                "stage": constants.AUTOML_STAGE,
                "duration_seconds": training_duration,
                "model_name": model_name,
                "metrics_count": len(metrics),
                "completed_at": end_time.isoformat()
            },
            f"AutoML stage completed successfully in {training_duration:.1f}s"
        )
        
        # Log high-level summary for pipeline summary
        summary_logger.log_automl_summary(
            model_name=model_name,
            task_type=target_info.task_type,
            target_column=target_info.name,
            training_shape=df_ml_ready.shape,
            metrics=metrics,
            training_duration=training_duration
        )
        
        return True
        
    except Exception as e:
        log.error(f"Unexpected error in AutoML stage: {e}")
        log.error("Full traceback:", exc_info=True)
        
        # Structured log: Unexpected error
        logger.log_structured_error(
            structured_log,
            "unexpected_error",
            f"Unexpected error in AutoML stage: {str(e)}",
            {"stage": constants.AUTOML_STAGE}
        )
        
        _update_status_failed(run_id, f"Unexpected error: {str(e)}")
        return False


def _update_status_failed(run_id: str, error_message: str) -> None:
    """
    Helper function to update status.json to failed state.
    
    Args:
        run_id: Run identifier
        error_message: Error message to include in status
    """
    try:
        status_data = {
            "stage": constants.AUTOML_STAGE,
            "status": "failed",
            "message": f"AutoML failed: {error_message}"
        }
        storage.write_json_atomic(run_id, constants.STATUS_FILENAME, status_data)
    except Exception as e:
        # If we can't even update status, log it but don't raise
        log = logger.get_stage_logger(run_id, constants.AUTOML_STAGE)
        log.error(f"Could not update status to failed: {e}")


def validate_automl_stage_inputs(run_id: str) -> bool:
    """
    Validate that all required inputs for the AutoML stage are available.
    
    Args:
        run_id: Run identifier
        
    Returns:
        True if all inputs are valid, False otherwise
    """
    log = logger.get_stage_logger(run_id, constants.AUTOML_STAGE)
    
    try:
        # Check if run directory exists
        run_dir = storage.get_run_dir(run_id)
        if not run_dir.exists():
            log.error(f"Run directory does not exist: {run_dir}")
            return False
        
        # Check if metadata.json exists
        metadata_path = run_dir / constants.METADATA_FILENAME
        if not metadata_path.exists():
            log.error(f"Metadata file does not exist: {metadata_path}")
            return False
        
        # Check if cleaned_data.csv exists
        cleaned_data_path = run_dir / constants.CLEANED_DATA_FILE
        if not cleaned_data_path.exists():
            log.error(f"Cleaned data file does not exist: {cleaned_data_path}")
            return False
        
        # Try to load and validate metadata structure
        try:
            metadata_dict = storage.read_json(run_id, constants.METADATA_FILENAME)
            
            # Check for required keys
            if 'target_info' not in metadata_dict:
                log.error("Missing 'target_info' in metadata")
                return False
            
            if 'prep_info' not in metadata_dict:
                log.error("Missing 'prep_info' in metadata - prep stage must be completed first")
                return False
            
            # Validate target_info can be parsed
            target_info_dict = metadata_dict['target_info']
            target_info = schemas.TargetInfo(**target_info_dict)
            
            # Validate prep_info exists and has expected structure
            prep_info = metadata_dict['prep_info']
            if not isinstance(prep_info, dict):
                log.error("prep_info is not a dictionary")
                return False
            
            # Check for key prep_info fields
            required_prep_fields = ['final_shape_after_prep', 'cleaning_steps_performed']
            for field in required_prep_fields:
                if field not in prep_info:
                    log.error(f"Missing required field in prep_info: {field}")
                    return False
            
            log.info("All AutoML stage inputs validated successfully")
            return True
            
        except Exception as e:
            log.error(f"Metadata validation failed: {e}")
            return False
        
    except Exception as e:
        log.error(f"Input validation failed: {e}")
        return False


def get_automl_stage_summary(run_id: str) -> Optional[dict]:
    """
    Get a summary of the AutoML stage results for a given run.
    
    Args:
        run_id: Run identifier
        
    Returns:
        Dictionary with AutoML stage summary or None if not available
    """
    try:
        metadata_dict = storage.read_json(run_id, constants.METADATA_FILENAME)
        automl_info = metadata_dict.get('automl_info')
        
        if not automl_info:
            return None
        
        # Check if model file exists
        model_file_path = storage.get_run_dir(run_id) / automl_info.get('pycaret_pipeline_path', '')
        
        return {
            'run_id': run_id,
            'tool_used': automl_info.get('tool_used'),
            'best_model_name': automl_info.get('best_model_name'),
            'task_type': automl_info.get('task_type'),
            'target_column': automl_info.get('target_column'),
            'performance_metrics': automl_info.get('performance_metrics', {}),
            'training_dataset_shape': automl_info.get('dataset_shape_for_training'),
            'model_file_exists': model_file_path.exists() if model_file_path else False,
            'completed_at': automl_info.get('automl_completed_at')
        }
        
    except Exception:
        return None 