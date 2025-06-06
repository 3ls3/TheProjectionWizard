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
    # Get logger for this run and stage
    log = logger.get_stage_logger(run_id, constants.AUTOML_STAGE)
    
    try:
        # Log stage start
        log.info(f"Starting AutoML stage for run {run_id}")
        log.info("="*50)
        log.info("AUTOML STAGE - PYCARET MODEL TRAINING")
        log.info("="*50)
        
        # Update status to running
        try:
            status_data = {
                "stage": constants.AUTOML_STAGE,
                "status": "running",
                "message": "AutoML training in progress..."
            }
            storage.write_json_atomic(run_id, constants.STATUS_FILENAME, status_data)
        except Exception as e:
            log.warning(f"Could not update status to running: {e}")
        
        # =============================
        # 1. LOAD INPUTS
        # =============================
        log.info("Loading inputs: metadata and cleaned data")
        
        # Load metadata.json
        try:
            metadata_dict = storage.read_json(run_id, constants.METADATA_FILENAME)
        except Exception as e:
            log.error(f"Failed to load metadata.json: {e}")
            _update_status_failed(run_id, f"Failed to load metadata: {str(e)}")
            return False
        
        if not metadata_dict:
            log.error("metadata.json is empty or invalid")
            _update_status_failed(run_id, "Empty or invalid metadata")
            return False
        
        # Convert target_info dict to Pydantic object (Critical: Task 6 learnings)
        target_info_dict = metadata_dict.get('target_info')
        if not target_info_dict:
            log.error("No target_info found in metadata")
            _update_status_failed(run_id, "Missing target information in metadata")
            return False
        
        try:
            target_info = schemas.TargetInfo(**target_info_dict)
            log.info(f"Loaded target info: column='{target_info.name}', task_type='{target_info.task_type}', ml_type='{target_info.ml_type}'")
        except Exception as e:
            log.error(f"Failed to parse target_info: {e}")
            _update_status_failed(run_id, f"Invalid target info format: {str(e)}")
            return False
        
        # Check for prep_info (this should exist after prep stage)
        prep_info_dict = metadata_dict.get('prep_info')
        if not prep_info_dict:
            log.error("No prep_info found in metadata - prep stage must be completed first")
            _update_status_failed(run_id, "Missing prep_info - data preparation stage required")
            return False
        
        log.info(f"Found prep_info: final_shape={prep_info_dict.get('final_shape_after_prep')}")
        
        # Load cleaned data
        try:
            cleaned_data_path = storage.get_run_dir(run_id) / constants.CLEANED_DATA_FILE
            if not cleaned_data_path.exists():
                log.error(f"Cleaned data file not found: {cleaned_data_path}")
                _update_status_failed(run_id, "Cleaned data file not found - data preparation stage required")
                return False
                
            df_ml_ready = pd.read_csv(cleaned_data_path)
            log.info(f"Loaded cleaned data: shape {df_ml_ready.shape}")
            
            if df_ml_ready.empty:
                log.error("Cleaned data is empty")
                _update_status_failed(run_id, "Cleaned data file is empty")
                return False
                
        except Exception as e:
            log.error(f"Failed to load cleaned data: {e}")
            _update_status_failed(run_id, f"Failed to read cleaned data: {str(e)}")
            return False
        
        # Define PyCaret model directory (should already exist from prep stage)
        pycaret_model_dir = storage.get_run_dir(run_id) / constants.MODEL_DIR
        if not pycaret_model_dir.exists():
            log.warning(f"Model directory does not exist, creating: {pycaret_model_dir}")
            pycaret_model_dir.mkdir(parents=True, exist_ok=True)
        
        log.info(f"PyCaret model directory: {pycaret_model_dir}")
        
        # =============================
        # 2. VALIDATE INPUTS FOR PYCARET
        # =============================
        if not test_mode:
            log.info("Validating inputs for PyCaret...")
            
            try:
                is_valid, validation_issues = pycaret_logic.validate_pycaret_inputs(
                    df_ml_ready=df_ml_ready,
                    target_column_name=target_info.name,
                    task_type=target_info.task_type
                )
                
                if not is_valid:
                    log.error("Input validation failed for PyCaret:")
                    for issue in validation_issues:
                        log.error(f"  - {issue}")
                    _update_status_failed(run_id, f"Input validation failed: {'; '.join(validation_issues)}")
                    return False
                
                log.info("Input validation passed for PyCaret")
                
            except Exception as e:
                log.error(f"Input validation error: {e}")
                _update_status_failed(run_id, f"Input validation error: {str(e)}")
                return False
        else:
            log.warning("Skipping PyCaret input validation due to test mode")
        
        # =============================
        # 3. RUN PYCARET AUTOML EXPERIMENT
        # =============================
        log.info("Starting PyCaret AutoML experiment...")
        
        try:
            # Use configuration from constants
            automl_config = constants.AUTOML_CONFIG
            session_id = automl_config.get('session_id', 123)
            
            final_pipeline, metrics, model_name = pycaret_logic.run_pycaret_experiment(
                df_ml_ready=df_ml_ready,
                target_column_name=target_info.name,
                task_type=target_info.task_type,
                run_id=run_id,
                pycaret_model_dir=pycaret_model_dir,
                session_id=session_id,
                top_n_models_to_compare=3,  # Can be configured later
                allow_lightgbm_and_xgboost=True,  # Allow all models by default
                test_mode=test_mode
            )
            
        except Exception as e:
            log.error(f"PyCaret experiment failed with exception: {e}")
            _update_status_failed(run_id, f"PyCaret experiment failed: {str(e)}")
            return False
        
        # =============================
        # 4. HANDLE PYCARET RESULTS
        # =============================
        log.info("Processing PyCaret experiment results...")
        
        if final_pipeline is None or metrics is None or model_name is None:
            log.error("PyCaret experiment failed - returned None values")
            log.error(f"Pipeline: {final_pipeline}")
            log.error(f"Metrics: {metrics}")
            log.error(f"Model name: {model_name}")
            _update_status_failed(run_id, "AutoML (PyCaret) experiment failed")
            return False
        
        log.info("PyCaret experiment completed successfully!")
        log.info(f"Best model: {model_name}")
        log.info(f"Performance metrics: {len(metrics)} metrics extracted")
        
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
            
        except Exception as e:
            log.error(f"Failed to update metadata: {e}")
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
            
        except Exception as e:
            log.warning(f"Could not update final status: {e}")
        
        # =============================
        # 7. LOG COMPLETION
        # =============================
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
        
        return True
        
    except Exception as e:
        log.error(f"Unexpected error in AutoML stage: {e}")
        log.error("Full traceback:", exc_info=True)
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