"""
AutoML Stage Runner for The Projection Wizard.
Orchestrates the complete AutoML stage using PyCaret for model training.
Refactored for GCS-based storage.
"""

import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import traceback
import tempfile
import io
import json

from common import logger, storage, constants, schemas
from api.utils.gcs_utils import (
    download_run_file, upload_run_file, check_run_file_exists, 
    PROJECT_BUCKET_NAME
)
from . import pycaret_logic


def run_automl_stage_gcs(run_id: str, 
                        test_mode: bool = False,
                        gcs_bucket_name: str = PROJECT_BUCKET_NAME) -> bool:
    """
    Execute the complete AutoML stage for a given run using GCS storage.

    This function orchestrates:
    1. Loading inputs (metadata.json, cleaned_data.csv) from GCS
    2. Running PyCaret AutoML experiment
    3. Saving model to GCS and updating metadata
    4. Updating pipeline status

    Args:
        run_id: Unique run identifier
        test_mode: Flag to allow AutoML to work with very small datasets for testing purposes
        gcs_bucket_name: GCS bucket name for storage

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
        log.info("Starting AutoML stage validation (GCS-based)...")

        # Structured log: Stage started
        logger.log_structured_event(
            structured_log,
            "stage_started",
            {"stage": constants.AUTOML_STAGE, "test_mode": test_mode, "storage_type": "gcs"},
            "AutoML stage validation started (GCS-based)"
        )

        # Check if validation stage failed - prevent AutoML from running
        try:
            status_data = storage.read_json(run_id, constants.STATUS_FILENAME)
            if status_data and status_data.get('stage') == constants.VALIDATION_STAGE:
                if status_data.get('status') == 'failed':
                    error_msg = "Cannot run AutoML: validation stage failed"
                    validation_errors = status_data.get('errors', [])
                    if validation_errors:
                        error_msg += f" - Reasons: {'; '.join(validation_errors[:3])}"

                    log.error(error_msg)
                    logger.log_structured_error(
                        structured_log,
                        "validation_failed_prerequisite",
                        error_msg,
                        {"stage": constants.AUTOML_STAGE, "validation_errors": validation_errors}
                    )
                    _update_status_failed(run_id, error_msg)
                    return False
        except Exception as e:
            log.warning(f"Could not check validation status: {e}")
            # Continue execution - don't fail on status check errors

        if not validate_automl_stage_inputs_gcs(run_id, gcs_bucket_name):
            log.error("AutoML stage input validation failed")
            logger.log_structured_error(
                structured_log,
                "input_validation_failed",
                "AutoML stage input validation failed",
                {"stage": constants.AUTOML_STAGE}
            )
            _update_status_failed(run_id, "Input validation failed")
            return False

        log.info("AutoML stage inputs validated successfully (GCS)")

        # Structured log: Validation passed
        logger.log_structured_event(
            structured_log,
            "input_validation_passed",
            {"stage": constants.AUTOML_STAGE, "storage_type": "gcs"},
            "AutoML stage input validation passed (GCS)"
        )

        # =============================
        # 2. LOAD METADATA AND DATA FROM GCS
        # =============================
        log.info("Loading metadata and cleaned data from GCS...")

        try:
            # Load metadata
            metadata_dict = storage.read_json(run_id, constants.METADATA_FILENAME)
            target_info_dict = metadata_dict['target_info']
            target_info = schemas.TargetInfo(**target_info_dict)

            log.info(f"Target info loaded: column='{target_info.name}', task_type='{target_info.task_type}'")

            # Load cleaned data from GCS
            if not check_run_file_exists(run_id, constants.CLEANED_DATA_FILE):
                raise Exception(f"Cleaned data file not found in GCS: {constants.CLEANED_DATA_FILE}")

            # Download cleaned data from GCS
            cleaned_data_bytes = download_run_file(run_id, constants.CLEANED_DATA_FILE)
            if cleaned_data_bytes is None:
                raise Exception("Failed to download cleaned data from GCS")

            # Load CSV from bytes
            df_ml_ready = pd.read_csv(io.BytesIO(cleaned_data_bytes))
            if df_ml_ready is None or df_ml_ready.empty:
                raise Exception("Cleaned data is empty or could not be loaded")

            log.info(f"Cleaned data loaded from GCS: shape={df_ml_ready.shape}")

            # Structured log: Data loaded
            logger.log_structured_event(
                structured_log,
                "data_loaded",
                {
                    "target_column": target_info.name,
                    "task_type": target_info.task_type,
                    "data_shape": {"rows": df_ml_ready.shape[0], "columns": df_ml_ready.shape[1]},
                    "source": "gcs"
                },
                f"Training data loaded from GCS: {df_ml_ready.shape}"
            )

        except Exception as e:
            log.error(f"Failed to load inputs from GCS: {e}")
            logger.log_structured_error(
                structured_log,
                "data_loading_failed",
                f"Failed to load inputs from GCS: {str(e)}",
                {"stage": constants.AUTOML_STAGE}
            )
            _update_status_failed(run_id, f"Failed to load inputs from GCS: {str(e)}")
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
                "task_type": target_info.task_type,
                "storage_type": "gcs"
            },
            "AutoML training started with PyCaret (GCS-based)"
        )

        try:
            # Run PyCaret AutoML with GCS support
            best_model, metrics, model_name = pycaret_logic.run_pycaret_experiment_gcs(
                df_ml_ready=df_ml_ready,
                target_column_name=target_info.name,
                task_type=target_info.task_type,
                run_id=run_id,
                gcs_bucket_name=gcs_bucket_name,
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
                    "tool": "PyCaret",
                    "storage_type": "gcs"
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
        # 4. VERIFY MODEL SAVED TO GCS
        # =============================
        log.info("Verifying trained model saved to GCS...")

        try:
            # Check if model was saved to GCS by the pycaret_experiment function
            model_filename = "pycaret_pipeline.pkl"
            model_gcs_path = f"{constants.MODEL_DIR}/{model_filename}"

            if check_run_file_exists(run_id, model_gcs_path):
                log.info(f"Model verified saved to GCS: {model_gcs_path}")

                # Structured log: Model saved
                logger.log_structured_event(
                    structured_log,
                    "model_saved",
                    {
                        "model_gcs_path": model_gcs_path,
                        "model_name": model_name,
                        "storage_type": "gcs"
                    },
                    f"Model saved to GCS: {model_gcs_path}"
                )
            else:
                raise FileNotFoundError(f"Model was not saved to GCS at expected location: {model_gcs_path}")

        except Exception as e:
            log.error(f"Failed to verify model in GCS: {e}")
            logger.log_structured_error(
                structured_log,
                "model_save_failed",
                f"Failed to verify model in GCS: {str(e)}",
                {"stage": constants.AUTOML_STAGE}
            )
            _update_status_failed(run_id, f"Failed to verify model in GCS: {str(e)}")
            return False

        # =============================
        # 5. UPDATE METADATA WITH AUTOML INFO
        # =============================
        log.info("Updating metadata with AutoML results...")

        try:
            # Create automl_info dictionary with GCS paths
            automl_info = {
                'tool_used': 'PyCaret',
                'best_model_name': model_name,
                'pycaret_pipeline_gcs_path': model_gcs_path,  # GCS path instead of local path
                'performance_metrics': metrics,
                'automl_completed_at': datetime.now(timezone.utc).isoformat(),
                'target_column': target_info.name,
                'task_type': target_info.task_type,
                'dataset_shape_for_training': list(df_ml_ready.shape),
                'storage_type': 'gcs',
                'gcs_bucket': gcs_bucket_name
            }

            # =============================
            # LOAD MODEL COMPARISON RESULTS
            # =============================
            try:
                log.info("Loading model comparison results...")
                
                # Check if model comparison results exist in GCS
                if check_run_file_exists(run_id, 'model_comparison_results.json'):
                    # Download model comparison results from GCS
                    comparison_bytes = download_run_file(run_id, 'model_comparison_results.json')
                    if comparison_bytes:
                        comparison_data = json.loads(comparison_bytes.decode('utf-8'))
                        
                        # Add model comparison results to automl_info
                        automl_info['model_comparison_results'] = comparison_data
                        automl_info['total_models_compared'] = comparison_data.get('total_models_compared', 0)
                        automl_info['model_comparison_available'] = True
                        
                        log.info(f"Added model comparison results for {comparison_data.get('total_models_compared', 0)} models")
                        
                        # Extract summary statistics for quick access
                        all_models = comparison_data.get('all_model_results', [])
                        if all_models:
                            # Get the best model metrics (first in list since PyCaret sorts by performance)
                            best_model_metrics = all_models[0].get('metrics', {})
                            automl_info['best_model_detailed_metrics'] = best_model_metrics
                            
                            # Create a summary of all models for quick reference
                            model_summary = []
                            for model_result in all_models[:5]:  # Top 5 models
                                summary_item = {
                                    'model_name': model_result.get('model_name', 'Unknown'),
                                    'rank': model_result.get('rank', 0)
                                }
                                
                                # Add key metrics based on task type
                                model_metrics = model_result.get('metrics', {})
                                if target_info.task_type == 'classification':
                                    # Classification metrics
                                    for metric in ['AUC', 'Accuracy', 'F1', 'Precision', 'Recall']:
                                        if metric in model_metrics:
                                            summary_item[metric] = model_metrics[metric]
                                else:
                                    # Regression metrics
                                    for metric in ['R2', 'RMSE', 'MAE', 'MAPE']:
                                        if metric in model_metrics:
                                            summary_item[metric] = model_metrics[metric]
                                
                                model_summary.append(summary_item)
                            
                            automl_info['top_models_summary'] = model_summary
                            log.info(f"Created summary for top {len(model_summary)} models")
                    else:
                        log.warning("Could not download model comparison results from GCS")
                        automl_info['model_comparison_available'] = False
                else:
                    log.warning("Model comparison results file not found in GCS")
                    automl_info['model_comparison_available'] = False
                    
            except Exception as comparison_error:
                log.error(f"Failed to load model comparison results: {comparison_error}")
                automl_info['model_comparison_available'] = False
                # Don't fail the entire process

            # =============================
            # EXTRACT CLASS LABELS (FOR CLASSIFICATION)
            # =============================
            if target_info.task_type == "classification":
                try:
                    log.info("Extracting class labels for classification task...")

                    # Download the trained model from GCS to extract class labels
                    model_bytes = download_run_file(run_id, model_gcs_path)
                    if model_bytes is None:
                        raise Exception("Could not download model from GCS for class label extraction")

                    # Load model from bytes
                    import joblib
                    with tempfile.NamedTemporaryFile() as tmp_file:
                        tmp_file.write(model_bytes)
                        tmp_file.flush()
                        trained_pipeline = joblib.load(tmp_file.name)

                    # Extract class labels from the model
                    class_labels = None

                    # Try different ways to extract class labels
                    if hasattr(trained_pipeline, 'classes_'):
                        # Sklearn-like models have classes_ attribute
                        class_labels = trained_pipeline.classes_.tolist()
                        log.info(f"Found classes_ attribute: {class_labels}")
                    elif hasattr(trained_pipeline, 'named_steps'):
                        # Pipeline with named steps - look for the final estimator
                        for step_name, step_model in trained_pipeline.named_steps.items():
                            if hasattr(step_model, 'classes_'):
                                class_labels = step_model.classes_.tolist()
                                log.info(f"Found classes_ in step '{step_name}': {class_labels}")
                                break
                    elif hasattr(trained_pipeline, '_final_estimator'):
                        # Another way to access final estimator
                        if hasattr(trained_pipeline._final_estimator, 'classes_'):
                            class_labels = trained_pipeline._final_estimator.classes_.tolist()
                            log.info(f"Found classes_ in _final_estimator: {class_labels}")

                    # If still no class labels found, try to infer from target column
                    if class_labels is None:
                        log.warning("Could not extract class labels from model, inferring from target column...")
                        unique_targets = sorted(df_ml_ready[target_info.name].dropna().unique())
                        class_labels = [str(label) for label in unique_targets]
                        log.info(f"Inferred class labels from target column: {class_labels}")

                    if class_labels is not None:
                        # Ensure class labels are JSON serializable
                        class_labels = [str(label) for label in class_labels]

                        # Add to automl_info
                        automl_info['class_labels'] = class_labels
                        log.info(f"Added {len(class_labels)} class labels to automl_info")

                        # Save class labels to separate JSON file in GCS for quick access
                        class_labels_data = {
                            'class_labels': class_labels,
                            'task_type': target_info.task_type,
                            'target_column': target_info.name,
                            'extracted_at': datetime.now(timezone.utc).isoformat(),
                            'extraction_method': 'model_classes_attribute',
                            'storage_type': 'gcs'
                        }

                        # Upload class labels to GCS
                        class_labels_json = io.BytesIO(
                            str.encode(
                                str(class_labels_data).replace("'", '"')  # Simple JSON conversion
                            )
                        )
                        upload_success = upload_run_file(run_id, 'class_labels.json', class_labels_json)
                        
                        if upload_success:
                            log.info("Class labels saved to GCS: class_labels.json")
                        else:
                            log.warning("Failed to save class labels to GCS")

                        # Structured log: Class labels extracted
                        logger.log_structured_event(
                            structured_log,
                            "class_labels_extracted",
                            {
                                "class_labels": class_labels,
                                "num_classes": len(class_labels),
                                "storage_type": "gcs"
                            },
                            f"Extracted {len(class_labels)} class labels"
                        )
                    else:
                        log.warning("Could not extract class labels from model")
                        automl_info['class_labels'] = None

                except Exception as class_error:
                    log.error(f"Failed to extract class labels: {class_error}")
                    log.error(f"Class label extraction traceback: {traceback.format_exc()}")
                    automl_info['class_labels'] = None
                    # Don't fail the entire process for this error

            else:
                # For regression tasks, class labels don't apply
                log.info("Skipping class label extraction for regression task")
                automl_info['class_labels'] = None

            # Add automl_info to existing metadata
            metadata_dict['automl_info'] = automl_info

            # Save updated metadata
            storage.write_json_atomic(run_id, constants.METADATA_FILENAME, metadata_dict)
            log.info("Metadata updated with AutoML results")

            # Structured log: Metadata updated
            logger.log_structured_event(
                structured_log,
                "metadata_updated",
                {
                    "automl_info_keys": list(automl_info.keys()),
                    "storage_type": "gcs"
                },
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
                "message": f"AutoML completed (GCS-based). Best model: {model_name}"
            }
            storage.write_json_atomic(run_id, constants.STATUS_FILENAME, status_data)
            log.info("Status updated to completed")

            # Structured log: Status updated
            logger.log_structured_event(
                structured_log,
                "status_updated",
                {"status": "completed", "stage": constants.AUTOML_STAGE, "storage_type": "gcs"},
                "AutoML stage status updated to completed (GCS-based)"
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
        log.info("AUTOML STAGE COMPLETED SUCCESSFULLY (GCS)")
        log.info("="*50)
        log.info(f"Dataset shape: {df_ml_ready.shape}")
        log.info(f"Target column: {target_info.name}")
        log.info(f"Task type: {target_info.task_type}")
        log.info(f"Best model: {model_name}")
        log.info(f"Performance metrics:")
        for metric_name, metric_value in metrics.items():
            log.info(f"  - {metric_name}: {metric_value}")
        log.info(f"GCS files created:")
        log.info(f"  - {model_gcs_path}")
        log.info(f"  - {constants.METADATA_FILENAME} (updated)")
        log.info(f"  - {constants.STATUS_FILENAME} (updated)")
        if target_info.task_type == "classification" and automl_info.get('class_labels'):
            log.info(f"  - class_labels.json (classification labels)")
            log.info(f"Class labels: {automl_info['class_labels']}")
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
                "completed_at": end_time.isoformat(),
                "storage_type": "gcs"
            },
            f"AutoML stage completed successfully in {training_duration:.1f}s (GCS-based)"
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


def run_automl_stage(run_id: str, test_mode: bool = False) -> bool:
    """
    Legacy compatibility function - redirects to GCS version.

    Args:
        run_id: Unique run identifier
        test_mode: Flag to allow AutoML to work with very small datasets for testing purposes

    Returns:
        True if stage completes successfully, False otherwise
    """
    logger_instance = logger.get_stage_logger(run_id, constants.AUTOML_STAGE)
    logger_instance.warning("Using legacy run_automl_stage function - redirecting to GCS version")
    return run_automl_stage_gcs(run_id, test_mode)


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


def validate_automl_stage_inputs_gcs(run_id: str, 
                                    gcs_bucket_name: str = PROJECT_BUCKET_NAME) -> bool:
    """
    Validate that all required inputs for the AutoML stage are available in GCS.

    Args:
        run_id: Run identifier
        gcs_bucket_name: GCS bucket name

    Returns:
        True if all inputs are valid, False otherwise
    """
    log = logger.get_stage_logger(run_id, constants.AUTOML_STAGE)

    try:
        # Check if metadata.json exists (this might still be in local storage)
        try:
            metadata_dict = storage.read_json(run_id, constants.METADATA_FILENAME)
        except Exception as e:
            log.error(f"Metadata file does not exist or cannot be read: {e}")
            return False

        # Check if cleaned_data.csv exists in GCS
        if not check_run_file_exists(run_id, constants.CLEANED_DATA_FILE):
            log.error(f"Cleaned data file does not exist in GCS: {constants.CLEANED_DATA_FILE}")
            return False

        # Try to load and validate metadata structure
        try:
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

            log.info("All AutoML stage inputs validated successfully (GCS)")
            return True

        except Exception as e:
            log.error(f"Metadata validation failed: {e}")
            return False

    except Exception as e:
        log.error(f"Input validation failed: {e}")
        return False


def validate_automl_stage_inputs(run_id: str) -> bool:
    """
    Legacy compatibility function - redirects to GCS version.

    Args:
        run_id: Run identifier

    Returns:
        True if all inputs are valid, False otherwise
    """
    logger_instance = logger.get_stage_logger(run_id, constants.AUTOML_STAGE)
    logger_instance.warning("Using legacy validate_automl_stage_inputs function - redirecting to GCS version")
    return validate_automl_stage_inputs_gcs(run_id)


def get_automl_stage_summary_gcs(run_id: str) -> Optional[dict]:
    """
    Get a summary of the AutoML stage results for a given run (GCS version).

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

        # Check if model file exists in GCS
        model_gcs_path = automl_info.get('pycaret_pipeline_gcs_path', '')
        model_file_exists = check_run_file_exists(run_id, model_gcs_path) if model_gcs_path else False

        return {
            'run_id': run_id,
            'tool_used': automl_info.get('tool_used'),
            'best_model_name': automl_info.get('best_model_name'),
            'task_type': automl_info.get('task_type'),
            'target_column': automl_info.get('target_column'),
            'performance_metrics': automl_info.get('performance_metrics', {}),
            'training_dataset_shape': automl_info.get('dataset_shape_for_training'),
            'model_file_exists': model_file_exists,
            'completed_at': automl_info.get('automl_completed_at'),
            'storage_type': automl_info.get('storage_type', 'unknown'),
            'gcs_bucket': automl_info.get('gcs_bucket', 'unknown')
        }

    except Exception:
        return None


def get_automl_stage_summary(run_id: str) -> Optional[dict]:
    """
    Legacy compatibility function - redirects to GCS version.

    Args:
        run_id: Run identifier

    Returns:
        Dictionary with AutoML stage summary or None if not available
    """
    return get_automl_stage_summary_gcs(run_id)
