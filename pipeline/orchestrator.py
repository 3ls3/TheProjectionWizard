"""
Pipeline orchestrator for The Projection Wizard.
Provides a simple interface for running multiple pipeline stages sequentially.
Refactored for GCS-based storage with comprehensive status management.
"""

import json
from datetime import datetime, timezone
from io import BytesIO

from common import logger
from api.utils.gcs_utils import (
    PROJECT_BUCKET_NAME, download_run_file, upload_run_file, GCSError
)


def _update_orchestrator_status_gcs(run_id: str, 
                                   gcs_bucket_name: str,
                                   orchestrator_status: str,
                                   message: str,
                                   current_step: str = None) -> bool:
    """
    Update the orchestrator-level status in status.json in GCS.
    
    Args:
        run_id: The ID of the run
        gcs_bucket_name: GCS bucket name
        orchestrator_status: Status of orchestrator ('processing', 'completed', 'failed')
        message: Status message
        current_step: Current pipeline step being processed
        
    Returns:
        True if status update successful, False otherwise
    """
    try:
        # Download current status.json
        status_bytes = download_run_file(run_id, "status.json")
        if not status_bytes:
            logger.error(f"Could not download status.json for orchestrator update: {run_id}")
            return False
        
        # Parse current status
        current_status = json.loads(status_bytes.decode('utf-8'))
        
        # Update orchestrator-specific fields
        pipeline_execution = {
            "status": orchestrator_status,
            "message": message,
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
        
        if current_step:
            pipeline_execution["current_orchestrator_step"] = current_step
            
        current_status["pipeline_execution"] = pipeline_execution
        current_status["last_updated"] = datetime.now(timezone.utc).isoformat()
        
        # If orchestrator completed successfully, update top-level status
        if orchestrator_status == "completed":
            current_status.update({
                "stage": "completed",
                "status": "completed", 
                "message": "All pipeline stages completed successfully",
                "progress_pct": 100
            })
        elif orchestrator_status == "failed":
            current_status.update({
                "status": "failed",
                "message": f"Pipeline failed at {current_step}" if current_step else "Pipeline orchestration failed"
            })
        
        # Upload updated status back to GCS
        status_json = json.dumps(current_status, indent=2).encode('utf-8')
        status_io = BytesIO(status_json)
        
        success = upload_run_file(run_id, "status.json", status_io)
        if success:
            logger.info(f"Updated orchestrator status to '{orchestrator_status}' for run {run_id}")
        return success
        
    except Exception as e:
        logger.error(f"Failed to update orchestrator status for run {run_id}: {str(e)}")
        return False


def run_from_schema_confirm_gcs(run_id: str, 
                               gcs_bucket_name: str = PROJECT_BUCKET_NAME) -> bool:
    """
    Run steps 3-7 synchronously after feature schemas are confirmed using GCS storage.
    
    This function orchestrates the execution of the main ML pipeline stages, managing
    their sequential execution, status updates, and error handling in a GCS environment.

    Args:
        run_id: The ID of the run to process
        gcs_bucket_name: GCS bucket name for storage

    Returns:
        True if all stages completed successfully, False if any stage failed
    """
    # Get logger for orchestration
    orchestrator_logger = logger.get_structured_logger(run_id, "pipeline_orchestrator")

    # Log orchestration start
    logger.log_structured_event(
        orchestrator_logger,
        "orchestrator_started",
        {
            "run_id": run_id,
            "gcs_bucket": gcs_bucket_name,
            "stages": ["validation", "prep", "automl", "explain"]
        },
        f"Starting GCS-based pipeline orchestration for run {run_id}"
    )

    # Update status to indicate orchestration has started
    if not _update_orchestrator_status_gcs(
        run_id, gcs_bucket_name, "processing", 
        "Automated pipeline execution started", "step_3_validation"
    ):
        logger.log_structured_error(
            orchestrator_logger,
            "orchestrator_status_update_failed",
            "Failed to update initial orchestrator status",
            {"run_id": run_id, "stage": "initialization"}
        )

    try:
        # Import stage runners - verify these are the correct GCS-aware function names
        from pipeline.step_3_validation import validation_runner
        from pipeline.step_4_prep import prep_runner  
        from pipeline.step_5_automl import automl_runner
        from pipeline.step_6_explain import explain_runner

        # Stage 3: Validation
        logger.log_structured_event(
            orchestrator_logger,
            "stage_started",
            {"stage": "step_3_validation", "stage_name": "Validation"},
            "Starting Stage 3: Validation (GCS)"
        )
        
        _update_orchestrator_status_gcs(
            run_id, gcs_bucket_name, "processing",
            "Running validation stage", "step_3_validation"
        )
        
        if not validation_runner.run_gcs_validation_stage(run_id, gcs_bucket_name):
            logger.log_structured_error(
                orchestrator_logger,
                "stage_failed", 
                "Stage 3 (validation) failed",
                {"stage": "step_3_validation", "run_id": run_id}
            )
            _update_orchestrator_status_gcs(
                run_id, gcs_bucket_name, "failed",
                "Pipeline failed at validation stage", "step_3_validation"
            )
            return False
            
        logger.log_structured_event(
            orchestrator_logger,
            "stage_completed",
            {"stage": "step_3_validation", "stage_name": "Validation"},
            "Stage 3 (validation) completed successfully"
        )

        # Stage 4: Data Preparation
        logger.log_structured_event(
            orchestrator_logger,
            "stage_started", 
            {"stage": "step_4_prep", "stage_name": "Data Preparation"},
            "Starting Stage 4: Data Preparation (GCS)"
        )
        
        _update_orchestrator_status_gcs(
            run_id, gcs_bucket_name, "processing",
            "Running data preparation stage", "step_4_prep"
        )
        
        if not prep_runner.run_preparation_stage_gcs(run_id, gcs_bucket_name):
            logger.log_structured_error(
                orchestrator_logger,
                "stage_failed",
                "Stage 4 (preparation) failed", 
                {"stage": "step_4_prep", "run_id": run_id}
            )
            _update_orchestrator_status_gcs(
                run_id, gcs_bucket_name, "failed",
                "Pipeline failed at data preparation stage", "step_4_prep"
            )
            return False
            
        logger.log_structured_event(
            orchestrator_logger,
            "stage_completed",
            {"stage": "step_4_prep", "stage_name": "Data Preparation"},
            "Stage 4 (preparation) completed successfully"
        )

        # Stage 5: AutoML
        logger.log_structured_event(
            orchestrator_logger,
            "stage_started",
            {"stage": "step_5_automl", "stage_name": "AutoML"},
            "Starting Stage 5: AutoML (GCS)"
        )
        
        _update_orchestrator_status_gcs(
            run_id, gcs_bucket_name, "processing",
            "Running AutoML stage", "step_5_automl"
        )
        
        if not automl_runner.run_automl_stage_gcs(run_id, gcs_bucket_name=gcs_bucket_name):
            logger.log_structured_error(
                orchestrator_logger,
                "stage_failed",
                "Stage 5 (automl) failed",
                {"stage": "step_5_automl", "run_id": run_id}
            )
            _update_orchestrator_status_gcs(
                run_id, gcs_bucket_name, "failed", 
                "Pipeline failed at AutoML stage", "step_5_automl"
            )
            return False
            
        logger.log_structured_event(
            orchestrator_logger,
            "stage_completed",
            {"stage": "step_5_automl", "stage_name": "AutoML"},
            "Stage 5 (automl) completed successfully"
        )

        # Stage 6: Explainability
        logger.log_structured_event(
            orchestrator_logger,
            "stage_started",
            {"stage": "step_6_explain", "stage_name": "Explainability"},
            "Starting Stage 6: Explainability (GCS)"
        )
        
        _update_orchestrator_status_gcs(
            run_id, gcs_bucket_name, "processing",
            "Running explainability stage", "step_6_explain"
        )
        
        if not explain_runner.run_explainability_stage_gcs(run_id, gcs_bucket_name):
            logger.log_structured_error(
                orchestrator_logger,
                "stage_failed",
                "Stage 6 (explain) failed",
                {"stage": "step_6_explain", "run_id": run_id}
            )
            _update_orchestrator_status_gcs(
                run_id, gcs_bucket_name, "failed",
                "Pipeline failed at explainability stage", "step_6_explain"
            )
            return False
            
        logger.log_structured_event(
            orchestrator_logger,
            "stage_completed",
            {"stage": "step_6_explain", "stage_name": "Explainability"},
            "Stage 6 (explain) completed successfully"
        )

        # All stages completed successfully - update final status
        _update_orchestrator_status_gcs(
            run_id, gcs_bucket_name, "completed",
            "All automated pipeline stages finished successfully"
        )
        
        logger.log_structured_event(
            orchestrator_logger,
            "orchestrator_completed",
            {
                "run_id": run_id,
                "gcs_bucket": gcs_bucket_name,
                "stages_completed": ["validation", "prep", "automl", "explain"],
                "total_stages": 4
            },
            f"All pipeline stages completed successfully for run {run_id} (GCS)"
        )
        
        return True

    except ImportError as e:
        error_msg = f"Failed to import pipeline stage modules: {str(e)}"
        logger.log_structured_error(
            orchestrator_logger,
            "orchestrator_import_error",
            error_msg,
            {"run_id": run_id, "error": str(e), "error_type": "ImportError"}
        )
        _update_orchestrator_status_gcs(
            run_id, gcs_bucket_name, "failed",
            f"Pipeline import error: {str(e)}", "import_stage"
        )
        return False
        
    except Exception as e:
        error_msg = f"Unexpected error in pipeline orchestrator (GCS): {str(e)}"
        logger.log_structured_error(
            orchestrator_logger,
            "orchestrator_unexpected_error", 
            error_msg,
            {
                "run_id": run_id,
                "error": str(e),
                "error_type": type(e).__name__,
                "gcs_bucket": gcs_bucket_name
            }
        )
        _update_orchestrator_status_gcs(
            run_id, gcs_bucket_name, "failed",
            f"Unexpected orchestrator error: {str(e)}", "unknown_stage"
        )
        return False


def run_from_schema_confirm(run_id: str) -> bool:
    """
    Legacy compatibility function - redirects to GCS version.
    Run steps 3-7 synchronously after feature schemas are confirmed.

    Args:
        run_id: The ID of the run to process

    Returns:
        True if all stages completed successfully, False if any stage failed
    """
    # Get logger for orchestration
    orchestrator_logger = logger.get_structured_logger(run_id, "pipeline_orchestrator_legacy")
    
    logger.log_structured_event(
        orchestrator_logger,
        "legacy_function_called",
        {"run_id": run_id, "redirecting_to": "run_from_schema_confirm_gcs"},
        "Using legacy run_from_schema_confirm function - redirecting to GCS version"
    )
    
    return run_from_schema_confirm_gcs(run_id)
