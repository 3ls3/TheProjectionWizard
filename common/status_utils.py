"""
Status utilities for The Projection Wizard.
Provides functions to assess the status of pipeline stages.
"""

from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field

from . import storage, constants, schemas


class StageStatusSummary(BaseModel):
    """
    Summary of a pipeline stage's status.
    
    This model provides a standardized way to understand the state of any pipeline stage,
    including whether it has completed successfully, failed, or is still pending.
    """
    status: str = Field(
        ..., 
        description="Current stage status: 'pending', 'running', 'completed_successfully', 'completed_with_warnings', 'failed_critically', 'unknown'"
    )
    message: Optional[str] = Field(
        None, 
        description="Context about the status, such as error messages or success confirmations"
    )
    can_proceed: bool = Field(
        ..., 
        description="Whether the pipeline can logically proceed from this stage to the next"
    )


def get_stage_status_summary(run_id: str, stage_name: str) -> StageStatusSummary:
    """
    Get the status summary for a specific pipeline stage.
    
    This function centralizes all status-checking logic by examining the appropriate
    status files (status.json, validation.json) for a given run and stage.
    
    Args:
        run_id: Unique run identifier
        stage_name: Stage name (e.g., constants.VALIDATION_STAGE, constants.PREP_STAGE)
        
    Returns:
        StageStatusSummary: Structured status information for the stage
        
    Examples:
        >>> summary = get_stage_status_summary("run_123", constants.VALIDATION_STAGE)
        >>> if summary.can_proceed:
        ...     print(f"Validation passed: {summary.message}")
        >>> else:
        ...     print(f"Validation failed: {summary.message}")
    """
    # Initial validation of inputs
    if not run_id or not run_id.strip():
        return StageStatusSummary(
            status="unknown",
            message="Invalid or missing run_id",
            can_proceed=False
        )
    
    if stage_name not in constants.PIPELINE_STAGES:
        return StageStatusSummary(
            status="unknown",
            message=f"Unrecognized stage: {stage_name}",
            can_proceed=False
        )
    
    try:
        # Handle validation stage with special logic
        if stage_name == constants.VALIDATION_STAGE:
            return _get_validation_stage_status(run_id)
        
        # Handle other stages (prep, automl, explain) using status.json
        else:
            return _get_standard_stage_status(run_id, stage_name)
            
    except Exception as e:
        print(f"Error checking status for stage {stage_name} in run {run_id}: {e}")
        return StageStatusSummary(
            status="unknown",
            message=f"Error determining stage status: {str(e)}",
            can_proceed=False
        )


def _get_validation_stage_status(run_id: str) -> StageStatusSummary:
    """
    Get status for the validation stage using validation.json and status.json.
    
    Args:
        run_id: Unique run identifier
        
    Returns:
        StageStatusSummary: Status summary for validation stage
    """
    try:
        # First check status.json for critical failures
        status_data = storage.read_json(run_id, constants.STATUS_FILENAME)
        if status_data:
            current_stage = status_data.get('stage')
            stage_status = status_data.get('status')
            
            # If status.json shows validation failed critically
            if current_stage == constants.VALIDATION_STAGE and stage_status == 'failed':
                return StageStatusSummary(
                    status="failed_critically",
                    message="Validation critically failed according to pipeline status",
                    can_proceed=False
                )
        
        # Check validation.json for detailed results
        validation_data = storage.read_json(run_id, constants.VALIDATION_FILENAME)
        if not validation_data:
            return StageStatusSummary(
                status="pending",
                message="Validation report not found - validation has not been run yet",
                can_proceed=False
            )
        
        # Parse validation results using the schema
        try:
            validation_summary = schemas.ValidationReportSummary(**validation_data)
        except Exception as e:
            print(f"Error parsing validation report for run {run_id}: {e}")
            return StageStatusSummary(
                status="unknown",
                message="Validation report exists but could not be parsed",
                can_proceed=False
            )
        
        # Calculate success rate
        if validation_summary.total_expectations == 0:
            success_rate = 0.0
        else:
            success_rate = validation_summary.successful_expectations / validation_summary.total_expectations
        
        # Determine if we can proceed (overall success OR success rate >= 95%)
        can_proceed = validation_summary.overall_success or success_rate >= 0.95
        
        # Determine status based on results
        if can_proceed:
            if validation_summary.overall_success:
                status = "completed_successfully"
                message = f"Validation passed: {validation_summary.successful_expectations}/{validation_summary.total_expectations} expectations succeeded"
            else:
                status = "completed_with_warnings"
                message = f"Validation passed with warnings: {success_rate:.1%} success rate ({validation_summary.successful_expectations}/{validation_summary.total_expectations} expectations)"
        else:
            status = "failed_critically"
            message = f"Validation failed: Only {success_rate:.1%} success rate ({validation_summary.successful_expectations}/{validation_summary.total_expectations} expectations)"
        
        return StageStatusSummary(
            status=status,
            message=message,
            can_proceed=can_proceed
        )
        
    except IOError as e:
        print(f"I/O error reading validation files for run {run_id}: {e}")
        return StageStatusSummary(
            status="unknown",
            message="Error reading validation files",
            can_proceed=False
        )


def _get_standard_stage_status(run_id: str, stage_name: str) -> StageStatusSummary:
    """
    Get status for standard stages (prep, automl, explain) using status.json.
    
    Args:
        run_id: Unique run identifier
        stage_name: Name of the stage to check
        
    Returns:
        StageStatusSummary: Status summary for the stage
    """
    try:
        status_data = storage.read_json(run_id, constants.STATUS_FILENAME)
        if not status_data:
            return StageStatusSummary(
                status="pending",
                message="Stage not yet started - no status file found",
                can_proceed=False
            )
        
        current_stage = status_data.get('stage')
        stage_status = status_data.get('status')
        
        if not current_stage or not stage_status:
            return StageStatusSummary(
                status="unknown",
                message="Status file exists but is missing required fields",
                can_proceed=False
            )
        
        # If this is the current stage and it's completed
        if current_stage == stage_name and stage_status == 'completed':
            return StageStatusSummary(
                status="completed_successfully",
                message="Stage completed successfully",
                can_proceed=True
            )
        
        # If this is the current stage and it failed
        if current_stage == stage_name and stage_status == 'failed':
            error_message = status_data.get('message', 'Stage failed')
            return StageStatusSummary(
                status="failed_critically",
                message=f"Stage failed: {error_message}",
                can_proceed=False
            )
        
        # If this is the current stage and it's running
        if current_stage == stage_name and stage_status in ['running', 'in_progress']:
            return StageStatusSummary(
                status="running",
                message="Stage is currently running",
                can_proceed=False
            )
        
        # Get stage order for comparison
        try:
            current_stage_index = constants.PIPELINE_STAGES.index(current_stage)
            target_stage_index = constants.PIPELINE_STAGES.index(stage_name)
        except ValueError:
            return StageStatusSummary(
                status="unknown",
                message="Unknown stage in pipeline sequence",
                can_proceed=False
            )
        
        # If the pipeline hasn't reached this stage yet
        if current_stage_index < target_stage_index:
            return StageStatusSummary(
                status="pending",
                message="Stage not yet started - pipeline has not reached this stage",
                can_proceed=False
            )
        
        # If the pipeline has moved past this stage, assume it was completed
        if current_stage_index > target_stage_index:
            return StageStatusSummary(
                status="completed_successfully",
                message="Stage was completed in a previous pipeline step",
                can_proceed=True
            )
        
        # Fallback case
        return StageStatusSummary(
            status="unknown",
            message="Could not determine stage status from available information",
            can_proceed=False
        )
        
    except IOError as e:
        print(f"I/O error reading status file for run {run_id}: {e}")
        return StageStatusSummary(
            status="unknown", 
            message="Error reading status file",
            can_proceed=False
        ) 