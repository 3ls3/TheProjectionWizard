"""
Shared error codes and error handling utilities for The Projection Wizard API.
Provides consistent error responses across all endpoints.
"""

from fastapi import HTTPException
from typing import Dict, Any, Optional


# Error codes for consistent API responses
class ErrorCodes:
    """Standard error codes used across the API."""

    # General errors
    INTERNAL_SERVER_ERROR = "internal_server_error"
    INVALID_REQUEST = "invalid_request"

    # Run-related errors
    RUN_NOT_FOUND = "run_not_found"
    RUN_ALREADY_EXISTS = "run_already_exists"

    # File upload errors
    INVALID_CSV = "invalid_csv"
    FILE_TOO_LARGE = "file_too_large"
    FILE_PARSE_ERROR = "file_parse_error"
    EMPTY_FILE = "empty_file"

    # Pipeline state errors
    TARGET_NOT_CONFIRMED = "target_not_confirmed"
    FEATURES_ALREADY_CONFIRMED = "features_already_confirmed"
    PIPELINE_FAILED = "pipeline_failed"
    PIPELINE_STILL_RUNNING = "pipeline_still_running"

    # Data errors
    TARGET_INFERENCE_FAILED = "target_inference_failed"
    SCHEMA_CONFIRMATION_FAILED = "schema_confirmation_failed"

    # Status/Results errors
    STATUS_NOT_AVAILABLE = "status_not_available"
    RESULTS_NOT_AVAILABLE = "results_not_available"
    MISSING_METADATA = "missing_metadata"


def create_error_detail(
    message: str,
    code: str,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a standardized error detail dictionary.

    Args:
        message: Human-readable error message
        code: Machine-readable error code
        context: Optional additional context information

    Returns:
        Standardized error detail dictionary
    """
    detail = {
        "detail": message,
        "code": code
    }

    if context:
        detail["context"] = context

    return detail


def raise_run_not_found(run_id: str) -> None:
    """Raise a standardized 404 error for missing runs."""
    raise HTTPException(
        status_code=404,
        detail=create_error_detail(
            message=f"Run '{run_id}' not found",
            code=ErrorCodes.RUN_NOT_FOUND,
            context={"run_id": run_id}
        )
    )


def raise_invalid_csv(
    message: str,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """Raise a standardized 400 error for invalid CSV files."""
    raise HTTPException(
        status_code=400,
        detail=create_error_detail(
            message=f"Invalid CSV file: {message}",
            code=ErrorCodes.INVALID_CSV,
            context=context
        )
    )


def raise_target_not_confirmed(run_id: str) -> None:
    """Raise a standardized 400 error when target hasn't been confirmed."""
    raise HTTPException(
        status_code=400,
        detail=create_error_detail(
            message="Target must be confirmed before proceeding",
            code=ErrorCodes.TARGET_NOT_CONFIRMED,
            context={"run_id": run_id}
        )
    )


def raise_features_already_confirmed(run_id: str) -> None:
    """Raise a standardized 400 error when features are already confirmed."""
    raise HTTPException(
        status_code=400,
        detail=create_error_detail(
            message="Features have already been confirmed for this run",
            code=ErrorCodes.FEATURES_ALREADY_CONFIRMED,
            context={"run_id": run_id}
        )
    )


def raise_pipeline_failed(
    run_id: str,
    message: str = "Pipeline execution failed"
) -> None:
    """Raise a standardized 400 error for pipeline failures."""
    raise HTTPException(
        status_code=400,
        detail=create_error_detail(
            message=message,
            code=ErrorCodes.PIPELINE_FAILED,
            context={"run_id": run_id}
        )
    )


def raise_pipeline_still_running(run_id: str) -> None:
    """Raise a standardized 202 error when pipeline is still running."""
    raise HTTPException(
        status_code=202,
        detail=create_error_detail(
            message="Pipeline still running - results not yet available",
            code=ErrorCodes.PIPELINE_STILL_RUNNING,
            context={"run_id": run_id}
        )
    )


def raise_status_not_available(run_id: str) -> None:
    """Raise a standardized 404 error when status is not available."""
    raise HTTPException(
        status_code=404,
        detail=create_error_detail(
            message=f"Status information not available for run '{run_id}'",
            code=ErrorCodes.STATUS_NOT_AVAILABLE,
            context={"run_id": run_id}
        )
    )


def raise_results_not_available(
    run_id: str,
    message: str = "Results not available"
) -> None:
    """Raise a standardized 500 error when results are not available."""
    raise HTTPException(
        status_code=500,
        detail=create_error_detail(
            message=message,
            code=ErrorCodes.RESULTS_NOT_AVAILABLE,
            context={"run_id": run_id}
        )
    )


def raise_internal_server_error(
    message: str,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """Raise a standardized 500 error for internal server errors."""
    raise HTTPException(
        status_code=500,
        detail=create_error_detail(
            message=f"Internal server error: {message}",
            code=ErrorCodes.INTERNAL_SERVER_ERROR,
            context=context
        )
    )
