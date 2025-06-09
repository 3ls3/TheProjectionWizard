"""
Pipeline API routes for The Projection Wizard.
Provides endpoints for the main ML pipeline functionality.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from uuid import uuid4
import pandas as pd
from datetime import datetime, timezone
from typing import List
import tempfile
from pathlib import Path

from common import storage, logger, constants
from .schema import (
    UploadResponse,
    TargetSuggestionResponse,
    TargetConfirmationRequest
)
from pipeline.step_2_schema import target_definition_logic as tgt_logic

router = APIRouter(prefix="/api")


def _new_run_id() -> str:
    """Generate a new unique run ID."""
    return uuid4().hex[:8]


def _validate_run_exists(run_id: str) -> bool:
    """Check if a run exists by checking for original_data.csv."""
    # Don't use get_run_dir as it creates the directory
    from pathlib import Path
    run_dir = Path(constants.DATA_DIR_NAME) / "runs" / run_id
    original_data_path = run_dir / constants.ORIGINAL_DATA_FILE
    return original_data_path.exists()


def _write_original_data(run_id: str, df: pd.DataFrame) -> None:
    """Write original data CSV file for a run."""
    run_dir = storage.get_run_dir(run_id)
    csv_path = run_dir / constants.ORIGINAL_DATA_FILE
    df.to_csv(csv_path, index=False)


def _create_preview(df: pd.DataFrame, num_rows: int = 5) -> List[List[str]]:
    """Create a preview of the first N rows as list of lists of strings."""
    # Get first N rows
    preview_df = df.head(num_rows)

    # Convert to list of lists, including header
    preview = []

    # Add header row
    preview.append(list(df.columns.astype(str)))

    # Add data rows, converting all values to strings
    for _, row in preview_df.iterrows():
        preview.append([str(val) for val in row.values])

    return preview


@router.post("/upload", response_model=UploadResponse, status_code=201)
async def upload_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file and initialize a new ML pipeline run.

    Creates a new run directory, saves the original data, and returns
    basic information about the uploaded dataset.

    Args:
        file: The CSV file to upload

    Returns:
        UploadResponse with run_id, shape, and data preview

    Raises:
        HTTPException: If file validation fails or processing errors occur
    """

    # 1. Validate content-type (must contain "csv")
    if not file.content_type or "csv" not in file.content_type.lower():
        if not file.filename or not file.filename.lower().endswith('.csv'):
            raise HTTPException(
                status_code=400,
                detail="File must be a CSV file (content-type or filename "
                       "must indicate CSV)"
            )

    # Generate new run ID
    run_id = _new_run_id()

    # Get logger for this operation
    upload_logger = logger.get_logger(run_id, "api_upload")

    try:
        # 2. Read file content into pandas DataFrame
        # Use temporary file to avoid memory issues with large files
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
            # Read file content
            content = await file.read()
            tmp.write(content)
            tmp.flush()

            # Read into pandas
            try:
                df = pd.read_csv(tmp.name)
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to parse CSV file: {str(e)}"
                )
            finally:
                # Clean up temp file
                Path(tmp.name).unlink(missing_ok=True)

        # Validate data shape
        if df.empty:
            raise HTTPException(
                status_code=400,
                detail="CSV file is empty or contains no valid data"
            )

        # 3. Generate run_id and create directory (already done above)
        # 4. Persist original CSV
        _write_original_data(run_id, df)

        # 5. Seed minimal metadata.json
        metadata = {
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "original_filename": file.filename or "uploaded.csv",
            "initial_rows": len(df),
            "initial_cols": len(df.columns),
            "initial_dtypes": {col: str(dtype)
                               for col, dtype in df.dtypes.items()}
        }

        storage.write_metadata(run_id, metadata)

        # Add entry to run index
        index_entry = {
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc),
            "original_filename": file.filename or "uploaded.csv",
            "status": "uploaded"
        }
        storage.append_to_run_index(index_entry)

        # 6. Return UploadResponse(run_id, shape, preview)
        shape = (len(df), len(df.columns))
        preview = _create_preview(df)

        # Log successful upload
        upload_logger.info(
            f"Successfully uploaded file '{file.filename}' as run {run_id}"
        )
        upload_logger.info(
            f"Dataset shape: {shape[0]} rows, {shape[1]} columns"
        )

        return UploadResponse(
            run_id=run_id,
            shape=shape,
            preview=preview
        )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Log unexpected errors
        upload_logger.error(f"Unexpected error during file upload: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during file upload: {str(e)}"
        )


@router.get("/target-suggestion", response_model=TargetSuggestionResponse)
def target_suggestion(run_id: str = Query(...)):
    """
    Get target column and task type suggestions for a run.

    Analyzes the uploaded dataset to suggest the most likely target column
    and task type (classification vs regression).

    Args:
        run_id: The ID of the run to analyze

    Returns:
        TargetSuggestionResponse with suggested column, task type, and
        confidence

    Raises:
        HTTPException: If run not found or analysis fails
    """

    # Validate run exists
    if not _validate_run_exists(run_id):
        raise HTTPException(
            status_code=404,
            detail=f"Run '{run_id}' not found"
        )

    # Get logger for this operation
    suggestion_logger = logger.get_logger(run_id, "api_target_suggestion")

    try:
        # Load original data
        df = storage.read_original_data(run_id)
        if df is None:
            raise HTTPException(
                status_code=404,
                detail=f"Original data not found for run '{run_id}'"
            )

        # Call pipeline logic to suggest target and task type
        suggested_col, suggested_task, suggested_ml_type = (
            tgt_logic.suggest_target_and_task(df)
        )

        if suggested_col is None or suggested_task is None:
            raise HTTPException(
                status_code=400,
                detail="Could not infer target column from the data"
            )

        # Log the suggestion
        suggestion_logger.info(
            f"Suggested target: '{suggested_col}', "
            f"task type: '{suggested_task}'"
        )

        # For now, set a static confidence score
        # In the future, this could be calculated based on heuristics
        confidence = 0.87

        return TargetSuggestionResponse(
            suggested_column=suggested_col,
            task_type=suggested_task,
            confidence=confidence
        )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Log unexpected errors
        suggestion_logger.error(
            f"Unexpected error during target suggestion: {str(e)}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during target analysis: {str(e)}"
        )


@router.post("/confirm-target", status_code=200)
def confirm_target(req: TargetConfirmationRequest):
    """
    Confirm the target column and task type for a run.

    Updates the run metadata with the confirmed target information,
    enabling the next stages of the pipeline.

    Args:
        req: Request containing run_id, confirmed column, and task type

    Returns:
        Success status message

    Raises:
        HTTPException: If run not found or update fails
    """

    # Validate run exists
    if not _validate_run_exists(req.run_id):
        raise HTTPException(
            status_code=404,
            detail=f"Run '{req.run_id}' not found"
        )

    # Get logger for this operation
    confirm_logger = logger.get_logger(req.run_id, "api_target_confirmation")

    try:
        # Load existing metadata.json
        metadata = storage.read_metadata(req.run_id)
        if metadata is None:
            # Create minimal metadata if it doesn't exist
            metadata = {
                "run_id": req.run_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        # Update metadata with target information
        metadata['target_info'] = {
            "name": req.confirmed_column,
            "task_type": req.task_type,
            "user_confirmed_at": datetime.now(timezone.utc).isoformat()
        }

        # Also set top-level task_type for convenience
        metadata['task_type'] = req.task_type

        # Persist updated metadata
        storage.write_metadata(req.run_id, metadata)

        # Log successful confirmation
        confirm_logger.info(
            f"Target confirmed: '{req.confirmed_column}', "
            f"task type: '{req.task_type}'"
        )

        return {"status": "ok"}

    except Exception as e:
        # Log unexpected errors
        confirm_logger.error(
            f"Unexpected error during target confirmation: {str(e)}"
        )
        raise HTTPException(
            status_code=500,
            detail=(
                f"Internal server error during target confirmation: {str(e)}"
            )
        )
