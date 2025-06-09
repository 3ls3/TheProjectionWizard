"""
Pipeline API routes for The Projection Wizard.
Provides endpoints for the main ML pipeline functionality.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from uuid import uuid4
import pandas as pd
from datetime import datetime, timezone
from typing import List
import tempfile
from pathlib import Path

from common import storage, logger, constants
from .schema import UploadResponse

router = APIRouter(prefix="/api")


def _new_run_id() -> str:
    """Generate a new unique run ID."""
    return uuid4().hex[:8]


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
