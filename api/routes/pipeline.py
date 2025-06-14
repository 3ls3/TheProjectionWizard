"""
Pipeline API routes for The Projection Wizard.
Provides endpoints for the main ML pipeline functionality.
"""

import asyncio
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
from fastapi import APIRouter, UploadFile, File, Query, HTTPException, BackgroundTasks, Body
from fastapi.responses import FileResponse, JSONResponse
from uuid import uuid4
import json
from io import BytesIO
from pathlib import Path
import traceback
import numpy as np

from common import storage, logger, constants
from common import result_utils, errors
from api.utils.gcs_utils import (
    upload_run_file, download_run_file, check_run_file_exists,
    GCSError, 
    PROJECT_BUCKET_NAME
)
from api.utils.io_helpers import (
    validate_run_exists_gcs,
    load_original_data_csv_gcs,
    load_metadata_json_gcs
)
from pipeline.step_7_predict.predict_logic import load_pipeline_gcs
from .schema import (
    UploadResponse,
    TargetSuggestionResponse,
    TargetConfirmationRequest,
    TargetConfirmationResponse,
    FeatureSuggestionResponse,
    FeatureConfirmationRequest,
    FeatureConfirmationResponse,
    PipelineStatusResponse,
    PredictionInputRequest,
    PredictionSchemaResponse,
    PredictionResponse,
    FinalResultsResponse,
    # Enhanced prediction models
    EnhancedPredictionSchemaResponse,
    SinglePredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionExplanationResponse,
    PredictionComparisonRequest,
    PredictionComparisonResponse,
    ShapExplanationResponse,
    EnhancedPredictionResponse
)
from pipeline.step_2_schema import target_definition_logic as tgt_logic
from pipeline.step_2_schema import feature_definition_logic as feat_logic
from pipeline.orchestrator import run_from_schema_confirm
from pipeline.step_7_predict.predict_logic import (
    generate_predictions,
    generate_batch_predictions
)
# Removed complex prediction logic imports for simplified approach

router = APIRouter(prefix="/api")


def _new_run_id() -> str:
    """Generate a new unique run ID."""
    return uuid4().hex[:8]


def _validate_run_exists(run_id: str) -> bool:
    """Check if a run exists by checking for original_data.csv in GCS."""
    from api.utils.gcs_utils import check_run_file_exists
    try:
        return check_run_file_exists(run_id, "original_data.csv")
    except Exception:
        # If GCS check fails, fall back to False
        return False


def _validate_target_confirmed(run_id: str) -> bool:
    """Check if target has been confirmed for this run."""
    from api.utils.gcs_utils import download_run_file
    try:
        # Download metadata from GCS
        metadata_bytes = download_run_file(run_id, "metadata.json")
        if metadata_bytes is None:
            return False
        
        metadata = json.loads(metadata_bytes.decode('utf-8'))
        return metadata is not None and 'target_info' in metadata
    except Exception:
        # If GCS check fails, fall back to False
        return False


def _upload_original_data_to_gcs(run_id: str, csv_content: bytes) -> None:
    """Upload original CSV data directly to GCS."""
    csv_io = BytesIO(csv_content)
    success = upload_run_file(run_id, "original_data.csv", csv_io)
    if not success:
        raise GCSError(f"Failed to upload original data CSV to GCS for run {run_id}")


def _create_and_upload_status_json(run_id: str) -> None:
    """Create initial status.json and upload to GCS."""
    upload_ts = datetime.now(timezone.utc).isoformat()
    
    status_data = {
        "run_id": run_id,
        "stage": "upload",
        "status": "completed",
        "message": "File uploaded successfully to GCS",
        "progress_pct": 5,
        "last_updated": upload_ts,
        "stages": {
            "upload": {
                "status": "completed",
                "message": "CSV file uploaded to GCS",
                "timestamp": upload_ts
            },
            "target_suggestion": {
                "status": "pending",
                "message": "Waiting for target column confirmation"
            },
            "feature_suggestion": {
                "status": "pending", 
                "message": "Waiting for feature schema confirmation"
            },
            "pipeline_execution": {
                "status": "pending",
                "message": "Automated pipeline stages not started"
            }
        }
    }
    
    status_json_content = json.dumps(status_data, indent=2)
    status_io = BytesIO(status_json_content.encode('utf-8'))
    
    success = upload_run_file(run_id, "status.json", status_io)
    if not success:
        raise GCSError(f"Failed to upload status.json to GCS for run {run_id}")


def _create_and_upload_metadata_json(run_id: str, filename: str, df: pd.DataFrame) -> None:
    """Create initial metadata.json and upload to GCS."""
    upload_ts = datetime.now(timezone.utc).isoformat()
    
    metadata = {
        "run_id": run_id,
        "timestamp": upload_ts,
        "original_filename": filename,
        "initial_rows": len(df),
        "initial_cols": len(df.columns),
        "initial_dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "storage": {
            "type": "gcs",
            "bucket": PROJECT_BUCKET_NAME,
            "csv_path": f"runs/{run_id}/original_data.csv",
            "metadata_path": f"runs/{run_id}/metadata.json",
            "status_path": f"runs/{run_id}/status.json"
        }
    }
    
    metadata_json_content = json.dumps(metadata, indent=2)
    metadata_io = BytesIO(metadata_json_content.encode('utf-8'))
    
    success = upload_run_file(run_id, "metadata.json", metadata_io)
    if not success:
        raise GCSError(f"Failed to upload metadata.json to GCS for run {run_id}")


def _format_value_for_preview(val):
    """Format a value for display in data preview, handling numeric types properly."""
    if pd.isna(val):
        return "nan"
    if isinstance(val, (int, np.integer)):
        return str(val)
    if isinstance(val, (float, np.floating)):
        # If it's a whole number, display as integer
        if val.is_integer():
            return str(int(val))
        else:
            return str(val)
    return str(val)


def _create_preview(df: pd.DataFrame, num_rows: int = 5) -> List[List[str]]:
    """Create a preview of the first N rows as list of lists of strings."""
    # Get first N rows
    preview_df = df.head(num_rows)

    # Convert to list of lists, including header
    preview = []

    # Add header row
    preview.append(list(df.columns.astype(str)))

    # Add data rows, converting all values to strings with proper formatting
    for _, row in preview_df.iterrows():
        preview.append([_format_value_for_preview(val) for val in row.values])

    return preview


def _run_pipeline_with_error_handling(run_id: str) -> None:
    """
    Wrapper for running the pipeline with proper error handling.
    Used as background task to prevent unhandled exceptions from crashing the API.
    """
    # Get logger for background task
    background_logger = logger.get_structured_logger(
        run_id, "api_background_pipeline"
    )

    try:
        logger.log_structured_event(
            background_logger,
            "background_pipeline_started",
            {"run_id": run_id},
            f"Starting background pipeline execution for run {run_id}"
        )

        # Run the actual pipeline
        success = run_from_schema_confirm(run_id)

        if success:
            logger.log_structured_event(
                background_logger,
                "background_pipeline_succeeded",
                {"run_id": run_id},
                f"Background pipeline completed successfully for run {run_id}"
            )
        else:
            logger.log_structured_error(
                background_logger,
                "background_pipeline_failed",
                f"Pipeline execution failed for run {run_id}",
                {"run_id": run_id, "success": False}
            )

    except Exception as e:
        logger.log_structured_error(
            background_logger,
            "background_pipeline_error",
            f"Unexpected error in background pipeline for run {run_id}: {str(e)}",
            {"run_id": run_id, "error": str(e), "error_type": type(e).__name__}
        )


@router.options("/upload")
async def upload_options():
    """Handle CORS preflight requests for upload endpoint."""
    return {"message": "OK"}


@router.post("/upload", response_model=UploadResponse, status_code=201)
async def upload_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file and initialize a new ML pipeline run.

    Uploads the CSV file to Google Cloud Storage along with initial 
    status.json and metadata.json files. All data is stored in GCS 
    under the runs/{run_id}/ prefix for cloud-native operation.

    Args:
        file: The CSV file to upload

    Returns:
        UploadResponse with run_id, shape, and data preview

    Raises:
        HTTPException: If file validation fails, GCS upload fails, or 
        other processing errors occur
    """

    # Generate new run ID and get logger
    run_id = _new_run_id()
    upload_logger = logger.get_structured_logger(run_id, "api_upload")

    # Log request start
    logger.log_structured_event(
        upload_logger,
        "api_request_started",
        {
            "endpoint": "upload",
            "filename": file.filename,
            "content_type": file.content_type
        },
        f"Starting file upload for new run {run_id}"
    )

    try:
        # 1. Validate content-type (must contain "csv")
        if not file.content_type or "csv" not in file.content_type.lower():
            if not file.filename or not file.filename.lower().endswith('.csv'):
                errors.raise_invalid_csv(
                    "Content-type or filename must indicate CSV format"
                )

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
                errors.raise_invalid_csv(
                    f"Failed to parse CSV: {str(e)}",
                    {"parse_error": str(e)}
                )
            finally:
                # Clean up temp file
                Path(tmp.name).unlink(missing_ok=True)

        # Validate data shape
        if df.empty:
            errors.raise_invalid_csv("File is empty or contains no valid data")

        # 3. Upload original CSV to GCS
        try:
            _upload_original_data_to_gcs(run_id, content)
        except GCSError as e:
            logger.log_structured_error(
                upload_logger,
                "gcs_upload_failed",
                f"Failed to upload CSV to GCS: {str(e)}",
                {"run_id": run_id, "error": str(e)}
            )
            errors.raise_internal_server_error(f"Failed to upload CSV to cloud storage: {str(e)}")

        # 4. Create and upload initial status.json to GCS
        try:
            _create_and_upload_status_json(run_id)
        except GCSError as e:
            logger.log_structured_error(
                upload_logger,
                "gcs_status_upload_failed",
                f"Failed to upload status.json to GCS: {str(e)}",
                {"run_id": run_id, "error": str(e)}
            )
            errors.raise_internal_server_error(f"Failed to upload status to cloud storage: {str(e)}")

        # 5. Create and upload metadata.json to GCS
        try:
            _create_and_upload_metadata_json(run_id, file.filename or "uploaded.csv", df)
        except GCSError as e:
            logger.log_structured_error(
                upload_logger,
                "gcs_metadata_upload_failed",
                f"Failed to upload metadata.json to GCS: {str(e)}",
                {"run_id": run_id, "error": str(e)}
            )
            errors.raise_internal_server_error(f"Failed to upload metadata to cloud storage: {str(e)}")

        # 6. Add entry to local run index (keeping this for backwards compatibility)
        try:
            index_entry = {
                "run_id": run_id,
                "timestamp": datetime.now(timezone.utc),
                "original_filename": file.filename or "uploaded.csv",
                "status": "uploaded"
            }
            storage.append_to_run_index(index_entry)
        except Exception as e:
            # Log but don't fail the request if run index fails
            logger.log_structured_error(
                upload_logger,
                "run_index_failed",
                f"Failed to update run index: {str(e)}",
                {"run_id": run_id, "error": str(e)}
            )

        # 7. Return UploadResponse(run_id, shape, preview)
        shape = (len(df), len(df.columns))
        preview = _create_preview(df)

        # Log successful completion
        logger.log_structured_event(
            upload_logger,
            "api_request_succeeded",
            {
                "endpoint": "upload",
                "run_id": run_id,
                "shape": shape,
                "filename": file.filename,
                "storage_type": "gcs",
                "bucket": PROJECT_BUCKET_NAME
            },
            f"Successfully uploaded file '{file.filename}' to GCS as run {run_id}"
        )

        return UploadResponse(
            run_id=run_id,
            shape=shape,
            preview=preview
        )

    except HTTPException:
        # Log handled HTTP exceptions
        logger.log_structured_error(
            upload_logger,
            "api_request_failed",
            "File upload validation failed",
            {"endpoint": "upload", "filename": file.filename}
        )
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Log unexpected errors
        logger.log_structured_error(
            upload_logger,
            "api_request_failed",
            f"Unexpected error during file upload: {str(e)}",
            {
                "endpoint": "upload",
                "error": str(e),
                "error_type": type(e).__name__,
                "filename": file.filename
            }
        )
        errors.raise_internal_server_error(f"File upload failed: {str(e)}")


def _get_column_statistics(series: pd.Series) -> "ColumnStatistics":
    """Get statistics for a single column."""
    from .schema import ColumnStatistics
    
    # Get sample values (first 3 non-null values)
    sample_values = []
    non_null_values = series.dropna()
    if len(non_null_values) > 0:
        sample_values = [_format_value_for_preview(val) for val in non_null_values.head(3).tolist()]
    
    return ColumnStatistics(
        unique_values=series.nunique(),
        missing_values=series.isna().sum(),
        missing_percentage=round((series.isna().sum() / len(series)) * 100, 1),
        data_type=str(series.dtype),
        sample_values=sample_values
    )


def _get_ml_type_options() -> Dict[str, List["MLTypeOption"]]:
    """Get ML type options grouped by task type."""
    from .schema import MLTypeOption
    
    # ML type descriptions from the Streamlit page
    ml_type_descriptions = {
        "binary_01": "Binary classification with 0/1 numeric labels",
        "binary_numeric": "Binary classification with numeric labels (not 0/1)",
        "binary_text_labels": "Binary classification with text labels (e.g., 'yes'/'no')",
        "binary_boolean": "Binary classification with True/False boolean values",
        "multiclass_int_labels": "Multi-class classification with integer labels",
        "multiclass_text_labels": "Multi-class classification with text labels",
        "high_cardinality_text": "Classification with many unique text categories (may need preprocessing)",
        "numeric_continuous": "Regression with continuous numeric values"
    }
    
    classification_types = [
        "binary_01", "binary_numeric", "binary_text_labels", "binary_boolean",
        "multiclass_int_labels", "multiclass_text_labels", "high_cardinality_text"
    ]
    
    regression_types = ["numeric_continuous"]
    
    return {
        "classification": [
            MLTypeOption(value=ml_type, description=ml_type_descriptions[ml_type])
            for ml_type in classification_types
        ],
        "regression": [
            MLTypeOption(value=ml_type, description=ml_type_descriptions[ml_type])
            for ml_type in regression_types
        ]
    }


@router.get("/target-suggestion", response_model=TargetSuggestionResponse)
def target_suggestion(run_id: str = Query(...)):
    """
    Get target column and task type suggestions for a run.

    Analyzes the uploaded dataset to suggest the most likely target column
    and task type, and provides all necessary data for frontend UI.

    Args:
        run_id: The ID of the run to analyze

    Returns:
        TargetSuggestionResponse with enhanced data for frontend

    Raises:
        HTTPException: If run not found or analysis fails
    """

    # Get logger for this operation
    suggestion_logger = logger.get_structured_logger(
        run_id, "api_target_suggestion"
    )

    # Log request start
    logger.log_structured_event(
        suggestion_logger,
        "api_request_started",
        {"endpoint": "target-suggestion", "run_id": run_id},
        f"Starting target suggestion for run {run_id}"
    )

    try:
        # Validate run exists
        if not _validate_run_exists(run_id):
            errors.raise_run_not_found(run_id)

        # Load original data
        df = storage.read_original_data(run_id)
        if df is None:
            errors.raise_results_not_available(
                run_id, f"Original data not found for run '{run_id}'"
            )

        # Call pipeline logic to suggest target and task type
        suggested_col, suggested_task, suggested_ml_type = (
            tgt_logic.suggest_target_and_task(df)
        )

        if suggested_col is None or suggested_task is None:
            raise HTTPException(
                status_code=400,
                detail=errors.create_error_detail(
                    "Could not infer target column from the data",
                    errors.ErrorCodes.TARGET_INFERENCE_FAILED,
                    {"run_id": run_id}
                )
            )

        # Get statistics for all columns
        columns_stats = {}
        for col in df.columns:
            columns_stats[col] = _get_column_statistics(df[col])

        # Create data preview
        data_preview = _create_preview(df, num_rows=5)

        # Get ML type options
        ml_type_options = _get_ml_type_options()

        # For now, set a static confidence score
        # In the future, this could be calculated based on heuristics
        confidence = 0.87

        # Log successful completion
        logger.log_structured_event(
            suggestion_logger,
            "api_request_succeeded",
            {
                "endpoint": "target-suggestion",
                "run_id": run_id,
                "suggested_column": suggested_col,
                "suggested_task_type": suggested_task,
                "suggested_ml_type": suggested_ml_type,
                "confidence": confidence,
                "total_columns": len(columns_stats)
            },
            f"Target suggestion completed for run {run_id}: "
            f"'{suggested_col}' ({suggested_task}, {suggested_ml_type})"
        )

        return TargetSuggestionResponse(
            columns=columns_stats,
            suggested_column=suggested_col,
            suggested_task_type=suggested_task,
            suggested_ml_type=suggested_ml_type,
            confidence=confidence,
            available_ml_types=ml_type_options,
            data_preview=data_preview
        )

    except HTTPException:
        # Log handled HTTP exceptions
        logger.log_structured_error(
            suggestion_logger,
            "api_request_failed",
            "Target suggestion failed",
            {"endpoint": "target-suggestion", "run_id": run_id}
        )
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Log unexpected errors
        logger.log_structured_error(
            suggestion_logger,
            "api_request_failed",
            f"Unexpected error during target suggestion: {str(e)}",
            {
                "endpoint": "target-suggestion",
                "run_id": run_id,
                "error": str(e),
                "error_type": type(e).__name__
            }
        )
        errors.raise_internal_server_error(f"Target analysis failed: {str(e)}")


@router.post("/confirm-target", response_model=TargetConfirmationResponse)
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

    # Get logger for this operation
    confirm_logger = logger.get_structured_logger(
        req.run_id, "api_target_confirmation"
    )

    # Log request start
    logger.log_structured_event(
        confirm_logger,
        "api_request_started",
        {
            "endpoint": "confirm-target",
            "run_id": req.run_id,
            "confirmed_column": req.confirmed_column,
            "task_type": req.task_type
        },
        f"Starting target confirmation for run {req.run_id}"
    )

    try:
        # Validate run exists
        if not _validate_run_exists(req.run_id):
            errors.raise_run_not_found(req.run_id)

        # Load existing metadata.json
        metadata = storage.read_metadata(req.run_id)
        if metadata is None:
            # Create minimal metadata if it doesn't exist
            metadata = {
                "run_id": req.run_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        # Update metadata with target information (use user-provided ml_type)
        metadata['target_info'] = {
            "name": req.confirmed_column,
            "task_type": req.task_type,
            "ml_type": req.ml_type,
            "user_confirmed_at": datetime.now(timezone.utc).isoformat()
        }

        # Also set top-level task_type for convenience
        metadata['task_type'] = req.task_type

        # Persist updated metadata
        storage.write_metadata(req.run_id, metadata)

        # Log successful completion
        logger.log_structured_event(
            confirm_logger,
            "api_request_succeeded",
            {
                "endpoint": "confirm-target",
                "run_id": req.run_id,
                "confirmed_column": req.confirmed_column,
                "task_type": req.task_type,
                "ml_type": req.ml_type
            },
            f"Target confirmed for run {req.run_id}: "
            f"'{req.confirmed_column}' ({req.task_type}, {req.ml_type})"
        )

        return TargetConfirmationResponse(
            target_info=metadata['target_info']
        )

    except HTTPException:
        # Log handled HTTP exceptions
        logger.log_structured_error(
            confirm_logger,
            "api_request_failed",
            "Target confirmation failed",
            {
                "endpoint": "confirm-target",
                "run_id": req.run_id,
                "confirmed_column": req.confirmed_column
            }
        )
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Log unexpected errors
        logger.log_structured_error(
            confirm_logger,
            "api_request_failed",
            f"Unexpected error during target confirmation: {str(e)}",
            {
                "endpoint": "confirm-target",
                "run_id": req.run_id,
                "error": str(e),
                "error_type": type(e).__name__
            }
        )
        errors.raise_internal_server_error(
            f"Target confirmation failed: {str(e)}"
        )


def _get_available_dtypes() -> Dict[str, str]:
    """Get available data types with user-friendly descriptions."""
    return {
        "object": "Text/String",
        "int64": "Integer Numbers",
        "float64": "Decimal Numbers", 
        "bool": "True/False (Boolean)",
        "datetime64[ns]": "Date/Time",
        "category": "Category"
    }


def _get_available_encoding_roles() -> Dict[str, str]:
    """Get available encoding roles with descriptions."""
    return {
        "numeric-continuous": "Numeric (Continuous) - e.g., price, temperature",
        "numeric-discrete": "Numeric (Discrete) - e.g., count, age",
        "categorical-nominal": "Categorical (No Order) - e.g., color, brand",
        "categorical-ordinal": "Categorical (Ordered) - e.g., small/medium/large",
        "text": "Text (for NLP or hashing) - e.g., descriptions, comments",
        "datetime": "Date/Time features - e.g., timestamp, date",
        "boolean": "Boolean (True/False) - e.g., is_active, has_discount",
        "target": "Target variable (prediction goal)"
    }


@router.get("/feature-suggestion", response_model=FeatureSuggestionResponse)
def feature_suggestion(run_id: str = Query(...), top_n: int = 5):
    """
    Get feature schema suggestions for a run.

    Analyzes the dataset to identify key features and suggest appropriate
    data types and encoding roles for each column. Provides all necessary
    data for building an interactive frontend UI.

    Args:
        run_id: The ID of the run to analyze
        top_n: Number of top features to prioritize (default: 5)

    Returns:
        FeatureSuggestionResponse with enhanced feature data for frontend

    Raises:
        HTTPException: If run not found, target not confirmed, or
        analysis fails
    """

    # Get logger for this operation
    feature_logger = logger.get_structured_logger(
        run_id, "api_feature_suggestion"
    )

    # Log request start
    logger.log_structured_event(
        feature_logger,
        "api_request_started",
        {
            "endpoint": "feature-suggestion",
            "run_id": run_id,
            "top_n": top_n
        },
        f"Starting feature suggestion for run {run_id}"
    )

    try:
        # Validate run exists
        if not _validate_run_exists(run_id):
            errors.raise_run_not_found(run_id)

        # Validate target has been confirmed
        if not _validate_target_confirmed(run_id):
            errors.raise_target_not_confirmed(run_id)

        # Load original data and metadata
        df = storage.read_original_data(run_id)
        if df is None:
            errors.raise_results_not_available(
                run_id, f"Original data not found for run '{run_id}'"
            )

        metadata = storage.read_metadata(run_id)
        target_info = metadata['target_info']

        # Get all feature schemas from pipeline logic
        all_initial_schemas = feat_logic.suggest_initial_feature_schemas(df)

        # Identify key features
        key_features = feat_logic.identify_key_features(
            df, target_info, num_features_to_surface=top_n
        )

        # Build enhanced feature schemas with statistics
        from .schema import FeatureSchema
        feature_schemas = {}
        target_column = target_info['name']
        
        for col in df.columns:
            if col == target_column:
                continue  # Skip target column
                
            # Get basic schema from pipeline logic
            basic_schema = all_initial_schemas[col]
            
            # Add enhanced information
            feature_schemas[col] = FeatureSchema(
                initial_dtype=basic_schema['initial_dtype'],
                suggested_encoding_role=basic_schema['suggested_encoding_role'],
                statistics=_get_column_statistics(df[col]),
                is_key_feature=col in key_features
            )

        # Create data preview
        data_preview = _create_preview(df, num_rows=5)

        # Get available options for UI dropdowns
        available_dtypes = _get_available_dtypes()
        available_encoding_roles = _get_available_encoding_roles()

        # Log successful completion
        logger.log_structured_event(
            feature_logger,
            "api_request_succeeded",
            {
                "endpoint": "feature-suggestion",
                "run_id": run_id,
                "total_features": len(feature_schemas),
                "key_features_count": len(key_features),
                "top_n": top_n
            },
            f"Feature suggestion completed for run {run_id}: "
            f"{len(feature_schemas)} features, {len(key_features)} key features"
        )

        return FeatureSuggestionResponse(
            feature_schemas=feature_schemas,
            key_features=key_features,
            available_dtypes=available_dtypes,
            available_encoding_roles=available_encoding_roles,
            target_info=target_info,
            data_preview=data_preview
        )

    except HTTPException:
        # Log handled HTTP exceptions
        logger.log_structured_error(
            feature_logger,
            "api_request_failed",
            "Feature suggestion failed",
            {"endpoint": "feature-suggestion", "run_id": run_id}
        )
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Log unexpected errors
        logger.log_structured_error(
            feature_logger,
            "api_request_failed",
            f"Unexpected error during feature suggestion: {str(e)}",
            {
                "endpoint": "feature-suggestion",
                "run_id": run_id,
                "error": str(e),
                "error_type": type(e).__name__
            }
        )
        errors.raise_internal_server_error(f"Feature analysis failed: {str(e)}")


@router.post("/confirm-features", response_model=FeatureConfirmationResponse)
def confirm_features(
    req: FeatureConfirmationRequest,
    background_tasks: BackgroundTasks,
):
    """
    Confirm feature schemas and start the automated pipeline stages.

    Updates the run metadata with confirmed feature information and
    schedules the execution of stages 3-7 in the background.

    Args:
        req: Request containing run_id and confirmed feature schemas
        background_tasks: FastAPI background tasks for async execution

    Returns:
        Status indicating pipeline has started

    Raises:
        HTTPException: If run not found, validation fails, or
        confirmation fails
    """

    # Get logger for this operation
    confirm_logger = logger.get_structured_logger(
        req.run_id, "api_feature_confirmation"
    )

    # Log request start
    logger.log_structured_event(
        confirm_logger,
        "api_request_started",
        {
            "endpoint": "confirm-features",
            "run_id": req.run_id,
            "confirmed_schemas_count": len(req.confirmed_schemas)
        },
        f"Starting feature confirmation for run {req.run_id}"
    )

    try:
        # Validate run exists
        if not _validate_run_exists(req.run_id):
            errors.raise_run_not_found(req.run_id)

        # Validate target has been confirmed
        if not _validate_target_confirmed(req.run_id):
            errors.raise_target_not_confirmed(req.run_id)

        # Load original data to get all initial schemas
        df = storage.read_original_data(req.run_id)
        if df is None:
            errors.raise_results_not_available(
                req.run_id, f"Original data not found for run '{req.run_id}'"
            )

        # Get all initial schemas
        all_initial_schemas = feat_logic.suggest_initial_feature_schemas(df)

        # Check if features have already been confirmed
        metadata = storage.read_metadata(req.run_id)
        if metadata and 'feature_schemas' in metadata:
            errors.raise_features_already_confirmed(req.run_id)

        # Confirm feature schemas using pipeline logic
        success = feat_logic.confirm_feature_schemas(
            req.run_id,
            req.confirmed_schemas,
            all_initial_schemas
        )

        if not success:
            raise HTTPException(
                status_code=400,
                detail=errors.create_error_detail(
                    "Schema confirmation failed",
                    errors.ErrorCodes.SCHEMA_CONFIRMATION_FAILED,
                    {"run_id": req.run_id}
                )
            )

        # Schedule background pipeline execution with error handling wrapper
        background_tasks.add_task(_run_pipeline_with_error_handling, req.run_id)

        # Create summary for response
        total_features = len(all_initial_schemas) - 1  # Exclude target column
        user_modified_count = len(req.confirmed_schemas)
        system_default_count = total_features - user_modified_count
        
        # Get key features that were modified
        key_features_modified = []
        if req.key_features_modified:
            key_features_modified = req.key_features_modified
        else:
            # Fallback: identify which confirmed schemas are key features
            metadata = storage.read_metadata(req.run_id)
            target_info = metadata['target_info']
            df = storage.read_original_data(req.run_id)
            if df is not None and target_info:
                key_features = feat_logic.identify_key_features(df, target_info, 5)
                key_features_modified = [col for col in req.confirmed_schemas.keys() if col in key_features]

        summary = {
            "total_features": total_features,
            "user_modified_count": user_modified_count,
            "system_default_count": system_default_count,
            "key_features_modified": key_features_modified,
            "pipeline_status": "started",
            "next_steps": [
                "Data validation will begin automatically",
                "You can check progress at /api/status",
                "Results will be available at /api/results when complete"
            ]
        }

        # Log successful completion
        logger.log_structured_event(
            confirm_logger,
            "api_request_succeeded",
            {
                "endpoint": "confirm-features",
                "run_id": req.run_id,
                "confirmed_schemas_count": len(req.confirmed_schemas),
                "background_task_scheduled": True,
                "summary": summary
            },
            f"Feature confirmation completed for run {req.run_id}, "
            f"background pipeline scheduled"
        )

        return FeatureConfirmationResponse(
            summary=summary
        )

    except HTTPException:
        # Log handled HTTP exceptions
        logger.log_structured_error(
            confirm_logger,
            "api_request_failed",
            "Feature confirmation failed",
            {
                "endpoint": "confirm-features",
                "run_id": req.run_id,
                "confirmed_schemas_count": len(req.confirmed_schemas)
            }
        )
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Log unexpected errors
        logger.log_structured_error(
            confirm_logger,
            "api_request_failed",
            f"Unexpected error during feature confirmation: {str(e)}",
            {
                "endpoint": "confirm-features",
                "run_id": req.run_id,
                "error": str(e),
                "error_type": type(e).__name__
            }
        )
        errors.raise_internal_server_error(
            f"Feature confirmation failed: {str(e)}"
        )


@router.get("/status", response_model=PipelineStatusResponse)
def get_status(run_id: str = Query(...)):
    """
    Get the current pipeline status for a run.

    Reads the status.json file to provide real-time information about
    pipeline progress, current stage, and completion status.

    Args:
        run_id: The ID of the run to check status for

    Returns:
        PipelineStatusResponse with stage, status, message, and progress

    Raises:
        HTTPException: If run not found or status unavailable
    """

    # Get logger for this operation
    status_logger = logger.get_structured_logger(run_id, "api_status")

    # Log request start
    logger.log_structured_event(
        status_logger,
        "api_request_started",
        {"endpoint": "status", "run_id": run_id},
        f"Starting status check for run {run_id}"
    )

    try:
        # Validate run exists
        if not _validate_run_exists(run_id):
            errors.raise_run_not_found(run_id)

        # Get status using result utils
        status_data = result_utils.get_pipeline_status(run_id)

        # Log successful completion
        logger.log_structured_event(
            status_logger,
            "api_request_succeeded",
            {
                "endpoint": "status",
                "run_id": run_id,
                "stage": status_data['stage'],
                "status": status_data['status'],
                "progress_pct": status_data['progress_pct']
            },
            f"Status retrieved for run {run_id}: "
            f"{status_data['stage']} ({status_data['status']})"
        )

        return PipelineStatusResponse(**status_data)

    except FileNotFoundError:
        # Log and raise status not available error
        logger.log_structured_error(
            status_logger,
            "api_request_failed",
            "Status file not found",
            {"endpoint": "status", "run_id": run_id}
        )
        errors.raise_status_not_available(run_id)
    except HTTPException:
        # Log handled HTTP exceptions
        logger.log_structured_error(
            status_logger,
            "api_request_failed",
            "Status check failed",
            {"endpoint": "status", "run_id": run_id}
        )
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Log unexpected errors
        logger.log_structured_error(
            status_logger,
            "api_request_failed",
            f"Unexpected error getting status: {str(e)}",
            {
                "endpoint": "status",
                "run_id": run_id,
                "error": str(e),
                "error_type": type(e).__name__
            }
        )
        errors.raise_internal_server_error(f"Status check failed: {str(e)}")


@router.get("/results", response_model=FinalResultsResponse)
def get_results(run_id: str = Query(...)):
    """
    Get the final pipeline results for a completed run.

    Returns model metrics, feature importance, and explainability information
    once the pipeline has completed all stages successfully.

    Args:
        run_id: The ID of the run to get results for

    Returns:
        FinalResultsResponse with model metrics, top features, and
        explainability

    Raises:
        HTTPException: If run not found, pipeline not completed, or results
        unavailable
    """

    # Get logger for this operation
    results_logger = logger.get_structured_logger(run_id, "api_results")

    # Log request start
    logger.log_structured_event(
        results_logger,
        "api_request_started",
        {"endpoint": "results", "run_id": run_id},
        f"Starting results retrieval for run {run_id}"
    )

    try:
        # Validate run exists
        if not _validate_run_exists(run_id):
            errors.raise_run_not_found(run_id)

        # Check pipeline status first
        status_data = result_utils.get_pipeline_status(run_id)

        if status_data['status'] == 'failed':
            errors.raise_pipeline_failed(
                run_id, "Pipeline failed - results not available"
            )
        elif status_data['status'] != 'completed':
            errors.raise_pipeline_still_running(run_id)

        # Pipeline completed - get results
        results_data = result_utils.build_results(run_id)

        # Log successful completion
        logger.log_structured_event(
            results_logger,
            "api_request_succeeded",
            {
                "endpoint": "results",
                "run_id": run_id,
                "metrics_count": len(results_data['model_metrics']),
                "features_count": len(results_data['top_features'])
            },
            f"Results retrieved for run {run_id}: "
            f"{len(results_data['model_metrics'])} metrics, "
            f"{len(results_data['top_features'])} features"
        )

        return FinalResultsResponse(**results_data)

    except HTTPException:
        # Log handled HTTP exceptions
        logger.log_structured_error(
            results_logger,
            "api_request_failed",
            "Results retrieval failed",
            {"endpoint": "results", "run_id": run_id}
        )
        # Re-raise HTTP exceptions as-is
        raise
    except FileNotFoundError as e:
        # Log and raise results not available error
        logger.log_structured_error(
            results_logger,
            "api_request_failed",
            f"Results files missing: {str(e)}",
            {
                "endpoint": "results",
                "run_id": run_id,
                "error": str(e)
            }
        )
        errors.raise_results_not_available(
            run_id, "Results files missing - pipeline may have failed"
        )
    except Exception as e:
        # Log unexpected errors
        logger.log_structured_error(
            results_logger,
            "api_request_failed",
            f"Unexpected error getting results: {str(e)}",
            {
                "endpoint": "results",
                "run_id": run_id,
                "error": str(e),
                "error_type": type(e).__name__
            }
        )
        errors.raise_internal_server_error(f"Results retrieval failed: {str(e)}")


@router.get("/prediction-schema", response_model=PredictionSchemaResponse)
def get_prediction_schema(run_id: str = Query(...)):
    """
    Get the input schema for building the prediction form.

    Returns information about numeric and categorical columns needed to
    create dynamic form inputs with proper validation ranges and options.

    Args:
        run_id: The ID of the run to get prediction schema for

    Returns:
        PredictionSchemaResponse with column schemas and target info

    Raises:
        HTTPException: If run not found, pipeline not completed, or
        prediction not ready
    """

    # Get logger for this operation
    schema_logger = logger.get_structured_logger(run_id, "api_prediction_schema")

    # Log request start
    logger.log_structured_event(
        schema_logger,
        "api_request_started",
        {"endpoint": "prediction-schema", "run_id": run_id},
        f"Starting prediction schema retrieval for run {run_id}"
    )

    try:
        # Validate run exists
        if not _validate_run_exists(run_id):
            errors.raise_run_not_found(run_id)

        # Check if prediction is ready
        prediction_readiness = result_utils.check_prediction_readiness_gcs(run_id)
        if not prediction_readiness.get("prediction_ready", False):
            missing_files = [k for k, v in prediction_readiness.items() 
                           if k != "prediction_ready" and not v]
            raise HTTPException(
                status_code=400,
                detail=errors.create_error_detail(
                    "Prediction not ready - missing required files",
                    errors.ErrorCodes.PREDICTION_NOT_READY,
                    {"run_id": run_id, "missing_files": missing_files}
                )
            )

        # Load necessary data
        metadata = storage.read_metadata(run_id)
        if not metadata:
            errors.raise_results_not_available(run_id, "Metadata not found")

        df_original = storage.read_original_data(run_id)
        if df_original is None:
            errors.raise_results_not_available(run_id, "Original data not found")

        # Get target info
        target_info = metadata.get('target_info', {})
        target_column = target_info.get('name')
        if not target_column:
            raise HTTPException(
                status_code=400,
                detail=errors.create_error_detail(
                    "Target column information not found",
                    errors.ErrorCodes.TARGET_INFO_MISSING,
                    {"run_id": run_id}
                )
            )

        # Import column mapper to get input schema
        from pipeline.step_7_predict.column_mapper import get_input_schema
        schema_info = get_input_schema(run_id, df_original, target_column)

        # Log successful completion
        logger.log_structured_event(
            schema_logger,
            "api_request_succeeded",
            {
                "endpoint": "prediction-schema",
                "run_id": run_id,
                "numeric_columns_count": len(schema_info.get('numeric_columns', {})),
                "categorical_columns_count": len(schema_info.get('categorical_columns', {})),
                "target_column": target_column
            },
            f"Prediction schema retrieved for run {run_id}"
        )

        return PredictionSchemaResponse(
            numeric_columns=schema_info.get('numeric_columns', {}),
            categorical_columns=schema_info.get('categorical_columns', {}),
            column_encoding_roles=schema_info.get('column_encoding_roles', {}),
            target_info=target_info
        )

    except HTTPException:
        # Log handled HTTP exceptions
        logger.log_structured_error(
            schema_logger,
            "api_request_failed",
            "Prediction schema retrieval failed",
            {"endpoint": "prediction-schema", "run_id": run_id}
        )
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Log unexpected errors
        logger.log_structured_error(
            schema_logger,
            "api_request_failed",
            f"Unexpected error getting prediction schema: {str(e)}",
            {
                "endpoint": "prediction-schema",
                "run_id": run_id,
                "error": str(e),
                "error_type": type(e).__name__
            }
        )
        errors.raise_internal_server_error(f"Prediction schema retrieval failed: {str(e)}")


@router.post("/predict", response_model=PredictionResponse)
def make_prediction(req: PredictionInputRequest):
    """
    Make a prediction using the trained model.

    Processes user input, transforms it to match model expectations,
    generates predictions, and returns results with confidence information.

    Args:
        req: Request containing run_id and input values

    Returns:
        PredictionResponse with prediction value and metadata

    Raises:
        HTTPException: If run not found, prediction not ready, or
        prediction fails
    """

    # Get logger for this operation
    predict_logger = logger.get_structured_logger(req.run_id, "api_predict")

    # Log request start
    logger.log_structured_event(
        predict_logger,
        "api_request_started",
        {
            "endpoint": "predict",
            "run_id": req.run_id,
            "input_features_count": len(req.input_values)
        },
        f"Starting prediction for run {req.run_id}"
    )

    try:
        # Validate run exists
        if not _validate_run_exists(req.run_id):
            errors.raise_run_not_found(req.run_id)

        # Check if prediction is ready
        prediction_readiness = result_utils.check_prediction_readiness_gcs(req.run_id)
        if not prediction_readiness.get("prediction_ready", False):
            missing_files = [k for k, v in prediction_readiness.items() 
                           if k != "prediction_ready" and not v]
            raise HTTPException(
                status_code=400,
                detail=errors.create_error_detail(
                    "Prediction not ready - missing required files",
                    errors.ErrorCodes.PREDICTION_NOT_READY,
                    {"run_id": req.run_id, "missing_files": missing_files}
                )
            )

        # Load necessary data
        metadata = storage.read_metadata(req.run_id)
        if not metadata:
            errors.raise_results_not_available(req.run_id, "Metadata not found")

        df_original = storage.read_original_data(req.run_id)
        if df_original is None:
            errors.raise_results_not_available(req.run_id, "Original data not found")

        # Get target and model info
        target_info = metadata.get('target_info', {})
        target_column = target_info.get('name')
        task_type = target_info.get('task_type', 'unknown')
        
        automl_info = metadata.get('automl_info', {})
        model_name = automl_info.get('best_model_name', 'Unknown')

        if not target_column:
            raise HTTPException(
                status_code=400,
                detail=errors.create_error_detail(
                    "Target column information not found",
                    errors.ErrorCodes.TARGET_INFO_MISSING,
                    {"run_id": req.run_id}
                )
            )

        # Transform user input to model format
        from pipeline.step_7_predict.column_mapper import encode_user_input_gcs
        encoded_df, encoding_issues = encode_user_input_gcs(
            req.input_values, req.run_id, df_original, target_column
        )

        if encoded_df is None or encoding_issues:
            raise HTTPException(
                status_code=400,
                detail=errors.create_error_detail(
                    "Failed to process input values",
                    errors.ErrorCodes.INPUT_ENCODING_FAILED,
                    {"run_id": req.run_id, "issues": encoding_issues}
                )
            )

        # Load the trained model
        from pipeline.step_7_predict.predict_logic import load_pipeline_gcs, generate_predictions
        model = load_pipeline_gcs(req.run_id)
        if model is None:
            raise HTTPException(
                status_code=500,
                detail=errors.create_error_detail(
                    "Failed to load trained model",
                    errors.ErrorCodes.MODEL_LOADING_FAILED,
                    {"run_id": req.run_id}
                )
            )

        # Generate prediction
        result_df = generate_predictions(model, encoded_df, target_column)
        prediction_value = result_df['prediction'].iloc[0]

        # Convert numpy types to Python types for JSON serialization
        if hasattr(prediction_value, 'item'):
            prediction_value = prediction_value.item()

        # Extract processed features (excluding prediction column)
        input_features = result_df.drop(columns=['prediction']).iloc[0].to_dict()

        # Log successful completion
        logger.log_structured_event(
            predict_logger,
            "api_request_succeeded",
            {
                "endpoint": "predict",
                "run_id": req.run_id,
                "prediction_value": prediction_value,
                "task_type": task_type,
                "model_name": model_name
            },
            f"Prediction completed for run {req.run_id}: {prediction_value}"
        )

        return PredictionResponse(
            prediction_value=prediction_value,
            confidence=None,  # Could be enhanced with model.predict_proba() for classification
            input_features=input_features,
            task_type=task_type,
            target_column=target_column,
            model_name=model_name
        )

    except HTTPException:
        # Log handled HTTP exceptions
        logger.log_structured_error(
            predict_logger,
            "api_request_failed",
            "Prediction failed",
            {"endpoint": "predict", "run_id": req.run_id}
        )
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Log unexpected errors
        logger.log_structured_error(
            predict_logger,
            "api_request_failed",
            f"Unexpected error during prediction: {str(e)}",
            {
                "endpoint": "predict",
                "run_id": req.run_id,
                "error": str(e),
                "error_type": type(e).__name__
            }
        )
        errors.raise_internal_server_error(f"Prediction failed: {str(e)}")


# Removed complex prediction endpoints - replaced with simplified /predict/enhanced


# Removed complex batch prediction and comparison endpoints for simplified approach
    """
    Compare multiple prediction scenarios side-by-side with sensitivity analysis.

    Args:
        req: Request containing run_id and scenarios to compare

    Returns:
        PredictionComparisonResponse with comparison analysis

    Raises:
        HTTPException: If run not found, prediction not ready, or comparison fails
    """
    from api.routes.schema import (
        PredictionComparisonResponse, ComparisonAnalysis, 
        PredictionProbabilities
    )

    # Get logger for this operation
    compare_logger = logger.get_structured_logger(req.run_id, "api_predict_compare")

    # Log request start
    logger.log_structured_event(
        compare_logger,
        "api_request_started",
        {
            "endpoint": "predict/compare",
            "run_id": req.run_id,
            "scenarios_count": len(req.scenarios)
        },
        f"Starting prediction comparison for run {req.run_id} with {len(req.scenarios)} scenarios"
    )

    try:
        # Validate run exists
        if not _validate_run_exists(req.run_id):
            errors.raise_run_not_found(req.run_id)

        # Check scenarios limit
        if len(req.scenarios) > 10:  # Reasonable limit for comparison
            raise HTTPException(
                status_code=400,
                detail=errors.create_error_detail(
                    "Too many scenarios to compare - maximum 10 scenarios per request",
                    "scenarios_limit_exceeded",
                    {"run_id": req.run_id, "requested_count": len(req.scenarios), "max_count": 10}
                )
            )

        # Check if prediction is ready
        prediction_readiness = result_utils.check_prediction_readiness_gcs(req.run_id)
        if not prediction_readiness.get("prediction_ready", False):
            missing_files = [k for k, v in prediction_readiness.items() 
                           if k != "prediction_ready" and not v]
            raise HTTPException(
                status_code=400,
                detail=errors.create_error_detail(
                    "Prediction not ready - missing required files",
                    errors.ErrorCodes.PREDICTION_NOT_READY,
                    {"run_id": req.run_id, "missing_files": missing_files}
                )
            )

        # Load necessary data
        metadata = storage.read_metadata(req.run_id)
        if not metadata:
            errors.raise_results_not_available(req.run_id, "Metadata not found")

        df_original = storage.read_original_data(req.run_id)
        if df_original is None:
            errors.raise_results_not_available(req.run_id, "Original data not found")

        # Get target and model info
        target_info = metadata.get('target_info', {})
        target_column = target_info.get('name')
        task_type = target_info.get('task_type', 'unknown')
        
        automl_info = metadata.get('automl_info', {})
        model_name = automl_info.get('best_model_name', 'Unknown')

        if not target_column:
            raise HTTPException(
                status_code=400,
                detail=errors.create_error_detail(
                    "Target column information not found",
                    errors.ErrorCodes.TARGET_INFO_MISSING,
                    {"run_id": req.run_id}
                )
            )

        # Load the trained model
        from pipeline.step_7_predict.predict_logic import load_pipeline_gcs, generate_enhanced_prediction
        from pipeline.step_7_predict.column_mapper import encode_user_input_gcs
        
        model = load_pipeline_gcs(req.run_id)
        if model is None:
            raise HTTPException(
                status_code=500,
                detail=errors.create_error_detail(
                    "Failed to load trained model",
                    errors.ErrorCodes.MODEL_LOADING_FAILED,
                    {"run_id": req.run_id}
                )
            )

        # Generate predictions for each scenario
        scenario_results = []
        all_predictions = []
        
        for scenario in req.scenarios:
            try:
                # Encode inputs
                encoded_df, encoding_issues = encode_user_input_gcs(
                    scenario.input_values, req.run_id, df_original, target_column
                )
                
                if encoded_df is None or encoding_issues:
                    raise ValueError(f"Input encoding failed: {encoding_issues}")
                
                # Generate prediction
                enhanced_pred = generate_enhanced_prediction(model, encoded_df, target_column, task_type)
                
                # Create probabilities if available
                probabilities = None
                if 'probabilities' in enhanced_pred and enhanced_pred['probabilities']:
                    prob_dict = enhanced_pred['probabilities']
                    class_probs = {k: v for k, v in prob_dict.items() if not k.startswith('_')}
                    probabilities = PredictionProbabilities(
                        class_probabilities=class_probs,
                        predicted_class=prob_dict.get('_predicted_class', ''),
                        confidence=prob_dict.get('_confidence', 0.0)
                    )
                
                # Update scenario with results
                scenario.prediction_value = enhanced_pred['prediction_value']
                scenario.probabilities = probabilities
                
                # Identify key differences (simplified logic)
                scenario.key_differences = []
                if len(scenario_results) > 0:
                    # Compare with first scenario
                    first_scenario = scenario_results[0]
                    for key, value in scenario.input_values.items():
                        if key in first_scenario.input_values and value != first_scenario.input_values[key]:
                            scenario.key_differences.append(f"{key}: {value} vs {first_scenario.input_values[key]}")
                
                scenario_results.append(scenario)
                all_predictions.append(enhanced_pred['prediction_value'])
                
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=errors.create_error_detail(
                        f"Failed to process scenario '{scenario.scenario_name}'",
                        errors.ErrorCodes.INPUT_ENCODING_FAILED,
                        {"run_id": req.run_id, "scenario_name": scenario.scenario_name, "error": str(e)}
                    )
                )

        # Analyze differences and sensitivity
        import numpy as np
        
        # Find most influential features (simplified)
        all_feature_names = set()
        for scenario in scenario_results:
            all_feature_names.update(scenario.input_values.keys())
        
        most_influential_features = list(all_feature_names)[:5]  # Top 5 for simplicity
        
        # Calculate prediction sensitivity (standard deviation of predictions)
        prediction_sensitivity = {}
        if len(all_predictions) > 1:
            pred_std = float(np.std(all_predictions))
            prediction_sensitivity = {
                "prediction_std": pred_std,
                "prediction_range": float(max(all_predictions) - min(all_predictions)),
                "coefficient_of_variation": pred_std / float(np.mean(all_predictions)) if np.mean(all_predictions) != 0 else 0.0
            }

        # Create scenario rankings (for classification: by confidence, for regression: by prediction value)
        scenario_rankings = None
        if task_type.lower() == "classification":
            scenario_rankings = sorted(
                [s.scenario_name for s in scenario_results],
                key=lambda name: next(
                    (s.probabilities.confidence for s in scenario_results if s.scenario_name == name and s.probabilities),
                    0.0
                ),
                reverse=True
            )
        elif task_type.lower() == "regression":
            scenario_rankings = sorted(
                [s.scenario_name for s in scenario_results],
                key=lambda name: next(
                    (s.prediction_value for s in scenario_results if s.scenario_name == name),
                    0.0
                ),
                reverse=True
            )

        # Create comparison analysis
        comparison_analysis = ComparisonAnalysis(
            most_influential_features=most_influential_features,
            prediction_sensitivity=prediction_sensitivity,
            scenario_rankings=scenario_rankings
        )

        # Log successful completion
        logger.log_structured_event(
            compare_logger,
            "api_request_succeeded",
            {
                "endpoint": "predict/compare",
                "run_id": req.run_id,
                "scenarios_count": len(scenario_results),
                "most_influential_features": most_influential_features,
                "prediction_sensitivity": prediction_sensitivity,
                "task_type": task_type,
                "model_name": model_name
            },
            f"Prediction comparison completed for run {req.run_id}: {len(scenario_results)} scenarios analyzed"
        )

        return PredictionComparisonResponse(
            scenarios=scenario_results,
            comparison_analysis=comparison_analysis,
            task_type=task_type,
            target_column=target_column,
            model_name=model_name,
            comparison_timestamp=datetime.now().isoformat()
        )

    except HTTPException:
        # Log handled HTTP exceptions
        logger.log_structured_error(
            compare_logger,
            "api_request_failed",
            "Prediction comparison failed",
            {"endpoint": "predict/compare", "run_id": req.run_id}
        )
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Log unexpected errors
        logger.log_structured_error(
            compare_logger,
            "api_request_failed",
            f"Unexpected error during prediction comparison: {str(e)}",
            {
                "endpoint": "predict/compare",
                "run_id": req.run_id,
                "error": str(e),
                "error_type": type(e).__name__
            }
        )
        errors.raise_internal_server_error(f"Prediction comparison failed: {str(e)}")


@router.get("/download/{run_id}/{filename}")
async def download_file(run_id: str, filename: str):
    """
    Download result files (plots, reports, etc.) for a completed run from GCS.

    Downloads files from GCS storage and streams them to the client,
    primarily for explainability plots and other generated artifacts.

    Args:
        run_id: The ID of the run
        filename: The name of the file to download

    Returns:
        FileResponse with the requested file content from GCS

    Raises:
        HTTPException: If run not found or file not available in GCS
    """

    # Get logger for this operation
    download_logger = logger.get_structured_logger(run_id, "api_download")

    # Log request start
    logger.log_structured_event(
        download_logger,
        "api_request_started",
        {"endpoint": "download", "run_id": run_id, "filename": filename},
        f"Starting GCS file download for run {run_id}: {filename}"
    )

    try:
        # Validate run exists
        if not _validate_run_exists(run_id):
            errors.raise_run_not_found(run_id)

        # Look for file in common GCS paths
        possible_gcs_paths = [
            f"plots/{filename}",                   # SHAP plots
            f"step_6_explain/{filename}",          # Explainability outputs
            f"step_5_automl/{filename}",           # Model artifacts
            f"model/{filename}",                   # Model files
            filename,                              # Root level files
        ]

        file_content = None
        found_gcs_path = None
        
        for gcs_path in possible_gcs_paths:
            if check_run_file_exists(run_id, gcs_path):
                file_content = download_run_file(run_id, gcs_path)
                if file_content:
                    found_gcs_path = gcs_path
                    break

        if file_content is None:
            logger.log_structured_error(
                download_logger,
                "api_request_failed",
                f"File not found in GCS: {filename}",
                {"endpoint": "download", "run_id": run_id, "filename": filename, "gcs_bucket": PROJECT_BUCKET_NAME}
            )
            raise HTTPException(
                status_code=404,
                detail=errors.create_error_detail(
                    f"File '{filename}' not found for run '{run_id}' in GCS",
                    "file_not_found",
                    {"run_id": run_id, "filename": filename, "storage_type": "gcs"}
                )
            )

        # Create a temporary file to serve via FileResponse
        import tempfile
        import os
        
        # Create temporary file with proper extension
        file_extension = Path(filename).suffix
        temp_fd, temp_path = tempfile.mkstemp(suffix=file_extension)
        
        try:
            # Write GCS content to temporary file
            with os.fdopen(temp_fd, 'wb') as tmp_file:
                tmp_file.write(file_content)
            
            # Log successful completion
            logger.log_structured_event(
                download_logger,
                "api_request_succeeded",
                {
                    "endpoint": "download",
                    "run_id": run_id,
                    "filename": filename,
                    "gcs_path": found_gcs_path,
                    "file_size": len(file_content),
                    "storage_type": "gcs",
                    "bucket": PROJECT_BUCKET_NAME
                },
                f"GCS file download initiated for run {run_id}: {filename} ({len(file_content)} bytes)"
            )

            # Return FileResponse that will clean up temp file after serving
            class TempFileResponse(FileResponse):
                def __init__(self, *args, **kwargs):
                    self.temp_path = kwargs.pop('temp_path', None)
                    super().__init__(*args, **kwargs)
                
                async def __call__(self, scope, receive, send):
                    try:
                        await super().__call__(scope, receive, send)
                    finally:
                        # Clean up temporary file
                        if self.temp_path and os.path.exists(self.temp_path):
                            os.unlink(self.temp_path)

            return TempFileResponse(
                path=temp_path,
                filename=filename,
                media_type="application/octet-stream",
                temp_path=temp_path
            )
            
        except Exception as e:
            # Clean up temp file if error occurs
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Log unexpected errors
        logger.log_structured_error(
            download_logger,
            "api_request_failed",
            f"Unexpected error downloading file from GCS: {str(e)}",
            {
                "endpoint": "download",
                "run_id": run_id,
                "filename": filename,
                "error": str(e),
                "error_type": type(e).__name__,
                "storage_type": "gcs"
            }
        )
        errors.raise_internal_server_error(f"File download from GCS failed: {str(e)}")


# Enhanced Prediction API Endpoints - Simplified and Consolidated

@router.post("/predict/enhanced", response_model=EnhancedPredictionResponse)
def make_enhanced_prediction(req: PredictionInputRequest):
    """
    Make an enhanced prediction combining the reliable prediction logic with feature importance from results.
    
    This endpoint:
    1. Uses the proven working prediction logic from /api/predict
    2. Enriches the response with feature importance data from /api/results
    3. Provides SHAP availability information without complex real-time SHAP calculations
    
    Args:
        req: Request containing run_id and input values

    Returns:
        EnhancedPredictionResponse with prediction and feature importance

    Raises:
        HTTPException: If run not found, prediction not ready, or prediction fails
    """
    
    # Get logger for this operation
    predict_logger = logger.get_structured_logger(req.run_id, "api_predict_enhanced")

    # Log request start
    logger.log_structured_event(
        predict_logger,
        "api_request_started",
        {
            "endpoint": "predict/enhanced",
            "run_id": req.run_id,
            "input_features_count": len(req.input_values)
        },
        f"Starting enhanced prediction for run {req.run_id}"
    )

    try:
        # Validate run exists
        if not _validate_run_exists(req.run_id):
            errors.raise_run_not_found(req.run_id)

        # Check if prediction is ready
        prediction_readiness = result_utils.check_prediction_readiness_gcs(req.run_id)
        if not prediction_readiness.get("prediction_ready", False):
            missing_files = [k for k, v in prediction_readiness.items() 
                           if k != "prediction_ready" and not v]
            raise HTTPException(
                status_code=400,
                detail=errors.create_error_detail(
                    "Prediction not ready - missing required files",
                    errors.ErrorCodes.PREDICTION_NOT_READY,
                    {"run_id": req.run_id, "missing_files": missing_files}
                )
            )

        # === STEP 1: Make prediction using proven working logic ===
        
        # Load necessary data
        metadata = storage.read_metadata(req.run_id)
        if not metadata:
            errors.raise_results_not_available(req.run_id, "Metadata not found")

        df_original = storage.read_original_data(req.run_id)
        if df_original is None:
            errors.raise_results_not_available(req.run_id, "Original data not found")

        # Get target and model info
        target_info = metadata.get('target_info', {})
        target_column = target_info.get('name')
        task_type = target_info.get('task_type', 'unknown')
        
        automl_info = metadata.get('automl_info', {})
        model_name = automl_info.get('best_model_name', 'Unknown')

        if not target_column:
            raise HTTPException(
                status_code=400,
                detail=errors.create_error_detail(
                    "Target column information not found",
                    errors.ErrorCodes.TARGET_INFO_MISSING,
                    {"run_id": req.run_id}
                )
            )

        # Transform user input to model format (using proven working logic)
        from pipeline.step_7_predict.column_mapper import encode_user_input_gcs
        encoded_df, encoding_issues = encode_user_input_gcs(
            req.input_values, req.run_id, df_original, target_column
        )

        if encoded_df is None or encoding_issues:
            raise HTTPException(
                status_code=400,
                detail=errors.create_error_detail(
                    "Failed to process input values",
                    errors.ErrorCodes.INPUT_ENCODING_FAILED,
                    {"run_id": req.run_id, "issues": encoding_issues}
                )
            )

        # Load the trained model
        from pipeline.step_7_predict.predict_logic import load_pipeline_gcs, generate_predictions
        model = load_pipeline_gcs(req.run_id)
        if model is None:
            raise HTTPException(
                status_code=500,
                detail=errors.create_error_detail(
                    "Failed to load trained model",
                    errors.ErrorCodes.MODEL_LOADING_FAILED,
                    {"run_id": req.run_id}
                )
            )

        # Generate prediction
        result_df = generate_predictions(model, encoded_df, target_column)
        prediction_value = result_df['prediction'].iloc[0]

        # Convert numpy types to Python types for JSON serialization
        if hasattr(prediction_value, 'item'):
            prediction_value = prediction_value.item()

        # Extract processed features (excluding prediction column)
        input_features = result_df.drop(columns=['prediction']).iloc[0].to_dict()

        # === STEP 2: Get feature importance from metadata ===
        
        feature_importance = []
        feature_importance_scores = {}
        shap_plot_available = False
        explainability_available = False
        
        try:
            # Get feature importance directly from metadata (faster and more reliable)
            explain_info = metadata.get('explain_info', {})
            
            # Check if SHAP feature importance is available
            if explain_info.get('feature_importance_scores'):
                feature_importance_scores = explain_info['feature_importance_scores']
                # Convert to sorted list for feature_importance (backwards compatibility)
                sorted_features = sorted(feature_importance_scores.items(), key=lambda x: x[1], reverse=True)
                feature_importance = [feature for feature, score in sorted_features]
                explainability_available = True
                
            elif metadata.get('feature_importance', {}).get('scores'):
                # Use top-level feature importance
                feature_importance_scores = metadata['feature_importance']['scores']
                feature_importance = metadata['feature_importance'].get('ranking', [])
                explainability_available = True
            
            # Check SHAP plot availability
            if explain_info:
                shap_plot_gcs_path = explain_info.get('shap_summary_plot_gcs_path')
                if shap_plot_gcs_path:
                    from api.utils.gcs_utils import check_run_file_exists
                    shap_plot_available = check_run_file_exists(req.run_id, shap_plot_gcs_path)
                else:
                    # Fallback check
                    shap_plot_available = check_run_file_exists(req.run_id, f"plots/shap_summary.png")
            
            # If no SHAP data, try to get from results as fallback
            if not feature_importance_scores:
                status_data = result_utils.get_pipeline_status(req.run_id)
                if status_data.get('status') == 'completed':
                    try:
                        results_data = result_utils.build_results(req.run_id)
                        if 'feature_importance_scores' in results_data:
                            feature_importance_scores = results_data['feature_importance_scores']
                        if 'top_features' in results_data:
                            feature_importance = results_data['top_features']
                    except Exception:
                        pass  # Fallback failed, continue with empty values
                    
        except Exception as e:
            # Log but don't fail the request if feature importance is not available
            logger.log_structured_event(
                predict_logger,
                "feature_importance_unavailable",
                {
                    "endpoint": "predict/enhanced", 
                    "run_id": req.run_id,
                    "error": str(e)
                },
                f"Feature importance not available for run {req.run_id}: {str(e)}"
            )

        # Log successful completion
        logger.log_structured_event(
            predict_logger,
            "api_request_succeeded",
            {
                "endpoint": "predict/enhanced",
                "run_id": req.run_id,
                "prediction_value": prediction_value,
                "task_type": task_type,
                "model_name": model_name,
                "feature_importance_count": len(feature_importance),
                "shap_plot_available": shap_plot_available,
                "explainability_available": explainability_available
            },
            f"Enhanced prediction completed for run {req.run_id}: {prediction_value}"
        )

        return EnhancedPredictionResponse(
            prediction_value=prediction_value,
            confidence=None,  # Could be enhanced with model.predict_proba() for classification
            input_features=input_features,
            feature_importance=feature_importance,
            feature_importance_scores=feature_importance_scores,
            task_type=task_type,
            target_column=target_column,
            model_name=model_name,
            prediction_timestamp=datetime.now(timezone.utc).isoformat(),
            shap_plot_available=shap_plot_available,
            explainability_available=explainability_available
        )

    except HTTPException:
        # Log handled HTTP exceptions
        logger.log_structured_error(
            predict_logger,
            "api_request_failed",
            "Enhanced prediction failed",
            {"endpoint": "predict/enhanced", "run_id": req.run_id}
        )
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Log unexpected errors
        logger.log_structured_error(
            predict_logger,
            "api_request_failed",
            f"Unexpected error during enhanced prediction: {str(e)}",
            {
                "endpoint": "predict/enhanced",
                "run_id": req.run_id,
                "error": str(e),
                "error_type": type(e).__name__
            }
        )
        errors.raise_internal_server_error(f"Enhanced prediction failed: {str(e)}")
