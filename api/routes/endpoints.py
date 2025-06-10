"""
Schema-related API routes for The Projection Wizard.
Provides endpoints for feature suggestions and schema operations.
Refactored for GCS-based storage.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, List, Any
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import from pipeline modules
from pipeline.step_2_schema.feature_definition_logic import (
    identify_key_features, 
    suggest_initial_feature_schemas
)
from api.utils.io_helpers import (
    load_original_data_csv_gcs, 
    load_metadata_json_gcs, 
    validate_run_exists_gcs, 
    validate_required_files_gcs,
    list_runs_gcs,
    DataLoadError
)
from api.utils.gcs_utils import PROJECT_BUCKET_NAME

router = APIRouter()


@router.get("/feature_suggestions")
async def get_feature_suggestions(
    run_id: str = Query(..., description="The ID of the run to analyze"),
    num_features: int = Query(7, description="Number of key features to identify", ge=1, le=50)
):
    """
    Get feature suggestions for a given run.
    
    Returns key features identified through importance analysis and initial
    dtype/encoding role suggestions for all columns.
    
    Args:
        run_id: The ID of the run to analyze
        num_features: Number of top features to identify (default: 7)
        
    Returns:
        JSON response containing:
        - key_features: List of identified important feature names
        - initial_suggestions: Dict mapping column names to schema suggestions
        - metadata: Additional information about the analysis
    """
    
    # Validate run exists in GCS
    if not validate_run_exists_gcs(run_id, PROJECT_BUCKET_NAME):
        raise HTTPException(
            status_code=404, 
            detail=f"Run '{run_id}' not found in GCS"
        )
    
    # Check required files exist in GCS
    file_status = validate_required_files_gcs(run_id, PROJECT_BUCKET_NAME)
    missing_files = [file for file, exists in file_status.items() if not exists]
    
    if missing_files:
        raise HTTPException(
            status_code=404,
            detail=f"Missing required files for run '{run_id}' in GCS: {', '.join(missing_files)}"
        )
    
    try:
        # Load data files from GCS
        df = load_original_data_csv_gcs(run_id, PROJECT_BUCKET_NAME)
        metadata = load_metadata_json_gcs(run_id, PROJECT_BUCKET_NAME)
        
        if df is None:
            raise HTTPException(
                status_code=404,
                detail=f"Could not load original_data.csv for run '{run_id}' from GCS"
            )
            
        if metadata is None:
            raise HTTPException(
                status_code=404,
                detail=f"Could not load metadata.json for run '{run_id}' from GCS"
            )
        
        # Extract target information from metadata
        target_info = metadata.get('target_info')
        if not target_info:
            raise HTTPException(
                status_code=400,
                detail=f"No target column information found in metadata for run '{run_id}'"
            )
        
        # Identify key features
        key_features = identify_key_features(
            df_original=df,
            target_info=target_info,
            num_features_to_surface=num_features
        )
        
        # Get initial schema suggestions for all columns
        initial_suggestions = suggest_initial_feature_schemas(df)
        
        # Prepare response
        response = {
            "key_features": key_features,
            "initial_suggestions": initial_suggestions,
            "metadata": {
                "run_id": run_id,
                "total_columns": len(df.columns),
                "total_rows": len(df),
                "target_column": target_info.get('name'),
                "task_type": target_info.get('task_type'),
                "num_key_features_requested": num_features,
                "num_key_features_identified": len(key_features),
                "storage_type": "gcs",
                "bucket": PROJECT_BUCKET_NAME
            }
        }
        
        return response
        
    except DataLoadError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze features for run '{run_id}' from GCS: {str(e)}"
        )


@router.get("/runs/{run_id}/info")
async def get_run_info(run_id: str):
    """
    Get basic information about a run from GCS.
    
    Args:
        run_id: The ID of the run
        
    Returns:
        JSON response with run information and file availability in GCS
    """
    
    # Check if run exists in GCS
    if not validate_run_exists_gcs(run_id, PROJECT_BUCKET_NAME):
        raise HTTPException(
            status_code=404,
            detail=f"Run '{run_id}' not found in GCS"
        )
    
    # Get file status from GCS
    file_status = validate_required_files_gcs(run_id, PROJECT_BUCKET_NAME)
    
    response = {
        "run_id": run_id,
        "files": file_status,
        "ready_for_analysis": all(file_status.values()),
        "storage_type": "gcs",
        "bucket": PROJECT_BUCKET_NAME
    }
    
    # If metadata is available in GCS, include some basic info
    if file_status.get("metadata_json"):
        try:
            metadata = load_metadata_json_gcs(run_id, PROJECT_BUCKET_NAME)
            if metadata:
                response["target_info"] = metadata.get('target_info')
                response["upload_timestamp"] = metadata.get('timestamp')
        except:
            pass  # Don't fail if metadata can't be loaded
    
    return response


@router.get("/runs")
async def list_runs():
    """
    List all available runs from GCS.
    
    Returns:
        JSON response with list of available run IDs from GCS storage
    """
    
    try:
        # Get all run IDs from GCS
        run_ids = list_runs_gcs(PROJECT_BUCKET_NAME)
        
        return {
            "runs": run_ids,
            "total_runs": len(run_ids),
            "storage_type": "gcs",
            "bucket": PROJECT_BUCKET_NAME
        }
        
    except DataLoadError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list runs from GCS: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list runs from GCS: {str(e)}"
        ) 