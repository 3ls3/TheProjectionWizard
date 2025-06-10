"""
Schema-related API routes for The Projection Wizard.
Provides endpoints for feature suggestions and schema operations.
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
    load_original_data_csv, 
    load_metadata_json, 
    validate_run_exists, 
    validate_required_files,
    DataLoadError
)

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
    
    # Validate run exists
    if not validate_run_exists(run_id):
        raise HTTPException(
            status_code=404, 
            detail=f"Run '{run_id}' not found"
        )
    
    # Check required files exist
    file_status = validate_required_files(run_id)
    missing_files = [file for file, exists in file_status.items() if not exists]
    
    if missing_files:
        raise HTTPException(
            status_code=404,
            detail=f"Missing required files for run '{run_id}': {', '.join(missing_files)}"
        )
    
    try:
        # Load data files
        df = load_original_data_csv(run_id)
        metadata = load_metadata_json(run_id)
        
        if df is None:
            raise HTTPException(
                status_code=404,
                detail=f"Could not load original_data.csv for run '{run_id}'"
            )
            
        if metadata is None:
            raise HTTPException(
                status_code=404,
                detail=f"Could not load metadata.json for run '{run_id}'"
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
                "num_key_features_identified": len(key_features)
            }
        }
        
        return response
        
    except DataLoadError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze features for run '{run_id}': {str(e)}"
        )


@router.get("/runs/{run_id}/info")
async def get_run_info(run_id: str):
    """
    Get basic information about a run.
    
    Args:
        run_id: The ID of the run
        
    Returns:
        JSON response with run information and file availability
    """
    
    # Check if run exists
    if not validate_run_exists(run_id):
        raise HTTPException(
            status_code=404,
            detail=f"Run '{run_id}' not found"
        )
    
    # Get file status
    file_status = validate_required_files(run_id)
    
    response = {
        "run_id": run_id,
        "files": file_status,
        "ready_for_analysis": all(file_status.values())
    }
    
    # If metadata is available, include some basic info
    if file_status.get("metadata_json"):
        try:
            metadata = load_metadata_json(run_id)
            if metadata:
                response["target_info"] = metadata.get('target_info')
                response["upload_timestamp"] = metadata.get('timestamp')
        except:
            pass  # Don't fail if metadata can't be loaded
    
    return response


@router.get("/runs")
async def list_runs():
    """
    List all available runs.
    
    Returns:
        JSON response with list of available run IDs
    """
    
    try:
        project_root = Path(__file__).parent.parent.parent
        runs_dir = project_root / "data" / "runs"
        
        if not runs_dir.exists():
            return {"runs": []}
        
        # Get all subdirectories in runs directory
        run_ids = [d.name for d in runs_dir.iterdir() if d.is_dir()]
        run_ids.sort()  # Sort alphabetically
        
        return {
            "runs": run_ids,
            "total_runs": len(run_ids)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list runs: {str(e)}"
        ) 