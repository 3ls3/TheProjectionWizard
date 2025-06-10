"""
Example integration of GCS utilities with FastAPI endpoints.
This demonstrates how the GCS utilities can be used in the actual API endpoints.
"""

from fastapi import FastAPI, UploadFile, HTTPException
from typing import Dict, Any
import uuid
from io import BytesIO
import json

from .utils.gcs_utils import (
    upload_run_file,
    download_run_file, 
    check_run_file_exists,
    list_run_files,
    GCSError
)

app = FastAPI()

@app.post("/upload")
async def upload_csv(file: UploadFile) -> Dict[str, Any]:
    """
    Example endpoint showing how to use GCS utilities for file upload.
    This mimics what the actual /upload endpoint would do.
    """
    try:
        # Generate a run ID
        run_id = str(uuid.uuid4())
        
        # Read the uploaded file into memory
        file_content = await file.read()
        file_io = BytesIO(file_content)
        
        # Upload to GCS using our utility function
        upload_success = upload_run_file(run_id, "original_data.csv", file_io)
        
        if not upload_success:
            raise HTTPException(status_code=500, detail="Failed to upload file to GCS")
        
        # Create initial metadata
        metadata = {
            "run_id": run_id,
            "original_filename": file.filename,
            "file_size": len(file_content),
            "upload_timestamp": "2025-06-10T12:00:00Z"  # In real app, use datetime.utcnow()
        }
        
        # Save metadata to GCS
        metadata_json = json.dumps(metadata, indent=2)
        metadata_io = BytesIO(metadata_json.encode('utf-8'))
        metadata_success = upload_run_file(run_id, "metadata.json", metadata_io)
        
        if not metadata_success:
            raise HTTPException(status_code=500, detail="Failed to upload metadata to GCS")
        
        return {
            "success": True,
            "run_id": run_id,
            "message": "File uploaded successfully to GCS",
            "preview": {
                "filename": file.filename,
                "size_bytes": len(file_content)
            }
        }
        
    except GCSError as e:
        raise HTTPException(status_code=500, detail=f"GCS Error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/runs/{run_id}/files")
async def list_run_files_endpoint(run_id: str) -> Dict[str, Any]:
    """
    Example endpoint showing how to list files for a run using GCS utilities.
    """
    try:
        files = list_run_files(run_id)
        
        return {
            "success": True,
            "run_id": run_id,
            "files": files,
            "count": len(files)
        }
        
    except GCSError as e:
        raise HTTPException(status_code=500, detail=f"GCS Error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/runs/{run_id}/metadata")
async def get_run_metadata(run_id: str) -> Dict[str, Any]:
    """
    Example endpoint showing how to download and parse metadata from GCS.
    """
    try:
        # Check if metadata file exists
        if not check_run_file_exists(run_id, "metadata.json"):
            raise HTTPException(status_code=404, detail="Run metadata not found")
        
        # Download metadata
        metadata_bytes = download_run_file(run_id, "metadata.json")
        
        if metadata_bytes is None:
            raise HTTPException(status_code=404, detail="Failed to download metadata")
        
        # Parse JSON
        metadata = json.loads(metadata_bytes.decode('utf-8'))
        
        return {
            "success": True,
            "run_id": run_id,
            "metadata": metadata
        }
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid JSON in metadata file")
    except GCSError as e:
        raise HTTPException(status_code=500, detail=f"GCS Error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/runs/{run_id}/status")
async def get_run_status(run_id: str) -> Dict[str, Any]:
    """
    Example endpoint showing how to check file existence and determine run status.
    """
    try:
        # Check which files exist for this run
        required_files = ["original_data.csv", "metadata.json"]
        optional_files = ["status.json", "results.json", "model.pkl"]
        
        file_status = {}
        for file_name in required_files + optional_files:
            file_status[file_name] = check_run_file_exists(run_id, file_name)
        
        # Determine overall status
        has_required = all(file_status[f] for f in required_files)
        has_results = file_status.get("results.json", False)
        
        if has_results:
            status = "completed"
        elif has_required:
            status = "processing" 
        else:
            status = "incomplete"
        
        return {
            "success": True,
            "run_id": run_id,
            "status": status,
            "files": file_status,
            "required_files_present": has_required
        }
        
    except GCSError as e:
        raise HTTPException(status_code=500, detail=f"GCS Error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}") 