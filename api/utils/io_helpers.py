"""
IO helper functions for loading data files from run directories.
Handles CSV and metadata loading with graceful error handling.
Refactored for GCS-based storage.
"""

import pandas as pd
import json
from pathlib import Path
from typing import Optional, Dict, Any
import io

from api.utils.gcs_utils import (
    download_run_file, check_run_file_exists, list_gcs_files, 
    PROJECT_BUCKET_NAME, GCSError
)


class DataLoadError(Exception):
    """Custom exception for data loading errors."""
    pass


def get_run_directory(run_id: str) -> Path:
    """
    Legacy compatibility function - Get the path to a run directory.
    
    Args:
        run_id: The ID of the run
        
    Returns:
        Path object for the run directory
        
    Note:
        This function is kept for backward compatibility but should not be used
        in GCS-based operations. Use GCS functions instead.
    """
    project_root = Path(__file__).parent.parent.parent
    return project_root / "data" / "runs" / run_id


def validate_run_exists_gcs(run_id: str, 
                           gcs_bucket_name: str = PROJECT_BUCKET_NAME) -> bool:
    """
    Check if a run exists in GCS by checking for marker files.
    
    Args:
        run_id: The ID of the run
        gcs_bucket_name: GCS bucket name for storage
        
    Returns:
        True if run directory exists in GCS, False otherwise
    """
    try:
        # Check for key marker files that indicate a run exists
        marker_files = ["original_data.csv", "metadata.json"]
        
        for marker_file in marker_files:
            if check_run_file_exists(run_id, marker_file):
                return True
                
        return False
        
    except GCSError:
        return False
    except Exception:
        return False


def validate_run_exists(run_id: str) -> bool:
    """
    Legacy compatibility function - redirects to GCS version.
    
    Args:
        run_id: The ID of the run
        
    Returns:
        True if run directory exists, False otherwise
    """
    return validate_run_exists_gcs(run_id)


def validate_required_files_gcs(run_id: str,
                                gcs_bucket_name: str = PROJECT_BUCKET_NAME) -> Dict[str, bool]:
    """
    Check which required files exist for a run in GCS.
    
    Args:
        run_id: The ID of the run
        gcs_bucket_name: GCS bucket name for storage
        
    Returns:
        Dictionary indicating which files exist in GCS
    """
    try:
        return {
            "run_directory": validate_run_exists_gcs(run_id, gcs_bucket_name),
            "original_data_csv": check_run_file_exists(run_id, "original_data.csv"),
            "metadata_json": check_run_file_exists(run_id, "metadata.json")
        }
    except Exception:
        # Return all False if any error occurs
        return {
            "run_directory": False,
            "original_data_csv": False,
            "metadata_json": False
        }


def validate_required_files(run_id: str) -> Dict[str, bool]:
    """
    Legacy compatibility function - redirects to GCS version.
    
    Args:
        run_id: The ID of the run
        
    Returns:
        Dictionary indicating which files exist
    """
    return validate_required_files_gcs(run_id)


def load_original_data_csv_gcs(run_id: str,
                              gcs_bucket_name: str = PROJECT_BUCKET_NAME) -> Optional[pd.DataFrame]:
    """
    Load the original_data.csv file for a given run from GCS.
    
    Args:
        run_id: The ID of the run
        gcs_bucket_name: GCS bucket name for storage
        
    Returns:
        pandas DataFrame with the original data, or None if file not found
        
    Raises:
        DataLoadError: If file exists but cannot be loaded
    """
    try:
        # Check if file exists first
        if not check_run_file_exists(run_id, "original_data.csv"):
            return None
            
        # Download file from GCS
        csv_bytes = download_run_file(run_id, "original_data.csv")
        if csv_bytes is None:
            return None
            
        # Load into pandas DataFrame
        csv_io = io.BytesIO(csv_bytes)
        df = pd.read_csv(csv_io)
        return df
        
    except GCSError:
        return None
    except Exception as e:
        raise DataLoadError(f"Failed to load original_data.csv for run {run_id} from GCS: {str(e)}")


def load_original_data_csv(run_id: str) -> Optional[pd.DataFrame]:
    """
    Legacy compatibility function - redirects to GCS version.
    
    Args:
        run_id: The ID of the run
        
    Returns:
        pandas DataFrame with the original data, or None if file not found
        
    Raises:
        DataLoadError: If file exists but cannot be loaded
    """
    return load_original_data_csv_gcs(run_id)


def load_metadata_json_gcs(run_id: str,
                          gcs_bucket_name: str = PROJECT_BUCKET_NAME) -> Optional[Dict[str, Any]]:
    """
    Load the metadata.json file for a given run from GCS.
    
    Args:
        run_id: The ID of the run
        gcs_bucket_name: GCS bucket name for storage
        
    Returns:
        Dictionary with metadata, or None if file not found
        
    Raises:
        DataLoadError: If file exists but cannot be loaded
    """
    try:
        # Check if file exists first
        if not check_run_file_exists(run_id, "metadata.json"):
            return None
            
        # Download file from GCS
        metadata_bytes = download_run_file(run_id, "metadata.json")
        if metadata_bytes is None:
            return None
            
        # Parse JSON content
        metadata_str = metadata_bytes.decode('utf-8')
        metadata = json.loads(metadata_str)
        return metadata
        
    except GCSError:
        return None
    except json.JSONDecodeError as e:
        raise DataLoadError(f"Invalid JSON in metadata.json for run {run_id}: {str(e)}")
    except Exception as e:
        raise DataLoadError(f"Failed to load metadata.json for run {run_id} from GCS: {str(e)}")


def load_metadata_json(run_id: str) -> Optional[Dict[str, Any]]:
    """
    Legacy compatibility function - redirects to GCS version.
    
    Args:
        run_id: The ID of the run
        
    Returns:
        Dictionary with metadata, or None if file not found
        
    Raises:
        DataLoadError: If file exists but cannot be loaded
    """
    return load_metadata_json_gcs(run_id)


def list_runs_gcs(gcs_bucket_name: str = PROJECT_BUCKET_NAME) -> list[str]:
    """
    List all available runs from GCS.
    
    Args:
        gcs_bucket_name: GCS bucket name for storage
        
    Returns:
        List of run IDs found in GCS
        
    Raises:
        DataLoadError: If listing fails
    """
    try:
        # List all files under the runs/ prefix
        all_files = list_gcs_files(gcs_bucket_name, "runs/")
        
        # Extract unique run IDs from file paths
        run_ids = set()
        for file_path in all_files:
            # File paths look like: runs/{run_id}/filename.ext
            path_parts = file_path.split('/')
            if len(path_parts) >= 3 and path_parts[0] == "runs":
                run_id = path_parts[1]
                if run_id:  # Ensure non-empty run_id
                    run_ids.add(run_id)
        
        # Convert to sorted list
        return sorted(list(run_ids))
        
    except GCSError as e:
        raise DataLoadError(f"Failed to list runs from GCS: {str(e)}")
    except Exception as e:
        raise DataLoadError(f"Failed to list runs from GCS: {str(e)}")


def list_runs() -> list[str]:
    """
    List all available runs - GCS version by default.
    
    Returns:
        List of run IDs found in storage
        
    Raises:
        DataLoadError: If listing fails
    """
    return list_runs_gcs() 