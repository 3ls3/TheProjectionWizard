"""
IO helper functions for loading data files from run directories.
Handles CSV and metadata loading with graceful error handling.
"""

import pandas as pd
import json
from pathlib import Path
from typing import Optional, Dict, Any


class DataLoadError(Exception):
    """Custom exception for data loading errors."""
    pass


def get_run_directory(run_id: str) -> Path:
    """
    Get the path to a run directory.
    
    Args:
        run_id: The ID of the run
        
    Returns:
        Path object for the run directory
    """
    project_root = Path(__file__).parent.parent.parent
    return project_root / "data" / "runs" / run_id


def load_original_data_csv(run_id: str) -> Optional[pd.DataFrame]:
    """
    Load the original_data.csv file for a given run.
    
    Args:
        run_id: The ID of the run
        
    Returns:
        pandas DataFrame with the original data, or None if file not found
        
    Raises:
        DataLoadError: If file exists but cannot be loaded
    """
    try:
        run_dir = get_run_directory(run_id)
        csv_path = run_dir / "original_data.csv"
        
        if not csv_path.exists():
            return None
            
        df = pd.read_csv(csv_path)
        return df
        
    except FileNotFoundError:
        return None
    except Exception as e:
        raise DataLoadError(f"Failed to load original_data.csv for run {run_id}: {str(e)}")


def load_metadata_json(run_id: str) -> Optional[Dict[str, Any]]:
    """
    Load the metadata.json file for a given run.
    
    Args:
        run_id: The ID of the run
        
    Returns:
        Dictionary with metadata, or None if file not found
        
    Raises:
        DataLoadError: If file exists but cannot be loaded
    """
    try:
        run_dir = get_run_directory(run_id)
        metadata_path = run_dir / "metadata.json"
        
        if not metadata_path.exists():
            return None
            
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            
        return metadata
        
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as e:
        raise DataLoadError(f"Invalid JSON in metadata.json for run {run_id}: {str(e)}")
    except Exception as e:
        raise DataLoadError(f"Failed to load metadata.json for run {run_id}: {str(e)}")


def validate_run_exists(run_id: str) -> bool:
    """
    Check if a run directory exists.
    
    Args:
        run_id: The ID of the run
        
    Returns:
        True if run directory exists, False otherwise
    """
    run_dir = get_run_directory(run_id)
    return run_dir.exists() and run_dir.is_dir()


def validate_required_files(run_id: str) -> Dict[str, bool]:
    """
    Check which required files exist for a run.
    
    Args:
        run_id: The ID of the run
        
    Returns:
        Dictionary indicating which files exist
    """
    run_dir = get_run_directory(run_id)
    
    return {
        "run_directory": run_dir.exists(),
        "original_data_csv": (run_dir / "original_data.csv").exists(),
        "metadata_json": (run_dir / "metadata.json").exists()
    } 