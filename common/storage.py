"""
Storage utilities for The Projection Wizard.
Provides run_id-centric file operations with atomic writing for critical files.
"""

import json
import tempfile
import shutil
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from .constants import DATA_DIR_NAME, RUNS_DIR_NAME, RUN_INDEX_FILENAME
from .schemas import RunIndexEntry


def get_run_dir(run_id: str) -> Path:
    """
    Helper function to consistently construct the run directory path.
    Ensures the directory exists, creating it if necessary.
    
    Args:
        run_id: Unique run identifier
        
    Returns:
        Path to the run directory
    """
    run_dir = Path(DATA_DIR_NAME) / RUNS_DIR_NAME / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create standard subdirectories
    (run_dir / "model").mkdir(exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)
    
    return run_dir


def write_json_atomic(run_id: str, filename: str, data: dict) -> None:
    """
    Write JSON data to file atomically to prevent corruption.
    Uses get_run_dir(run_id) to determine the base path.
    
    Args:
        run_id: Unique run identifier
        filename: Name of the JSON file (e.g., 'metadata.json')
        data: Data to write as JSON
        
    Raises:
        IOError: If writing fails
    """
    run_dir = get_run_dir(run_id)
    filepath = run_dir / filename
    
    # Write to temporary file first
    with tempfile.NamedTemporaryFile(
        mode='w', 
        dir=run_dir, 
        delete=False,
        suffix='.tmp'
    ) as tmp_file:
        json.dump(data, tmp_file, indent=2, default=str)
        tmp_file.flush()
        temp_path = Path(tmp_file.name)
    
    # Atomic move to final location
    try:
        shutil.move(str(temp_path), str(filepath))
    except Exception as e:
        # Clean up temp file if move fails
        temp_path.unlink(missing_ok=True)
        raise IOError(f"Failed to write {filepath}: {e}")


def read_json(run_id: str, filename: str) -> Optional[dict]:
    """
    Read JSON data from file with error handling.
    Uses get_run_dir(run_id) to determine the base path.
    
    Args:
        run_id: Unique run identifier
        filename: Name of the JSON file (e.g., 'metadata.json')
        
    Returns:
        JSON data as dictionary, or None if file doesn't exist
        
    Raises:
        IOError: If file exists but cannot be read or parsed
    """
    run_dir = get_run_dir(run_id)
    filepath = run_dir / filename
    
    if not filepath.exists():
        return None
        
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        raise IOError(f"Failed to read {filepath}: {e}")


def append_to_run_index(run_entry_data: dict) -> None:
    """
    Append a new entry to the run index CSV file.
    Creates the file and header if it doesn't exist.
    
    Args:
        run_entry_data: Dictionary representing a row (keys matching RunIndexEntry fields)
    """
    run_index_path = Path(DATA_DIR_NAME) / RUNS_DIR_NAME / RUN_INDEX_FILENAME
    
    # Ensure parent directory exists
    run_index_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file exists to determine if we need to write header
    file_exists = run_index_path.exists()
    
    # Get field names from RunIndexEntry model
    fieldnames = list(RunIndexEntry.model_fields.keys())
    
    # Open file in append mode
    with open(run_index_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header if file is new
        if not file_exists:
            writer.writeheader()
        
        # Write the data row
        writer.writerow(run_entry_data)


# Convenience functions for common operations
def write_metadata(run_id: str, metadata: dict) -> None:
    """Write metadata.json for a run."""
    write_json_atomic(run_id, "metadata.json", metadata)


def read_metadata(run_id: str) -> Optional[dict]:
    """Read metadata.json for a run."""
    return read_json(run_id, "metadata.json")


def write_status(run_id: str, status: dict) -> None:
    """Write status.json for a run."""
    write_json_atomic(run_id, "status.json", status)


def read_status(run_id: str) -> Optional[dict]:
    """Read status.json for a run."""
    return read_json(run_id, "status.json")


def read_original_data(run_id: str) -> Optional["pd.DataFrame"]:
    """
    Read the original data CSV file for a run.
    
    Args:
        run_id: Unique run identifier
        
    Returns:
        pandas DataFrame with the original data, or None if file doesn't exist
        
    Raises:
        IOError: If file exists but cannot be read or parsed
    """
    import pandas as pd
    from .constants import ORIGINAL_DATA_FILE
    
    run_dir = get_run_dir(run_id)
    filepath = run_dir / ORIGINAL_DATA_FILE
    
    if not filepath.exists():
        return None
        
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise IOError(f"Failed to read original data from {filepath}: {e}")


def read_cleaned_data(run_id: str) -> Optional["pd.DataFrame"]:
    """
    Read the cleaned data CSV file for a run.
    
    Args:
        run_id: Unique run identifier
        
    Returns:
        pandas DataFrame with the cleaned data, or None if file doesn't exist
        
    Raises:
        IOError: If file exists but cannot be read or parsed
    """
    from .constants import CLEANED_DATA_FILE
    
    run_dir = get_run_dir(run_id)
    filepath = run_dir / CLEANED_DATA_FILE
    
    if not filepath.exists():
        return None
        
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise IOError(f"Failed to read cleaned data from {filepath}: {e}")


def get_run_dir_path(run_id: str) -> Path:
    """
    Get the path to the run directory (alias for get_run_dir for compatibility).
    
    Args:
        run_id: Unique run identifier
        
    Returns:
        Path to the run directory
    """
    return get_run_dir(run_id)


def list_runs() -> List[str]:
    """
    List all available run IDs.
    
    Returns:
        List of run ID strings
    """
    runs_dir = Path(DATA_DIR_NAME) / RUNS_DIR_NAME
    if not runs_dir.exists():
        return []
        
    return [d.name for d in runs_dir.iterdir() if d.is_dir()]


def get_artifact_path(run_id: str, artifact_name: str) -> Path:
    """
    Get the full path for a run artifact.
    
    Args:
        run_id: Unique run identifier
        artifact_name: Name of the artifact file
        
    Returns:
        Full path to the artifact
    """
    return get_run_dir(run_id) / artifact_name 