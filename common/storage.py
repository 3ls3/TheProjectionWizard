"""
Storage utilities for The Projection Wizard.
Provides atomic file writing operations to prevent corruption of critical files.
"""

import json
import tempfile
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import uuid

from .constants import RUNS_DIR, METADATA_FILE, STATUS_FILE


def generate_run_id() -> str:
    """
    Generate a unique run ID using timestamp and UUID.
    
    Returns:
        Unique run identifier string
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"{timestamp}_{unique_id}"


def get_run_dir(run_id: str) -> Path:
    """
    Get the directory path for a specific run.
    
    Args:
        run_id: Unique run identifier
        
    Returns:
        Path to the run directory
    """
    return RUNS_DIR / run_id


def create_run_directory(run_id: str) -> Path:
    """
    Create the directory structure for a new run.
    
    Args:
        run_id: Unique run identifier
        
    Returns:
        Path to the created run directory
    """
    run_dir = get_run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (run_dir / "model").mkdir(exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)
    
    return run_dir


def write_json_atomic(filepath: Path, data: Dict[str, Any]) -> None:
    """
    Write JSON data to file atomically to prevent corruption.
    
    Args:
        filepath: Target file path
        data: Data to write as JSON
        
    Raises:
        IOError: If writing fails
    """
    filepath = Path(filepath)
    
    # Ensure parent directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to temporary file first
    with tempfile.NamedTemporaryFile(
        mode='w', 
        dir=filepath.parent, 
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


def read_json_safe(filepath: Path) -> Optional[Dict[str, Any]]:
    """
    Safely read JSON data from file with error handling.
    
    Args:
        filepath: File path to read from
        
    Returns:
        JSON data as dictionary, or None if file doesn't exist or is invalid
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        return None
        
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Failed to read {filepath}: {e}")
        return None


def write_metadata(run_id: str, metadata: Dict[str, Any]) -> None:
    """
    Write metadata.json for a run atomically.
    
    Args:
        run_id: Unique run identifier
        metadata: Metadata dictionary
    """
    run_dir = get_run_dir(run_id)
    metadata_path = run_dir / METADATA_FILE
    write_json_atomic(metadata_path, metadata)


def read_metadata(run_id: str) -> Optional[Dict[str, Any]]:
    """
    Read metadata.json for a run.
    
    Args:
        run_id: Unique run identifier
        
    Returns:
        Metadata dictionary or None if not found
    """
    run_dir = get_run_dir(run_id)
    metadata_path = run_dir / METADATA_FILE
    return read_json_safe(metadata_path)


def write_status(run_id: str, status: Dict[str, Any]) -> None:
    """
    Write status.json for a run atomically.
    
    Args:
        run_id: Unique run identifier
        status: Status dictionary
    """
    run_dir = get_run_dir(run_id)
    status_path = run_dir / STATUS_FILE
    write_json_atomic(status_path, status)


def read_status(run_id: str) -> Optional[Dict[str, Any]]:
    """
    Read status.json for a run.
    
    Args:
        run_id: Unique run identifier
        
    Returns:
        Status dictionary or None if not found
    """
    run_dir = get_run_dir(run_id)
    status_path = run_dir / STATUS_FILE
    return read_json_safe(status_path)


def list_runs() -> List[str]:
    """
    List all available run IDs.
    
    Returns:
        List of run ID strings
    """
    if not RUNS_DIR.exists():
        return []
        
    return [d.name for d in RUNS_DIR.iterdir() if d.is_dir()]


def get_latest_run() -> Optional[str]:
    """
    Get the most recently created run ID.
    
    Returns:
        Latest run ID or None if no runs exist
    """
    runs = list_runs()
    if not runs:
        return None
        
    # Sort by directory creation time
    run_dirs = [(run_id, get_run_dir(run_id).stat().st_ctime) for run_id in runs]
    run_dirs.sort(key=lambda x: x[1], reverse=True)
    
    return run_dirs[0][0]


def cleanup_run(run_id: str, confirm: bool = False) -> bool:
    """
    Delete all artifacts for a run.
    
    Args:
        run_id: Unique run identifier
        confirm: If True, actually delete the files
        
    Returns:
        True if deletion was successful or would be successful
    """
    run_dir = get_run_dir(run_id)
    
    if not run_dir.exists():
        return False
        
    if confirm:
        try:
            shutil.rmtree(run_dir)
            return True
        except Exception as e:
            print(f"Failed to delete run {run_id}: {e}")
            return False
    else:
        # Dry run - just check if deletion would be possible
        return True


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