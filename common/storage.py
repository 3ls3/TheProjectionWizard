"""
Storage utilities for The Projection Wizard.
Provides run_id-centric file operations with GCS backend for cloud-native operation.

=============================================================================
⚠️  PARALLEL DEVELOPMENT COORDINATION REQUIRED ⚠️
=============================================================================

This file provides the CORE storage interface used by all components:
- API developer: Uses for file upload/download, run management
- Pipeline developer: Uses for reading/writing stage artifacts
- Testing developer: Uses for test fixture management

COLLABORATION PROTOCOL:
1. 🗣️  ANNOUNCE in Slack: "Need to add storage function for [feature]"
2. ⏳ COORDINATE with team - storage changes affect everyone!
3. 📝 ADD new functions without modifying existing ones
4. 🔄 FOLLOW existing patterns for consistency
5. ✅ TEST with existing pipeline to ensure no breaks
6. 📢 NOTIFY team: "Added new storage functions - available for use"

SAFE PATTERNS:
✅ Add new specialized read/write functions
✅ Follow atomic writing pattern for critical files
✅ Use existing error handling patterns
✅ Add utility functions that complement existing ones

DANGEROUS PATTERNS:
❌ Changing function signatures (breaks all callers)
❌ Modifying file naming conventions (breaks existing runs)
❌ Changing atomic write behavior (can cause corruption)
❌ Removing existing functions (breaks imports)

EXAMPLE SAFE ADDITION:
```python
def read_api_cache(run_id: str) -> Optional[Dict]:
    '''Read API-specific cache data.'''
    return read_json(run_id, "api_cache.json")

def write_api_response(run_id: str, endpoint: str, data: Dict) -> None:
    '''Write API response data with atomic operations.'''
    filename = f"api_response_{endpoint}.json"
    write_json_atomic(run_id, filename, data)
```

If changing core storage behavior, discuss in #projection-wizard Slack first!

=============================================================================
⚠️  GCS BACKEND REFACTORING - CLOUD-NATIVE OPERATION ⚠️
=============================================================================

This storage module has been refactored to use Google Cloud Storage (GCS) as the backend
instead of local filesystem operations. This enables cloud-native operation in environments
like Google Cloud Run where local filesystem is ephemeral.

GCS Storage Structure:
- Bucket: projection-wizard-runs-mvp-w23
- Object paths: runs/{run_id}/{filename}
- Examples: runs/abc123/metadata.json, runs/abc123/original_data.csv

Key Changes:
- write_json_atomic() now uploads JSON directly to GCS (inherently atomic)
- read_json() downloads and parses JSON from GCS
- Data file functions use GCS download with BytesIO for pandas
- list_runs() queries GCS bucket instead of local directories
- Local filesystem functions (get_run_dir, etc.) are deprecated for GCS operations

Backward Compatibility:
- Function signatures remain unchanged for existing pipeline code
- Error handling patterns preserved
- Same return types and behaviors expected

=============================================================================
"""

import json
import tempfile
import shutil
import csv
import io
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

# GCS imports
from api.utils.gcs_utils import (
    upload_run_file, 
    download_run_file, 
    check_run_file_exists, 
    list_gcs_files,
    PROJECT_BUCKET_NAME,
    GCSError
)
from api.utils.io_helpers import list_runs_gcs

from .constants import DATA_DIR_NAME, RUNS_DIR_NAME, RUN_INDEX_FILENAME
from .schemas import RunIndexEntry


# =============================================================================
# CORE GCS-BACKED I/O FUNCTIONS
# =============================================================================

def write_json_atomic(run_id: str, filename: str, data: dict) -> None:
    """
    Write JSON data to GCS atomically (GCS uploads are inherently atomic).
    Uses the GCS backend instead of local filesystem.
    
    Args:
        run_id: Unique run identifier
        filename: Name of the JSON file (e.g., 'metadata.json')
        data: Data to write as JSON
        
    Raises:
        IOError: If GCS upload fails
    """
    try:
        # Convert data to JSON bytes
        json_str = json.dumps(data, indent=2, default=str)
        json_bytes = json_str.encode('utf-8')
        
        # Upload to GCS using BytesIO
        json_buffer = io.BytesIO(json_bytes)
        success = upload_run_file(run_id, filename, json_buffer)
        
        if not success:
            raise IOError(f"Failed to upload {filename} to GCS for run {run_id}")
            
    except GCSError as e:
        raise IOError(f"GCS error writing {filename} for run {run_id}: {str(e)}")
    except Exception as e:
        raise IOError(f"Failed to write {filename} for run {run_id}: {str(e)}")


def read_json(run_id: str, filename: str) -> Optional[dict]:
    """
    Read JSON data from GCS with error handling.
    Uses the GCS backend instead of local filesystem.
    
    Args:
        run_id: Unique run identifier
        filename: Name of the JSON file (e.g., 'metadata.json')
        
    Returns:
        JSON data as dictionary, or None if file doesn't exist
        
    Raises:
        IOError: If file exists but cannot be read or parsed
    """
    try:
        # Download from GCS
        file_bytes = download_run_file(run_id, filename)
        
        if file_bytes is None:
            return None
            
        # Decode and parse JSON
        json_str = file_bytes.decode('utf-8')
        return json.loads(json_str)
        
    except json.JSONDecodeError as e:
        raise IOError(f"Invalid JSON in {filename} for run {run_id}: {str(e)}")
    except GCSError as e:
        raise IOError(f"GCS error reading {filename} for run {run_id}: {str(e)}")
    except Exception as e:
        raise IOError(f"Failed to read {filename} for run {run_id}: {str(e)}")


def read_original_data(run_id: str) -> Optional["pd.DataFrame"]:
    """
    Read the original data CSV file for a run from GCS.
    
    Args:
        run_id: Unique run identifier
        
    Returns:
        pandas DataFrame with the original data, or None if file doesn't exist
        
    Raises:
        IOError: If file exists but cannot be read or parsed
    """
    from .constants import ORIGINAL_DATA_FILE
    
    try:
        # Download CSV from GCS
        csv_bytes = download_run_file(run_id, ORIGINAL_DATA_FILE)
        
        if csv_bytes is None:
            return None
            
        # Load CSV from bytes using BytesIO
        csv_buffer = io.BytesIO(csv_bytes)
        return pd.read_csv(csv_buffer)
        
    except GCSError as e:
        raise IOError(f"GCS error reading original data for run {run_id}: {str(e)}")
    except Exception as e:
        raise IOError(f"Failed to read original data for run {run_id}: {str(e)}")


def read_cleaned_data(run_id: str) -> Optional["pd.DataFrame"]:
    """
    Read the cleaned data CSV file for a run from GCS.
    
    Args:
        run_id: Unique run identifier
        
    Returns:
        pandas DataFrame with the cleaned data, or None if file doesn't exist
        
    Raises:
        IOError: If file exists but cannot be read or parsed
    """
    from .constants import CLEANED_DATA_FILE
    
    try:
        # Download CSV from GCS
        csv_bytes = download_run_file(run_id, CLEANED_DATA_FILE)
        
        if csv_bytes is None:
            return None
            
        # Load CSV from bytes using BytesIO
        csv_buffer = io.BytesIO(csv_bytes)
        return pd.read_csv(csv_buffer)
        
    except GCSError as e:
        raise IOError(f"GCS error reading cleaned data for run {run_id}: {str(e)}")
    except Exception as e:
        raise IOError(f"Failed to read cleaned data for run {run_id}: {str(e)}")


def list_runs() -> List[str]:
    """
    List all available run IDs from GCS bucket.
    Uses GCS backend instead of local directory scanning.
    
    Returns:
        List of run ID strings
    """
    try:
        return list_runs_gcs(PROJECT_BUCKET_NAME)
    except Exception as e:
        # Return empty list if GCS listing fails rather than raising
        # This maintains backward compatibility with existing code
        return []


# =============================================================================
# CONVENIENCE FUNCTIONS (unchanged signatures)
# =============================================================================

def write_metadata(run_id: str, metadata: dict) -> None:
    """Write metadata.json for a run to GCS."""
    write_json_atomic(run_id, "metadata.json", metadata)


def read_metadata(run_id: str) -> Optional[dict]:
    """Read metadata.json for a run from GCS."""
    return read_json(run_id, "metadata.json")


def write_status(run_id: str, status: dict) -> None:
    """Write status.json for a run to GCS."""
    write_json_atomic(run_id, "status.json", status)


def read_status(run_id: str) -> Optional[dict]:
    """Read status.json for a run from GCS."""
    return read_json(run_id, "status.json")


# =============================================================================
# DEPRECATED/LOCAL-ONLY FUNCTIONS
# =============================================================================

def get_run_dir(run_id: str) -> Path:
    """
    ⚠️  DEPRECATED FOR GCS OPERATIONS ⚠️
    
    Helper function to construct local run directory path.
    
    WARNING: This function creates LOCAL directories and is NOT compatible
    with GCS-based storage. It should only be used for:
    - Legacy local-only operations
    - Temporary file operations that need local paths
    - Testing with local fixtures
    
    For GCS operations, use the string-based object names directly:
    - GCS object path: f"runs/{run_id}/{filename}"
    - Use upload_run_file(), download_run_file(), etc. instead
    
    Args:
        run_id: Unique run identifier
        
    Returns:
        Path to the local run directory (creates if necessary)
    """
    run_dir = Path(DATA_DIR_NAME) / RUNS_DIR_NAME / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create standard subdirectories for local operations
    (run_dir / "model").mkdir(exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)
    
    return run_dir


def get_run_dir_path(run_id: str) -> Path:
    """
    ⚠️  DEPRECATED FOR GCS OPERATIONS ⚠️
    
    Alias for get_run_dir() - see deprecation warning above.
    """
    return get_run_dir(run_id)


def get_artifact_path(run_id: str, artifact_name: str) -> Path:
    """
    ⚠️  DEPRECATED FOR GCS OPERATIONS ⚠️
    
    Get the local path for a run artifact.
    
    WARNING: This returns a LOCAL Path object and should not be used
    for GCS operations. For GCS, use the string-based object names:
    - GCS object name: f"runs/{run_id}/{artifact_name}"
    - Use check_run_file_exists(), download_run_file(), etc.
    
    Args:
        run_id: Unique run identifier
        artifact_name: Name of the artifact file
        
    Returns:
        Local path to the artifact (deprecated for GCS)
    """
    return get_run_dir(run_id) / artifact_name


def append_to_run_index(run_entry_data: dict) -> None:
    """
    ⚠️  LOCAL-ONLY FUNCTION - NOT GCS COMPATIBLE ⚠️
    
    Append a new entry to the local run index CSV file.
    
    WARNING: This function maintains a LOCAL CSV index file which is
    incompatible with cloud-native GCS storage. Consider whether this
    local index is still needed, since run listing can be done via:
    - list_runs() -> queries GCS directly
    - list_runs_gcs() -> comprehensive GCS-based run discovery
    
    This function is preserved for backward compatibility but should
    be evaluated for removal in favor of GCS-based run discovery.
    
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


# =============================================================================
# GCS UTILITY FUNCTIONS
# =============================================================================

def check_run_exists(run_id: str) -> bool:
    """
    Check if a run exists in GCS by looking for metadata.json.
    
    Args:
        run_id: Unique run identifier
        
    Returns:
        True if run exists (has metadata.json), False otherwise
    """
    try:
        return check_run_file_exists(run_id, "metadata.json")
    except Exception:
        return False


def get_run_files(run_id: str) -> List[str]:
    """
    Get list of all files for a specific run from GCS.
    
    Args:
        run_id: Unique run identifier
        
    Returns:
        List of filenames in the run directory
    """
    try:
        # Use existing GCS function to list files
        from api.utils.gcs_utils import list_run_files
        return list_run_files(run_id)
    except Exception:
        return []


def delete_run_file(run_id: str, filename: str) -> bool:
    """
    Delete a specific file for a run from GCS.
    
    Args:
        run_id: Unique run identifier
        filename: Name of file to delete
        
    Returns:
        True if deletion successful, False otherwise
    """
    try:
        from api.utils.gcs_utils import delete_gcs_file
        object_path = f"runs/{run_id}/{filename}"
        return delete_gcs_file(PROJECT_BUCKET_NAME, object_path)
    except Exception:
        return False 