# GCS (Google Cloud Storage) Integration

This module provides utilities for interacting with Google Cloud Storage for the ML pipeline backend.

## Overview

The GCS utilities handle all file operations for ML pipeline runs, including:
- Uploading CSV files and metadata
- Downloading processed data and results
- Checking file existence
- Listing files in run directories
- Deleting files when needed

## Files

- `gcs_utils.py` - Core GCS utility functions
- `example_gcs_integration.py` - Example FastAPI endpoints using GCS utilities

## Configuration

- **GCS Bucket**: `projection-wizard-runs-mvp-w23` (Europe-West1)
- **Authentication**: Uses Application Default Credentials (ADC)
- **File Structure**: `runs/{run_id}/{filename}`

## Core Functions

### Generic Functions
- `upload_to_gcs(bucket_name, source, destination)` - Upload file to GCS
- `download_from_gcs(bucket_name, source)` - Download file from GCS 
- `check_gcs_file_exists(bucket_name, path)` - Check if file exists
- `list_gcs_files(bucket_name, prefix)` - List files with prefix
- `delete_gcs_file(bucket_name, path)` - Delete file from GCS

### Convenience Functions (Pre-configured for project bucket)
- `upload_run_file(run_id, filename, source)` - Upload to run directory
- `download_run_file(run_id, filename)` - Download from run directory
- `check_run_file_exists(run_id, filename)` - Check file in run directory
- `list_run_files(run_id)` - List files in run directory

## Usage Examples

### Upload a CSV file
```python
from api.utils.gcs_utils import upload_run_file
from io import BytesIO

# Upload from BytesIO
file_content = BytesIO(csv_data.encode())
success = upload_run_file("run123", "original_data.csv", file_content)

# Upload from file path
success = upload_run_file("run123", "original_data.csv", "/path/to/file.csv")
```

### Download and process metadata
```python
from api.utils.gcs_utils import download_run_file
import json

# Download metadata
metadata_bytes = download_run_file("run123", "metadata.json")
if metadata_bytes:
    metadata = json.loads(metadata_bytes.decode('utf-8'))
```

### Check run status
```python
from api.utils.gcs_utils import check_run_file_exists

# Check if required files exist
has_data = check_run_file_exists("run123", "original_data.csv")
has_metadata = check_run_file_exists("run123", "metadata.json")
has_results = check_run_file_exists("run123", "results.json")
```

## Error Handling

All functions raise `GCSError` for GCS-specific issues:
- Permission denied (403)
- File not found (404)
- Network or service errors

Example error handling:
```python
from api.utils.gcs_utils import GCSError, upload_run_file

try:
    upload_run_file("run123", "data.csv", file_content)
except GCSError as e:
    logger.error(f"GCS operation failed: {e}")
    # Handle error appropriately
```

## Authentication Setup

For local development:
```bash
# Unset any service account credentials
unset GOOGLE_APPLICATION_CREDENTIALS

# Login with your user account
gcloud auth application-default login
```

For production (Cloud Run), the service account attached to the Cloud Run service must have the `Storage Object Admin` role on the bucket.

## Testing

The integration was tested successfully with:
- File upload (BytesIO and file path)
- File download  
- File existence checking
- File listing
- File deletion
- Error handling

All operations work correctly with the configured bucket and permissions. 