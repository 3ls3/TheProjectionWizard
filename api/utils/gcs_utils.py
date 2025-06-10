"""
GCS (Google Cloud Storage) utility functions for ML pipeline file operations.
Handles all interactions with GCS bucket for run-specific data storage.
"""

import logging
from io import BytesIO
from typing import Optional, List, Union
from pathlib import Path

from google.cloud import storage
from google.cloud.exceptions import NotFound, Forbidden
from google.api_core import exceptions as gcs_exceptions

# Configure logging
logger = logging.getLogger(__name__)

# Initialize GCS client once at module level for reuse
# This will automatically use Application Default Credentials (ADC)
try:
    _gcs_client = storage.Client()
    logger.info("GCS client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize GCS client: {str(e)}")
    _gcs_client = None


class GCSError(Exception):
    """Custom exception for GCS operations."""
    pass


def upload_to_gcs(
    bucket_name: str, 
    source_file_name_or_blob: Union[str, BytesIO], 
    destination_blob_name: str
) -> bool:
    """
    Upload a file to Google Cloud Storage.
    
    Args:
        bucket_name: Name of the GCS bucket
        source_file_name_or_blob: Either a local file path (str) or BytesIO object
        destination_blob_name: Target path in GCS bucket (e.g., "runs/run123/file.csv")
        
    Returns:
        True if upload successful, False otherwise
        
    Raises:
        GCSError: If upload fails with specific error details
    """
    if _gcs_client is None:
        logger.error("GCS client not initialized")
        raise GCSError("GCS client not available")
    
    try:
        bucket = _gcs_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        
        if isinstance(source_file_name_or_blob, (str, Path)):
            # Upload from local file path
            file_path = str(source_file_name_or_blob)
            blob.upload_from_filename(file_path)
            logger.info(f"Successfully uploaded {file_path} to gs://{bucket_name}/{destination_blob_name}")
        else:
            # Upload from BytesIO or file-like object
            source_file_name_or_blob.seek(0)  # Reset pointer to beginning
            blob.upload_from_file(source_file_name_or_blob)
            logger.info(f"Successfully uploaded file-like object to gs://{bucket_name}/{destination_blob_name}")
        
        return True
        
    except Forbidden as e:
        error_msg = f"Permission denied uploading to gs://{bucket_name}/{destination_blob_name}: {str(e)}"
        logger.error(error_msg)
        raise GCSError(error_msg)
    except Exception as e:
        error_msg = f"Failed to upload to gs://{bucket_name}/{destination_blob_name}: {str(e)}"
        logger.error(error_msg)
        raise GCSError(error_msg)


def download_from_gcs(bucket_name: str, source_blob_name: str) -> Optional[bytes]:
    """
    Download a file from Google Cloud Storage.
    
    Args:
        bucket_name: Name of the GCS bucket
        source_blob_name: Path to the file in GCS bucket
        
    Returns:
        File content as bytes if successful, None if file not found
        
    Raises:
        GCSError: If download fails with specific error details
    """
    if _gcs_client is None:
        logger.error("GCS client not initialized")
        raise GCSError("GCS client not available")
    
    try:
        bucket = _gcs_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        
        if not blob.exists():
            logger.warning(f"File not found: gs://{bucket_name}/{source_blob_name}")
            return None
            
        content = blob.download_as_bytes()
        logger.info(f"Successfully downloaded gs://{bucket_name}/{source_blob_name} ({len(content)} bytes)")
        return content
        
    except NotFound:
        logger.warning(f"File not found: gs://{bucket_name}/{source_blob_name}")
        return None
    except Forbidden as e:
        error_msg = f"Permission denied downloading gs://{bucket_name}/{source_blob_name}: {str(e)}"
        logger.error(error_msg)
        raise GCSError(error_msg)
    except Exception as e:
        error_msg = f"Failed to download gs://{bucket_name}/{source_blob_name}: {str(e)}"
        logger.error(error_msg)
        raise GCSError(error_msg)


def download_to_file(bucket_name: str, source_blob_name: str, destination_file_path: str) -> bool:
    """
    Download a file from GCS directly to a local file path.
    
    Args:
        bucket_name: Name of the GCS bucket
        source_blob_name: Path to the file in GCS bucket
        destination_file_path: Local file path to save the downloaded file
        
    Returns:
        True if download successful, False if file not found
        
    Raises:
        GCSError: If download fails with specific error details
    """
    if _gcs_client is None:
        logger.error("GCS client not initialized")
        raise GCSError("GCS client not available")
    
    try:
        bucket = _gcs_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        
        if not blob.exists():
            logger.warning(f"File not found: gs://{bucket_name}/{source_blob_name}")
            return False
            
        blob.download_to_filename(destination_file_path)
        logger.info(f"Successfully downloaded gs://{bucket_name}/{source_blob_name} to {destination_file_path}")
        return True
        
    except NotFound:
        logger.warning(f"File not found: gs://{bucket_name}/{source_blob_name}")
        return False
    except Forbidden as e:
        error_msg = f"Permission denied downloading gs://{bucket_name}/{source_blob_name}: {str(e)}"
        logger.error(error_msg)
        raise GCSError(error_msg)
    except Exception as e:
        error_msg = f"Failed to download gs://{bucket_name}/{source_blob_name} to {destination_file_path}: {str(e)}"
        logger.error(error_msg)
        raise GCSError(error_msg)


def check_gcs_file_exists(bucket_name: str, blob_name: str) -> bool:
    """
    Check if a file exists in Google Cloud Storage.
    
    Args:
        bucket_name: Name of the GCS bucket
        blob_name: Path to the file in GCS bucket
        
    Returns:
        True if file exists, False otherwise
        
    Raises:
        GCSError: If check fails due to permissions or other errors
    """
    if _gcs_client is None:
        logger.error("GCS client not initialized")
        raise GCSError("GCS client not available")
    
    try:
        bucket = _gcs_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        exists = blob.exists()
        
        logger.debug(f"File existence check for gs://{bucket_name}/{blob_name}: {exists}")
        return exists
        
    except Forbidden as e:
        error_msg = f"Permission denied checking gs://{bucket_name}/{blob_name}: {str(e)}"
        logger.error(error_msg)
        raise GCSError(error_msg)
    except Exception as e:
        error_msg = f"Failed to check existence of gs://{bucket_name}/{blob_name}: {str(e)}"
        logger.error(error_msg)
        raise GCSError(error_msg)


def list_gcs_files(bucket_name: str, prefix: str) -> List[str]:
    """
    List files in Google Cloud Storage with a given prefix.
    
    Args:
        bucket_name: Name of the GCS bucket
        prefix: Prefix to filter files (simulates directory structure)
        
    Returns:
        List of blob names (full paths) matching the prefix
        
    Raises:
        GCSError: If listing fails due to permissions or other errors
    """
    if _gcs_client is None:
        logger.error("GCS client not initialized")
        raise GCSError("GCS client not available")
    
    try:
        bucket = _gcs_client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)
        
        blob_names = [blob.name for blob in blobs]
        logger.info(f"Listed {len(blob_names)} files in gs://{bucket_name} with prefix '{prefix}'")
        
        return blob_names
        
    except Forbidden as e:
        error_msg = f"Permission denied listing gs://{bucket_name} with prefix '{prefix}': {str(e)}"
        logger.error(error_msg)
        raise GCSError(error_msg)
    except Exception as e:
        error_msg = f"Failed to list files in gs://{bucket_name} with prefix '{prefix}': {str(e)}"
        logger.error(error_msg)
        raise GCSError(error_msg)


def delete_gcs_file(bucket_name: str, blob_name: str) -> bool:
    """
    Delete a file from Google Cloud Storage.
    
    Args:
        bucket_name: Name of the GCS bucket
        blob_name: Path to the file in GCS bucket
        
    Returns:
        True if deletion successful, False if file not found
        
    Raises:
        GCSError: If deletion fails due to permissions or other errors
    """
    if _gcs_client is None:
        logger.error("GCS client not initialized")
        raise GCSError("GCS client not available")
    
    try:
        bucket = _gcs_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        if not blob.exists():
            logger.warning(f"File not found for deletion: gs://{bucket_name}/{blob_name}")
            return False
            
        blob.delete()
        logger.info(f"Successfully deleted gs://{bucket_name}/{blob_name}")
        return True
        
    except NotFound:
        logger.warning(f"File not found for deletion: gs://{bucket_name}/{blob_name}")
        return False
    except Forbidden as e:
        error_msg = f"Permission denied deleting gs://{bucket_name}/{blob_name}: {str(e)}"
        logger.error(error_msg)
        raise GCSError(error_msg)
    except Exception as e:
        error_msg = f"Failed to delete gs://{bucket_name}/{blob_name}: {str(e)}"
        logger.error(error_msg)
        raise GCSError(error_msg)


# Convenience functions for the specific bucket used in this project
PROJECT_BUCKET_NAME = "projection-wizard-runs-mvp-w23"


def upload_run_file(run_id: str, file_name: str, source: Union[str, BytesIO]) -> bool:
    """
    Convenience function to upload a file to a specific run directory in our project bucket.
    
    Args:
        run_id: The run ID
        file_name: Name of the file (e.g., "original_data.csv", "metadata.json")
        source: Source file path or BytesIO object
        
    Returns:
        True if upload successful, False otherwise
    """
    destination_path = f"runs/{run_id}/{file_name}"
    return upload_to_gcs(PROJECT_BUCKET_NAME, source, destination_path)


def download_run_file(run_id: str, file_name: str) -> Optional[bytes]:
    """
    Convenience function to download a file from a specific run directory in our project bucket.
    
    Args:
        run_id: The run ID
        file_name: Name of the file to download
        
    Returns:
        File content as bytes if successful, None if not found
    """
    source_path = f"runs/{run_id}/{file_name}"
    return download_from_gcs(PROJECT_BUCKET_NAME, source_path)


def check_run_file_exists(run_id: str, file_name: str) -> bool:
    """
    Convenience function to check if a file exists in a specific run directory.
    
    Args:
        run_id: The run ID
        file_name: Name of the file to check
        
    Returns:
        True if file exists, False otherwise
    """
    file_path = f"runs/{run_id}/{file_name}"
    return check_gcs_file_exists(PROJECT_BUCKET_NAME, file_path)


def list_run_files(run_id: str) -> List[str]:
    """
    Convenience function to list all files in a specific run directory.
    
    Args:
        run_id: The run ID
        
    Returns:
        List of file names in the run directory
    """
    prefix = f"runs/{run_id}/"
    full_paths = list_gcs_files(PROJECT_BUCKET_NAME, prefix)
    
    # Extract just the file names (remove the run directory prefix)
    file_names = []
    for path in full_paths:
        if path.startswith(prefix):
            file_name = path[len(prefix):]
            if file_name:  # Skip empty strings (directory entries)
                file_names.append(file_name)
    
    return file_names 