"""
Core ingestion logic for The Projection Wizard.
Handles confirmation of uploaded data in GCS and run state initialization.
This step assumes data has already been uploaded to GCS by the /api/upload endpoint.
"""

import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, BinaryIO, Union
import io
import json
import logging
import tempfile

# Import common modules
from common import constants, schemas
from api.utils.gcs_utils import (
    download_from_gcs, upload_to_gcs, check_gcs_file_exists,
    PROJECT_BUCKET_NAME, download_run_file, upload_run_file, check_run_file_exists
)

# Configure logging for this module
logger = logging.getLogger(__name__)


def run_gcs_ingestion(run_id: str, gcs_bucket_name: str = PROJECT_BUCKET_NAME) -> bool:
    """
    Process data ingestion from GCS for a given run_id.
    This function assumes data has already been uploaded to GCS by the /api/upload endpoint.
    
    This step:
    1. Updates status to "ingestion processing"
    2. Confirms original_data.csv exists in GCS
    3. Performs basic validation by downloading and reading CSV
    4. Updates metadata.json with ingestion completion timestamp
    5. Updates status to "completed" and ready for next step
    
    Args:
        run_id: Unique run identifier
        gcs_bucket_name: GCS bucket name (defaults to PROJECT_BUCKET_NAME)
        
    Returns:
        True if ingestion successful, False otherwise
        
    Raises:
        Exception: If critical errors occur during ingestion
    """
    logger.info(f"Starting GCS ingestion for run_id: {run_id}")
    
    try:
        # Step 1: Update status to "ingestion started"
        if not _update_status_to_processing(run_id, gcs_bucket_name):
            logger.error(f"Failed to update status to processing for run_id: {run_id}")
            return False
        
        # Step 2: Confirm original_data.csv exists in GCS
        metadata = _download_and_validate_metadata(run_id, gcs_bucket_name)
        if not metadata:
            logger.error(f"Failed to download or validate metadata for run_id: {run_id}")
            _update_status_to_failed(run_id, gcs_bucket_name, "Failed to download or validate metadata")
            return False
        
        # Step 3: Confirm original_data.csv exists and perform basic validation
        csv_validation_result = _validate_original_csv(run_id, gcs_bucket_name, metadata)
        if not csv_validation_result['success']:
            logger.error(f"CSV validation failed for run_id: {run_id} - {csv_validation_result['error']}")
            _update_status_to_failed(run_id, gcs_bucket_name, csv_validation_result['error'])
            return False
        
        # Step 4: Update metadata with ingestion completion and any updated stats
        if not _update_metadata_with_ingestion_completion(run_id, gcs_bucket_name, metadata, csv_validation_result):
            logger.error(f"Failed to update metadata with ingestion completion for run_id: {run_id}")
            _update_status_to_failed(run_id, gcs_bucket_name, "Failed to update metadata with ingestion completion")
            return False
        
        # Step 5: Update status to completed
        if not _update_status_to_completed(run_id, gcs_bucket_name):
            logger.error(f"Failed to update status to completed for run_id: {run_id}")
            return False
        
        logger.info(f"GCS ingestion completed successfully for run_id: {run_id}")
        return True
        
    except Exception as e:
        logger.error(f"Critical error during GCS ingestion for run_id {run_id}: {str(e)}", exc_info=True)
        _update_status_to_failed(run_id, gcs_bucket_name, f"Critical error: {str(e)}")
        raise


def _update_status_to_processing(run_id: str, gcs_bucket_name: str) -> bool:
    """Update status.json to indicate ingestion processing has started."""
    try:
        # Download current status.json
        status_bytes = download_run_file(run_id, constants.STATUS_FILE)
        if not status_bytes:
            logger.error(f"Could not download status.json for run_id: {run_id}")
            return False
        
        # Parse current status
        current_status = json.loads(status_bytes.decode('utf-8'))
        
        # Update status fields
        current_status.update({
            'current_stage': constants.INGEST_STAGE,
            'current_stage_name': constants.STAGE_DISPLAY_NAMES[constants.INGEST_STAGE],
            'status': 'processing',
            'message': 'Starting data ingestion...',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'last_updated': datetime.now(timezone.utc).isoformat()
        })
        
        # Upload updated status back to GCS
        status_json = json.dumps(current_status, indent=2).encode('utf-8')
        status_io = io.BytesIO(status_json)
        
        success = upload_run_file(run_id, constants.STATUS_FILE, status_io)
        if success:
            logger.info(f"Updated status to processing for run_id: {run_id}")
        return success
        
    except Exception as e:
        logger.error(f"Failed to update status to processing for run_id {run_id}: {str(e)}")
        return False


def _download_and_validate_metadata(run_id: str, gcs_bucket_name: str) -> Optional[dict]:
    """Download and validate metadata.json from GCS."""
    try:
        # Download metadata.json
        metadata_bytes = download_run_file(run_id, constants.METADATA_FILE)
        if not metadata_bytes:
            logger.error(f"Could not download metadata.json for run_id: {run_id}")
            return None
        
        # Parse metadata
        metadata = json.loads(metadata_bytes.decode('utf-8'))
        
        # Validate required fields
        if 'run_id' not in metadata or metadata['run_id'] != run_id:
            logger.error(f"Metadata run_id mismatch for run_id: {run_id}")
            return None
        
        if 'original_filename' not in metadata:
            logger.error(f"Metadata missing original_filename for run_id: {run_id}")
            return None
        
        logger.info(f"Successfully downloaded and validated metadata for run_id: {run_id}")
        return metadata
        
    except Exception as e:
        logger.error(f"Failed to download/validate metadata for run_id {run_id}: {str(e)}")
        return None


def _validate_original_csv(run_id: str, gcs_bucket_name: str, metadata: dict) -> dict:
    """Validate that original_data.csv exists and can be read from GCS."""
    try:
        # Check if original_data.csv exists in GCS
        csv_exists = check_run_file_exists(run_id, constants.ORIGINAL_DATA_FILE)
        if not csv_exists:
            return {
                'success': False,
                'error': f"Original data CSV not found in GCS at runs/{run_id}/{constants.ORIGINAL_DATA_FILE}"
            }
        
        logger.info(f"Confirmed original_data.csv exists in GCS for run_id: {run_id}")
        
        # Download and validate CSV can be read
        csv_bytes = download_run_file(run_id, constants.ORIGINAL_DATA_FILE)
        if not csv_bytes:
            return {
                'success': False,
                'error': "Failed to download original_data.csv from GCS"
            }
        
        # Try to read CSV with pandas to validate it's a proper CSV
        try:
            df = pd.read_csv(io.BytesIO(csv_bytes))
            
            # Extract basic statistics
            initial_rows = df.shape[0]
            initial_cols = df.shape[1]
            initial_dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
            memory_usage_mb = df.memory_usage(deep=True).sum() / (1024**2)
            missing_values_total = df.isnull().sum().sum()
            
            logger.info(f"CSV validation successful for run_id {run_id}: {initial_rows} rows, {initial_cols} columns")
            
            return {
                'success': True,
                'stats': {
                    'initial_rows': int(initial_rows),
                    'initial_cols': int(initial_cols),
                    'initial_dtypes': initial_dtypes,
                    'memory_usage_mb': float(memory_usage_mb),
                    'missing_values_total': int(missing_values_total),
                    'column_names': list(df.columns)
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Failed to parse CSV with pandas: {str(e)}"
            }
            
    except Exception as e:
        logger.error(f"Error validating CSV for run_id {run_id}: {str(e)}")
        return {
            'success': False,
            'error': f"Unexpected error during CSV validation: {str(e)}"
        }


def _update_metadata_with_ingestion_completion(run_id: str, gcs_bucket_name: str, 
                                               metadata: dict, validation_result: dict) -> bool:
    """Update metadata.json with ingestion completion timestamp and validated stats."""
    try:
        # Add ingestion completion timestamp
        metadata['ingestion_completed_at'] = datetime.now(timezone.utc).isoformat()
        
        # Update stats if validation was successful and provided new data
        if validation_result.get('success') and 'stats' in validation_result:
            stats = validation_result['stats']
            # Update any stats that might have been estimated during upload
            metadata['initial_rows'] = stats['initial_rows']
            metadata['initial_cols'] = stats['initial_cols']
            metadata['initial_dtypes'] = stats['initial_dtypes']
            
            # Add new stats if not already present
            if 'memory_usage_mb' not in metadata:
                metadata['memory_usage_mb'] = stats['memory_usage_mb']
            if 'missing_values_total' not in metadata:
                metadata['missing_values_total'] = stats['missing_values_total']
            if 'column_names' not in metadata:
                metadata['column_names'] = stats['column_names']
        
        # Upload updated metadata back to GCS
        metadata_json = json.dumps(metadata, indent=2, default=str).encode('utf-8')
        metadata_io = io.BytesIO(metadata_json)
        
        success = upload_run_file(run_id, constants.METADATA_FILE, metadata_io)
        if success:
            logger.info(f"Updated metadata with ingestion completion for run_id: {run_id}")
        return success
        
    except Exception as e:
        logger.error(f"Failed to update metadata with ingestion completion for run_id {run_id}: {str(e)}")
        return False


def _update_status_to_completed(run_id: str, gcs_bucket_name: str) -> bool:
    """Update status.json to indicate ingestion completed successfully."""
    try:
        # Download current status.json
        status_bytes = download_run_file(run_id, constants.STATUS_FILE)
        if not status_bytes:
            logger.error(f"Could not download status.json for run_id: {run_id}")
            return False
        
        # Parse current status
        current_status = json.loads(status_bytes.decode('utf-8'))
        
        # Update status fields
        current_status.update({
            'status': 'completed',
            'message': 'Data ingestion completed successfully. Ready for schema definition.',
            'next_stage': constants.SCHEMA_STAGE,
            'next_stage_name': constants.STAGE_DISPLAY_NAMES[constants.SCHEMA_STAGE],
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'last_updated': datetime.now(timezone.utc).isoformat()
        })
        
        # Upload updated status back to GCS
        status_json = json.dumps(current_status, indent=2).encode('utf-8')
        status_io = io.BytesIO(status_json)
        
        success = upload_run_file(run_id, constants.STATUS_FILE, status_io)
        if success:
            logger.info(f"Updated status to completed for run_id: {run_id}")
        return success
        
    except Exception as e:
        logger.error(f"Failed to update status to completed for run_id {run_id}: {str(e)}")
        return False


def _update_status_to_failed(run_id: str, gcs_bucket_name: str, error_message: str) -> bool:
    """Update status.json to indicate ingestion failed."""
    try:
        # Try to download current status.json, create minimal if not available
        status_bytes = download_run_file(run_id, constants.STATUS_FILE)
        if status_bytes:
            current_status = json.loads(status_bytes.decode('utf-8'))
        else:
            logger.warning(f"Could not download status.json for run_id: {run_id}, creating minimal status")
            current_status = {
                'run_id': run_id,
                'current_stage': constants.INGEST_STAGE
            }
        
        # Update status fields
        current_status.update({
            'status': 'failed',
            'message': f'Data ingestion failed: {error_message}',
            'error_details': error_message,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'last_updated': datetime.now(timezone.utc).isoformat()
        })
        
        # Upload updated status back to GCS
        status_json = json.dumps(current_status, indent=2).encode('utf-8')
        status_io = io.BytesIO(status_json)
        
        success = upload_run_file(run_id, constants.STATUS_FILE, status_io)
        if success:
            logger.info(f"Updated status to failed for run_id: {run_id}")
        return success
        
    except Exception as e:
        logger.error(f"Failed to update status to failed for run_id {run_id}: {str(e)}")
        return False


# Legacy function maintained for backward compatibility
def run_ingestion(uploaded_file_object: Union[BinaryIO, object], base_runs_path_str: str) -> str:
    """
    Legacy ingestion function maintained for backward compatibility.
    This function is deprecated in favor of run_gcs_ingestion().
    
    For new implementations, use run_gcs_ingestion() which assumes
    data is already uploaded to GCS by the /api/upload endpoint.
    """
    logger.warning("run_ingestion() is deprecated. Use run_gcs_ingestion() for GCS-based workflows.")
    
    # This legacy implementation would need to be adapted if still needed
    # For now, raising an exception to encourage migration to GCS workflow
    raise NotImplementedError(
        "Legacy local file ingestion is no longer supported. "
        "Use the /api/upload endpoint followed by run_gcs_ingestion()."
    ) 