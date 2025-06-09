"""
Core ingestion logic for The Projection Wizard.
Handles CSV file upload, run initialization, and metadata creation.
"""

import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, BinaryIO, Union
import io

# Import common modules
from common import constants, schemas, utils, storage, logger


def run_ingestion(uploaded_file_object: Union[BinaryIO, object], base_runs_path_str: str) -> str:
    """
    Process uploaded CSV file and initialize a new pipeline run.
    
    Args:
        uploaded_file_object: File object from Streamlit's st.file_uploader or similar
        base_runs_path_str: Base directory path for runs (e.g., "data/runs")
        
    Returns:
        Generated run_id string
        
    Raises:
        Exception: If critical errors occur during ingestion
    """
    # Generate run_id
    run_id = utils.generate_run_id()
    
    # Setup logging for this run
    logger_instance = logger.get_stage_logger(run_id=run_id, stage=constants.INGEST_STAGE)
    structured_log = logger.get_stage_structured_logger(run_id=run_id, stage=constants.INGEST_STAGE)
    
    logger_instance.info("Starting data ingestion process")
    
    # Structured log: Ingestion started
    logger.log_structured_event(
        structured_log,
        "ingestion_started",
        {
            "run_id": run_id,
            "stage": constants.INGEST_STAGE,
            "base_runs_path": base_runs_path_str
        },
        f"Data ingestion started for run {run_id}"
    )
    
    # Initialize variables for error handling
    initial_rows = None
    initial_cols = None 
    initial_dtypes = None
    csv_read_successful = False
    errors_list = []
    original_filename = "unknown_file.csv"
    
    try:
        # Construct and create run directory
        run_dir_path = storage.get_run_dir(run_id)
        logger_instance.info(f"Created run directory: {run_dir_path}")
        
        # Structured log: Run directory created
        logger.log_structured_event(
            structured_log,
            "run_directory_created",
            {
                "run_directory": str(run_dir_path),
                "run_id": run_id
            },
            f"Run directory created: {run_dir_path}"
        )
        
        # Save original data
        original_data_path = run_dir_path / constants.ORIGINAL_DATA_FILE
        logger_instance.info(f"Saving original data to: {original_data_path}")
        
        # Handle different file object types
        try:
            # Check if it's a Streamlit UploadedFile or similar
            if hasattr(uploaded_file_object, 'getvalue'):
                # Streamlit UploadedFile
                file_content = uploaded_file_object.getvalue()
                original_filename = getattr(uploaded_file_object, 'name', 'uploaded_file.csv')
            elif hasattr(uploaded_file_object, 'read'):
                # Standard file object
                file_content = uploaded_file_object.read()
                original_filename = getattr(uploaded_file_object, 'name', 'uploaded_file.csv')
                # Reset file pointer if possible
                if hasattr(uploaded_file_object, 'seek'):
                    uploaded_file_object.seek(0)
            else:
                raise ValueError("Unsupported file object type")
            
            # Write file content
            with open(original_data_path, 'wb') as f:
                f.write(file_content)
            
            logger_instance.info("Successfully saved original data file")
            
            # Structured log: File saved
            logger.log_structured_event(
                structured_log,
                "file_saved",
                {
                    "original_filename": original_filename,
                    "saved_path": constants.ORIGINAL_DATA_FILE,
                    "file_size_bytes": len(file_content)
                },
                f"Original data file saved: {original_filename}"
            )
            
        except Exception as e:
            error_msg = f"Failed to save original data: {str(e)}"
            logger_instance.error(error_msg)
            errors_list.append(error_msg)
            
            # Structured log: File save failed
            logger.log_structured_error(
                structured_log,
                "file_save_failed",
                error_msg,
                {"stage": constants.INGEST_STAGE, "original_filename": original_filename}
            )
            
            # Continue with limited metadata creation
            original_filename = "upload_failed.csv"
        
        # Read basic data statistics (with error handling)
        logger_instance.info("Attempting to read and analyze CSV data")
        
        try:
            # Try to read the CSV file
            df = pd.read_csv(original_data_path)
            
            # Extract basic statistics
            initial_rows = df.shape[0]
            initial_cols = df.shape[1]
            initial_dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
            
            csv_read_successful = True
            logger_instance.info(f"Successfully read CSV: {initial_rows} rows, {initial_cols} columns")
            
            # Log basic data info
            logger_instance.info(f"Column dtypes: {initial_dtypes}")
            
            # Structured log: CSV parsed successfully
            logger.log_structured_event(
                structured_log,
                "csv_parsed_successfully",
                {
                    "rows": int(initial_rows),
                    "columns": int(initial_cols),
                    "column_names": list(df.columns),
                    "dtypes": initial_dtypes,
                    "memory_usage_mb": float(df.memory_usage(deep=True).sum() / 1024**2),
                    "missing_values_total": int(df.isnull().sum().sum())
                },
                f"CSV parsed successfully: {initial_rows} rows × {initial_cols} columns"
            )
            
            # Log data quality metrics as structured metrics (converting numpy types to Python types)
            logger.log_structured_metric(
                structured_log,
                "dataset_rows",
                int(initial_rows),
                "data_quality",
                {"dataset_columns": int(initial_cols)}
            )
            
            logger.log_structured_metric(
                structured_log,
                "dataset_columns",
                int(initial_cols),
                "data_quality",
                {"dataset_rows": int(initial_rows)}
            )
            
            total_missing = int(df.isnull().sum().sum())
            total_cells = int(initial_rows * initial_cols)
            missing_percentage = float((total_missing / total_cells) * 100)
            logger.log_structured_metric(
                structured_log,
                "missing_values_percentage",
                missing_percentage,
                "data_quality",
                {"total_missing": total_missing, "total_cells": total_cells}
            )
            
        except Exception as e:
            error_msg = f"Failed to read or parse uploaded CSV: {str(e)}"
            logger_instance.error(error_msg)
            errors_list.append(error_msg)
            csv_read_successful = False
            
            # Structured log: CSV parsing failed
            logger.log_structured_error(
                structured_log,
                "csv_parsing_failed",
                error_msg,
                {"stage": constants.INGEST_STAGE, "file": original_filename}
            )
            
            # Set fallback values
            initial_rows = None
            initial_cols = None
            initial_dtypes = None
            
        # Create initial metadata.json
        logger_instance.info("Creating initial metadata")
        
        try:
            metadata_model = schemas.BaseMetadata(
                run_id=run_id,
                timestamp=datetime.now(timezone.utc),
                original_filename=original_filename,
                initial_rows=initial_rows,
                initial_cols=initial_cols,
                initial_dtypes=initial_dtypes
            )
            
            # Convert to dictionary for JSON serialization
            metadata_dict = metadata_model.model_dump(mode='json')
            
            # Write metadata atomically
            storage.write_json_atomic(
                run_id=run_id,
                filename=constants.METADATA_FILE,
                data=metadata_dict
            )
            
            logger_instance.info("Successfully created metadata.json")
            
            # Structured log: Metadata created
            logger.log_structured_event(
                structured_log,
                "metadata_created",
                {
                    "file": constants.METADATA_FILE,
                    "run_id": run_id,
                    "has_data_stats": csv_read_successful,
                    "metadata_keys": list(metadata_dict.keys())
                },
                f"Initial metadata created: {constants.METADATA_FILE}"
            )
            
        except Exception as e:
            error_msg = f"Failed to create metadata.json: {str(e)}"
            logger_instance.error(error_msg)
            errors_list.append(error_msg)
            
            # Structured log: Metadata creation failed
            logger.log_structured_error(
                structured_log,
                "metadata_creation_failed",
                error_msg,
                {"stage": constants.INGEST_STAGE}
            )
        
        # Create initial status.json
        logger_instance.info("Creating initial status")
        
        try:
            # Determine status based on CSV read success
            if csv_read_successful and len(errors_list) == 0:
                status_val = 'completed'
                message_val = 'Ingestion successful.'
            elif csv_read_successful and len(errors_list) > 0:
                status_val = 'completed'
                message_val = 'Ingestion completed with warnings.'
            else:
                status_val = 'failed'
                message_val = 'Failed to read or parse uploaded CSV.'
            
            status_model = schemas.StageStatus(
                stage=constants.INGEST_STAGE,
                status=status_val,
                message=message_val,
                errors=errors_list if errors_list else None
            )
            
            # Convert to dictionary for JSON serialization
            status_dict = status_model.model_dump(mode='json')
            
            # Write status atomically
            storage.write_json_atomic(
                run_id=run_id,
                filename=constants.STATUS_FILE,
                data=status_dict
            )
            
            logger_instance.info(f"Successfully created status.json with status: {status_val}")
            
            # Structured log: Status created
            logger.log_structured_event(
                structured_log,
                "status_created",
                {
                    "file": constants.STATUS_FILE,
                    "status": status_val,
                    "message": message_val,
                    "errors_count": len(errors_list),
                    "csv_read_successful": csv_read_successful
                },
                f"Initial status created: {status_val}"
            )
            
        except Exception as e:
            error_msg = f"Failed to create status.json: {str(e)}"
            logger_instance.error(error_msg)
            
            # Structured log: Status creation failed
            logger.log_structured_error(
                structured_log,
                "status_creation_failed",
                error_msg,
                {"stage": constants.INGEST_STAGE}
            )
            # This is critical - we should still try to continue
        
        # Append to run index (if ingestion was successful enough)
        logger_instance.info("Adding entry to run index")
        
        try:
            # Determine final status for index
            if csv_read_successful and len(errors_list) == 0:
                index_status = "Ingestion Completed"
            elif csv_read_successful:
                index_status = "Ingestion Completed with Warnings"
            else:
                index_status = "Ingestion Failed: CSV Parse Error"
            
            # Prepare run entry data
            run_entry_data = {
                'run_id': run_id,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'original_filename': original_filename,
                'status': index_status
            }
            
            # Append to run index
            storage.append_to_run_index(run_entry_data=run_entry_data)
            
            logger_instance.info("Successfully added entry to run index")
            
            # Structured log: Run index updated
            logger.log_structured_event(
                structured_log,
                "run_index_updated",
                {
                    "run_id": run_id,
                    "index_status": index_status,
                    "original_filename": original_filename
                },
                f"Run index updated: {index_status}"
            )
            
        except Exception as e:
            error_msg = f"Failed to update run index: {str(e)}"
            logger_instance.error(error_msg)
            
            # Structured log: Run index update failed
            logger.log_structured_error(
                structured_log,
                "run_index_update_failed",
                error_msg,
                {"stage": constants.INGEST_STAGE, "run_id": run_id}
            )
            # Not critical for the run to continue
        
        # Log completion
        final_status = "successful" if csv_read_successful else "failed"
        logger_instance.info(f"Ingestion process completed with status: {final_status}")
        
        # Structured log: Ingestion completed
        logger.log_structured_event(
            structured_log,
            "ingestion_completed",
            {
                "run_id": run_id,
                "success": csv_read_successful,
                "final_status": final_status,
                "errors_count": len(errors_list),
                "data_rows": initial_rows,
                "data_columns": initial_cols,
                "original_filename": original_filename,
                "artifacts_created": [
                    constants.ORIGINAL_DATA_FILE,
                    constants.METADATA_FILE,
                    constants.STATUS_FILE
                ]
            },
            f"Ingestion completed: {final_status} for {original_filename}"
        )
        
        return run_id
        
    except Exception as e:
        # Critical error - log and re-raise
        logger_instance.error(f"Critical error during ingestion: {str(e)}", exc_info=True)
        
        # Structured log: Critical error
        logger.log_structured_error(
            structured_log,
            "critical_ingestion_error",
            f"Critical error during ingestion: {str(e)}",
            {"stage": constants.INGEST_STAGE, "run_id": run_id}
        )
        
        raise 