"""
Prep Stage Runner for The Projection Wizard.
Orchestrates the complete data preparation stage including cleaning, encoding, and profiling.
"""

import pandas as pd
from pathlib import Path
from typing import Optional
from datetime import datetime

from common import logger, storage, constants, schemas
from . import cleaning_logic, encoding_logic, profiling_logic


def run_preparation_stage(run_id: str) -> bool:
    """
    Execute the complete data preparation stage for a given run.
    
    This function orchestrates:
    1. Loading inputs (metadata.json, original_data.csv)
    2. Data cleaning (missing values, duplicates)
    3. Feature encoding (ML-ready transformations)
    4. Data profiling (ydata-profiling report)
    5. Saving outputs and updating metadata
    
    Args:
        run_id: Unique run identifier
        
    Returns:
        True if stage completes successfully, False otherwise
    """
    # Get loggers for this run and stage
    log = logger.get_stage_logger(run_id, constants.PREP_STAGE)
    structured_log = logger.get_stage_structured_logger(run_id, constants.PREP_STAGE)
    
    # Track timing
    start_time = datetime.now()
    
    try:
        # Log stage start
        log.info(f"Starting data preparation stage for run {run_id}")
        
        # Structured log: Stage started
        logger.log_structured_event(
            structured_log,
            "stage_started",
            {"stage": constants.PREP_STAGE},
            "Data preparation stage started"
        )
        
        # Update status to running
        try:
            status_data = {
                "stage": constants.PREP_STAGE,
                "status": "running",
                "message": "Data preparation in progress..."
            }
            storage.write_json_atomic(run_id, constants.STATUS_FILENAME, status_data)
        except Exception as e:
            log.warning(f"Could not update status to running: {e}")
        
        # =============================
        # 1. LOAD INPUTS
        # =============================
        log.info("Loading inputs: metadata and original data")
        
        # Load metadata.json
        try:
            metadata_dict = storage.read_json(run_id, constants.METADATA_FILENAME)
        except Exception as e:
            log.error(f"Failed to load metadata.json: {e}")
            logger.log_structured_error(
                structured_log,
                "metadata_load_failed",
                f"Failed to load metadata.json: {str(e)}",
                {"stage": constants.PREP_STAGE}
            )
            _update_status_failed(run_id, f"Failed to load metadata: {str(e)}")
            return False
        
        if not metadata_dict:
            log.error("metadata.json is empty or invalid")
            logger.log_structured_error(
                structured_log,
                "metadata_empty",
                "metadata.json is empty or invalid",
                {"stage": constants.PREP_STAGE}
            )
            _update_status_failed(run_id, "Empty or invalid metadata")
            return False
        
        # Convert target_info dict to Pydantic object (Critical: Task 5 learnings)
        target_info_dict = metadata_dict.get('target_info')
        if not target_info_dict:
            log.error("No target_info found in metadata")
            logger.log_structured_error(
                structured_log,
                "target_info_missing",
                "No target_info found in metadata",
                {"stage": constants.PREP_STAGE}
            )
            _update_status_failed(run_id, "Missing target information in metadata")
            return False
        
        try:
            target_info = schemas.TargetInfo(**target_info_dict)
            log.info(f"Loaded target info: column='{target_info.name}', ml_type='{target_info.ml_type}'")
        except Exception as e:
            log.error(f"Failed to parse target_info: {e}")
            logger.log_structured_error(
                structured_log,
                "target_info_parse_failed",
                f"Failed to parse target_info: {str(e)}",
                {"stage": constants.PREP_STAGE}
            )
            _update_status_failed(run_id, f"Invalid target info format: {str(e)}")
            return False
        
        # Convert feature_schemas dict to Pydantic objects
        feature_schemas_dict = metadata_dict.get('feature_schemas', {})
        if not feature_schemas_dict:
            log.warning("No feature_schemas found in metadata - will use inference")
            feature_schemas_obj = {}
        else:
            try:
                feature_schemas_obj = {
                    col: schemas.FeatureSchemaInfo(**schema_dict) 
                    for col, schema_dict in feature_schemas_dict.items()
                }
                log.info(f"Loaded feature schemas for {len(feature_schemas_obj)} columns")
            except Exception as e:
                log.error(f"Failed to parse feature_schemas: {e}")
                logger.log_structured_error(
                    structured_log,
                    "feature_schemas_parse_failed",
                    f"Failed to parse feature_schemas: {str(e)}",
                    {"stage": constants.PREP_STAGE}
                )
                _update_status_failed(run_id, f"Invalid feature schemas format: {str(e)}")
                return False
        
        # Load original data
        try:
            original_data_path = storage.get_run_dir(run_id) / constants.ORIGINAL_DATA_FILENAME
            if not original_data_path.exists():
                log.error(f"Original data file not found: {original_data_path}")
                logger.log_structured_error(
                    structured_log,
                    "original_data_not_found",
                    f"Original data file not found: {original_data_path}",
                    {"stage": constants.PREP_STAGE}
                )
                _update_status_failed(run_id, "Original data file not found")
                return False
                
            df_orig = pd.read_csv(original_data_path)
            log.info(f"Loaded original data: shape {df_orig.shape}")
            
            if df_orig.empty:
                log.error("Original data is empty")
                logger.log_structured_error(
                    structured_log,
                    "original_data_empty",
                    "Original data file is empty",
                    {"stage": constants.PREP_STAGE}
                )
                _update_status_failed(run_id, "Original data file is empty")
                return False
            
            # Structured log: Data loaded
            logger.log_structured_event(
                structured_log,
                "data_loaded",
                {
                    "data_shape": {"rows": df_orig.shape[0], "columns": df_orig.shape[1]},
                    "target_column": target_info.name,
                    "feature_schemas_count": len(feature_schemas_obj)
                },
                f"Original data loaded: {df_orig.shape}"
            )
                
        except Exception as e:
            log.error(f"Failed to load original data: {e}")
            logger.log_structured_error(
                structured_log,
                "data_load_failed",
                f"Failed to load original data: {str(e)}",
                {"stage": constants.PREP_STAGE}
            )
            _update_status_failed(run_id, f"Failed to read original data: {str(e)}")
            return False
        
        # =============================
        # 2. DATA CLEANING
        # =============================
        log.info("Starting data cleaning...")
        
        try:
            df_cleaned, cleaning_steps = cleaning_logic.clean_data(
                df_original=df_orig,
                feature_schemas=feature_schemas_obj,
                target_info=target_info,
                cleaning_config=None  # Using defaults for MVP
            )
            log.info(f"Cleaning completed: {df_orig.shape} → {df_cleaned.shape}")
            log.info(f"Cleaning steps performed: {len(cleaning_steps)}")
            for step in cleaning_steps:
                log.info(f"  - {step}")
            
            # Structured log: Cleaning completed
            logger.log_structured_event(
                structured_log,
                "cleaning_completed",
                {
                    "input_shape": {"rows": df_orig.shape[0], "columns": df_orig.shape[1]},
                    "output_shape": {"rows": df_cleaned.shape[0], "columns": df_cleaned.shape[1]},
                    "cleaning_steps": cleaning_steps,
                    "rows_removed": df_orig.shape[0] - df_cleaned.shape[0]
                },
                f"Data cleaning completed: {df_orig.shape} → {df_cleaned.shape}"
            )
                
        except Exception as e:
            log.error(f"Data cleaning failed: {e}")
            logger.log_structured_error(
                structured_log,
                "cleaning_failed",
                f"Data cleaning failed: {str(e)}",
                {"stage": constants.PREP_STAGE}
            )
            _update_status_failed(run_id, f"Data cleaning failed: {str(e)}")
            return False
        
        # =============================
        # 3. FEATURE ENCODING
        # =============================
        log.info("Starting feature encoding...")
        
        try:
            df_encoded, encoders_info = encoding_logic.encode_features(
                df_cleaned=df_cleaned,
                feature_schemas=feature_schemas_obj,
                target_info=target_info,
                run_id=run_id
            )
            log.info(f"Encoding completed: {df_cleaned.shape} → {df_encoded.shape}")
            log.info(f"Encoders/scalers saved: {len(encoders_info)}")
            for encoder_name in encoders_info.keys():
                log.info(f"  - {encoder_name}")
            
            # Structured log: Encoding completed
            logger.log_structured_event(
                structured_log,
                "encoding_completed",
                {
                    "input_shape": {"rows": df_cleaned.shape[0], "columns": df_cleaned.shape[1]},
                    "output_shape": {"rows": df_encoded.shape[0], "columns": df_encoded.shape[1]},
                    "encoders_count": len(encoders_info),
                    "encoder_types": list(encoders_info.keys()),
                    "columns_added": df_encoded.shape[1] - df_cleaned.shape[1]
                },
                f"Feature encoding completed: {df_cleaned.shape} → {df_encoded.shape}"
            )
                
        except Exception as e:
            log.error(f"Feature encoding failed: {e}")
            logger.log_structured_error(
                structured_log,
                "encoding_failed",
                f"Feature encoding failed: {str(e)}",
                {"stage": constants.PREP_STAGE}
            )
            _update_status_failed(run_id, f"Feature encoding failed: {str(e)}")
            return False
        
        # =============================
        # 4. SAVE CLEANED DATA
        # =============================
        log.info("Saving cleaned and encoded data...")
        
        try:
            cleaned_data_path = storage.get_run_dir(run_id) / constants.CLEANED_DATA_FILE
            df_encoded.to_csv(cleaned_data_path, index=False)
            log.info(f"Cleaned data saved to: {cleaned_data_path}")
            log.info(f"Final data shape: {df_encoded.shape}")
            
            # Structured log: Data saved
            logger.log_structured_event(
                structured_log,
                "cleaned_data_saved",
                {
                    "file_path": constants.CLEANED_DATA_FILE,
                    "file_size_bytes": cleaned_data_path.stat().st_size if cleaned_data_path.exists() else None,
                    "final_shape": {"rows": df_encoded.shape[0], "columns": df_encoded.shape[1]}
                },
                f"Cleaned data saved: {constants.CLEANED_DATA_FILE}"
            )
            
        except Exception as e:
            log.error(f"Failed to save cleaned data: {e}")
            logger.log_structured_error(
                structured_log,
                "save_cleaned_data_failed",
                f"Failed to save cleaned data: {str(e)}",
                {"stage": constants.PREP_STAGE}
            )
            _update_status_failed(run_id, f"Failed to save cleaned data: {str(e)}")
            return False
        
        # =============================
        # 5. GENERATE PROFILING REPORT
        # =============================
        log.info("Generating data profiling report...")
        
        # Use run_id in filename for uniqueness, or could use PROFILE_REPORT_FILE constant
        profile_report_path = storage.get_run_dir(run_id) / f"{run_id}_profile.html"
        
        try:
            profiling_success = profiling_logic.generate_profile_report_with_fallback(
                df_final_prepared=df_encoded,
                report_path=profile_report_path,
                title=f"Data Profile for Run {run_id}",
                run_id=run_id
            )
            
            if profiling_success:
                log.info(f"Profiling report generated: {profile_report_path}")
                # Structured log: Profiling completed
                logger.log_structured_event(
                    structured_log,
                    "profiling_completed",
                    {
                        "report_path": profile_report_path.name,
                        "file_size_bytes": profile_report_path.stat().st_size if profile_report_path.exists() else None
                    },
                    f"Profiling report generated: {profile_report_path.name}"
                )
            else:
                log.warning("Profiling report generation failed, but continuing...")
                logger.log_structured_event(
                    structured_log,
                    "profiling_failed",
                    {"non_critical": True},
                    "Profiling report generation failed (non-critical)"
                )
                
        except Exception as e:
            log.warning(f"Profiling report failed (non-critical): {e}")
            profiling_success = False
            logger.log_structured_event(
                structured_log,
                "profiling_error",
                {"error": str(e), "non_critical": True},
                f"Profiling report error (non-critical): {str(e)}"
            )
        
        # =============================
        # 6. UPDATE METADATA
        # =============================
        log.info("Updating metadata with prep results...")
        
        try:
            # Create prep_info dictionary
            prep_info = {
                'cleaning_steps_performed': cleaning_steps,
                'encoders_scalers_info': encoders_info,
                'cleaned_data_path': constants.CLEANED_DATA_FILE,
                'profiling_report_path': profile_report_path.name if profiling_success else None,
                'final_shape_after_prep': list(df_encoded.shape)
            }
            
            # Add prep_info to existing metadata
            metadata_dict['prep_info'] = prep_info
            
            # Save updated metadata
            storage.write_json_atomic(run_id, constants.METADATA_FILENAME, metadata_dict)
            log.info("Metadata updated with prep results")
            
            # Structured log: Metadata updated
            logger.log_structured_event(
                structured_log,
                "metadata_updated",
                {
                    "prep_info_keys": list(prep_info.keys()),
                    "cleaning_steps_count": len(cleaning_steps),
                    "encoders_count": len(encoders_info)
                },
                "Metadata updated with prep results"
            )
            
        except Exception as e:
            log.error(f"Failed to update metadata: {e}")
            logger.log_structured_error(
                structured_log,
                "metadata_update_failed",
                f"Failed to update metadata: {str(e)}",
                {"stage": constants.PREP_STAGE}
            )
            _update_status_failed(run_id, f"Failed to update metadata: {str(e)}")
            return False
        
        # =============================
        # 7. UPDATE STATUS TO COMPLETED
        # =============================
        try:
            status_data = {
                "stage": constants.PREP_STAGE,
                "status": "completed",
                "message": "Data preparation completed successfully."
            }
            storage.write_json_atomic(run_id, constants.STATUS_FILENAME, status_data)
            log.info("Status updated to completed")
            
        except Exception as e:
            log.warning(f"Could not update final status: {e}")
        
        # =============================
        # 8. LOG COMPLETION
        # =============================
        log.info("="*50)
        log.info("DATA PREPARATION STAGE COMPLETED SUCCESSFULLY")
        log.info("="*50)
        log.info(f"Original data shape: {df_orig.shape}")
        log.info(f"Final data shape: {df_encoded.shape}")
        log.info(f"Cleaning steps: {len(cleaning_steps)}")
        log.info(f"Encoders saved: {len(encoders_info)}")
        log.info(f"Profiling report: {'✅' if profiling_success else '❌'}")
        log.info(f"Output files:")
        log.info(f"  - {constants.CLEANED_DATA_FILE}")
        log.info(f"  - {constants.METADATA_FILENAME} (updated)")
        log.info(f"  - {constants.STATUS_FILENAME} (updated)")
        if profiling_success:
            log.info(f"  - {profile_report_path.name}")
        log.info("="*50)
        
        return True
        
    except Exception as e:
        log.error(f"Unexpected error in prep stage: {e}")
        log.error("Full traceback:", exc_info=True)
        _update_status_failed(run_id, f"Unexpected error: {str(e)}")
        return False


def _update_status_failed(run_id: str, error_message: str) -> None:
    """
    Helper function to update status.json to failed state.
    
    Args:
        run_id: Run identifier
        error_message: Error message to include in status
    """
    try:
        status_data = {
            "stage": constants.PREP_STAGE,
            "status": "failed",
            "message": f"Data preparation failed: {error_message}"
        }
        storage.write_json_atomic(run_id, constants.STATUS_FILENAME, status_data)
    except Exception as e:
        # If we can't even update status, log it but don't raise
        log = logger.get_stage_logger(run_id, constants.PREP_STAGE)
        log.error(f"Could not update status to failed: {e}")


def validate_prep_stage_inputs(run_id: str) -> bool:
    """
    Validate that all required inputs for the prep stage are available.
    
    Args:
        run_id: Run identifier
        
    Returns:
        True if all inputs are valid, False otherwise
    """
    log = logger.get_stage_logger(run_id, constants.PREP_STAGE)
    
    try:
        # Check if run directory exists
        run_dir = storage.get_run_dir(run_id)
        if not run_dir.exists():
            log.error(f"Run directory does not exist: {run_dir}")
            return False
        
        # Check if metadata.json exists
        metadata_path = run_dir / constants.METADATA_FILENAME
        if not metadata_path.exists():
            log.error(f"Metadata file does not exist: {metadata_path}")
            return False
        
        # Check if original_data.csv exists
        original_data_path = run_dir / constants.ORIGINAL_DATA_FILENAME
        if not original_data_path.exists():
            log.error(f"Original data file does not exist: {original_data_path}")
            return False
        
        # Try to load and validate metadata structure
        try:
            metadata_dict = storage.read_json(run_id, constants.METADATA_FILENAME)
            
            # Check for required keys
            if 'target_info' not in metadata_dict:
                log.error("Missing 'target_info' in metadata")
                return False
            
            # Validate target_info can be parsed
            target_info_dict = metadata_dict['target_info']
            schemas.TargetInfo(**target_info_dict)
            
            # Feature schemas are optional, but if present, validate format
            feature_schemas_dict = metadata_dict.get('feature_schemas', {})
            if feature_schemas_dict:
                # Try to parse at least one schema to validate format
                first_schema = next(iter(feature_schemas_dict.values()))
                schemas.FeatureSchemaInfo(**first_schema)
            
            log.info("All prep stage inputs validated successfully")
            return True
            
        except Exception as e:
            log.error(f"Metadata validation failed: {e}")
            return False
        
    except Exception as e:
        log.error(f"Input validation failed: {e}")
        return False


# Additional utility functions for testing and debugging
def get_prep_stage_summary(run_id: str) -> Optional[dict]:
    """
    Get a summary of the prep stage results for a given run.
    
    Args:
        run_id: Run identifier
        
    Returns:
        Dictionary with prep stage summary or None if not available
    """
    try:
        metadata_dict = storage.read_json(run_id, constants.METADATA_FILENAME)
        prep_info = metadata_dict.get('prep_info')
        
        if not prep_info:
            return None
        
        return {
            'run_id': run_id,
            'final_shape': prep_info.get('final_shape_after_prep'),
            'cleaning_steps_count': len(prep_info.get('cleaning_steps_performed', [])),
            'encoders_count': len(prep_info.get('encoders_scalers_info', {})),
            'has_profile_report': prep_info.get('profiling_report_path') is not None,
            'cleaned_data_available': (storage.get_run_dir(run_id) / constants.CLEANED_DATA_FILE).exists()
        }
        
    except Exception:
        return None 