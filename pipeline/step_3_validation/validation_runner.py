"""
Data validation runner for The Projection Wizard.
Orchestrates the validation process using Great Expectations.
Refactored for GCS-based storage.
"""

import pandas as pd
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
import json
import io
import logging

from common import constants
from common.schemas import FeatureSchemaInfo, TargetInfo, ValidationInfo
from api.utils.gcs_utils import (
    download_run_file, upload_run_file, check_run_file_exists,
    PROJECT_BUCKET_NAME
)
from .ge_logic import generate_ge_suite_from_metadata_gcs, run_ge_validation_from_gcs

# Configure logging for this module
logger = logging.getLogger(__name__)


def run_gcs_validation_stage(run_id: str, gcs_bucket_name: str = PROJECT_BUCKET_NAME) -> bool:
    """
    Run the data validation stage according to the GCS specification.
    
    Args:
        run_id: The ID of the run to validate
        gcs_bucket_name: GCS bucket name
        
    Returns:
        True if successful, False otherwise
    """
    # Track timing
    start_time = datetime.now()
    
    try:
        logger.info(f"Starting GCS validation stage for run {run_id}")
        
        # Step 1: Update status.json to "processing"
        try:
            status_data = {
                'stage': constants.VALIDATION_STAGE,
                'status': 'processing',
                'message': 'Running data validation with Great Expectations',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'errors': []
            }
            status_bytes = json.dumps(status_data, indent=2).encode('utf-8')
            upload_run_file(run_id, constants.STATUS_FILENAME, io.BytesIO(status_bytes))
            logger.info("Updated status.json to processing")
        except Exception as e:
            logger.error(f"Failed to update status.json: {str(e)}")
            return False
        
        # Step 2: Download and validate metadata.json
        try:
            metadata_bytes = download_run_file(run_id, constants.METADATA_FILENAME)
            if metadata_bytes is None:
                error_msg = f"Could not read metadata.json for run {run_id}"
                logger.error(error_msg)
                _update_status_failed(run_id, error_msg)
                return False
            
            metadata_dict = json.loads(metadata_bytes.decode('utf-8'))
            logger.info("Downloaded and parsed metadata.json from GCS")
        except Exception as e:
            error_msg = f"Failed to load metadata.json: {str(e)}"
            logger.error(error_msg)
            _update_status_failed(run_id, error_msg)
            return False
        
        # Step 3: Extract and validate feature schemas and target info
        try:
            # Extract feature_schemas and target_info
            feature_schemas_dict = metadata_dict.get('feature_schemas', {})
            target_info_dict = metadata_dict.get('target_info')
            
            # Convert feature_schemas dictionaries to FeatureSchemaInfo objects
            feature_schemas = {}
            if feature_schemas_dict:
                for col_name, schema_dict in feature_schemas_dict.items():
                    feature_schemas[col_name] = FeatureSchemaInfo(**schema_dict)
            
            if not feature_schemas:
                error_msg = "Feature schemas not found in metadata. Please complete schema confirmation first."
                logger.error(error_msg)
                _update_status_failed(run_id, error_msg)
                return False
            
            # Convert target_info dictionary to TargetInfo object if it exists
            target_info = None
            if target_info_dict:
                target_info = TargetInfo(**target_info_dict)
            
            logger.info(f"Found feature schemas for {len(feature_schemas)} columns")
        except Exception as e:
            error_msg = f"Failed to parse schema information: {str(e)}"
            logger.error(error_msg)
            _update_status_failed(run_id, error_msg)
            return False
        
        # Step 4: Download and load original_data.csv
        try:
            csv_bytes = download_run_file(run_id, constants.ORIGINAL_DATA_FILENAME)
            if csv_bytes is None:
                error_msg = f"Could not download original_data.csv for run {run_id}"
                logger.error(error_msg)
                _update_status_failed(run_id, error_msg)
                return False
            
            df = pd.read_csv(io.BytesIO(csv_bytes))
            logger.info(f"Loaded original data with shape {df.shape}")
        except Exception as e:
            error_msg = f"Failed to load original_data.csv: {str(e)}"
            logger.error(error_msg)
            _update_status_failed(run_id, error_msg)
            return False
        
        # Step 5: Generate Great Expectations suite
        try:
            ge_suite_dict = generate_ge_suite_from_metadata_gcs(run_id, gcs_bucket_name)
            if ge_suite_dict is None:
                error_msg = "Failed to generate Great Expectations suite"
                logger.error(error_msg)
                _update_status_failed(run_id, error_msg)
                return False
            
            expectations_count = len(ge_suite_dict.get('expectations', []))
            logger.info(f"Generated GE suite with {expectations_count} expectations")
        except Exception as e:
            error_msg = f"Failed to generate GE suite: {str(e)}"
            logger.error(error_msg)
            _update_status_failed(run_id, error_msg)
            return False
        
        # Step 6: Run Great Expectations validation
        try:
            from .ge_logic import run_ge_validation_on_dataframe
            ge_results_dict = run_ge_validation_on_dataframe(df, ge_suite_dict, run_id)
            if ge_results_dict is None:
                error_msg = "Failed to run Great Expectations validation"
                logger.error(error_msg)
                _update_status_failed(run_id, error_msg)
                return False
            
            logger.info("Completed GE validation")
        except Exception as e:
            error_msg = f"Failed to run GE validation: {str(e)}"
            logger.error(error_msg)
            _update_status_failed(run_id, error_msg)
            return False
        
        # Step 7: Process validation results
        try:
            # Extract key validation metrics
            overall_success = ge_results_dict.get("success", False)
            stats = ge_results_dict.get("statistics", {})
            total_expectations = stats.get("evaluated_expectations", 0)
            successful_expectations = stats.get("successful_expectations", 0)
            failed_expectations = stats.get("unsuccessful_expectations", 0)
            success_rate = (successful_expectations / total_expectations * 100) if total_expectations > 0 else 0
            
            logger.info(f"Validation results: {success_rate:.1f}% success rate ({successful_expectations}/{total_expectations})")
        except Exception as e:
            error_msg = f"Failed to process validation results: {str(e)}"
            logger.error(error_msg)
            _update_status_failed(run_id, error_msg)
            return False
        
        # Step 8: Prepare and save validation.json
        try:
            validation_summary = {
                "overall_success": overall_success,
                "total_expectations": total_expectations,
                "successful_expectations": successful_expectations,
                "failed_expectations": failed_expectations,
                "run_time_s": ge_results_dict.get("meta", {}).get("run_time", 0),
                "ge_version": ge_results_dict.get("meta", {}).get("great_expectations_version", "unknown"),
                "results_ge_native": ge_results_dict  # Store the full raw GE result
            }
            
            # Upload validation.json to GCS
            validation_bytes = json.dumps(validation_summary, indent=2).encode('utf-8')
            upload_run_file(run_id, constants.VALIDATION_FILENAME, io.BytesIO(validation_bytes))
            logger.info("Saved validation.json to GCS")
        except Exception as e:
            error_msg = f"Failed to save validation.json: {str(e)}"
            logger.error(error_msg)
            _update_status_failed(run_id, error_msg)
            return False
        
        # Step 9: Update metadata.json with validation info
        try:
            validation_info = {
                'passed': validation_summary['overall_success'], 
                'report_filename': constants.VALIDATION_FILENAME,
                'total_expectations_evaluated': validation_summary['total_expectations'],
                'successful_expectations': validation_summary['successful_expectations']
            }
            metadata_dict['validation_info'] = validation_info
            
            # Upload updated metadata.json to GCS
            metadata_bytes = json.dumps(metadata_dict, indent=2).encode('utf-8')
            upload_run_file(run_id, constants.METADATA_FILENAME, io.BytesIO(metadata_bytes))
            logger.info("Updated metadata.json with validation info")
        except Exception as e:
            error_msg = f"Failed to update metadata.json: {str(e)}"
            logger.error(error_msg)
            _update_status_failed(run_id, error_msg)
            return False
        
        # Step 10: Determine pipeline continuation and update status.json
        try:
            # Determine if pipeline should continue based on validation success rate
            validation_success_threshold = constants.VALIDATION_CONFIG.get("pipeline_failure_threshold", 0.90)  # 90% threshold
            success_rate_decimal = successful_expectations / total_expectations if total_expectations > 0 else 0.0
            
            if success_rate_decimal < validation_success_threshold:
                # Validation failed - stop pipeline execution
                status_value = 'failed'
                failure_reasons = []
                failure_reasons.append(f"Validation success rate {success_rate:.1f}% is below required threshold {validation_success_threshold*100:.1f}%")
                failure_reasons.append(f"Failed expectations: {failed_expectations}/{total_expectations}")
                
                # Add critical failure details if available
                critical_failures = _extract_critical_failures(ge_results_dict)
                if critical_failures:
                    failure_reasons.extend(critical_failures[:5])  # Limit to first 5 critical failures
                    
                message = f"Pipeline stopped: Validation failed with {failed_expectations} critical issues"
                
            elif not validation_summary['overall_success']:
                # Validation had issues but passed threshold - continue with warning
                status_value = 'completed'
                message = f"Validation completed with warnings: {failed_expectations} non-critical issues found"
            else:
                # Validation fully passed
                status_value = 'completed' 
                message = "Validation checks completed successfully"
            
            # Update status.json
            status_data = {
                'stage': constants.VALIDATION_STAGE,
                'status': status_value,
                'message': message,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'errors': failure_reasons if status_value == 'failed' else []
            }
            status_bytes = json.dumps(status_data, indent=2).encode('utf-8')
            upload_run_file(run_id, constants.STATUS_FILENAME, io.BytesIO(status_bytes))
            logger.info(f"Updated status.json with status: {status_value}")
        except Exception as e:
            error_msg = f"Failed to update final status: {str(e)}"
            logger.error(error_msg)
            _update_status_failed(run_id, error_msg)
            return False
        
        # Step 11: Log final results
        try:
            # Calculate stage duration
            end_time = datetime.now()
            stage_duration = (end_time - start_time).total_seconds()
            
            # Log overall result
            if status_value == 'failed':
                logger.error(f"❌ Pipeline STOPPED: Validation failed with {failed_expectations}/{total_expectations} expectations ({success_rate:.1f}%)")
                return False  # Signal pipeline failure
            elif validation_summary['overall_success']:
                logger.info(f"✅ Validation PASSED: {validation_summary['successful_expectations']}/{validation_summary['total_expectations']} expectations")
            else:
                logger.warning(f"⚠️ Validation completed with warnings: {validation_summary['successful_expectations']}/{validation_summary['total_expectations']} expectations passed ({success_rate:.1f}%)")
            
            logger.info(f"Validation stage completed in {stage_duration:.1f}s")
            return True
        except Exception as e:
            logger.error(f"Error in final logging: {str(e)}")
            return status_value != 'failed'  # Return success unless explicitly failed
        
    except Exception as e:
        error_msg = f"Validation stage failed: {str(e)}"
        logger.error(error_msg)
        _update_status_failed(run_id, error_msg)
        return False


def _update_status_failed(run_id: str, error_message: str) -> None:
    """
    Update status.json with failed status and error message.
    
    Args:
        run_id: The run ID
        error_message: Error message to record
    """
    try:
        status_data = {
            'stage': constants.VALIDATION_STAGE,
            'status': 'failed',
            'message': error_message,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'errors': [error_message]
        }
        status_bytes = json.dumps(status_data, indent=2).encode('utf-8')
        upload_run_file(run_id, constants.STATUS_FILENAME, io.BytesIO(status_bytes))
        logger.info("Updated status.json with failed status")
    except Exception as status_error:
        logger.error(f"Failed to update status.json with error: {str(status_error)}")


def _extract_critical_failures(validation_results: Dict[str, Any]) -> List[str]:
    """
    Extract a list of critical failure descriptions from validation results.
    
    Args:
        validation_results: Full GE validation results dictionary
        
    Returns:
        List of critical failure descriptions
    """
    critical_failures = []
    
    try:
        results = validation_results.get("results", [])
        
        for result in results:
            if not result.get("success", True):
                # Extract expectation details
                expectation_type = result.get("expectation_config", {}).get("expectation_type", "unknown")
                column = result.get("expectation_config", {}).get("kwargs", {}).get("column", "table_level")
                
                # Identify critical failures that warrant pipeline stoppage
                if expectation_type in [
                    "expect_column_to_exist",
                    "expect_table_columns_to_match_ordered_list", 
                    "expect_column_values_to_be_of_type"
                ]:
                    critical_failures.append(f"{expectation_type} failed for {column}")
    except Exception as e:
        logger.warning(f"Error extracting critical failures: {str(e)}")
    
    return critical_failures


def _extract_failed_expectations(validation_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract a summary of failed expectations from validation results.
    
    Args:
        validation_results: Full GE validation results dictionary
        
    Returns:
        Dictionary summarizing failed expectations
    """
    failed_summary = {
        "total_failed": 0,
        "by_column": {},
        "by_expectation_type": {},
        "critical_failures": []
    }
    
    results = validation_results.get("results", [])
    
    for result in results:
        if not result.get("success", True):
            failed_summary["total_failed"] += 1
            
            # Extract expectation details
            expectation_type = result.get("expectation_config", {}).get("expectation_type", "unknown")
            column = result.get("expectation_config", {}).get("kwargs", {}).get("column", "table_level")
            
            # Count by column
            if column not in failed_summary["by_column"]:
                failed_summary["by_column"][column] = 0
            failed_summary["by_column"][column] += 1
            
            # Count by expectation type
            if expectation_type not in failed_summary["by_expectation_type"]:
                failed_summary["by_expectation_type"][expectation_type] = 0
            failed_summary["by_expectation_type"][expectation_type] += 1
            
            # Identify critical failures (data type mismatches, missing columns, etc.)
            if expectation_type in [
                "expect_column_to_exist",
                "expect_table_columns_to_match_ordered_list",
                "expect_column_values_to_be_of_type"
            ]:
                failed_summary["critical_failures"].append({
                    "expectation_type": expectation_type,
                    "column": column,
                    "result": result.get("result", {})
                })
    
    return failed_summary


def get_gcs_validation_summary(run_id: str, gcs_bucket_name: str = PROJECT_BUCKET_NAME) -> Optional[Dict[str, Any]]:
    """
    Get a summary of validation results for a run from GCS.
    
    Args:
        run_id: The ID of the run
        gcs_bucket_name: GCS bucket name
        
    Returns:
        Dictionary with validation summary or None if not found
    """
    try:
        validation_bytes = download_run_file(run_id, constants.VALIDATION_FILENAME)
        if validation_bytes is None:
            return None
        
        return json.loads(validation_bytes.decode('utf-8'))
    except Exception as e:
        logger.error(f"Failed to get validation summary from GCS: {str(e)}")
        return None


def check_gcs_validation_status(run_id: str, gcs_bucket_name: str = PROJECT_BUCKET_NAME) -> str:
    """
    Check the validation status for a run from GCS.
    
    Args:
        run_id: The ID of the run
        gcs_bucket_name: GCS bucket name
        
    Returns:
        Status string: 'not_started', 'passed', 'failed', 'completed_with_warnings', or 'error'
    """
    try:
        validation_summary = get_gcs_validation_summary(run_id, gcs_bucket_name)
        if validation_summary is None:
            return 'not_started'
        
        if validation_summary.get('overall_success', False):
            return 'passed'
        else:
            # Check if it was a critical failure by looking at status.json
            status_bytes = download_run_file(run_id, constants.STATUS_FILENAME)
            if status_bytes:
                status_data = json.loads(status_bytes.decode('utf-8'))
                if status_data.get('status') == 'failed':
                    return 'failed'
            
            return 'completed_with_warnings'
            
    except Exception as e:
        logger.error(f"Failed to check validation status from GCS: {str(e)}")
        return 'error'


# Legacy compatibility functions (redirects to GCS versions)
def run_validation_stage(run_id: str) -> bool:
    """
    Legacy compatibility function - redirects to GCS version.
    """
    logger.warning("Using legacy run_validation_stage function - redirecting to GCS version")
    return run_gcs_validation_stage(run_id)


def get_validation_summary(run_id: str) -> Optional[Dict[str, Any]]:
    """
    Legacy compatibility function - redirects to GCS version.
    """
    logger.warning("Using legacy get_validation_summary function - redirecting to GCS version")
    return get_gcs_validation_summary(run_id)


def check_validation_status(run_id: str) -> str:
    """
    Legacy compatibility function - redirects to GCS version.
    """
    logger.warning("Using legacy check_validation_status function - redirecting to GCS version")
    return check_gcs_validation_status(run_id) 