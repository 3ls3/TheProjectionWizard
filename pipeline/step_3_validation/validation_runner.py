"""
Data validation runner for The Projection Wizard.
Orchestrates the validation process using Great Expectations.
"""

import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
import sys
sys.path.append(str(Path(__file__).parent.parent))

from common import storage, logger, constants
from .ge_logic import generate_ge_suite_from_metadata, run_ge_validation_on_dataframe


def run_validation_stage(run_id: str) -> bool:
    """
    Run the data validation stage according to the specification.
    
    Args:
        run_id: The ID of the run to validate
        
    Returns:
        True if successful, False otherwise
    """
    # Get loggers
    logger_instance = logger.get_stage_logger(run_id, constants.VALIDATION_STAGE)
    structured_log = logger.get_stage_structured_logger(run_id, constants.VALIDATION_STAGE)
    
    # Track timing
    start_time = datetime.now()
    
    try:
        logger_instance.info(f"Starting validation stage for run {run_id}")
        
        # Structured log: Stage started
        logger.log_structured_event(
            structured_log,
            "stage_started",
            {"stage": constants.VALIDATION_STAGE},
            "Validation stage started"
        )
        
        # Load metadata.json
        metadata_dict = storage.read_json(run_id, constants.METADATA_FILENAME)
        if metadata_dict is None:
            logger_instance.error(f"Could not read metadata.json for run {run_id}")
            logger.log_structured_error(
                structured_log,
                "metadata_load_failed",
                f"Could not read metadata.json for run {run_id}",
                {"stage": constants.VALIDATION_STAGE}
            )
            return False
        
        # Extract feature_schemas and target_info
        feature_schemas = metadata_dict.get('feature_schemas', {})
        target_info_dict = metadata_dict.get('target_info')
        
        # Convert target_info dictionary to TargetInfo object if it exists
        target_info = None
        if target_info_dict:
            from common.schemas import TargetInfo
            target_info = TargetInfo(**target_info_dict)
        
        # Convert feature_schemas dictionaries to FeatureSchemaInfo objects
        converted_feature_schemas = {}
        if feature_schemas:
            from common.schemas import FeatureSchemaInfo
            for col_name, schema_dict in feature_schemas.items():
                converted_feature_schemas[col_name] = FeatureSchemaInfo(**schema_dict)
        
        if not converted_feature_schemas:
            logger_instance.error("Feature schemas not found in metadata. Please complete schema confirmation first.")
            logger.log_structured_error(
                structured_log,
                "feature_schemas_missing",
                "Feature schemas not found in metadata",
                {"stage": constants.VALIDATION_STAGE}
            )
            return False
        
        logger_instance.info(f"Found feature schemas for {len(converted_feature_schemas)} columns")
        
        # Structured log: Metadata loaded
        logger.log_structured_event(
            structured_log,
            "metadata_loaded",
            {
                "feature_schemas_count": len(converted_feature_schemas),
                "has_target_info": target_info is not None,
                "target_column": target_info.name if target_info else None
            },
            f"Metadata loaded: {len(converted_feature_schemas)} feature schemas"
        )
        
        # Load original_data.csv
        df = pd.read_csv(storage.get_run_dir(run_id) / constants.ORIGINAL_DATA_FILENAME)
        logger_instance.info(f"Loaded original data with shape {df.shape}")
        
        # Structured log: Data loaded
        logger.log_structured_event(
            structured_log,
            "data_loaded",
            {
                "data_shape": {"rows": df.shape[0], "columns": df.shape[1]},
                "file": constants.ORIGINAL_DATA_FILENAME
            },
            f"Original data loaded: {df.shape}"
        )
        
        # Generate GE Suite
        ge_suite_dict = generate_ge_suite_from_metadata(
            converted_feature_schemas, 
            target_info, 
            list(df.columns),
            run_id
        )
        logger_instance.info(f"Generated GE suite with {len(ge_suite_dict.get('expectations', []))} expectations")
        
        # Structured log: GE suite generated
        expectations_count = len(ge_suite_dict.get('expectations', []))
        logger.log_structured_event(
            structured_log,
            "ge_suite_generated",
            {
                "expectations_count": expectations_count,
                "suite_name": ge_suite_dict.get('expectation_suite_name', 'unknown')
            },
            f"Generated Great Expectations suite with {expectations_count} expectations"
        )
        
        # Run GE Validation
        ge_results_dict = run_ge_validation_on_dataframe(df, ge_suite_dict, run_id)
        logger_instance.info("Completed GE validation")
        
        # Extract key validation metrics
        overall_success = ge_results_dict.get("success", False)
        stats = ge_results_dict.get("statistics", {})
        total_expectations = stats.get("evaluated_expectations", 0)
        successful_expectations = stats.get("successful_expectations", 0)
        failed_expectations = stats.get("unsuccessful_expectations", 0)
        success_rate = (successful_expectations / total_expectations * 100) if total_expectations > 0 else 0
        
        # Structured log: Validation completed
        logger.log_structured_event(
            structured_log,
            "validation_completed",
            {
                "overall_success": overall_success,
                "total_expectations": total_expectations,
                "successful_expectations": successful_expectations,
                "failed_expectations": failed_expectations,
                "success_rate": success_rate,
                "runtime_seconds": ge_results_dict.get("meta", {}).get("run_time", 0)
            },
            f"Validation completed: {success_rate:.1f}% success rate ({successful_expectations}/{total_expectations})"
        )
        
        # Log individual validation metrics
        logger.log_structured_metric(
            structured_log,
            "validation_success_rate",
            success_rate,
            "data_quality",
            {"total_expectations": total_expectations, "successful": successful_expectations}
        )
        
        logger.log_structured_metric(
            structured_log,
            "expectations_evaluated",
            total_expectations,
            "data_quality",
            {"successful": successful_expectations, "failed": failed_expectations}
        )
        
        # Prepare validation.json Content
        validation_summary = {
            "overall_success": overall_success,
            "total_expectations": total_expectations,
            "successful_expectations": successful_expectations,
            "failed_expectations": failed_expectations,
            "run_time_s": ge_results_dict.get("meta", {}).get("run_time", 0),  # Path might vary
            "ge_version": ge_results_dict.get("meta", {}).get("great_expectations_version", "unknown"),
            "results_ge_native": ge_results_dict  # Store the full raw GE result
        }
        
        # Save validation.json
        storage.write_json_atomic(run_id, constants.VALIDATION_FILENAME, validation_summary)
        logger_instance.info("Saved validation.json")
        
        # Structured log: Validation results saved
        logger.log_structured_event(
            structured_log,
            "validation_results_saved",
            {
                "file": constants.VALIDATION_FILENAME,
                "overall_success": overall_success
            },
            f"Validation results saved to {constants.VALIDATION_FILENAME}"
        )
        
        # Update metadata.json
        validation_info = {
            'passed': validation_summary['overall_success'], 
            'report_filename': constants.VALIDATION_FILENAME,
            'total_expectations_evaluated': validation_summary['total_expectations'],
            'successful_expectations': validation_summary['successful_expectations']
        }
        metadata_dict['validation_info'] = validation_info
        storage.write_metadata(run_id, metadata_dict)
        logger_instance.info("Updated metadata.json with validation info")
        
        # Structured log: Metadata updated
        logger.log_structured_event(
            structured_log,
            "metadata_updated",
            {
                "validation_info_keys": list(validation_info.keys()),
                "validation_passed": validation_info['passed']
            },
            "Metadata updated with validation results"
        )
        
        # Update status.json
        status_value = 'completed' if validation_summary['overall_success'] else 'completed'  # Stage completed regardless of validation results
        status_data = {
            'stage': constants.VALIDATION_STAGE,
            'status': status_value,
            'message': "Validation checks completed.",
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'errors': []
        }
        storage.write_status(run_id, status_data)
        logger_instance.info("Updated status.json")
        
        # Structured log: Status updated
        logger.log_structured_event(
            structured_log,
            "status_updated",
            {
                "status": status_value,
                "stage": constants.VALIDATION_STAGE
            },
            f"Status updated to {status_value}"
        )
        
        # Calculate stage duration
        end_time = datetime.now()
        stage_duration = (end_time - start_time).total_seconds()
        
        # Structured log: Stage completed
        logger.log_structured_event(
            structured_log,
            "stage_completed",
            {
                "stage": constants.VALIDATION_STAGE,
                "success": True,
                "duration_seconds": stage_duration,
                "validation_success_rate": success_rate,
                "completed_at": end_time.isoformat()
            },
            f"Validation stage completed successfully in {stage_duration:.1f}s"
        )
        
        # Log overall result
        if validation_summary['overall_success']:
            logger_instance.info(f"✅ Validation PASSED: {validation_summary['successful_expectations']}/{validation_summary['total_expectations']} expectations")
        else:
            logger_instance.warning(f"⚠️ Validation had issues: {validation_summary['successful_expectations']}/{validation_summary['total_expectations']} expectations passed")
        
        return True
        
    except Exception as e:
        error_msg = f"Validation stage failed: {str(e)}"
        logger_instance.error(error_msg)
        
        # Structured log: Stage failed
        logger.log_structured_error(
            structured_log,
            "stage_failed",
            error_msg,
            {"stage": constants.VALIDATION_STAGE}
        )
        
        # Update status.json with error
        try:
            status_data = {
                'stage': constants.VALIDATION_STAGE,
                'status': 'failed',
                'message': error_msg,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'errors': [str(e)]
            }
            storage.write_status(run_id, status_data)
        except Exception as status_error:
            logger_instance.error(f"Failed to update status.json with error: {str(status_error)}")
        
        return False


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


def get_validation_summary(run_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a summary of validation results for a run.
    
    Args:
        run_id: The ID of the run
        
    Returns:
        Dictionary with validation summary or None if not found
    """
    try:
        metadata_dict = storage.read_metadata(run_id)
        if metadata_dict and 'validation_summary' in metadata_dict:
            return metadata_dict['validation_summary']
        return None
    except Exception:
        return None


def check_validation_status(run_id: str) -> str:
    """
    Check the validation status for a run.
    
    Args:
        run_id: The ID of the run
        
    Returns:
        Status string: 'not_started', 'passed', 'failed', 'completed_with_warnings', or 'error'
    """
    try:
        validation_summary = get_validation_summary(run_id)
        if validation_summary is None:
            return 'not_started'
        
        if validation_summary.get('validation_passed', False):
            return 'passed'
        else:
            return 'completed_with_warnings'
            
    except Exception:
        return 'error' 