"""
Result aggregation utilities for The Projection Wizard.
Functions to collect and format final pipeline results for the API.
Refactored for GCS-based storage.
"""

import json
import io
from typing import Dict, Optional
from pathlib import Path
from common import constants
from api.utils.gcs_utils import (
    download_run_file, check_run_file_exists, list_run_files,
    PROJECT_BUCKET_NAME, GCSError
)

# Stage progress mapping for consistent progress reporting
STAGE_PROGRESS_MAP = {
    "ingest": 0,
    "schema": 10,
    "validation": 30,
    "prep": 50,
    "automl": 70,
    "explain": 90,
    "completed": 100,
    "failed": 100,
}

# Map internal stage names to clean API names
STAGE_NAME_MAP = {
    "step_1_ingest": "ingest",
    "step_2_schema": "schema",
    "step_3_validation": "validation",
    "step_4_prep": "prep",
    "step_5_automl": "automl",
    "step_6_explain": "explain",
}


def read_json_from_gcs(run_id: str, filename: str) -> Optional[dict]:
    """
    Read JSON file from GCS for a specific run.
    
    Args:
        run_id: The ID of the run
        filename: Name of the JSON file
        
    Returns:
        Parsed JSON data or None if file not found
    """
    try:
        file_bytes = download_run_file(run_id, filename)
        if file_bytes is None:
            return None
        
        return json.loads(file_bytes.decode('utf-8'))
    except (json.JSONDecodeError, GCSError):
        return None


def check_file_size_gcs(run_id: str, filename: str) -> int:
    """
    Get the size of a file in GCS (in bytes).
    
    Args:
        run_id: The ID of the run
        filename: Name of the file
        
    Returns:
        File size in bytes, or 0 if file doesn't exist
    """
    try:
        # We can approximate file size by downloading and measuring
        # For better performance, this could be improved by using GCS metadata
        file_bytes = download_run_file(run_id, filename)
        return len(file_bytes) if file_bytes else 0
    except:
        return 0


def build_results_gcs(run_id: str, 
                     gcs_bucket_name: str = PROJECT_BUCKET_NAME) -> Dict:
    """
    Build comprehensive results dictionary from pipeline artifacts in GCS.

    Reads artifacts created by all pipeline stages from GCS to construct a 
    complete results payload for the API that matches the Streamlit results page.

    Args:
        run_id: The ID of the run to collect results for
        gcs_bucket_name: GCS bucket name for storage

    Returns:
        Dictionary ready for FinalResultsResponse containing:
        - Core results (backwards compatible): model_metrics, top_features, explainability, download_url
        - Enhanced information: run_summary, pipeline_status, validation_summary, etc.

    Raises:
        FileNotFoundError: If required result files are missing
        ValueError: If results are incomplete or invalid
    """

    # Read metadata.json which contains aggregated results
    metadata = read_json_from_gcs(run_id, constants.METADATA_FILENAME)
    if metadata is None:
        raise FileNotFoundError(f"Metadata not found for run {run_id} in GCS")

    # Read status information
    status_data = read_json_from_gcs(run_id, constants.STATUS_FILENAME)

    # Read validation summary
    validation_data = read_json_from_gcs(run_id, constants.VALIDATION_FILENAME)

    # Extract automl_info and ensure it's valid
    automl_info = metadata.get('automl_info', {})
    if not automl_info:
        raise ValueError(f"AutoML information missing from metadata for run {run_id}")

    # Extract core model metrics
    model_metrics = automl_info.get('performance_metrics', {})
    if not model_metrics:
        raise ValueError(f"Model performance metrics missing for run {run_id}")

    # Extract top features - prioritize SHAP feature importance from explain_info
    top_features = []
    feature_importance_scores = {}
    
    # First, try to get SHAP feature importance from explain_info or top-level metadata
    explain_info = metadata.get('explain_info', {})
    if explain_info.get('feature_importance_scores'):
        feature_importance_scores = explain_info['feature_importance_scores']
        # Convert to sorted list for top_features (backwards compatibility)
        sorted_features = sorted(feature_importance_scores.items(), key=lambda x: x[1], reverse=True)
        top_features = [feature for feature, score in sorted_features]
    elif metadata.get('feature_importance', {}).get('ranking'):
        # Use top-level feature importance ranking
        top_features = metadata['feature_importance']['ranking']
        feature_importance_scores = metadata['feature_importance'].get('scores', {})
    else:
        # Fallback to automl feature importance
        top_features = automl_info.get('feature_importance', [])
        if not top_features:
            top_features = automl_info.get('top_features', [])

    # Extract explainability info
    explain_info = metadata.get('explain_info', {})
    explainability = {
        "available": "true" if explain_info else "false",
        "method": explain_info.get('tool_used', 'None'),
        "plot_filename": None
    }

    # Check for SHAP plot availability in GCS
    if explain_info:
        shap_plot_gcs_path = explain_info.get('shap_summary_plot_gcs_path')
        if shap_plot_gcs_path and check_run_file_exists(run_id, shap_plot_gcs_path):
            explainability["plot_filename"] = shap_plot_gcs_path.split('/')[-1]  # Extract filename
        else:
            # Fallback to standard naming
            shap_filename = f"{constants.PLOTS_DIR}/{constants.SHAP_SUMMARY_PLOT}"
            if check_run_file_exists(run_id, shap_filename):
                explainability["plot_filename"] = constants.SHAP_SUMMARY_PLOT

    # Build enhanced run summary
    run_summary = {
        "run_id": run_id,
        "original_filename": metadata.get('original_filename', 'Unknown'),
        "upload_timestamp": metadata.get('timestamp'),
        "target_column": automl_info.get('target_column'),
        "task_type": automl_info.get('task_type'),
        "model_name": automl_info.get('best_model_name'),
        "storage_type": "gcs",
        "gcs_bucket": gcs_bucket_name
    }

    # Calculate initial shape
    if 'initial_rows' in metadata and 'initial_cols' in metadata:
        run_summary["initial_shape"] = [metadata['initial_rows'], metadata['initial_cols']]
    elif 'data_stats' in metadata:
        data_stats = metadata['data_stats']
        if 'shape' in data_stats:
            run_summary["initial_shape"] = data_stats['shape']

    # Pipeline Status Info
    pipeline_status = {
        "stage": "completed",
        "status": "completed",
        "message": None,
        "errors": None
    }
    
    if status_data:
        pipeline_status.update({
            "stage": status_data.get('stage', 'completed'),
            "status": status_data.get('status', 'completed'),
            "message": status_data.get('message'),
            "errors": status_data.get('errors', []) if status_data.get('errors') else None
        })

    # Validation Summary Info
    validation_summary = None
    if validation_data:
        validation_summary = {
            "overall_success": validation_data.get('overall_success', False),
            "total_expectations": validation_data.get('total_expectations', 0),
            "successful_expectations": validation_data.get('successful_expectations', 0),
            "failed_expectations": validation_data.get('failed_expectations', 0)
        }

    # Data Prep Summary
    data_prep_summary = None
    prep_info = metadata.get('prep_info')
    if prep_info:
        final_shape = prep_info.get('final_shape_after_prep')
        if final_shape and len(final_shape) == 2:
            final_shape = tuple(final_shape)
        
        # Check if profiling report exists in GCS
        profiling_available = False
        profiling_filename = None
        profile_filename = prep_info.get('profiling_report_filename')
        if profile_filename and check_run_file_exists(run_id, profile_filename):
            profiling_available = True
            profiling_filename = profile_filename
        
        data_prep_summary = {
            "final_shape": final_shape,
            "cleaning_steps": prep_info.get('cleaning_steps_performed', []),
            "profiling_report_available": profiling_available,
            "profiling_report_filename": profiling_filename
        }

    # AutoML Summary with Model Comparison Results
    model_file_available = check_run_file_exists(run_id, f"{constants.MODEL_DIR}/pycaret_pipeline.pkl")
    automl_summary = {
        "tool_used": automl_info.get('tool_used'),
        "best_model_name": automl_info.get('best_model_name'),
        "target_column": automl_info.get('target_column'),
        "task_type": automl_info.get('task_type'),
        "performance_metrics": model_metrics,
        "model_file_available": model_file_available,
        
        # Model comparison results
        "model_comparison_available": automl_info.get('model_comparison_available', False),
        "total_models_compared": automl_info.get('total_models_compared', 0),
        "top_models_summary": automl_info.get('top_models_summary', []),
        "all_model_results": automl_info.get('model_comparison_results', {}).get('all_model_results', []) if automl_info.get('model_comparison_results') else []
    }

    # Explainability Summary
    explainability_summary = None
    if explain_info:
        shap_available = explainability.get("plot_filename") is not None
        
        explainability_summary = {
            "tool_used": explain_info.get('tool_used'),
            "features_explained": explain_info.get('features_explained'),
            "samples_used": explain_info.get('samples_used_for_explanation'),
            "shap_plot_available": shap_available,
            "shap_plot_filename": explainability.get("plot_filename")
        }

    # Available Downloads with file sizes from GCS
    available_downloads = {
        "original_data": False,
        "cleaned_data": False,
        "metadata_json": False,
        "validation_report": False,
        "profile_report": False,
        "model_artifacts": False,
        "shap_plot": False,
        "pipeline_log": False,
        "file_sizes": {}
    }

    # Check each file and calculate sizes from GCS
    file_checks = [
        ("original_data", constants.ORIGINAL_DATA_FILENAME),
        ("cleaned_data", constants.CLEANED_DATA_FILE),
        ("metadata_json", constants.METADATA_FILENAME),
        ("validation_report", constants.VALIDATION_FILENAME),
        ("pipeline_log", constants.PIPELINE_LOG_FILENAME),
    ]

    for file_key, filename in file_checks:
        if check_run_file_exists(run_id, filename):
            available_downloads[file_key] = True
            file_size = check_file_size_gcs(run_id, filename)
            available_downloads["file_sizes"][file_key] = file_size / 1024  # KB

    # Check profile report
    if data_prep_summary and data_prep_summary["profiling_report_available"]:
        available_downloads["profile_report"] = True
        file_size = check_file_size_gcs(run_id, data_prep_summary["profiling_report_filename"])
        available_downloads["file_sizes"]["profile_report"] = file_size / 1024

    # Check model artifacts - list files in model directory
    try:
        all_files = list_run_files(run_id)
        model_files = [f for f in all_files if f.startswith(f"{constants.MODEL_DIR}/")]
        if model_files:
            available_downloads["model_artifacts"] = True
            # Calculate total size of model files
            total_size = sum(check_file_size_gcs(run_id, f) for f in model_files)
            available_downloads["file_sizes"]["model_artifacts"] = total_size / 1024
    except:
        pass  # Model directory check is optional

    # Check SHAP plot
    if explainability_summary and explainability_summary["shap_plot_available"]:
        available_downloads["shap_plot"] = True
        file_size = check_file_size_gcs(run_id, explainability_summary["shap_plot_filename"])
        available_downloads["file_sizes"]["shap_plot"] = file_size / 1024

    # Check prediction readiness
    prediction_readiness = check_prediction_readiness_gcs(run_id, gcs_bucket_name)

    # === ASSEMBLE FINAL RESPONSE ===
    results = {
        # Core results (backwards compatibility)
        "model_metrics": model_metrics,
        "top_features": top_features,
        "explainability": explainability,
        "download_url": None,
        
        # Feature importance information
        "feature_importance_scores": feature_importance_scores,
        "feature_importance_available": bool(feature_importance_scores),
        
        # Enhanced detailed information
        "run_summary": run_summary,
        "pipeline_status": pipeline_status,
        "validation_summary": validation_summary,
        "data_prep_summary": data_prep_summary,
        "automl_summary": automl_summary,
        "explainability_summary": explainability_summary,
        "available_downloads": available_downloads,
        
        # Prediction readiness information
        "prediction_readiness": prediction_readiness
    }

    return results


def build_results(run_id: str) -> Dict:
    """
    Legacy compatibility function - redirects to GCS version.
    
    Args:
        run_id: The ID of the run to collect results for

    Returns:
        Dictionary ready for FinalResultsResponse
    """
    return build_results_gcs(run_id)


def get_pipeline_status_gcs(run_id: str,
                           gcs_bucket_name: str = PROJECT_BUCKET_NAME) -> Dict:
    """
    Get current pipeline status from status.json in GCS.

    Args:
        run_id: The ID of the run to check status for
        gcs_bucket_name: GCS bucket name for storage

    Returns:
        Dictionary containing stage, status, message, and progress_pct

    Raises:
        FileNotFoundError: If status.json is missing in GCS
    """

    # Read status.json from GCS
    status_data = read_json_from_gcs(run_id, "status.json")
    
    if status_data is None:
        raise FileNotFoundError(f"Status file not found for run {run_id} in GCS")

    # Map internal stage names to clean API names
    stage = status_data.get("stage", "unknown")
    clean_stage = STAGE_NAME_MAP.get(stage, stage)

    # Get status with default
    status = status_data.get("status", "unknown")

    # Calculate progress percentage using the improved mapping
    if status == "completed":
        progress_pct = 100
    elif status == "failed":
        progress_pct = 100  # Show 100% for failed to indicate completion
    elif status == "running":
        progress_pct = STAGE_PROGRESS_MAP.get(clean_stage, 50)
    else:
        # For pending or unknown status
        progress_pct = STAGE_PROGRESS_MAP.get(clean_stage, 0)

    # Ensure message is informative even if missing
    message = status_data.get("message")
    if not message:
        if status == "completed":
            message = f"Pipeline completed successfully at {clean_stage} stage"
        elif status == "failed":
            message = f"Pipeline failed at {clean_stage} stage"
        elif status == "running":
            message = f"Pipeline running: {clean_stage} stage in progress"
        else:
            message = f"Pipeline {status}: {clean_stage} stage"

    return {
        "stage": clean_stage,
        "status": status,
        "message": message,
        "progress_pct": progress_pct
    }


def check_prediction_readiness_gcs(run_id: str, 
                                  gcs_bucket_name: str = PROJECT_BUCKET_NAME) -> Dict[str, bool]:
    """
    Check if prediction functionality is ready for this run.
    
    Args:
        run_id: The ID of the run to check
        gcs_bucket_name: GCS bucket name for storage
        
    Returns:
        Dictionary with prediction readiness flags
    """
    readiness = {
        "prediction_ready": False,
        "model_file_available": False,
        "column_mapping_available": False,
        "original_data_available": False,
        "metadata_available": False
    }
    
    try:
        # Check for trained model file
        model_path = f"{constants.MODEL_DIR}/pycaret_pipeline.pkl"
        readiness["model_file_available"] = check_run_file_exists(run_id, model_path)
        
        # Check for column mapping (needed for input transformation)
        readiness["column_mapping_available"] = check_run_file_exists(run_id, "column_mapping.json")
        
        # Check for original data (needed for input schema generation)
        readiness["original_data_available"] = check_run_file_exists(run_id, constants.ORIGINAL_DATA_FILENAME)
        
        # Check for metadata (needed for target info)
        readiness["metadata_available"] = check_run_file_exists(run_id, constants.METADATA_FILENAME)
        
        # Prediction is ready if all required files are available
        readiness["prediction_ready"] = all([
            readiness["model_file_available"],
            readiness["column_mapping_available"], 
            readiness["original_data_available"],
            readiness["metadata_available"]
        ])
        
    except Exception:
        # If any error occurs, prediction is not ready
        pass
    
    return readiness


def get_pipeline_status(run_id: str) -> Dict:
    """
    Legacy compatibility function - redirects to GCS version.
    
    Args:
        run_id: The ID of the run to check status for

    Returns:
        Dictionary containing stage, status, message, and progress_pct
    """
    return get_pipeline_status_gcs(run_id)
