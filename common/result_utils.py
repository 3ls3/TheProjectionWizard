"""
Result aggregation utilities for The Projection Wizard.
Functions to collect and format final pipeline results for the API.
"""

import json
from typing import Dict
from pathlib import Path
from common import storage, constants


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


def build_results(run_id: str) -> Dict:
    """
    Build comprehensive results dictionary from pipeline artifacts.

    Reads artifacts created by all pipeline stages to construct a complete
    results payload for the API that matches the Streamlit results page.

    Args:
        run_id: The ID of the run to collect results for

    Returns:
        Dictionary ready for FinalResultsResponse containing:
        - Core results (backwards compatible): model_metrics, top_features, explainability, download_url
        - Enhanced information: run_summary, pipeline_status, validation_summary, etc.

    Raises:
        FileNotFoundError: If required result files are missing
        ValueError: If results are incomplete or invalid
    """

    # Get run directory path
    run_dir = storage.get_run_dir(run_id)

    # Read metadata.json which contains aggregated results
    metadata = storage.read_metadata(run_id)
    if metadata is None:
        raise FileNotFoundError(f"Metadata not found for run {run_id}")

    # Read status information
    status_data = None
    try:
        status_data = storage.read_json(run_id, constants.STATUS_FILENAME)
    except Exception:
        pass  # Status file may not exist in some cases

    # Read validation summary
    validation_data = None
    try:
        validation_data = storage.read_json(run_id, constants.VALIDATION_FILENAME)
    except Exception:
        pass  # Validation file may not exist

    # Check that AutoML stage completed for core metrics
    if 'automl_info' not in metadata:
        raise FileNotFoundError(f"AutoML results not found for run {run_id}")

    # Check that Explain stage completed for core explainability
    if 'explain_info' not in metadata:
        raise FileNotFoundError(f"Explainability results not found for run {run_id}")

    # === BUILD CORE RESULTS (backwards compatibility) ===
    automl_info = metadata['automl_info']

    # Extract model metrics
    model_metrics = {}
    if 'performance_metrics' in automl_info:
        for metric, value in automl_info['performance_metrics'].items():
            model_metrics[metric.lower()] = float(value)
    else:
        raise ValueError(f"Performance metrics missing from AutoML results for run {run_id}")

    # Extract top features from feature schemas
    top_features = []
    if 'feature_schemas' in metadata:
        target_column = metadata.get('target_info', {}).get('name', '')
        top_features = [
            col for col in metadata['feature_schemas'].keys()
            if col != target_column
        ]

    # Create explainability mapping (placeholder implementation)
    explainability = {}
    for i, feature in enumerate(top_features):
        importance = max(0.1, 1.0 - (i * 0.15))
        explainability[feature] = f"{importance:.2f}"

    # === BUILD ENHANCED INFORMATION ===

    # Run Summary
    run_summary = {
        "run_id": run_id,
        "timestamp": metadata.get('created_at') or metadata.get('timestamp'),
        "original_filename": metadata.get('original_filename'),
        "initial_shape": None,
        "target_info": metadata.get('target_info')
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
        
        # Check if profiling report exists
        profiling_available = False
        profiling_filename = None
        if 'profiling_report_path' in prep_info:
            profile_path = run_dir / prep_info['profiling_report_path']
            if profile_path.exists():
                profiling_available = True
                profiling_filename = profile_path.name
        
        data_prep_summary = {
            "final_shape": final_shape,
            "cleaning_steps": prep_info.get('cleaning_steps_performed', []),
            "profiling_report_available": profiling_available,
            "profiling_report_filename": profiling_filename
        }

    # AutoML Summary
    automl_summary = {
        "tool_used": automl_info.get('tool_used'),
        "best_model_name": automl_info.get('best_model_name'),
        "target_column": automl_info.get('target_column'),
        "task_type": automl_info.get('task_type'),
        "performance_metrics": model_metrics,
        "model_file_available": (run_dir / constants.MODEL_DIR / 'pycaret_pipeline.pkl').exists()
    }

    # Explainability Summary
    explainability_summary = None
    explain_info = metadata.get('explain_info')
    if explain_info:
        shap_available = False
        shap_filename = None
        if 'shap_summary_plot_path' in explain_info:
            shap_path = run_dir / explain_info['shap_summary_plot_path']
            if shap_path.exists():
                shap_available = True
                shap_filename = shap_path.name
        
        explainability_summary = {
            "tool_used": explain_info.get('tool_used'),
            "features_explained": explain_info.get('features_explained'),
            "samples_used": explain_info.get('samples_used_for_explanation'),
            "shap_plot_available": shap_available,
            "shap_plot_filename": shap_filename
        }

    # Available Downloads with file sizes
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

    # Check each file and calculate sizes
    file_checks = [
        ("original_data", run_dir / constants.ORIGINAL_DATA_FILENAME),
        ("cleaned_data", run_dir / constants.CLEANED_DATA_FILE),
        ("metadata_json", run_dir / constants.METADATA_FILENAME),
        ("validation_report", run_dir / constants.VALIDATION_FILENAME),
        ("pipeline_log", run_dir / constants.PIPELINE_LOG_FILENAME),
    ]

    for file_key, file_path in file_checks:
        if file_path.exists():
            available_downloads[file_key] = True
            available_downloads["file_sizes"][file_key] = file_path.stat().st_size / 1024  # KB

    # Check profile report
    if data_prep_summary and data_prep_summary["profiling_report_available"]:
        available_downloads["profile_report"] = True
        profile_path = run_dir / prep_info['profiling_report_path']
        available_downloads["file_sizes"]["profile_report"] = profile_path.stat().st_size / 1024

    # Check model artifacts directory
    model_dir = run_dir / constants.MODEL_DIR
    if model_dir.exists() and any(model_dir.iterdir()):
        available_downloads["model_artifacts"] = True
        # Calculate total size of model directory
        total_size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
        available_downloads["file_sizes"]["model_artifacts"] = total_size / 1024

    # Check SHAP plot
    if explainability_summary and explainability_summary["shap_plot_available"]:
        available_downloads["shap_plot"] = True
        shap_path = run_dir / explain_info['shap_summary_plot_path']
        available_downloads["file_sizes"]["shap_plot"] = shap_path.stat().st_size / 1024

    # === ASSEMBLE FINAL RESPONSE ===
    results = {
        # Core results (backwards compatibility)
        "model_metrics": model_metrics,
        "top_features": top_features,
        "explainability": explainability,
        "download_url": None,
        
        # Enhanced detailed information
        "run_summary": run_summary,
        "pipeline_status": pipeline_status,
        "validation_summary": validation_summary,
        "data_prep_summary": data_prep_summary,
        "automl_summary": automl_summary,
        "explainability_summary": explainability_summary,
        "available_downloads": available_downloads
    }

    return results


def get_pipeline_status(run_id: str) -> Dict:
    """
    Get current pipeline status from status.json.

    Args:
        run_id: The ID of the run to check status for

    Returns:
        Dictionary containing stage, status, message, and progress_pct

    Raises:
        FileNotFoundError: If status.json is missing
    """

    # Read status.json directly
    run_dir = storage.get_run_dir(run_id)
    status_path = run_dir / "status.json"

    if not status_path.exists():
        raise FileNotFoundError(f"Status file not found for run {run_id}")

    with open(status_path, 'r') as f:
        status_data = json.load(f)

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
