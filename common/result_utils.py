"""
Result aggregation utilities for The Projection Wizard.
Functions to collect and format final pipeline results for the API.
"""

import json
from typing import Dict
from common import storage


def build_results(run_id: str) -> Dict:
    """
    Build final results dictionary from pipeline artifacts.

    Reads artifacts created by AutoML (step 5) and Explain (step 6) stages
    to construct a complete results payload for the API.

    Args:
        run_id: The ID of the run to collect results for

    Returns:
        Dictionary ready for FinalResultsResponse containing:
        - model_metrics: Performance metrics from AutoML
        - top_features: List of most important features
        - explainability: Feature importance mapping
        - download_url: Optional download link (currently None)

    Raises:
        FileNotFoundError: If required result files are missing
        ValueError: If results are incomplete or invalid
    """

    # Read metadata.json which contains aggregated results
    metadata = storage.read_metadata(run_id)
    if metadata is None:
        raise FileNotFoundError(f"Metadata not found for run {run_id}")

    # Check that AutoML stage completed
    if 'automl_info' not in metadata:
        raise FileNotFoundError(f"AutoML results not found for run {run_id}")

    # Check that Explain stage completed
    if 'explain_info' not in metadata:
        raise FileNotFoundError(
            f"Explainability results not found for run {run_id}"
        )

    automl_info = metadata['automl_info']

    # Extract model metrics
    model_metrics = {}
    if 'performance_metrics' in automl_info:
        # Convert all values to float and ensure consistency
        for metric, value in automl_info['performance_metrics'].items():
            model_metrics[metric.lower()] = float(value)
    else:
        raise ValueError(
            f"Performance metrics missing from AutoML results for run {run_id}"
        )

    # Extract top features from feature schemas
    # For now, use all non-target features as "top features"
    # In the future, this could be based on actual feature importance scores
    top_features = []
    if 'feature_schemas' in metadata:
        target_column = metadata.get('target_info', {}).get('name', '')
        top_features = [
            col for col in metadata['feature_schemas'].keys()
            if col != target_column
        ]

    # Create explainability mapping
    # For now, create placeholder values since we don't have SHAP values
    # In production, this would read actual SHAP values from files
    explainability = {}
    for i, feature in enumerate(top_features):
        # Generate decreasing importance scores as placeholders
        importance = max(0.1, 1.0 - (i * 0.15))
        explainability[feature] = f"{importance:.2f}"

    # For now, download_url is None (could be implemented later)
    download_url = None

    results = {
        "model_metrics": model_metrics,
        "top_features": top_features,
        "explainability": explainability,
        "download_url": download_url
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

    # Map internal stage names to cleaner API names
    stage_mapping = {
        "step_3_validation": "validation",
        "step_4_prep": "prep",
        "step_5_automl": "automl",
        "step_6_explain": "explain"
    }

    stage = status_data.get("stage", "unknown")
    clean_stage = stage_mapping.get(stage, stage)

    # Calculate rough progress percentage
    progress_pct = None
    status = status_data.get("status", "unknown")

    if status == "completed":
        progress_pct = 100
    elif status == "failed":
        progress_pct = 0
    elif status == "running":
        # Map stages to rough progress percentages
        stage_progress = {
            "validation": 20,
            "prep": 40,
            "automl": 70,
            "explain": 90
        }
        progress_pct = stage_progress.get(clean_stage, 50)

    return {
        "stage": clean_stage,
        "status": status,
        "message": status_data.get("message"),
        "progress_pct": progress_pct
    }
