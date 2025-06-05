"""
Constants and configuration settings for The Projection Wizard.
Contains default paths, stage names, and core AutoML settings.
"""

from pathlib import Path
from typing import Dict, List

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = DATA_DIR / "runs"
FIXTURES_DIR = DATA_DIR / "fixtures"

# Pipeline stage constants mapped to actual bucket directory names
INGEST_STAGE = "step_1_ingest"
SCHEMA_STAGE = "step_2_schema"
VALIDATION_STAGE = "step_3_validation"
PREP_STAGE = "step_4_prep"
AUTOML_STAGE = "step_5_automl"
EXPLAIN_STAGE = "step_6_explain"
RESULTS_STAGE = "results"  # UI stage, not a bucket

# Pipeline stage names (using directory-mapped constants)
PIPELINE_STAGES = [
    INGEST_STAGE,
    SCHEMA_STAGE,
    VALIDATION_STAGE,
    PREP_STAGE,
    AUTOML_STAGE,
    EXPLAIN_STAGE,
    RESULTS_STAGE
]

# Mapping for easier lookup and validation
STAGE_TO_DIRECTORY = {
    INGEST_STAGE: INGEST_STAGE,
    SCHEMA_STAGE: SCHEMA_STAGE,
    VALIDATION_STAGE: VALIDATION_STAGE,
    PREP_STAGE: PREP_STAGE,
    AUTOML_STAGE: AUTOML_STAGE,
    EXPLAIN_STAGE: EXPLAIN_STAGE,
    RESULTS_STAGE: "ui"  # Results stage maps to UI directory
}

# Human-readable stage names for UI display
STAGE_DISPLAY_NAMES = {
    INGEST_STAGE: "Data Ingestion",
    SCHEMA_STAGE: "Schema Validation", 
    VALIDATION_STAGE: "Data Validation",
    PREP_STAGE: "Data Preparation",
    AUTOML_STAGE: "Model Training",
    EXPLAIN_STAGE: "Model Explanation",
    RESULTS_STAGE: "Results & Downloads"
}

# Directory names
DATA_DIR_NAME = "data"
RUNS_DIR_NAME = "runs"

# File names for run artifacts
ORIGINAL_DATA_FILE = "original_data.csv"
ORIGINAL_DATA_FILENAME = "original_data.csv"  # Alias for storage compatibility
CLEANED_DATA_FILE = "cleaned_data.csv"
METADATA_FILE = "metadata.json"
METADATA_FILENAME = "metadata.json"  # Alias for storage compatibility
STATUS_FILE = "status.json"
STATUS_FILENAME = "status.json"  # Alias for storage compatibility
VALIDATION_FILE = "validation.json"
VALIDATION_FILENAME = "validation.json"  # Alias for spec compatibility
PIPELINE_LOG_FILE = "pipeline.log"
PIPELINE_LOG_FILENAME = "pipeline.log"  # Alias for logger compatibility
PROFILE_REPORT_FILE = "ydata_profile.html"
RUN_INDEX_FILE = "index.csv"
RUN_INDEX_FILENAME = "index.csv"  # Alias for storage compatibility

# Model artifacts
MODEL_DIR = "model"
MODEL_FILE = "model.joblib"
SCALER_FILE = "scaler.pkl"

# Plots and visualizations
PLOTS_DIR = "plots"
SHAP_SUMMARY_PLOT = "shap_summary.png"

# AutoML configuration
AUTOML_CONFIG = {
    "train_size": 0.8,
    "session_id": 123,
    "compare_models_fold": 5,
    "compare_models_sort": "Accuracy",  # For classification
    "compare_models_sort_reg": "RMSE",  # For regression
    "finalize_model_estimator": "best",
    "optimize_threshold": False,
    "probability_threshold": 0.5
}

# Data validation thresholds
VALIDATION_CONFIG = {
    "missing_value_threshold": 0.3,  # 30% missing values triggers warning
    "duplicate_threshold": 0.1,      # 10% duplicates triggers warning
    "cardinality_threshold": 50,     # High cardinality categorical threshold
    "outlier_std_threshold": 3,      # Standard deviations for outlier detection
    "success_threshold": 80.0        # Minimum percentage of expectations that must pass
}

# Schema detection configuration
SCHEMA_CONFIG = {
    "max_categorical_cardinality": 20,  # Auto-detect as categorical if unique values <= this
    "min_numeric_threshold": 0.8,      # Fraction of numeric values to consider numeric type
    "importance_top_k": 10,            # Number of top important features to surface
    "mutual_info_random_state": 42
}

# UI configuration
UI_CONFIG = {
    "page_title": "The Projection Wizard",
    "page_icon": "🔮",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
    "max_upload_size_mb": 200
}

# Task types
TASK_TYPES = ["classification", "regression"]

# Target ML types for encoding
TARGET_ML_TYPES = [
    "binary_01",
    "binary_numeric", 
    "binary_text_labels",
    "binary_boolean",
    "multiclass_int_labels",
    "multiclass_text_labels",
    "high_cardinality_text",
    "numeric_continuous",
    "unknown_type"
]

# Status values
STATUS_VALUES = ["pending", "running", "completed", "failed", "cancelled"]

# Column encoding roles
ENCODING_ROLES = [
    "numeric-continuous",
    "numeric-discrete", 
    "categorical-nominal",
    "categorical-ordinal",
    "text",
    "datetime",
    "boolean",
    "target"
]

# Default cleaning strategies
CLEANING_STRATEGIES = {
    "missing_values": {
        "numeric": "median",
        "categorical": "mode",
        "text": "drop_row"
    },
    "duplicates": "drop",
    "outliers": "iqr_cap"  # IQR-based capping
}

# Helper functions for stage management
def get_stage_directory(stage: str) -> str:
    """
    Get the directory name for a given stage.
    
    Args:
        stage: Stage name (e.g., 'step_1_ingest')
        
    Returns:
        Directory name for the stage
        
    Raises:
        ValueError: If stage is not recognized
    """
    if stage not in STAGE_TO_DIRECTORY:
        raise ValueError(f"Unknown stage: {stage}. Valid stages: {list(STAGE_TO_DIRECTORY.keys())}")
    return STAGE_TO_DIRECTORY[stage]


def get_stage_display_name(stage: str) -> str:
    """
    Get the human-readable display name for a stage.
    
    Args:
        stage: Stage name (e.g., 'step_1_ingest')
        
    Returns:
        Human-readable display name
    """
    return STAGE_DISPLAY_NAMES.get(stage, stage)


def is_valid_stage(stage: str) -> bool:
    """
    Check if a stage name is valid.
    
    Args:
        stage: Stage name to validate
        
    Returns:
        True if stage is valid, False otherwise
    """
    return stage in PIPELINE_STAGES 