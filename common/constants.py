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

# Pipeline stage names
PIPELINE_STAGES = [
    "ingest",
    "schema", 
    "validation",
    "prep",
    "automl",
    "explain",
    "results"
]

# File names for run artifacts
ORIGINAL_DATA_FILE = "original_data.csv"
CLEANED_DATA_FILE = "cleaned_data.csv"
METADATA_FILE = "metadata.json"
STATUS_FILE = "status.json"
VALIDATION_FILE = "validation.json"
PIPELINE_LOG_FILE = "pipeline.log"
PROFILE_REPORT_FILE = "ydata_profile.html"
RUN_INDEX_FILE = "index.csv"

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
    "outlier_std_threshold": 3       # Standard deviations for outlier detection
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