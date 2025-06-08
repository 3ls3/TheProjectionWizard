# Metadata.json Usage Across Pipeline Steps

This document provides a comprehensive overview of how `metadata.json` is created, updated, and read across all pipeline steps in The Projection Wizard. Understanding this flow is crucial for contributors to avoid corrupting shared state and extend pipeline logic safely.

## Overview

The `metadata.json` file serves as the central state store for each pipeline run, containing configuration, intermediate results, and artifacts information that flows between pipeline steps. Each step both consumes metadata from previous steps and adds its own metadata for subsequent steps.

## Pipeline Step Analysis

### Step 1: Ingest (`pipeline/step_1_ingest/ingest_logic.py`)

**Purpose:** Initialize the pipeline run and create the foundational metadata structure.

- **Reads:** None (this step initializes `metadata.json`)
- **Writes:**
  - `run_id`: Unique identifier for this pipeline run
  - `timestamp`: UTC timestamp when ingestion occurred
  - `original_filename`: Name of the uploaded CSV file
  - `initial_rows`: Number of rows in the original dataset
  - `initial_cols`: Number of columns in the original dataset  
  - `initial_dtypes`: Dictionary mapping column names to their pandas data types

**Implementation:** Uses `schemas.BaseMetadata` Pydantic model to ensure structure validation before writing.

**Dependencies:** None - this is the entry point.

**Notes:** All subsequent steps depend on this foundational metadata. If ingestion fails, these fields may be `None`.

---

### Step 2: Schema Definition

#### Target Definition (`pipeline/step_2_schema/target_definition_logic.py`)

**Purpose:** Confirm the target column and task type for ML training.

- **Reads:** 
  - Entire existing metadata via `storage.read_metadata(run_id)`
- **Writes:**
  - `target_info`: Dictionary containing:
    - `name`: Target column name
    - `task_type`: "classification" or "regression"
    - `ml_type`: Specific ML-ready format (e.g., "binary_01", "multiclass_text_labels")
    - `user_confirmed_at`: UTC timestamp of confirmation
  - `task_type`: Top-level convenience field (same as `target_info.task_type`)

**Implementation:** Validates inputs against `constants.TASK_TYPES` and `constants.TARGET_ML_TYPES` before writing.

#### Feature Schema Definition (`pipeline/step_2_schema/feature_definition_logic.py`)

**Purpose:** Confirm data types and encoding roles for all feature columns.

- **Reads:**
  - Entire existing metadata via `storage.read_metadata(run_id)`
- **Writes:**
  - `feature_schemas`: Dictionary where keys are column names and values contain:
    - `dtype`: Final data type for the column
    - `encoding_role`: How to encode the feature (e.g., "numeric-continuous", "categorical-nominal")
    - `source`: "user_confirmed" or "system_defaulted"
  - `feature_schemas_confirmed_at`: UTC timestamp of confirmation

**Dependencies:** Target definition must be completed first (for excluding target from feature schemas).

**Notes:** This step can use system defaults for columns not explicitly reviewed by users.

---

### Step 3: Validation (`pipeline/step_3_validation/validation_runner.py`)

**Purpose:** Validate data quality using Great Expectations based on confirmed schemas.

- **Reads:**
  - `feature_schemas`: Used to generate appropriate validation expectations
  - `target_info`: Used for target-specific validation rules
- **Writes:**
  - `validation_info`: Dictionary containing:
    - `passed`: Boolean indicating overall validation success
    - `report_filename`: Name of the validation report file
    - `total_expectations_evaluated`: Number of validation rules checked
    - `successful_expectations`: Number of validation rules that passed

**Implementation:** 
- Converts metadata dictionaries to Pydantic objects (`FeatureSchemaInfo`, `TargetInfo`)
- Generates Great Expectations suite based on encoding roles
- Saves detailed validation results to separate `validation.json` file

**Dependencies:** Requires completed schema definition (both target and features).

**Notes:** Validation can complete even if some expectations fail - the stage reports completion status vs. data quality status separately.

---

### Step 4: Prep (`pipeline/step_4_prep/prep_runner.py`)

**Purpose:** Clean and encode data for ML readiness.

- **Reads:**
  - `target_info`: Used for target-aware cleaning and encoding
  - `feature_schemas`: Used to determine appropriate cleaning/encoding strategies per column
- **Writes:**
  - `prep_info`: Dictionary containing:
    - `cleaning_steps_performed`: List of strings describing cleaning actions taken
    - `encoders_scalers_info`: Dictionary with paths and metadata for saved encoders/scalers
    - `cleaned_data_path`: Path to the cleaned CSV file (typically "cleaned_data.csv")
    - `profiling_report_path`: Path to the ydata-profiling HTML report (if generated)
    - `final_shape_after_prep`: List `[rows, columns]` of the final dataset shape

**Implementation:**
- Orchestrates cleaning (`cleaning_logic.py`), encoding (`encoding_logic.py`), and profiling (`profiling_logic.py`)
- Saves ML-ready artifacts (encoders, scalers) to the `model/` directory
- Converts metadata dictionaries to Pydantic objects for type safety

**Dependencies:** Requires validation stage completion to ensure data quality checks are done.

**Notes:** The `encoders_scalers_info` contains paths to serialized sklearn transformers needed for future predictions.

---

### Step 5: AutoML (`pipeline/step_5_automl/automl_runner.py`)

**Purpose:** Train ML models using PyCaret and save the best performing model.

- **Reads:**
  - `target_info`: Used for task type and target column identification
  - `prep_info`: Validates that data preparation was completed
- **Writes:**
  - `automl_info`: Dictionary containing:
    - `tool_used`: "PyCaret" (identifies the AutoML tool)
    - `best_model_name`: Name of the best performing model
    - `pycaret_pipeline_path`: Relative path to saved PyCaret pipeline (typically "model/pycaret_pipeline.pkl")
    - `performance_metrics`: Dictionary of model performance metrics (accuracy, F1, etc.)
    - `automl_completed_at`: UTC timestamp of completion
    - `target_column`: Target column name (for convenience)
    - `task_type`: Task type (for convenience)
    - `dataset_shape_for_training`: List `[rows, columns]` used for training

**Implementation:**
- Delegates actual training to `pycaret_logic.py`
- Validates that required preprocessing artifacts exist
- Saves trained PyCaret pipeline to disk

**Dependencies:** Requires prep stage completion for cleaned data and feature encoders.

**Notes:** The saved PyCaret pipeline contains the entire preprocessing + model pipeline for easy deployment.

---

### Step 6: Explainability (`pipeline/step_6_explain/explain_runner.py`)

**Purpose:** Generate model explanations using SHAP for model interpretability.

- **Reads:**
  - `automl_info`: Used to locate the trained model (`pycaret_pipeline_path`) and get model metadata
  - `target_info`: Used for task-specific explanation approaches and target column identification
- **Writes:**
  - `explain_info`: Dictionary containing:
    - `tool_used`: "SHAP" (identifies the explainability tool)
    - `explanation_type`: "global_summary" (type of explanation generated)
    - `shap_summary_plot_path`: Relative path to SHAP summary plot image (typically "plots/shap_summary.png")
    - `explain_completed_at`: UTC timestamp of completion
    - `target_column`: Target column name (for convenience)
    - `task_type`: Task type (for convenience)  
    - `features_explained`: Number of features included in explanation
    - `samples_used_for_explanation`: Number of data samples used for SHAP analysis

**Implementation:**
- Loads the trained PyCaret pipeline from `automl_info.pycaret_pipeline_path`
- Uses `shap_logic.py` to generate SHAP explanations and summary plots
- Saves visualization artifacts to `plots/` directory
- Validates metadata using `schemas.TargetInfo` and `schemas.AutoMLInfo` Pydantic objects

**Dependencies:** Requires AutoML stage completion for trained model and cleaned data from prep stage.

**Notes:** SHAP explanations help users understand which features most influence model predictions. The stage generates global feature importance visualizations.

## Metadata Evolution Example

Here's how `metadata.json` evolves through a typical pipeline run:

### After Step 1 (Ingest):
```json
{
  "run_id": "run_20241201_143022_abc123",
  "timestamp": "2024-12-01T14:30:22.123456Z",
  "original_filename": "customer_data.csv",
  "initial_rows": 1000,
  "initial_cols": 15,
  "initial_dtypes": {
    "customer_id": "int64",
    "age": "int64",
    "income": "float64",
    "purchased": "bool"
  }
}
```

### After Step 2 (Schema):
```json
{
  // ... previous fields ...
  "target_info": {
    "name": "purchased",
    "task_type": "classification", 
    "ml_type": "binary_boolean",
    "user_confirmed_at": "2024-12-01T14:35:15.789Z"
  },
  "task_type": "classification",
  "feature_schemas": {
    "customer_id": {
      "dtype": "int64",
      "encoding_role": "identifier_ignore", 
      "source": "user_confirmed"
    },
    "age": {
      "dtype": "int64",
      "encoding_role": "numeric-continuous",
      "source": "system_defaulted"
    }
    // ... other columns ...
  },
  "feature_schemas_confirmed_at": "2024-12-01T14:35:15.789Z"
}
```

### After Step 3 (Validation):
```json
{
  // ... previous fields ...
  "validation_info": {
    "passed": true,
    "report_filename": "validation.json",
    "total_expectations_evaluated": 25,
    "successful_expectations": 23
  }
}
```

### After Step 4 (Prep):
```json
{
  // ... previous fields ...
  "prep_info": {
    "cleaning_steps_performed": [
      "Starting data cleaning with 1000 rows and 15 columns",
      "Imputed 45 NaNs with median (32.5) for numeric column: age",
      "Removed 12 duplicate rows",
      "Cleaning completed with 988 rows and 15 columns"
    ],
    "encoders_scalers_info": {
      "age_scaler": {
        "type": "StandardScaler",
        "path": "model/age_scaler.joblib",
        "column": "age",
        "mean": 32.5,
        "scale": 12.3
      },
      "income_scaler": {
        "type": "StandardScaler", 
        "path": "model/income_scaler.joblib",
        "column": "income"
      }
    },
    "cleaned_data_path": "cleaned_data.csv",
    "profiling_report_path": "run_20241201_143022_abc123_profile.html",
    "final_shape_after_prep": [988, 18]
  }
}
```

### After Step 5 (AutoML):
```json
{
  // ... previous fields ...
  "automl_info": {
    "tool_used": "PyCaret",
    "best_model_name": "RandomForestClassifier", 
    "pycaret_pipeline_path": "model/pycaret_pipeline.pkl",
    "performance_metrics": {
      "AUC": 0.892,
      "Accuracy": 0.845,
      "F1": 0.821,
      "Precision": 0.834,
      "Recall": 0.809
    },
    "automl_completed_at": "2024-12-01T14:52:33.456Z",
    "target_column": "purchased",
    "task_type": "classification",
    "dataset_shape_for_training": [988, 18]
  }
}
```

### After Step 6 (Explainability):
```json
{
  // ... previous fields ...
  "explain_info": {
    "tool_used": "SHAP",
    "explanation_type": "global_summary",
    "shap_summary_plot_path": "plots/shap_summary.png",
    "explain_completed_at": "2024-12-01T14:55:12.789Z",
    "target_column": "purchased",
    "task_type": "classification",
    "features_explained": 17,
    "samples_used_for_explanation": 100
  }
}
```

## Best Practices for Contributors

### Reading Metadata
- Always use `storage.read_metadata(run_id)` or `storage.read_json(run_id, constants.METADATA_FILENAME)`
- Validate that required keys exist before accessing them
- Convert dictionary data to Pydantic objects when available for type safety

### Writing Metadata
- Always read existing metadata first, then update specific keys
- Use `storage.write_metadata(run_id, metadata_dict)` for atomic writes
- Include timestamps for audit trails
- Validate data types before writing

### Error Handling
- If required metadata is missing, fail gracefully with clear error messages
- Check for partially completed stages using status.json in addition to metadata
- Don't assume metadata structure - always validate

### Extending the Pipeline
- Add new metadata fields to existing dictionaries rather than top-level keys when possible
- Document new metadata fields in this file
- Consider backward compatibility when modifying existing metadata structure
- Use Pydantic schemas when adding complex metadata structures

## Common Pitfalls to Avoid

1. **Race Conditions:** Always read-modify-write metadata atomically
2. **Missing Dependencies:** Check that prerequisite stages have completed before reading their metadata
3. **Type Assumptions:** Metadata values may be `None` if previous stages failed
4. **Path Issues:** Store relative paths in metadata, resolve to absolute paths when needed
5. **Serialization:** Ensure all metadata values are JSON-serializable

This documentation should be updated whenever metadata structure changes are made to the pipeline. 