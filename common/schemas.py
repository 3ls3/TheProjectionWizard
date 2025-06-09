"""
Pydantic data models for The Projection Wizard.
Defines schemas for metadata.json, status.json, and validation.json.

=============================================================================
⚠️  PARALLEL DEVELOPMENT COORDINATION REQUIRED ⚠️
=============================================================================

This file is CRITICAL for data consistency between components:
- API developer: Adds request/response models for endpoints
- Pipeline developer: May extend metadata models with new stage info
- Testing developer: Uses models for test data validation

COLLABORATION PROTOCOL:
1. 🗣️  ANNOUNCE in Slack: "Need to add/modify schema for [feature]"
2. ⏳ WAIT for team discussion - schema changes affect everyone!
3. 📝 ADD new models at bottom with clear docstrings
4. 🔄 EXTEND existing models using inheritance when possible
5. 🚫 NEVER remove or rename fields in existing models
6. ✅ TEST that all existing code still works
7. 📢 NOTIFY team: "Updated schemas.py - new models available"

SAFE PATTERNS:
✅ Add new Pydantic models for your features
✅ Extend existing models with Optional fields
✅ Create request/response models for API endpoints
✅ Use inheritance: class MyModel(BaseExistingModel)

DANGEROUS PATTERNS:
❌ Removing fields from existing models (breaks deserialization)
❌ Renaming fields (breaks all existing JSON files)
❌ Changing field types (breaks validation)
❌ Making optional fields required (breaks existing data)

EXAMPLE SAFE ADDITION:
```python
# API Request Models (Tim - Dec 2024)
class APIUploadRequest(BaseModel):
    filename: str
    file_size: int
    
class APIUploadResponse(BaseModel):
    run_id: str
    status: str
```

If modifying existing models, discuss in #projection-wizard Slack first!
=============================================================================
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, Literal
from pydantic import BaseModel, Field, validator

from .constants import STATUS_VALUES, TASK_TYPES, ENCODING_ROLES, PIPELINE_STAGES, TARGET_ML_TYPES


class BaseMetadata(BaseModel):
    """Base metadata for pipeline runs."""
    run_id: str
    timestamp: datetime
    original_filename: str
    initial_rows: Optional[int] = None
    initial_cols: Optional[int] = None
    initial_dtypes: Optional[Dict[str, str]] = None


class StageStatus(BaseModel):
    """Status information for a specific pipeline stage."""
    stage: str
    status: Literal['pending', 'in_progress', 'completed', 'failed']
    message: Optional[str] = None
    errors: Optional[List[str]] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    @validator('stage')
    def validate_stage(cls, v):
        if v not in PIPELINE_STAGES:
            raise ValueError(f'stage must be one of {PIPELINE_STAGES}')
        return v


class RunIndexEntry(BaseModel):
    """Entry for the run index CSV file."""
    run_id: str
    timestamp: datetime
    original_filename: str
    status: str  # Final status of the run, e.g., 'completed', 'failed_at_validation'


class ColumnSchema(BaseModel):
    """Schema information for a single column."""
    name: str
    dtype: str
    encoding_role: str = Field(..., description="Role for ML encoding")
    is_target: bool = False
    missing_count: int = 0
    unique_count: int = 0
    importance_score: Optional[float] = None
    suggested_encoding: Optional[str] = None
    
    @validator('encoding_role')
    def validate_encoding_role(cls, v):
        if v not in ENCODING_ROLES:
            raise ValueError(f'encoding_role must be one of {ENCODING_ROLES}')
        return v


class DataStats(BaseModel):
    """Basic statistics about the dataset."""
    n_rows: int
    n_cols: int
    missing_values_total: int
    duplicate_rows: int
    memory_usage_mb: float
    

class TargetInfo(BaseModel):
    """Information about the target variable for step 2 schema confirmation."""
    name: str = Field(..., description="Target column name")
    task_type: str = Field(..., description="classification or regression")
    ml_type: str = Field(..., description="ML-ready type for target encoding")
    user_confirmed_at: Optional[datetime] = None
    
    @validator('task_type')
    def validate_task_type(cls, v):
        if v not in TASK_TYPES:
            raise ValueError(f'task_type must be one of {TASK_TYPES}')
        return v
    
    @validator('ml_type')
    def validate_ml_type(cls, v):
        if v not in TARGET_ML_TYPES:
            raise ValueError(f'ml_type must be one of {TARGET_ML_TYPES}')
        return v


class FeatureSchemaInfo(BaseModel):
    """Schema information for a feature column after user confirmation."""
    dtype: str = Field(..., description="The final decided data type for the column")
    encoding_role: str = Field(..., description="The encoding role for ML processing")
    source: Literal['user_confirmed', 'system_defaulted'] = Field(..., description="Source of the schema decision")
    initial_dtype_suggestion: Optional[str] = None
    
    @validator('encoding_role')
    def validate_encoding_role(cls, v):
        if v not in ENCODING_ROLES:
            raise ValueError(f'encoding_role must be one of {ENCODING_ROLES}')
        return v


class MetadataWithTarget(BaseMetadata):
    """Metadata model that includes target information from step 2 schema confirmation."""
    target_info: Optional[TargetInfo] = None
    task_type: Optional[str] = None  # Top-level convenience field


class ValidationInfo(BaseModel):
    """Validation information for metadata.json."""
    passed: bool
    report_filename: str
    total_expectations_evaluated: Optional[int] = None
    successful_expectations: Optional[int] = None


class ValidationReportSummary(BaseModel):
    """Summary model for the top part of validation.json."""
    overall_success: bool
    total_expectations: int
    successful_expectations: int
    failed_expectations: int
    run_time_s: Optional[float] = None
    ge_version: Optional[str] = None
    results_ge_native: dict  # Raw, untyped GE result object


class MetadataWithFullSchema(MetadataWithTarget):
    """Metadata model that includes both target and feature schema information."""
    feature_schemas: Optional[Dict[str, FeatureSchemaInfo]] = None
    feature_schemas_confirmed_at: Optional[datetime] = None
    validation_info: Optional[ValidationInfo] = None


class AutoMLInfo(BaseModel):
    """Information about the AutoML stage results."""
    tool_used: str = Field(..., description="AutoML tool used (e.g., 'PyCaret')")
    best_model_name: Optional[str] = None
    pycaret_pipeline_path: Optional[str] = None  # Path relative to run_dir, e.g., "model/pycaret_pipeline.pkl"
    performance_metrics: Optional[Dict[str, float]] = None
    automl_completed_at: Optional[str] = None
    target_column: Optional[str] = None
    task_type: Optional[str] = None
    dataset_shape_for_training: Optional[List[int]] = None


class ExplainInfo(BaseModel):
    """Information about the explainability stage results."""
    tool_used: str = Field(..., description="Explainability tool used (e.g., 'SHAP')")
    explanation_type: str = Field(..., description="Type of explanation generated (e.g., 'global_summary')")
    shap_summary_plot_path: Optional[str] = None  # Path relative to run_dir, e.g., "plots/shap_summary.png"
    explain_completed_at: Optional[str] = None
    target_column: Optional[str] = None  # For display convenience
    task_type: Optional[str] = None      # For display convenience
    features_explained: Optional[int] = None
    samples_used_for_explanation: Optional[int] = None


class MetadataWithAutoML(MetadataWithFullSchema):
    """Metadata model that includes AutoML information in addition to prep results."""
    automl_info: Optional[AutoMLInfo] = None


class MetadataWithExplain(MetadataWithAutoML):
    """Metadata model that includes explainability information in addition to AutoML results."""
    explain_info: Optional[ExplainInfo] = None


class DetailedTargetInfo(BaseModel):
    """Detailed information about the target variable for later stages."""
    column_name: str
    task_type: str = Field(..., description="classification or regression")
    encoding_role: str
    unique_values: int
    class_distribution: Optional[Dict[str, int]] = None
    target_stats: Optional[Dict[str, float]] = None  # For regression targets
    
    @validator('task_type')
    def validate_task_type(cls, v):
        if v not in TASK_TYPES:
            raise ValueError(f'task_type must be one of {TASK_TYPES}')
        return v


class ModelInfo(BaseModel):
    """Information about the trained model."""
    model_name: str
    model_type: str
    performance_metrics: Dict[str, float]
    training_time_seconds: float
    feature_importance: Optional[Dict[str, float]] = None
    cv_scores: Optional[List[float]] = None
    best_params: Optional[Dict[str, Any]] = None


class ValidationResult(BaseModel):
    """Results from Great Expectations validation."""
    total_expectations: int
    successful_expectations: int
    failed_expectations: int
    success_rate: float
    critical_failures: List[str] = []
    warnings: List[str] = []
    
    
class ProcessingStep(BaseModel):
    """Information about a data processing step."""
    step_name: str
    step_type: str
    description: str
    parameters: Dict[str, Any] = {}
    timestamp: datetime
    rows_before: int
    rows_after: int
    columns_affected: List[str] = []


class RunMetadata(BaseModel):
    """Complete metadata for a pipeline run."""
    run_id: str
    created_at: datetime
    updated_at: datetime
    original_filename: str
    data_stats: DataStats
    target_info: Optional[DetailedTargetInfo] = None
    column_schemas: Dict[str, ColumnSchema] = {}
    processing_steps: List[ProcessingStep] = []
    model_info: Optional[ModelInfo] = None
    validation_summary: Optional[ValidationResult] = None
    artifact_paths: Dict[str, str] = {}
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class RunStatus(BaseModel):
    """Status information for a pipeline run."""
    run_id: str
    current_stage: str
    status: str = Field(..., description="Current status of the run")
    progress_percentage: float = Field(0.0, ge=0.0, le=100.0)
    last_updated: datetime
    errors: List[str] = []
    warnings: List[str] = []
    stage_timings: Dict[str, float] = {}  # Stage name -> duration in seconds
    
    @validator('status')
    def validate_status(cls, v):
        if v not in STATUS_VALUES:
            raise ValueError(f'status must be one of {STATUS_VALUES}')
        return v
        
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ValidationSummary(BaseModel):
    """Summary of data validation results."""
    run_id: str
    validation_timestamp: datetime
    overall_success: bool
    total_checks: int
    passed_checks: int
    failed_checks: int
    critical_failures: List[str] = []
    warnings: List[str] = []
    data_quality_score: float = Field(..., ge=0.0, le=1.0)
    expectation_results: Dict[str, Any] = {}  # Raw GE results
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class RunIndex(BaseModel):
    """Index entry for a completed run."""
    run_id: str
    created_at: datetime
    original_filename: str
    target_column: Optional[str] = None
    task_type: Optional[str] = None
    model_name: Optional[str] = None
    final_score: Optional[float] = None
    data_rows: int
    data_cols: int
    status: str
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        } 