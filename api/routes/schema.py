"""
API-facing Pydantic models for The Projection Wizard.
Defines clean request/response schemas for FastAPI endpoints.

These models are the single source of truth for the API contract between
frontend and backend. They are intentionally thin and do not leak internal
pipeline metadata structures.
"""

from pydantic import BaseModel, Field
from typing import Dict, Optional, Literal, List, Tuple, Any


class UploadResponse(BaseModel):
    """Response model for file upload endpoint."""
    api_version: Literal["v1"] = "v1"
    run_id: str
    shape: Tuple[int, int]
    preview: List[List[str]] = Field(
        ..., description="First 5 rows as raw strings"
    )


class ColumnStatistics(BaseModel):
    """Statistics for a single column."""
    unique_values: int
    missing_values: int
    missing_percentage: float
    data_type: str
    sample_values: List[str] = Field(
        default_factory=list, description="Sample values for preview"
    )


class MLTypeOption(BaseModel):
    """An ML type option with description."""
    value: str
    description: str


class TargetSuggestionResponse(BaseModel):
    """Enhanced response model for target column suggestion endpoint."""
    api_version: Literal["v1"] = "v1"
    
    # All available columns with their statistics
    columns: Dict[str, ColumnStatistics]
    
    # AI suggestions
    suggested_column: str
    suggested_task_type: Literal["classification", "regression"]
    suggested_ml_type: str
    confidence: Optional[float] = None
    
    # Available options for UI dropdowns
    available_task_types: List[str] = Field(
        default_factory=lambda: ["classification", "regression"]
    )
    available_ml_types: Dict[str, List[MLTypeOption]] = Field(
        default_factory=dict, description="ML type options grouped by task type"
    )
    
    # Data preview
    data_preview: List[List[str]] = Field(
        default_factory=list, description="Sample data rows"
    )


class TargetConfirmationRequest(BaseModel):
    """Enhanced request model for target column confirmation endpoint."""
    run_id: str
    confirmed_column: str
    task_type: Literal["classification", "regression"]
    ml_type: str = Field(..., description="The ML type for the target variable")


class TargetConfirmationResponse(BaseModel):
    """Response model for target confirmation endpoint."""
    api_version: Literal["v1"] = "v1"
    status: Literal["success"] = "success"
    message: str = "Target configuration saved successfully"
    target_info: Dict[str, Any] = Field(
        default_factory=dict, description="Confirmed target information"
    )


class FeatureSchema(BaseModel):
    """Schema information for a feature."""
    initial_dtype: str
    suggested_encoding_role: str
    
    # Enhanced information for UI
    statistics: ColumnStatistics
    is_key_feature: bool = False


class FeatureSuggestionResponse(BaseModel):
    """Enhanced response model for feature schema suggestions endpoint."""
    api_version: Literal["v1"] = "v1"
    
    # All feature schemas with enhanced information
    feature_schemas: Dict[str, FeatureSchema]
    
    # Key features identified (ordered by importance)
    key_features: List[str] = Field(
        default_factory=list, description="Key features ordered by importance"
    )
    
    # Available options for UI dropdowns
    available_dtypes: Dict[str, str] = Field(
        default_factory=dict, description="Available data types with descriptions"
    )
    available_encoding_roles: Dict[str, str] = Field(
        default_factory=dict, description="Available encoding roles with descriptions"
    )
    
    # Target information for context
    target_info: Dict[str, Any] = Field(
        default_factory=dict, description="Target column information"
    )
    
    # Data preview
    data_preview: List[List[str]] = Field(
        default_factory=list, description="Sample data rows"
    )


class FeatureConfirmationRequest(BaseModel):
    """Enhanced request model for feature schema confirmation endpoint."""
    run_id: str
    confirmed_schemas: Dict[str, Dict[str, str]] = Field(
        ..., description="User-confirmed schemas with 'final_dtype' and 'final_encoding_role'"
    )
    
    # Optional metadata for better tracking
    total_features_reviewed: Optional[int] = Field(
        None, description="Total number of features the user reviewed"
    )
    key_features_modified: Optional[List[str]] = Field(
        None, description="List of key features that were modified by the user"
    )


class FeatureConfirmationResponse(BaseModel):
    """Response model for feature confirmation endpoint."""
    api_version: Literal["v1"] = "v1"
    status: Literal["pipeline_started"] = "pipeline_started"
    message: str = "Feature schemas confirmed and pipeline started"
    
    # Summary of what was confirmed
    summary: Dict[str, Any] = Field(
        default_factory=dict, description="Summary of confirmed features and next steps"
    )


class PipelineStatusResponse(BaseModel):
    """Response model for pipeline status endpoint."""
    api_version: Literal["v1"] = "v1"
    stage: str = Field(
        ..., description="Current pipeline stage (e.g. 'prep', 'automl', 'completed')"
    )
    status: Literal["pending", "running", "completed", "failed"]
    message: Optional[str] = None
    progress_pct: Optional[int] = Field(
        None, description="Coarse progress percentage (0-100)"
    )


class RunSummary(BaseModel):
    """Summary information about the pipeline run."""
    run_id: str
    timestamp: Optional[str] = None
    original_filename: Optional[str] = None
    initial_shape: Optional[Tuple[int, int]] = None
    target_info: Optional[Dict[str, Any]] = None


class PipelineStatusInfo(BaseModel):
    """Detailed pipeline status information."""
    stage: str
    status: Literal["pending", "running", "completed", "failed"]
    message: Optional[str] = None
    errors: Optional[List[str]] = None


class ValidationSummaryInfo(BaseModel):
    """Validation summary information."""
    overall_success: bool
    total_expectations: int
    successful_expectations: int
    failed_expectations: int


class DataPrepSummary(BaseModel):
    """Data preparation summary information."""
    final_shape: Optional[Tuple[int, int]] = None
    cleaning_steps: List[str] = Field(default_factory=list)
    profiling_report_available: bool = False
    profiling_report_filename: Optional[str] = None


class AutoMLSummary(BaseModel):
    """AutoML model summary information."""
    tool_used: Optional[str] = None
    best_model_name: Optional[str] = None
    target_column: Optional[str] = None
    task_type: Optional[str] = None
    performance_metrics: Dict[str, float] = Field(default_factory=dict)
    model_file_available: bool = False


class ExplainabilitySummary(BaseModel):
    """Model explainability summary information."""
    tool_used: Optional[str] = None
    features_explained: Optional[int] = None
    samples_used: Optional[int] = None
    shap_plot_available: bool = False
    shap_plot_filename: Optional[str] = None


class AvailableDownloads(BaseModel):
    """Information about available downloadable files."""
    original_data: bool = False
    cleaned_data: bool = False
    metadata_json: bool = False
    validation_report: bool = False
    profile_report: bool = False
    model_artifacts: bool = False
    shap_plot: bool = False
    pipeline_log: bool = False
    
    # File size information (in KB)
    file_sizes: Dict[str, float] = Field(default_factory=dict)


class FinalResultsResponse(BaseModel):
    """Comprehensive response model for final pipeline results endpoint."""
    api_version: Literal["v1"] = "v1"
    
    # Core results (original structure maintained for backwards compatibility)
    model_metrics: Dict[str, float]
    top_features: List[str]
    explainability: Dict[str, str]
    download_url: Optional[str] = None
    
    # Enhanced detailed information
    run_summary: RunSummary
    pipeline_status: PipelineStatusInfo
    validation_summary: Optional[ValidationSummaryInfo] = None
    data_prep_summary: Optional[DataPrepSummary] = None
    automl_summary: Optional[AutoMLSummary] = None
    explainability_summary: Optional[ExplainabilitySummary] = None
    available_downloads: AvailableDownloads


__all__ = [
    "UploadResponse",
    "ColumnStatistics",
    "MLTypeOption",
    "TargetSuggestionResponse",
    "TargetConfirmationRequest",
    "TargetConfirmationResponse",
    "FeatureSchema",
    "FeatureSuggestionResponse",
    "FeatureConfirmationRequest",
    "FeatureConfirmationResponse",
    "PipelineStatusResponse",
    "RunSummary",
    "PipelineStatusInfo",
    "ValidationSummaryInfo",
    "DataPrepSummary",
    "AutoMLSummary",
    "ExplainabilitySummary",
    "AvailableDownloads",
    "FinalResultsResponse",
]
