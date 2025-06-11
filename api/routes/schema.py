"""
API-facing Pydantic models for The Projection Wizard.
Defines clean request/response schemas for FastAPI endpoints.

These models are the single source of truth for the API contract between
frontend and backend. They are intentionally thin and do not leak internal
pipeline metadata structures.
"""

from pydantic import BaseModel, Field
from typing import Dict, Optional, Literal, List, Tuple, Any, Union


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
    status: Literal["pending", "running", "processing", "completed", "failed"]
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
    status: Literal["pending", "running", "processing", "completed", "failed"]
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


class PredictionReadiness(BaseModel):
    """Information about prediction readiness."""
    prediction_ready: bool = False
    model_file_available: bool = False
    column_mapping_available: bool = False
    original_data_available: bool = False
    metadata_available: bool = False


class PredictionInputRequest(BaseModel):
    """Request model for making predictions."""
    run_id: str
    input_values: Dict[str, Any] = Field(
        ..., description="Raw user input values in original column format"
    )


class PredictionSchemaResponse(BaseModel):
    """Response with input schema for building prediction form."""
    api_version: Literal["v1"] = "v1"
    numeric_columns: Dict[str, Dict[str, float]] = Field(
        default_factory=dict, 
        description="Numeric columns with min, max, mean, std values"
    )
    categorical_columns: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Categorical columns with options and default values"
    )
    target_info: Dict[str, Any] = Field(
        default_factory=dict, description="Target column information"
    )


class PredictionResponse(BaseModel):
    """Response with prediction results."""
    api_version: Literal["v1"] = "v1"
    prediction_value: Any = Field(..., description="The predicted value")
    confidence: Optional[float] = Field(None, description="Prediction confidence (0-1)")
    input_features: Dict[str, Any] = Field(
        default_factory=dict, description="Processed features sent to model"
    )
    task_type: str
    target_column: str
    model_name: Optional[str] = None


# Enhanced Prediction API Schema Models for Option 2

class SliderConfig(BaseModel):
    """Configuration for slider-based numeric inputs."""
    min_value: float
    max_value: float
    default_value: float
    step_size: float
    suggested_value: Optional[float] = None
    display_format: str = ".3f"  # Format for displaying values


class CategoricalConfig(BaseModel):
    """Configuration for categorical inputs."""
    options: List[str]
    default_option: str
    display_names: Optional[Dict[str, str]] = None  # Map option -> friendly name
    descriptions: Optional[Dict[str, str]] = None  # Map option -> description


class FeatureMetadata(BaseModel):
    """Metadata about a feature for UI rendering."""
    importance_rank: Optional[int] = None
    importance_score: Optional[float] = None
    correlation_with_target: Optional[float] = None
    description: Optional[str] = None
    unit: Optional[str] = None
    category: Optional[str] = None  # e.g., "demographic", "behavioral", "financial"


class EnhancedPredictionSchemaResponse(BaseModel):
    """Enhanced schema response with rich UI metadata."""
    api_version: Literal["v1"] = "v1"
    
    # Enhanced numeric columns with slider configs
    numeric_columns: Dict[str, SliderConfig] = Field(default_factory=dict)
    
    # Enhanced categorical columns with UI configs
    categorical_columns: Dict[str, CategoricalConfig] = Field(default_factory=dict)
    
    # Feature metadata for all input features
    feature_metadata: Dict[str, FeatureMetadata] = Field(default_factory=dict)
    
    # Target information
    target_info: Dict[str, Any] = Field(default_factory=dict)
    
    # Model information
    model_info: Dict[str, Any] = Field(default_factory=dict)
    
    # Validation rules
    validation_rules: Dict[str, Any] = Field(default_factory=dict)


class PredictionProbabilities(BaseModel):
    """Class probabilities for classification tasks."""
    class_probabilities: Dict[str, float] = Field(default_factory=dict)
    predicted_class: str
    confidence: float


class FeatureContribution(BaseModel):
    """Model for individual feature contributions."""
    feature_name: str
    contribution_value: float
    feature_value: Union[float, int, str]
    contribution_direction: str  # 'positive', 'negative', 'neutral'
    shap_value: Optional[float] = None  # Real SHAP value when available


class DetailedExplanation(BaseModel):
    """Model for detailed prediction explanations."""
    prediction_id: str
    explanation_type: str  # 'shap', 'lime', 'model_importance'
    feature_explanations: List[FeatureContribution]
    global_explanation: str
    explanation_confidence: float
    shap_base_value: Optional[float] = None
    shap_values: Optional[Dict[str, float]] = None
    explanation_timestamp: str
    

class ShapExplanationResponse(BaseModel):
    """Response model for SHAP explanations."""
    prediction_id: str
    shap_values: Dict[str, float]
    shap_base_value: float
    feature_contributions: List[FeatureContribution]
    top_contributing_features: List[str]
    explanation_summary: str
    shap_available: bool
    fallback_used: bool
    explanation_timestamp: str


class PredictionConfidenceInterval(BaseModel):
    """Confidence interval for regression predictions."""
    lower_bound: float
    upper_bound: float
    confidence_level: float = 0.95


class SinglePredictionResponse(BaseModel):
    """Enhanced single prediction response."""
    api_version: Literal["v1"] = "v1"
    
    # Core prediction
    prediction_value: Any
    
    # Enhanced prediction information
    probabilities: Optional[PredictionProbabilities] = None
    confidence_interval: Optional[PredictionConfidenceInterval] = None
    
    # Feature analysis
    feature_contributions: List[FeatureContribution] = Field(default_factory=list)
    top_contributing_features: List[str] = Field(default_factory=list)
    
    # Input information
    input_features: Dict[str, Any] = Field(default_factory=dict)
    processed_features: Dict[str, Any] = Field(default_factory=dict)
    
    # Model metadata
    task_type: str
    target_column: str
    model_name: Optional[str] = None
    prediction_timestamp: str
    
    # Unique prediction ID for explanations
    prediction_id: str


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions."""
    run_id: str
    inputs: List[Dict[str, Any]] = Field(..., description="List of input dictionaries")
    include_explanations: bool = False
    include_confidence: bool = True


class BatchPredictionItem(BaseModel):
    """Single item in batch prediction response."""
    input_index: int
    prediction_value: Any
    probabilities: Optional[PredictionProbabilities] = None
    confidence_interval: Optional[PredictionConfidenceInterval] = None
    prediction_id: str


class BatchPredictionSummary(BaseModel):
    """Summary statistics for batch predictions."""
    total_predictions: int
    prediction_distribution: Dict[str, int] = Field(default_factory=dict)  # For classification
    prediction_range: Optional[Dict[str, float]] = None  # For regression (min, max, mean, std)
    processing_time_seconds: float


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""
    api_version: Literal["v1"] = "v1"
    
    predictions: List[BatchPredictionItem]
    summary: BatchPredictionSummary
    
    # Model metadata
    task_type: str
    target_column: str
    model_name: Optional[str] = None
    batch_timestamp: str


class PredictionExplanationResponse(BaseModel):
    """Detailed explanation for a specific prediction."""
    api_version: Literal["v1"] = "v1"
    
    prediction_id: str
    prediction_value: Any
    
    # SHAP values and feature contributions
    shap_values: Dict[str, float] = Field(default_factory=dict)
    feature_contributions: List[FeatureContribution] = Field(default_factory=list)
    
    # Counterfactual analysis
    counterfactuals: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Feature importance for this prediction
    local_feature_importance: Dict[str, float] = Field(default_factory=dict)
    
    # Model behavior insights
    prediction_confidence: float
    similar_cases_count: Optional[int] = None
    model_uncertainty: Optional[float] = None


class ComparisonScenario(BaseModel):
    """A prediction scenario for comparison."""
    scenario_name: str
    input_values: Dict[str, Any]
    prediction_value: Any
    probabilities: Optional[PredictionProbabilities] = None
    key_differences: List[str] = Field(default_factory=list)


class PredictionComparisonRequest(BaseModel):
    """Request for comparing multiple prediction scenarios."""
    run_id: str
    scenarios: List[ComparisonScenario]


class ComparisonAnalysis(BaseModel):
    """Analysis of differences between scenarios."""
    most_influential_features: List[str]
    prediction_sensitivity: Dict[str, float] = Field(default_factory=dict)
    scenario_rankings: Optional[List[str]] = None  # For classification/regression ordering


class PredictionComparisonResponse(BaseModel):
    """Response for prediction comparison."""
    api_version: Literal["v1"] = "v1"
    
    scenarios: List[ComparisonScenario]
    comparison_analysis: ComparisonAnalysis
    
    # Model metadata
    task_type: str
    target_column: str
    model_name: Optional[str] = None
    comparison_timestamp: str


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
    
    # Prediction readiness information
    prediction_readiness: PredictionReadiness


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
    "PredictionReadiness",
    "PredictionInputRequest",
    "PredictionSchemaResponse",
    "PredictionResponse",
    "FinalResultsResponse",
    "SliderConfig",
    "CategoricalConfig",
    "FeatureMetadata",
    "EnhancedPredictionSchemaResponse",
    "PredictionProbabilities",
    "FeatureContribution",
    "PredictionConfidenceInterval",
    "SinglePredictionResponse",
    "BatchPredictionRequest",
    "BatchPredictionItem",
    "BatchPredictionSummary",
    "BatchPredictionResponse",
    "PredictionExplanationResponse",
    "ComparisonScenario",
    "PredictionComparisonRequest",
    "ComparisonAnalysis",
    "PredictionComparisonResponse",
    "DetailedExplanation",
    "ShapExplanationResponse",
]
