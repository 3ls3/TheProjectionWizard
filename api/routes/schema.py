"""
API-facing Pydantic models for The Projection Wizard.
Defines clean request/response schemas for FastAPI endpoints.

These models are the single source of truth for the API contract between
frontend and backend. They are intentionally thin and do not leak internal
pipeline metadata structures.
"""

from pydantic import BaseModel, Field
from typing import Dict, Optional, Literal, List, Tuple


class UploadResponse(BaseModel):
    """Response model for file upload endpoint."""
    api_version: Literal["v1"] = "v1"
    run_id: str
    shape: Tuple[int, int]
    preview: List[List[str]] = Field(
        ..., description="First 5 rows as raw strings"
    )


class TargetSuggestionResponse(BaseModel):
    """Response model for target column suggestion endpoint."""
    api_version: Literal["v1"] = "v1"
    suggested_column: str
    task_type: Literal["classification", "regression"]
    confidence: Optional[float] = None


class TargetConfirmationRequest(BaseModel):
    """Request model for target column confirmation endpoint."""
    run_id: str
    confirmed_column: str
    task_type: Literal["classification", "regression"]


class FeatureSuggestionResponse(BaseModel):
    """Response model for feature schema suggestions endpoint."""
    api_version: Literal["v1"] = "v1"
    feature_schemas: Dict[str, Dict[str, str]] = Field(
        ..., description="Column names mapped to schema suggestions"
    )


class FeatureConfirmationRequest(BaseModel):
    """Request model for feature schema confirmation endpoint."""
    run_id: str
    confirmed_schemas: Dict[str, Dict[str, str]]


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


class FinalResultsResponse(BaseModel):
    """Response model for final pipeline results endpoint."""
    api_version: Literal["v1"] = "v1"
    model_metrics: Dict[str, float]
    top_features: List[str]
    explainability: Dict[str, str]
    download_url: Optional[str] = None


__all__ = [
    "UploadResponse",
    "TargetSuggestionResponse",
    "TargetConfirmationRequest",
    "FeatureSuggestionResponse",
    "FeatureConfirmationRequest",
    "PipelineStatusResponse",
    "FinalResultsResponse",
]
