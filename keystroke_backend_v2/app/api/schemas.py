from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class HealthResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "healthy"})
    model_loaded: bool = Field(..., json_schema_extra={"example": True})


class ModelInfoResponse(BaseModel):
    model_name: str
    feature_set: str
    normalization: str
    classes_count: int
    validation_metrics: Dict[str, Any]
    strictly_public_data_only: bool


class TopKPrediction(BaseModel):
    key: str
    probability: float


class SinglePredictResponse(BaseModel):
    prediction: str
    confidence: float
    status: str
    top_k: List[TopKPrediction]
    quality_info: Optional[Dict[str, Any]] = None


class EventPredictionItem(BaseModel):
    event_index: int
    timestamp_s: float
    prediction: str
    confidence: float
    status: str
    top_k: List[TopKPrediction]


class AnalyzeResponse(BaseModel):
    counts: List[int] = Field(default_factory=lambda: [0] * 27)
    formatted: str = ""
    detected_events_count: int
    reconstructed_sequence: str
    events: List[EventPredictionItem]
    quality_info: Optional[Dict[str, Any]] = None
