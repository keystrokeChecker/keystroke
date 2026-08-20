import io
from fastapi import APIRouter, File, Form, UploadFile, HTTPException, Depends
from typing import List

from app.api.schemas import HealthResponse, ModelInfoResponse, SinglePredictResponse, AnalyzeResponse
from app.inference.engine import InferenceEngine

router = APIRouter()
engine = InferenceEngine()


@router.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="ok", model_loaded=engine.is_loaded)


@router.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    if not engine.is_loaded:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded.")
    meta = engine.metadata
    return ModelInfoResponse(
        model_name=meta.get("model_name", "Unknown"),
        feature_set=meta.get("feature_set", "Unknown"),
        normalization=meta.get("normalization", "Unknown"),
        classes_count=meta.get("classes_count", 27),
        validation_metrics=meta.get("validation_metrics", {}),
        strictly_public_data_only=meta.get("strictly_public_data_only", True)
    )


@router.post("/predict_single", response_model=SinglePredictResponse)
async def predict_single(file: UploadFile = File(...)):
    if not engine.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    audio_bytes = await file.read()
    try:
        res = engine.predict_single_clip(audio_bytes)
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


@router.post("/analyze", response_model=AnalyzeResponse)
@router.post("/analyze_with_timestamps", response_model=AnalyzeResponse)
async def analyze_continuous(file: UploadFile = File(...), method: str = Form("default")):
    if not engine.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    audio_bytes = await file.read()
    try:
        res = engine.analyze_continuous_audio(audio_bytes)
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Analysis error: {str(e)}")


@router.post("/batch_predict", response_model=List[SinglePredictResponse])
async def batch_predict(files: List[UploadFile] = File(...)):
    if not engine.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    results = []
    for f in files:
        audio_bytes = await f.read()
        res = engine.predict_single_clip(audio_bytes)
        results.append(res)
    return results
