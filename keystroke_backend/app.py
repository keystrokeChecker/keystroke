import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from src.predictor import predict_keystroke_counts
from src.yamnet_config import CLASSIFIER_THRESHOLD, SENSITIVITY_DELTA, GAP_THRESHOLD, MERGE_GAP_SECONDS
from segmenter import format_output

app = FastAPI(
    title="Keystroke Audio Analyzer",
    description="Analyze WAV audio recorded from physical keyboard typing and return keystroke counts per word.",
    version="0.1",
)

VALID_METHODS = {"yamnet"}


@app.get("/health")
async def health():
    return {"status": "ok"}


async def _save_upload_file_tmp(upload_file: UploadFile) -> str:
    suffix = Path(upload_file.filename).suffix or ".wav"
    if suffix.lower() not in {".wav", ".wave"}:
        suffix = ".wav"

    try:
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        contents = await upload_file.read()
        if not contents:
            raise ValueError("Uploaded file is empty")
        tmp_file.write(contents)
        tmp_file.flush()
        tmp_file.close()
        return tmp_file.name
    finally:
        await upload_file.close()


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    method: str = Form("yamnet"),
    threshold: float = Form(CLASSIFIER_THRESHOLD),
    delta: float = Form(SENSITIVITY_DELTA),
    gap_threshold: float = Form(GAP_THRESHOLD),
    merge_gap_seconds: float = Form(MERGE_GAP_SECONDS),
):
    method = method.lower()
    if method not in VALID_METHODS:
        raise HTTPException(
            status_code=400,
            detail=f"method must be one of {sorted(VALID_METHODS)}",
        )

    wav_path = None
    try:
        wav_path = await _save_upload_file_tmp(file)
        counts = predict_keystroke_counts(
            wav_path,
            threshold=threshold,
            delta=delta,
            gap_threshold=gap_threshold,
            merge_gap_seconds=merge_gap_seconds,
        )
        return {"counts": counts, "formatted": format_output(counts)}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Audio processing failed. Check the server logs for details.",
        )
    finally:
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

