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

VALID_METHODS = {"yamnet", "rule", "ml"}


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
        print(f"Validation Error: method {method} not in {VALID_METHODS}")
        raise HTTPException(
            status_code=400,
            detail=f"method must be one of {sorted(VALID_METHODS)}",
        )
    
    # Map all legacy methods to yamnet
    method = "yamnet"

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
        formatted = format_output(counts)
        print(f"\n---> KEYSTROKES DETECTED: {formatted} <---")
        return {"counts": counts, "formatted": formatted}
    except FileNotFoundError as exc:
        print(f"FileNotFoundError: {exc}")
        raise HTTPException(status_code=400, detail=str(exc))
    except ValueError as exc:
        print(f"ValueError: {exc}")
        raise HTTPException(status_code=400, detail=str(exc))
    except EOFError as exc:
        print(f"EOFError (Corrupted Audio) on {wav_path}: {exc}")
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty or corrupted.")
    except Exception as exc:
        print(f"Unexpected Exception on {wav_path}: {exc.__class__.__name__} - {exc}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail="Audio processing failed. Check the server logs for details.",
        )
    # Removing the finally block to keep the file for debugging if it fails
    # Success case will clean it up:
    if wav_path and os.path.exists(wav_path):
        os.remove(wav_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

