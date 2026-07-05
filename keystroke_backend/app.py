import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from predict import predict_ml_based, predict_rule_based
from segmenter import format_output

app = FastAPI(
    title="Keystroke Audio Analyzer",
    description="Analyze WAV audio recorded from physical keyboard typing and return keystroke counts per word.",
    version="0.1",
)

VALID_METHODS = {"rule", "ml"}


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
    method: str = Form("rule"),
    threshold: float = Form(0.4),
    delta: float = Form(0.07),
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

        if method == "rule":
            counts = predict_rule_based(wav_path, threshold=threshold, delta=delta)
        else:
            counts = predict_ml_based(wav_path, threshold=threshold, delta=delta)

        formatted = format_output(counts)
        return {"counts": counts, "formatted": formatted}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
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
