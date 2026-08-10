from datetime import datetime
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from src.predictor import predict_keystroke_counts
from src.yamnet_config import CLASSIFIER_THRESHOLD

app = FastAPI(
    title="Keystroke Audio Analyzer",
    description="Analyze WAV audio with rule-based onset detection and YAMNet filtering.",
    version="0.1",
)

UPLOADS_DIR = Path(__file__).resolve().parent / "data" / "uploads"

@app.get("/health")
async def health():
    return {"status": "ok"}


async def _save_upload_file(upload_file: UploadFile) -> str:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    destination = UPLOADS_DIR / (
        f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
        f"{uuid4().hex[:8]}.wav"
    )
    try:
        contents = await upload_file.read()
        if not contents:
            raise ValueError("Uploaded file is empty")
        destination.write_bytes(contents)
        return str(destination)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await upload_file.close()


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    method: str = Form("yamnet"),
    threshold: float = Form(CLASSIFIER_THRESHOLD),
    typing_speed: str = Form("auto"),
):
    if method.lower() != "yamnet":
        raise HTTPException(status_code=400, detail="method must be 'yamnet'")
    if not 0.0 <= threshold <= 1.0:
        raise HTTPException(status_code=400, detail="threshold must be between 0 and 1")
    if typing_speed.lower() not in {"auto", "fast", "medium", "slow"}:
        raise HTTPException(
            status_code=400,
            detail="typing_speed must be auto, fast, medium, or slow",
        )

    wav_path = None
    try:
        wav_path = await _save_upload_file(file)
        result = predict_keystroke_counts(
            wav_path,
            threshold=threshold,
            typing_speed=typing_speed.lower(),
        )
        formatted = result["formatted"]
        print(f"\n---> KEYSTROKES DETECTED: {formatted} <---")
        return {**result, "saved_file": Path(wav_path).name}
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
