"""
STODS MRI AI Microservice
Exposes POST /predict — accepts before/after MRI scans (NIfTI),
runs segmentation + volume comparison, returns CV-only JSON.

Important:
- This endpoint does NOT calculate mismatch_alert.
- mismatch_alert should be calculated later by comparing CV result with NLP surgical report result.
"""
import os
import uuid
import shutil
import traceback
import time
from datetime import datetime, timezone

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from inference import generate_report

app = FastAPI(title="STODS MRI AI Microservice")

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXT = (".nii", ".nii.gz")
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "500"))
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024


def validate_filename(upload: UploadFile) -> None:
    """Validate uploaded file name and extension."""
    if not upload.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")

    filename = upload.filename.lower()
    if not filename.endswith(ALLOWED_EXT):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type for '{upload.filename}'. Expected .nii or .nii.gz",
        )


def save_upload(upload: UploadFile, subdir: str) -> str:
    """Save uploaded scan to a temporary local path."""
    validate_filename(upload)

    target_dir = os.path.join(UPLOAD_DIR, subdir)
    os.makedirs(target_dir, exist_ok=True)

    safe_original_name = os.path.basename(upload.filename)
    filename = f"{uuid.uuid4()}_{safe_original_name}"
    path = os.path.join(target_dir, filename)

    total_size = 0
    chunk_size = 1024 * 1024  # 1 MB

    with open(path, "wb") as f:
        while True:
            chunk = upload.file.read(chunk_size)
            if not chunk:
                break

            total_size += len(chunk)
            if total_size > MAX_UPLOAD_SIZE_BYTES:
                f.close()
                if os.path.exists(path):
                    os.remove(path)
                raise HTTPException(
                    status_code=413,
                    detail=f"File '{upload.filename}' is too large. Max size is {MAX_UPLOAD_SIZE_MB} MB.",
                )

            f.write(chunk)

    return path


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(
    patient_id: str = Form(...),
    before_scan: UploadFile = File(...),
    after_scan: UploadFile = File(...),
):
    """
    CV-only endpoint:
    before MRI + after MRI -> segmentation -> volume comparison -> organ status.
    """
    before_path = None
    after_path = None
    inference_id = str(uuid.uuid4())
    started_at = time.time()

    try:
        before_path = save_upload(before_scan, "before")
        after_path = save_upload(after_scan, "after")

        result = generate_report(before_path, after_path)
        processing_time_sec = round(time.time() - started_at, 3)

        return {
            "status": "success",
            "inference_id": inference_id,
            "patient_id": patient_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "processing_time_sec": processing_time_sec,
            "model_version": result.get("model_version"),
            "device": result.get("device"),
            "organs": result["organs"],
            "volumes": result["volumes"],
            "details": result["details"],
        }

    except HTTPException:
        raise
    except Exception:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "inference_id": inference_id,
                "message": "Internal server error during MRI inference.",
            },
        )
    finally:
        for p in [before_path, after_path]:
            if p and os.path.exists(p):
                os.remove(p)
