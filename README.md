# STODS MRI AI Microservice

Standalone FastAPI service. The Django backend calls it as an external AI API.

This endpoint is **CV-only**:

```text
before MRI + after MRI
→ segmentation
→ organ volume calculation
→ before/after volume comparison
→ organ status
```

It does **not** calculate `mismatch_alert`.  
`mismatch_alert` should be calculated later by comparing this CV result with the NLP surgical report result.

---

## Setup

```bash
pip install -r requirements.txt
```

Place your trained checkpoint at:

```text
models/unetr_lite_v3_best.pt
```

Or set a custom path:

```bash
set MODEL_PATH=D:\path\to\unetr_lite_v3_best.pt
```

Linux/macOS:

```bash
export MODEL_PATH=/path/to/unetr_lite_v3_best.pt
```

The checkpoint must match:

```text
UNETR_Lite_v3_DS
base=24
num_classes=5
labels: 0 background, 1 Liver, 2 R_Kidney, 3 L_Kidney, 4 Spleen
ROI_SIZE=(96, 96, 72)
```

---

## Run

```bash
uvicorn app:app --host 0.0.0.0 --port 8001
```

Docs:

```text
http://localhost:8001/docs
```

Health check:

```text
GET /health
```

---

## Endpoint

### POST `/predict`

Request type: `multipart/form-data`

Fields:

| Field | Type | Required | Description |
|---|---|---:|---|
| `patient_id` | string | yes | Patient or verification id from backend |
| `before_scan` | file | yes | Before-operation MRI scan, `.nii` or `.nii.gz` |
| `after_scan` | file | yes | After-operation MRI scan, `.nii` or `.nii.gz` |

---

## Example request from Django/backend

```python
import requests

AI_URL = "http://ai-service-url/predict"

with open(before_path, "rb") as before_file, open(after_path, "rb") as after_file:
    files = {
        "before_scan": before_file,
        "after_scan": after_file,
    }
    data = {
        "patient_id": str(patient_id),
    }

    response = requests.post(AI_URL, data=data, files=files, timeout=300)
    response.raise_for_status()
    ai_result = response.json()
```

---

## Success response

```json
{
  "status": "success",
  "inference_id": "b33b9f0d-2f27-4d48-9e5b-3b7316b2e0f4",
  "patient_id": "321",
  "timestamp": "2026-06-13T00:00:00+00:00",
  "processing_time_sec": 12.384,
  "model_version": "UNETR_Lite_v3_DS_v1",
  "device": "cuda",
  "organs": {
    "Liver": "present",
    "R_Kidney": "removed",
    "L_Kidney": "present",
    "Spleen": "present"
  },
  "volumes": {
    "before": {
      "Liver": 1289.061,
      "R_Kidney": 171.126,
      "L_Kidney": 164.093,
      "Spleen": 139.036
    },
    "after": {
      "Liver": 1289.061,
      "R_Kidney": 0.0,
      "L_Kidney": 164.093,
      "Spleen": 139.036
    }
  },
  "details": {
    "R_Kidney": {
      "before_ml": 171.126,
      "after_ml": 0.0,
      "delta_ml": -171.126,
      "delta_pct": -100.0,
      "status": "removed"
    }
  }
}
```

---

## Status classification per organ

| Status | Rule |
|---|---|
| `removed` | `delta_pct <= -50%` |
| `reduced` | `-50% < delta_pct < -10%` |
| `present` | `abs(delta_pct) <= 10%` |
| `increased` | `delta_pct > 10%` |
| `missing` | before volume ≈ 0 and after volume ≈ 0 |
| `appeared` | before volume ≈ 0 and after volume > 0 |

---

## Internal pipeline

1. Preprocess: load NIfTI, RAS orientation, spacing `(1.5, 1.5, 3.0)`, crop foreground, intensity scaling `0.5–99.5`, resize/pad/crop to `(96, 96, 72)`.
2. Segmentation: `UNETR_Lite_v3_DS`, sliding window inference, main output head only.
3. Volume calculation: voxel count × voxel volume in mL.
4. CV comparison: before/after volume change and organ status.

---

## Important note about mismatch_alert

Do not calculate `mismatch_alert` inside this microservice.

Correct flow:

```text
CV endpoint result + NLP surgical report result
→ backend comparison logic
→ mismatch_alert
```
