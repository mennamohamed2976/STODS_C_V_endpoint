"""
Runs the trained UNETR_Lite_v3_DS model on a preprocessed scan and
computes per-organ volumes (in mL) from the predicted segmentation mask.

This module returns CV-only results. It does NOT calculate mismatch_alert.
mismatch_alert should be calculated later by comparing CV output with NLP surgical report output.
"""
import os
import numpy as np
import torch

from monai.inferers import sliding_window_inference

from model_def import UNETR_Lite_v3_DS

# ---- Config: must match training notebook exactly ----
N_CLASSES = 5  # 0 background + 4 organs
ROI_SIZE = (96, 96, 72)
LABELS = {1: "Liver", 2: "R_Kidney", 3: "L_Kidney", 4: "Spleen"}

MODEL_PATH = os.getenv("MODEL_PATH", "models/unetr_lite_v3_best.pt")
MODEL_VERSION = "UNETR_Lite_v3_DS_v1"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None


def get_model():
    """Lazy-load the model once and cache it."""
    global _model

    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model checkpoint not found: {MODEL_PATH}. "
                "Place unetr_lite_v3_best.pt under models/ or set MODEL_PATH."
            )

        model = UNETR_Lite_v3_DS(in_channels=1, num_classes=N_CLASSES, base=24).to(DEVICE)
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        model.eval()
        _model = model

    return _model


def predict_segmentation(image_tensor: torch.Tensor) -> np.ndarray:
    """
    image_tensor: (C, D, H, W) preprocessed image tensor.
    Returns: 3D numpy array (D, H, W) of predicted class labels (0..4).
    """
    model = get_model()
    x = image_tensor.unsqueeze(0).to(DEVICE).float()  # (1, C, D, H, W)

    def predictor(inp):
        return model(inp)[0]  # main head logits only

    with torch.no_grad():
        logits = sliding_window_inference(
            inputs=x,
            roi_size=ROI_SIZE,
            sw_batch_size=1,
            predictor=predictor,
            overlap=0.5,
            mode="gaussian",
        )
        pred = torch.argmax(logits, dim=1)  # (1, D, H, W)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pred.squeeze(0).cpu().numpy().astype(np.uint8)


def voxel_volume_ml(meta) -> float:
    """
    Compute voxel volume in mL from MONAI meta dict.
    Falls back to preprocessing spacing if metadata is unavailable.
    """
    try:
        affine = meta["affine"] if "affine" in meta else meta.get("original_affine")
        if affine is not None:
            affine = np.asarray(affine)
            spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
            return float(spacing[0] * spacing[1] * spacing[2]) / 1000.0
    except Exception:
        pass

    # fallback: PIXDIM used in preprocessing: 1.5 x 1.5 x 3.0 mm
    return (1.5 * 1.5 * 3.0) / 1000.0


def organ_volumes_ml(pred_mask: np.ndarray, voxel_ml: float) -> dict:
    """Compute volume in mL for each organ label."""
    volumes = {}
    for label_id, name in LABELS.items():
        voxel_count = int(np.sum(pred_mask == label_id))
        volumes[name] = round(voxel_count * voxel_ml, 3)
    return volumes


def segment_and_measure(image_path: str) -> dict:
    """Preprocess one scan, segment it, and compute organ volumes."""
    from preprocess import preprocess_image

    image_tensor, meta = preprocess_image(image_path)
    pred_mask = predict_segmentation(image_tensor)
    voxel_ml = voxel_volume_ml(meta)
    return organ_volumes_ml(pred_mask, voxel_ml)


def generate_report(before_path: str, after_path: str) -> dict:
    """Run segmentation on before/after scans and compare organ volumes."""
    from compare import compare_volumes

    before_volumes = segment_and_measure(before_path)
    after_volumes = segment_and_measure(after_path)

    result = compare_volumes(before_volumes, after_volumes)
    result["model_version"] = MODEL_VERSION
    result["device"] = str(DEVICE)
    return result
