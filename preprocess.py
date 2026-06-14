"""
Preprocessing pipeline for incoming NIfTI scans.
Mirrors preprocess_FIXED.ipynb test transform used during inference.
"""
import warnings
warnings.filterwarnings("ignore")

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    Orientationd,
    Spacingd,
    CropForegroundd,
    ScaleIntensityRangePercentilesd,
    ResizeWithPadOrCropd,
)

# Same spacing used in preprocess/training pipeline
PIXDIM = (1.5, 1.5, 3.0)
CROP_MARGIN = 10
SPATIAL_SIZE = (96, 96, 72)


def get_inference_transform():
    """
    Load -> ChannelFirst -> RAS orientation -> Spacing -> CropForeground
    -> intensity scaling 0.5-99.5 percentile -> ResizeWithPadOrCrop.
    """
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=PIXDIM, mode=("bilinear",)),
        CropForegroundd(keys=["image"], source_key="image", margin=CROP_MARGIN),
        ScaleIntensityRangePercentilesd(
            keys=["image"],
            lower=0.5,
            upper=99.5,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        ResizeWithPadOrCropd(keys=["image"], spatial_size=SPATIAL_SIZE),
        EnsureTyped(keys=["image"], track_meta=True),
    ])


def preprocess_image(image_path: str):
    """
    Run preprocessing on a single NIfTI file path.
    Returns preprocessed tensor (C, D, H, W) and metadata dict.
    """
    tf = get_inference_transform()
    data = tf({"image": image_path})
    img = data["image"]
    meta = data["image"].meta if hasattr(data["image"], "meta") else {}
    return img, meta
