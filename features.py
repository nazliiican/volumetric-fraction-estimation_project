"""Hand-crafted image features for the Random Forest baseline.

Each feature is a simple, interpretable summary of the whole image: global
colour statistics (RGB + HSV means/stds), per-channel colour histograms, and an
optional Local Binary Pattern (LBP) texture histogram."""

import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.colors import rgb_to_hsv

try:
    from skimage.feature import local_binary_pattern
    _HAS_SKIMAGE = True
except Exception:
    local_binary_pattern = None
    _HAS_SKIMAGE = False

_RESAMPLE = getattr(
    getattr(Image, "Resampling", Image),
    "BILINEAR"
)

_LBP_P = 8                 
_LBP_R = 1.0               
_LBP_BINS = _LBP_P + 2     

_warned_no_skimage = False


def load_image(image_path, image_size=224):
    """Load an image as an RGB array resized to ``image_size x image_size``.
    """
    img = Image.open(image_path)
    img.draft("RGB", (image_size, image_size))
    img = img.convert("RGB").resize((image_size, image_size), _RESAMPLE)
    return np.asarray(img, dtype=np.uint8)


def _normalized_histogram(channel, bins, value_range=(0.0, 1.0)):
    """Histogram of a single channel, normalized to sum to 1."""
    hist, _ = np.histogram(channel, bins=bins, range=value_range)
    total = hist.sum()
    if total == 0:
        return np.full(bins, 1.0 / bins)
    return hist / total


def extract_color_features(image_path, image_size=224, hist_bins=16):
    """Colour statistics and normalized colour histograms (RGB + HSV).
    """
    rgb = load_image(image_path, image_size).astype(np.float64) / 255.0
    hsv = rgb_to_hsv(rgb)

    features = {}

    for i, name in enumerate("RGB"):
        features[f"rgb_mean_{name}"] = float(rgb[:, :, i].mean())
        features[f"rgb_std_{name}"] = float(rgb[:, :, i].std())
    for i, name in enumerate("HSV"):
        features[f"hsv_mean_{name}"] = float(hsv[:, :, i].mean())
        features[f"hsv_std_{name}"] = float(hsv[:, :, i].std())

    for i, name in enumerate("RGB"):
        hist = _normalized_histogram(rgb[:, :, i], hist_bins)
        for b in range(hist_bins):
            features[f"rgb_hist_{name}_{b:02d}"] = float(hist[b])
    for i, name in enumerate("HSV"):
        hist = _normalized_histogram(hsv[:, :, i], hist_bins)
        for b in range(hist_bins):
            features[f"hsv_hist_{name}_{b:02d}"] = float(hist[b])

    return features


def extract_texture_features(image_path, image_size=224):
    """Normalized uniform Local Binary Pattern histogram (texture).
    """
    global _warned_no_skimage
    if not _HAS_SKIMAGE:
        if not _warned_no_skimage:
            print(
                "[features] scikit-image not available -- skipping LBP texture "
                "features (using colour features only)."
            )
            _warned_no_skimage = True
        return {}

    rgb = load_image(image_path, image_size)            
    gray = (rgb @ np.array([0.299, 0.587, 0.114])).astype(np.uint8)

    if local_binary_pattern is None:
        return {}
    
    lbp = local_binary_pattern(gray, _LBP_P, _LBP_R, method="uniform")
    hist = _normalized_histogram(lbp, _LBP_BINS, value_range=(0, _LBP_BINS))
    return {f"lbp_{b:02d}": float(hist[b]) for b in range(_LBP_BINS)}


def extract_features_for_dataframe(
    image_df, image_size=224, hist_bins=16, include_texture=True
):
    """Extract features for every image row -> one DataFrame row per image.

    The returned DataFrame keeps the ``Code`` and ``image_path`` columns,
    followed by the named feature columns."""
   
    try:
        from tqdm import tqdm

        iterator = tqdm(
            image_df.iterrows(), total=len(image_df), desc="Extracting features"
        )
    except Exception:  
        iterator = image_df.iterrows()

    rows = []
    for _, row in iterator:
        feats = {"Code": row["Code"], "image_path": row["image_path"]}
        feats.update(extract_color_features(row["image_path"], image_size, hist_bins))
        if include_texture:
            feats.update(extract_texture_features(row["image_path"], image_size))
        rows.append(feats)

    return pd.DataFrame(rows)
