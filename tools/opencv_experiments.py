"""Ball-color experiments built around ViT detections and OpenCV.

OpenCV 1 prepares the glare-resistant crops for GPT color inference.
OpenCV 2 uses the same crops with a local KMeans dominant-color plus LAB
prototype classifier. It is deliberately lightweight and deterministic; it is
not a downloaded or trained neural-network color model.

All image inputs/outputs are RGB NumPy arrays unless otherwise noted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image


COLOR_LABELS = ("red", "orange", "yellow", "green", "blue", "purple", "pink")
BALL_LABELS = {"ball", "toy ball", "sports ball"}

# Approximate sRGB representatives.  Classification occurs in LAB so Euclidean
# distance better reflects visible color differences than raw RGB distance.
COLOR_PROTOTYPES_RGB: Mapping[str, tuple[int, int, int]] = {
    "red": (220, 45, 38), "orange": (239, 125, 32), "yellow": (238, 208, 38),
    "green": (43, 152, 66), "blue": (45, 103, 210), "purple": (112, 61, 161),
    "pink": (224, 75, 160),
}


@dataclass(frozen=True)
class BallColorPrediction:
    """One detector proposal plus its local bottom-half color estimate."""

    object_label: str
    bbox: tuple[int, int, int, int]
    predicted_color: str
    confidence: float
    processed_crop: np.ndarray
    dominant_rgb: tuple[int, int, int]


def normalize_bbox(bbox: Sequence[int | float]) -> tuple[int, int, int, int]:
    """Normalize x1/y1/x2/y2 coordinates, accepting either corner ordering."""
    if len(bbox) != 4:
        raise ValueError("bbox must contain four coordinates: x1, y1, x2, y2")
    x1, y1, x2, y2 = (round(float(value)) for value in bbox)
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def crop_detection(image: np.ndarray | Image.Image, bbox: Sequence[int | float]) -> np.ndarray:
    """Crop a detection safely; coordinates outside the image are clipped."""
    rgb = np.asarray(image.convert("RGB") if isinstance(image, Image.Image) else image)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("image must be an RGB image with shape (height, width, 3)")
    x1, y1, x2, y2 = normalize_bbox(bbox)
    height, width = rgb.shape[:2]
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((0, 0, 3), dtype=np.uint8)
    return rgb[y1:y2, x1:x2].copy()


def apply_bottom_half_mask(crop: np.ndarray) -> np.ndarray:
    """Retain only the lower half of a detected physical ball.

    In this OV7670 setup, the upper arc has the strongest lamp glare and is
    often clipped to pink/white/orange.  ViT boxes target the physical upper
    sphere rather than its lower water reflection, so the crop's bottom half is
    the most stable evidence of the ball's base pigment.  Zeroing the rest
    makes that rule explicit to either the local extractor or GPT.
    """
    if crop.size == 0:
        return crop.copy()
    masked = np.zeros_like(crop)
    masked[crop.shape[0] // 2 :, :, :] = crop[crop.shape[0] // 2 :, :, :]
    return masked


def preprocess_color_region(crop: np.ndarray, size: int = 96) -> np.ndarray:
    """Denoise, normalize local brightness, and suppress dark background.

    The result remains RGB so it can go directly to a PIL/GPT image input.
    A small CLAHE pass on LAB lightness improves dim-ball contrast without
    changing hue/chroma or claiming to perform camera white balance. HSV/LAB
    conversions then identify near-black tank background and make the local
    color distance calculation robust.
    """
    if crop.size == 0:
        return crop.copy()
    resized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(resized, (3, 3), 0)
    lab = cv2.cvtColor(blurred, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4)).apply(lab[:, :, 0])
    normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    hsv = cv2.cvtColor(normalized, cv2.COLOR_RGB2HSV)
    # Keep modestly lit, colored pixels; black water/tank regions carry no
    # useful pigment signal.  Low-saturation white/gray balls remain valid.
    valid = hsv[:, :, 2] >= 28
    processed = np.zeros_like(blurred)
    processed[valid] = normalized[valid]
    return processed


def _valid_pixels(processed_crop: np.ndarray) -> np.ndarray:
    if processed_crop.size == 0:
        return np.empty((0, 3), dtype=np.uint8)
    hsv = cv2.cvtColor(processed_crop, cv2.COLOR_RGB2HSV)
    return processed_crop[hsv[:, :, 2] >= 28]


def _kmeans_dominant_rgb(processed_crop: np.ndarray, clusters: int = 3) -> tuple[np.ndarray, float]:
    """Return the largest non-black KMeans cluster and its coverage ratio."""
    pixels = _valid_pixels(processed_crop)
    if len(pixels) == 0:
        return np.array((0, 0, 0), dtype=np.uint8), 0.0
    k = min(clusters, len(pixels))
    _compactness, labels, centers = cv2.kmeans(
        pixels.astype(np.float32), k, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5),
        3, cv2.KMEANS_PP_CENTERS,
    )
    counts = np.bincount(labels.ravel(), minlength=k)
    index = int(np.argmax(counts))
    return np.clip(np.rint(centers[index]), 0, 255).astype(np.uint8), float(counts[index] / len(pixels))


def _lab_prototypes() -> dict[str, np.ndarray]:
    values = np.array(list(COLOR_PROTOTYPES_RGB.values()), dtype=np.uint8).reshape(-1, 1, 3)
    lab = cv2.cvtColor(values, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    return dict(zip(COLOR_PROTOTYPES_RGB, lab))


LAB_PROTOTYPES = _lab_prototypes()


def predict_color(crop: np.ndarray) -> tuple[str, float, tuple[int, int, int]]:
    """Predict a discrete color from a preprocessed, bottom-half-only crop.

    This is a local seven-class color classifier: KMeans estimates the dominant
    base pigment, then nearest LAB prototype chooses the competition label.
    """
    dominant, coverage = _kmeans_dominant_rgb(crop)
    if coverage == 0.0:
        return "black", 0.0, (0, 0, 0)
    lab = cv2.cvtColor(dominant.reshape(1, 1, 3), cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)
    distances = {label: float(np.linalg.norm(lab - prototype)) for label, prototype in LAB_PROTOTYPES.items()}
    label, distance = min(distances.items(), key=lambda item: item[1])
    # Coverage distinguishes a stable dominant pigment from a mixed crop;
    # LAB distance distinguishes a prototypical color from an ambiguous one.
    confidence = max(0.0, min(1.0, (0.40 + 0.60 * coverage) * np.exp(-distance / 70.0)))
    return label, confidence, tuple(int(value) for value in dominant)


def is_ball_detection(detection: Mapping[str, Any]) -> bool:
    """Accept common detector labels while rejecting unrelated object proposals."""
    label = str(detection.get("label", "")).strip().lower()
    return label in BALL_LABELS or "ball" in label


def detection_bbox(detection: Mapping[str, Any]) -> tuple[int, int, int, int]:
    """Read either a ``bbox`` field or the explicit x1/y1/x2/y2 convention."""
    bbox = detection.get("bbox")
    if bbox is None:
        bbox = (detection["x1"], detection["y1"], detection["x2"], detection["y2"])
    return normalize_bbox(bbox)


def run_pipeline(image: np.ndarray | Image.Image, detections: Iterable[Mapping[str, Any]]) -> list[BallColorPrediction]:
    """Run OpenCV 2 local color classification on all ball/toy-ball detections."""
    results: list[BallColorPrediction] = []
    for detection in detections:
        if not is_ball_detection(detection):
            continue
        bbox = detection_bbox(detection)
        crop = crop_detection(image, bbox)
        masked = apply_bottom_half_mask(crop)
        processed = preprocess_color_region(masked)
        color, confidence, dominant_rgb = predict_color(processed)
        if confidence <= 0.0:
            continue
        results.append(BallColorPrediction(
            object_label=str(detection.get("label", "ball")), bbox=bbox,
            predicted_color=color, confidence=confidence,
            processed_crop=processed, dominant_rgb=dominant_rgb,
        ))
    return results


def make_contact_sheet(predictions: Sequence[BallColorPrediction]) -> bytes:
    """Encode processed crops into a labeled vertical JPEG for OpenCV 1/GPT."""
    if not predictions:
        return b""
    tiles = [Image.fromarray(item.processed_crop) for item in predictions if item.processed_crop.size]
    if not tiles:
        return b""
    side = max(tile.width for tile in tiles)
    sheet = Image.new("RGB", (side, (side + 18) * len(tiles)), "black")
    for index, tile in enumerate(tiles, start=1):
        sheet.paste(tile.resize((side, side)), (0, (index - 1) * (side + 18) + 18))
    import io
    encoded = io.BytesIO()
    sheet.save(encoded, format="JPEG", quality=90)
    return encoded.getvalue()
