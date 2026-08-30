"""Display-side color calibration for the OV7670 serial stream.

The board's tutorial stream has a pronounced scene-dependent color cast.  This
module uses the brightest few percent of a frame as an approximate white
reference (a standard ``white-patch`` white-balance method) and applies one
RGB gain per frame.  It deliberately runs before any future vision pipeline.
"""

from __future__ import annotations

import numpy as np
import cv2
from PIL import Image


class AutoWhiteBalance:
    """Neutralize a color cast while preserving the frame's relative colors."""

    bright_percentile = 96
    min_gain = 0.45
    max_gain = 2.8
    shadow_gamma = 0.58

    def apply(self, image: Image.Image) -> Image.Image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
        luminance = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
        brightest = rgb[luminance >= np.percentile(luminance, self.bright_percentile)]
        if brightest.size == 0:
            return image

        # Median is robust to a few saturated pixels. A bright window/white
        # card should have equal R/G/B after correction.
        reference = np.median(brightest, axis=0)
        reference = np.maximum(reference, 1.0)
        target = float(np.mean(reference))
        gains = np.clip(target / reference, self.min_gain, self.max_gain)
        corrected = np.clip(rgb * gains, 0, 255).astype(np.uint8)

        # The bright window makes the small OV7670 sensor underexpose the room.
        # A gentle gamma curve raises shadow detail while keeping highlights
        # bounded, then modest CLAHE restores local contrast in the dark areas.
        lifted = np.power(corrected.astype(np.float32) / 255.0, self.shadow_gamma)
        lifted = np.clip(lifted * 255.0, 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(lifted, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=1.4, tileGridSize=(8, 8)).apply(lab[:, :, 0])
        return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), "RGB")


def apply_rgb_gains(image: Image.Image, gains: tuple[float, float, float]) -> Image.Image:
    """Apply a measured, fixed RGB correction without exposure processing."""
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    corrected = np.clip(pixels * np.asarray(gains, dtype=np.float32), 0, 255).astype(np.uint8)
    return Image.fromarray(corrected, "RGB")
