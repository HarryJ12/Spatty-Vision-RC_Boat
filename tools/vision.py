"""Small, local OpenCV pipeline for the OV7670 dashboard.

The OV7670 frame arrives as RGB from ``do_we_know_camera``.  This module keeps
all computer vision on the Mac: it identifies saturated, round, non-blue
regions and reports the strongest candidate's color and image coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Detection:
    color: str
    x: int
    y: int
    area: float
    circularity: float


# OpenCV hue is 0–179. Blue is deliberately absent: it is the competition
# tank/wall color and must never become a ball candidate.
COLOR_RANGES: dict[str, tuple[tuple[int, int, int], ...]] = {
    "Red": ((0, 10, 80), (170, 179, 80)),
    "Orange": ((11, 25, 80),),
    "Yellow": ((26, 38, 90),),
    "Green": ((39, 84, 70),),
    "Purple": ((125, 169, 65),),
}


class VisionAnalyzer:
    """Find a single strongest colored-ball candidate in a camera frame."""

    min_area = 18.0
    min_circularity = 0.58

    def analyze(self, image: Image.Image) -> tuple[Image.Image, Detection | None]:
        rgb = np.asarray(image.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        best: Detection | None = None

        for color, hue_ranges in COLOR_RANGES.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for low_hue, high_hue, minimum_value in hue_ranges:
                lower = np.array((low_hue, 85, minimum_value), dtype=np.uint8)
                upper = np.array((high_hue, 255, 255), dtype=np.uint8)
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                if area < self.min_area or perimeter <= 0:
                    continue
                circularity = 4.0 * np.pi * area / (perimeter * perimeter)
                if circularity < self.min_circularity:
                    continue
                moments = cv2.moments(contour)
                if moments["m00"] == 0:
                    continue
                candidate = Detection(
                    color=color,
                    x=round(moments["m10"] / moments["m00"]),
                    y=round(moments["m01"] / moments["m00"]),
                    area=area,
                    circularity=circularity,
                )
                if best is None or candidate.area * candidate.circularity > best.area * best.circularity:
                    best = candidate

        if best is not None:
            cv2.circle(bgr, (best.x, best.y), max(6, round(np.sqrt(best.area / np.pi))), (83, 244, 173), 1)
            cv2.putText(
                bgr,
                f"{best.color} ({best.x}, {best.y})",
                (4, 13),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                (83, 244, 173),
                1,
                cv2.LINE_AA,
            )
        return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), best

