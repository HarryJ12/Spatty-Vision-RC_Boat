"""Run the bottom-half OpenCV color experiment against an image file.

Example:
  python3 tools/demo_opencv_experiments.py path/to/test-frame.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from opencv_experiments import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--detection", action="append", default=[], metavar="LABEL,X1,Y1,X2,Y2",
        help="Example: --detection ball,20,30,95,105 (repeat for each ball)",
    )
    args = parser.parse_args()
    detections = []
    for raw in args.detection:
        label, x1, y1, x2, y2 = raw.split(",")
        detections.append({"label": label, "bbox": (int(x1), int(y1), int(x2), int(y2))})
    if not detections:
        # Demonstration only: substitute ViT/OWL boxes in real use.
        width, height = Image.open(args.image).size
        detections = [{"label": "ball", "bbox": (width // 4, height // 4, width * 3 // 4, height * 3 // 4)}]
    for index, result in enumerate(run_pipeline(Image.open(args.image), detections), start=1):
        print(
            f"Ball {index}: {result.predicted_color} "
            f"({result.confidence:.0%} confidence), bbox={result.bbox}, "
            f"dominant_rgb={result.dominant_rgb}"
        )


if __name__ == "__main__":
    main()
