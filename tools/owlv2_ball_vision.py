"""Local OWLv2 ViT zero-shot ball detection for the Levithian dashboard."""

from __future__ import annotations

import io
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class SportsBall:
    """One local ball proposal with confidence, center, and source box."""

    confidence_percent: int
    x: int
    y: int
    mask_points: int = 0
    left: int = 0
    top: int = 0
    right: int = 0
    bottom: int = 0


@dataclass(frozen=True)
class SportsBallResult:
    balls: tuple[SportsBall, ...]
    status: str
    completed_at: float = 0.0
    latency_seconds: float | None = None
    is_analyzing: bool = False
    has_started: bool = False


MODEL_NAME = "google/owlv2-base-patch16-ensemble"
# OWLv2's generic prompts are empirically more reliable than pool-specific
# wording on the supplied reference frames. Reflection/color reasoning belongs
# in the downstream OpenAI color prompt, not in this local detector query.
BALL_QUERIES = ["a sports ball", "a tennis ball", "a floating ball"]
# On the live tank scene, true balls score in the 70% range while reflections
# and unrelated objects cluster below 50%. Reject those weak proposals before
# they reach Candidate Close-Up or downstream color classification.
CONFIDENCE_THRESHOLD = 0.50
MIN_BOX_SIDE_PIXELS = 18
MAX_BALLS = 5
# The camera can deliver new frames much faster than the dashboard should change
# its decision.  Run a deliberate, bounded control loop instead of treating
# every arriving serial frame as a fresh decision opportunity.
# A new local decision begins at most once every five seconds. The live preview
# is not throttled; only model work is.
MINIMUM_START_INTERVAL_SECONDS = 5.0


class Owlv2BallRunner:
    """Run a local ViT detector in a background thread on the latest frame."""

    def __init__(
        self,
        result_display_seconds: float = 4.0,
        minimum_start_interval_seconds: float = MINIMUM_START_INTERVAL_SECONDS,
    ) -> None:
        self.result_display_seconds = result_display_seconds
        self.minimum_start_interval_seconds = minimum_start_interval_seconds
        self._lock = threading.Lock()
        self._in_flight = False
        self._last_started = 0.0
        self._processor = None
        self._model = None
        self._device = None
        self.result = SportsBallResult((), "OWLv2 ViT ready")

    def submit(
        self, jpeg: bytes, on_complete: Callable[[SportsBallResult], None] | None = None
    ) -> bool:
        with self._lock:
            visible = self.result.completed_at and time.time() - self.result.completed_at < self.result_display_seconds
            elapsed_since_start = time.monotonic() - self._last_started
            if (
                self._in_flight
                or visible
                or elapsed_since_start < self.minimum_start_interval_seconds
            ):
                return False
            self._in_flight = True
            self._last_started = time.monotonic()
            self.result = SportsBallResult(
                self.result.balls,
                "Deepwater objective sweep...",
                completed_at=self.result.completed_at,
                latency_seconds=self.result.latency_seconds,
                is_analyzing=True,
                has_started=True,
            )
        threading.Thread(target=self._infer, args=(jpeg, on_complete), daemon=True).start()
        return True

    def seconds_until_next_start(self) -> float:
        """Return the remaining cadence hold without exposing mutable state."""
        with self._lock:
            return max(0.0, self.minimum_start_interval_seconds - (time.monotonic() - self._last_started))

    def snapshot(self) -> SportsBallResult:
        with self._lock:
            return self.result

    def _load_model(self):
        if self._model is None:
            import torch
            from transformers import Owlv2ForObjectDetection, Owlv2Processor

            self._device = "mps" if torch.backends.mps.is_available() else "cpu"
            self._processor = Owlv2Processor.from_pretrained(MODEL_NAME)
            self._model = Owlv2ForObjectDetection.from_pretrained(MODEL_NAME).to(self._device).eval()
        return self._processor, self._model, self._device

    def _infer(self, jpeg: bytes, on_complete: Callable[[SportsBallResult], None] | None = None) -> None:
        started_at = time.monotonic()
        try:
            import torch

            image = Image.open(io.BytesIO(jpeg)).convert("RGB")
            processor, model, device = self._load_model()
            inputs = processor(text=[BALL_QUERIES], images=image, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            target_sizes = torch.tensor([image.size[::-1]], device=device)
            detections = processor.post_process_object_detection(
                outputs=outputs, target_sizes=target_sizes, threshold=CONFIDENCE_THRESHOLD
            )[0]

            # Multiple text phrases can match one ball. Retain the highest score
            # per overlapping box using a lightweight IoU suppression pass.
            candidates = []
            for box, score in zip(detections["boxes"].cpu().tolist(), detections["scores"].cpu().tolist()):
                left, top, right, bottom = box
                if right - left < MIN_BOX_SIDE_PIXELS or bottom - top < MIN_BOX_SIDE_PIXELS:
                    continue
                candidates.append((float(score), left, top, right, bottom))
            candidates.sort(reverse=True)
            kept = []
            for candidate in candidates:
                score, left, top, right, bottom = candidate
                duplicate = False
                for _, kl, kt, kr, kb in kept:
                    intersection = max(0, min(right, kr) - max(left, kl)) * max(0, min(bottom, kb) - max(top, kt))
                    union = (right - left) * (bottom - top) + (kr - kl) * (kb - kt) - intersection
                    if union and intersection / union > 0.30:
                        duplicate = True
                        break
                if not duplicate:
                    kept.append(candidate)
            balls = tuple(sorted((
                SportsBall(
                    round(score * 100), round((left + right) / 2), round((top + bottom) / 2), 0,
                    round(left), round(top), round(right), round(bottom),
                )
                for score, left, top, right, bottom in kept
            ), key=lambda ball: ball.x))[:MAX_BALLS]
            result = SportsBallResult(
                balls,
                "NO BALL DETECTED" if not balls else f"Detected {len(balls)} zero-shot ball" + ("s" if len(balls) != 1 else ""),
                completed_at=time.time(),
                latency_seconds=time.monotonic() - started_at,
                is_analyzing=False,
                has_started=True,
            )
        except Exception as exc:
            result = SportsBallResult((), f"OWLv2 error: {exc}", time.time(), time.monotonic() - started_at, False, True)
        with self._lock:
            self.result = result
            self._in_flight = False
        if on_complete is not None:
            on_complete(result)
