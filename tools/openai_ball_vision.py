"""Rate-limited GPT-5.4 vision inference for the Levithian camera dashboard."""

from __future__ import annotations

import base64
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path


ALLOWED_COLORS = ("green", "purple", "pink", "yellow", "orange", "red", "blue")


PROMPT = """You are the perception module for a student boat competition. Inspect
ONLY the supplied camera frame. Your only task is to identify visible physical
floating balls and their real-world colors. Do not give navigation, collection,
motor, target, or general scene advice.

Scene and camera conditions:
- The dim water tank, walls, and shadows are dark green/blue. Other boats,
  mechanisms, 3D-printed/metal/plastic parts, hands, cards, text, and tank
  features are background, not balls.
- Target balls are plain solid-color spheres. They can move on the water, touch,
  overlap, or be partly cut off by a frame edge.
- This OV7670 feed is low-resolution and can be blurred, dark, neon, or shifted
  magenta/green when the boat or balls move. Use the visible rounded shape and
  the consistent broad color, but do not invent a ball from corruption.

Count and reflection rule:
- Scan left to right. Return every distinct physical rounded/spherical floating
  ball exactly once, including a recognizable partial ball.
- A bright cap, patch, band, logo-like mark, or shadow on a rounded sphere is a
  surface feature of that same ball, never another ball.
- A real upper ball commonly has a directly-below water reflection, producing a
  vertical figure-8. Count and classify the UPPER physical sphere once. Ignore
  the lower reflected duplicate even when it is circular, bright, purple-shifted,
  or nearly as large as the real ball.
- Do not count water, a highlight, a wheel, or a mechanical part as a ball.

Color rule:
- Every ball has a bright curved top glare cap. It may match the ball, or shift
  pink/red/orange/white. It is part of the same ball, but its tiny clipped or
  saturated center is weak evidence.
- Infer color mainly from the broad, non-reflected middle/lower body and rim of
  the UPPER sphere. The surrounding cap can support that conclusion only when it
  agrees. Never use the lower reflection as color evidence.
- Blue is valid only for a clearly spherical object with a broad cobalt/azure
  body, never blue water or a small cyan glare patch.
- Allowed colors only: green, purple, pink, yellow, orange, red, blue. Green is
  distinctly green; yellow is yellow/gold; blue is cobalt/azure; purple is
  violet/indigo; pink is rose/fuchsia/magenta. Keep neon warm colors distinct:
  orange has an amber/yellow-orange broad rim, red is warm crimson with little
  yellow, and pink has a stronger rose/magenta/purple-blue cast. A hot-pink or
  white glare alone must not turn an orange/red ball pink.

For each clearly visible ball, choose the best defensible base color and lower
confidence when lighting or motion makes it uncertain. If no ball is clearly
identifiable, return an empty list. Return balls in strict left-to-right order.

Return valid JSON only, with no Markdown or extra keys. Include only balls at
50–100 confidence. Use exactly:
{"balls":[{"color":"green","confidence_percent":86}]}
"""


CROP_PROMPT = """You are the color-classification module for a student boat
competition. Inspect ONLY the supplied candidate close-up sheet. Each tile is a
padded crop produced by a local ViT detector, ordered top-to-bottom in the same
left-to-right order as the original camera frame. Classify color only; do not
give navigation or scene advice.

Each tile is intended to contain one physical floating ball, but may also contain
dark blue/green tank background, a directly-below water reflection, or a bright
top glare cap. For a real ball, classify only the upper physical sphere once;
ignore its lower reflected duplicate. Infer color mainly from the broad
middle/lower body and rim. A small bright cap may be white, pink, red, or orange
glare and cannot determine color by itself.

Allowed colors only: green, purple, pink, yellow, orange, red, blue. Keep neon
warm colors distinct: orange has an amber/yellow-orange body/rim, red is warm
crimson with little yellow, and pink is rose/fuchsia/magenta with more
purple/blue. Do not call blue from background water or a small cyan glare patch.
If a tile does not clearly contain a physical ball, omit it. Return valid JSON
only, in tile order, with 50–100 confidence and no extra keys:
{"balls":[{"color":"green","confidence_percent":86}]}
"""

# Meta v2 receives a contact sheet made from same-frame, padded OWLv2 proposals
# rather than the entire camera frame.  Keep the full-frame guardrails in
# ``PROMPT`` and add only the input-format and proposal-handoff rules here.
META_V2_PROMPT = """The supplied image is a vertical contact sheet of numbered,
same-frame close-ups proposed by a local ViT detector. Tile 1 is the first
candidate, then tile 2, and so on. Each candidate has a supplied position and
confidence in the original frame. Treat every tile as strong but non-binding
evidence that it contains one physical ball: inspect it yourself, omit a clear
false proposal, and return at most one result per tile in top-to-bottom order.

Each crop may include dark tank background, a neighboring object, lamp glare,
or the ball's lower water reflection. Do not report the reflection as another
ball. Use the broad non-reflective body/rim, especially the lower physical half,
for its base color. The bright upper semicircle may be glare and is not by
itself the true color.

""" + PROMPT


OPENCV_CROP_PROMPT = """The supplied image is a vertical contact sheet of
same-frame ViT ball proposals after local OpenCV processing. Tiles are ordered
top-to-bottom in the original frame's left-to-right order. The OpenCV layer
uses the lower physical half, suppresses near-black tank pixels, reduces noise,
and normalizes local brightness. It deliberately does not apply an unmeasured
white-balance transform, so reason from the remaining broad body/rim rather
than assuming the pixels are perfectly color-calibrated.

Black regions are removed background or masked glare, not evidence of a black
ball. Each tile is strong but non-binding evidence of one ball: validate it,
omit a clear false proposal, and return at most one result per tile in
top-to-bottom order. Use the supplied ViT coordinate/confidence handoff as
helpful spatial evidence.

""" + PROMPT

ENV_FILE = Path(os.getenv("AZURE_OPENAI_ENV_FILE", "/Users/hjoshi/Documents/PlatformIO/.env.local"))


def load_env_values(names: set[str]) -> dict[str, str]:
    """Read only explicitly requested Azure settings from the designated file."""
    values = {name: os.environ[name] for name in names if os.getenv(name)}
    try:
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line.startswith("export "):
                line = line[7:].lstrip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            name = name.strip()
            if name in names and name not in values:
                values[name] = value.strip().strip('"').strip("'")
    except OSError:
        pass
    return values


@dataclass(frozen=True)
class BallVisionResult:
    balls: tuple[tuple[str, int], ...]
    status: str
    completed_at: float = 0.0
    latency_seconds: float | None = None
    is_analyzing: bool = False
    has_started: bool = False


class BallVisionRunner:
    """Submit at most one frame every interval; never persist frames or API keys."""

    def __init__(
        self,
        result_display_seconds: float = 4.0,
        prompt: str = PROMPT,
        minimum_start_interval_seconds: float = 7.0,
    ) -> None:
        self.result_display_seconds = result_display_seconds
        self.prompt = prompt
        self.minimum_start_interval_seconds = minimum_start_interval_seconds
        values = load_env_values({
            "AZURE_OPENAI_API_KEY",
            "CLINICAL_DATA_AZURE_OPENAI_ENDPOINT",
            "CLINICAL_DATA_AZURE_OPENAI_API_VERSION",
        })
        self.api_key = values.get("AZURE_OPENAI_API_KEY")
        self.endpoint = values.get("CLINICAL_DATA_AZURE_OPENAI_ENDPOINT")
        self.api_version = values.get("CLINICAL_DATA_AZURE_OPENAI_API_VERSION")
        self.model = "gpt-5.4"
        self._last_started = 0.0
        self._lock = threading.Lock()
        self._in_flight = False
        missing = [name for name, value in (("AZURE_OPENAI_API_KEY", self.api_key), ("CLINICAL_DATA_AZURE_OPENAI_ENDPOINT", self.endpoint), ("CLINICAL_DATA_AZURE_OPENAI_API_VERSION", self.api_version)) if not value]
        self.result = BallVisionResult((), "GPT-5.4 vision waiting for " + ", ".join(missing) if missing else "GPT-5.4 vision ready")

    def submit(self, jpeg: bytes, detector_context: str = "") -> bool:
        if not all((self.api_key, self.endpoint, self.api_version)):
            return False
        with self._lock:
            result_is_still_displayed = (
                self.result.completed_at
                and time.time() - self.result.completed_at < self.result_display_seconds
            )
            elapsed_since_start = time.monotonic() - self._last_started
            if (
                self._in_flight
                or result_is_still_displayed
                or elapsed_since_start < self.minimum_start_interval_seconds
            ):
                return False
            self._in_flight = True
            self._last_started = time.monotonic()
            self.result = BallVisionResult(
                self.result.balls,
                "Deepwater objective sweep...",
                completed_at=self.result.completed_at,
                latency_seconds=self.result.latency_seconds,
                is_analyzing=True,
                has_started=True,
            )
        threading.Thread(target=self._infer, args=(jpeg, detector_context), daemon=True).start()
        return True

    def snapshot(self) -> BallVisionResult:
        with self._lock:
            return self.result

    def seconds_until_next_start(self) -> float:
        """Return the remaining cadence hold without exposing mutable state."""
        with self._lock:
            return max(0.0, self.minimum_start_interval_seconds - (time.monotonic() - self._last_started))

    def record_no_ball_detected(self) -> None:
        """Record a local upstream no-candidate result without making a cloud call."""
        with self._lock:
            if self._in_flight:
                return
            self.result = BallVisionResult((), "NO BALL DETECTED", completed_at=time.time(), has_started=True)

    def _infer(self, jpeg: bytes, detector_context: str = "") -> None:
        started_at = time.monotonic()
        try:
            try:
                from openai import AzureOpenAI
                import httpx
            except ImportError:
                raise RuntimeError("OpenAI SDK is not installed. Run pip install -r tools/requirements-camera-ui.txt")

            encoded = base64.b64encode(jpeg).decode("ascii")
            response = AzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.endpoint,
                api_version=self.api_version,
                # The copied .env.local CA paths are stale on this Mac.  Use the
                # verified macOS trust store instead of inheriting those paths.
                http_client=httpx.Client(timeout=30.0, trust_env=False),
            ).chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.prompt},
                    {"role": "user", "content": [
                        *([{"type": "text", "text": detector_context}] if detector_context else []),
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded}", "detail": "low"},
                        },
                    ]},
                ],
                max_completion_tokens=200,
            )
            payload = json.loads(response.choices[0].message.content or "")
            balls = payload.get("balls", [])
            if not isinstance(balls, list):
                raise ValueError("Response did not contain a balls list")
            clean: list[tuple[str, int]] = []
            allowed = set(ALLOWED_COLORS)
            for ball in balls:
                color = str(ball.get("color", "")).lower()
                confidence = ball.get("confidence_percent")
                if color in allowed and isinstance(confidence, int) and 50 <= confidence <= 100:
                    clean.append((color, confidence))
            status = "NO BALL DETECTED" if not clean else "Detected: " + ", ".join(color.title() for color, _ in clean)
            result = BallVisionResult(
                tuple(clean),
                status,
                completed_at=time.time(),
                latency_seconds=time.monotonic() - started_at,
                is_analyzing=False,
                has_started=True,
            )
        except Exception as exc:  # Keep the camera dashboard available on API failures.
            result = BallVisionResult(
                (),
                f"AI vision error: {exc}",
                completed_at=time.time(),
                latency_seconds=time.monotonic() - started_at,
                is_analyzing=False,
                has_started=True,
            )
        with self._lock:
            self.result = result
            self._in_flight = False
