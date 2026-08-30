#!/usr/bin/env python3
"""Levithian Camera View — local web dashboard for the OV7670 serial stream."""

from __future__ import annotations

import glob
import io
import json
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from PIL import Image
from urllib.parse import parse_qs, urlparse

from do_we_know_camera import ArduinoStreamReader
from openai_ball_vision import (
    META_V2_PROMPT,
    OPENCV_CROP_PROMPT,
    BallVisionResult,
    BallVisionRunner,
)
from opencv_experiments import make_contact_sheet, run_pipeline
from owlv2_ball_vision import Owlv2BallRunner


HOST = "127.0.0.1"
PORT = 8765
BACKGROUND_IMAGE = Path(__file__).parent / "assets" / "levithian-deep-sea.png"
# Keep the preview live, but make each perception decision deliberate.  The
# selected runner receives only the newest frame once this interval expires.
INFERENCE_INTERVAL_SECONDS = 5.0


@dataclass(frozen=True)
class CandidateCrop:
    """A padded ViT proposal preview from the exact analyzed camera frame."""

    frame: int
    candidate: int
    confidence_percent: int
    image: bytes


class OpenCvColorRunner:
    """Expose OCV2's local results through the same dashboard interface."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.result = BallVisionResult((), "OCV2 ready")

    def begin(self) -> None:
        with self._lock:
            self.result = BallVisionResult(
                self.result.balls, "Deepwater objective sweep...",
                completed_at=self.result.completed_at,
                latency_seconds=self.result.latency_seconds,
                is_analyzing=True, has_started=True,
            )

    def complete(self, predictions) -> None:
        balls = tuple((item.predicted_color, round(item.confidence * 100)) for item in predictions)
        with self._lock:
            self.result = BallVisionResult(
                balls,
                "NO BALL DETECTED" if not balls else "Detected: " + ", ".join(color.title() for color, _ in balls),
                completed_at=time.time(), latency_seconds=0.0, is_analyzing=False, has_started=True,
            )

    def snapshot(self) -> BallVisionResult:
        with self._lock:
            return self.result

    def seconds_until_next_start(self) -> float:
        return 0.0


class CameraState:
    def __init__(self) -> None:
        self.frames: queue.Queue[Image.Image] = queue.Queue(maxsize=1)
        self.status_messages: queue.Queue[str] = queue.Queue()
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.jpeg: bytes | None = None
        self.status = "Connecting to Arduino…"
        self.frame_number = 0
        self.frame_times: deque[float] = deque(maxlen=30)
        self.frame_rate_fps = 0.0
        self.candidate_crops: tuple[CandidateCrop, ...] = ()
        self.candidate_crop_frame = 0
        self.opencv_crops: tuple[CandidateCrop, ...] = ()
        self.opencv_crop_frame = 0
        self.color_profile = "Raw camera RGB (no profile)"
        self.vision_mode = "vit"
        # Each mode is synchronized to a fresh serial camera frame: show the
        # sweep while that frame is analyzed, then retain its result until the
        # following fresh frame begins the next analysis.
        # Camera frames may arrive quickly, but all perception modes are capped
        # to one fresh decision every seven seconds. Each runner is single-flight
        # and receives the newest available frame when its cooldown expires.
        self.vit_runner = Owlv2BallRunner(
            result_display_seconds=0.0,
            minimum_start_interval_seconds=INFERENCE_INTERVAL_SECONDS,
        )
        self.openai_runner = BallVisionRunner(
            result_display_seconds=0.0,
            minimum_start_interval_seconds=INFERENCE_INTERVAL_SECONDS,
        )
        self.hybrid_runner = BallVisionRunner(
            result_display_seconds=0.0,
            minimum_start_interval_seconds=INFERENCE_INTERVAL_SECONDS,
        )
        self.meta_v2_runner = BallVisionRunner(
            result_display_seconds=0.0,
            prompt=META_V2_PROMPT,
            minimum_start_interval_seconds=INFERENCE_INTERVAL_SECONDS,
        )
        self.ocv1_runner = BallVisionRunner(
            result_display_seconds=0.0,
            prompt=OPENCV_CROP_PROMPT,
            minimum_start_interval_seconds=INFERENCE_INTERVAL_SECONDS,
        )
        self.ocv2_runner = OpenCvColorRunner()
        self.vision_runners = {
            "vit": self.vit_runner,
            "openai": self.openai_runner,
            "vit_openai": self.hybrid_runner,
            "meta_v2": self.meta_v2_runner,
            "ocv1": self.ocv1_runner,
            "ocv2": self.ocv2_runner,
        }

    @property
    def ball_vision(self):
        return self.vision_runners[self.vision_mode]

    def set_vision_mode(self, mode: str) -> bool:
        if mode not in self.vision_runners:
            return False
        with self.lock:
            self.vision_mode = mode
        return True

    def start(self) -> None:
        threading.Thread(target=self.wait_for_arduino, daemon=True).start()

    def wait_for_arduino(self) -> None:
        """Keep retrying after USB reconnects or a macOS CH340 port error."""
        converter_started = False
        while True:
            ports = glob.glob("/dev/cu.usbserial-*")
            if ports:
                reader = ArduinoStreamReader(ports[0], self.frames, self.status_messages)
                reader.start()
                if not converter_started:
                    threading.Thread(target=self.convert_frames, daemon=True).start()
                    converter_started = True
                # A serial-open error can happen if macOS recreates the CH340
                # device after a reconnect. Wait for this reader to end, then
                # scan again rather than requiring a dashboard restart.
                reader.join()
            with self.lock:
                self.status = "Arduino not found — waiting for USB reconnect…"
            time.sleep(1)

    def convert_frames(self) -> None:
        while True:
            try:
                while True:
                    self.status = self.status_messages.get_nowait()
            except queue.Empty:
                pass
            try:
                image = self.frames.get(timeout=0.25)
            except queue.Empty:
                continue
            # The physical module is mounted sideways; this orientation is the
            # second 90° counterclockwise turn requested for the dashboard.
            image = image.rotate(180, expand=True)
            encoded = io.BytesIO()
            image.save(encoded, format="JPEG", quality=85)
            with self.condition:
                self.jpeg = encoded.getvalue()
                self.status = "Live feed online"
                self.frame_number += 1
                frame_number = self.frame_number
                self.frame_times.append(time.monotonic())
                if len(self.frame_times) > 1:
                    elapsed = self.frame_times[-1] - self.frame_times[0]
                    self.frame_rate_fps = (len(self.frame_times) - 1) / elapsed if elapsed else 0.0
                self.condition.notify_all()
            jpeg = self.jpeg
            with self.lock:
                mode = self.vision_mode
            if mode == "vit_openai":
                # This hybrid mode uses same-frame ViT count/coordinates as
                # explicit hints, then asks GPT-5.4 to validate and classify.
                self.vit_runner.submit(jpeg, lambda result: self.submit_openai_with_vit(jpeg, result))
            elif mode == "meta_v2":
                self.vit_runner.submit(
                    jpeg,
                    lambda result: self.submit_meta_v2(jpeg, frame_number, result),
                )
            elif mode == "ocv1":
                self.vit_runner.submit(jpeg, lambda result: self.submit_ocv1(jpeg, frame_number, result))
            elif mode == "ocv2":
                started = self.vit_runner.submit(jpeg, lambda result: self.submit_ocv2(jpeg, frame_number, result))
                if started:
                    self.ocv2_runner.begin()
            elif mode == "vit":
                self.vit_runner.submit(jpeg, lambda result: self.store_candidate_crops(jpeg, frame_number, result))
            else:
                self.ball_vision.submit(jpeg)

    def store_candidate_crops(self, jpeg: bytes, frame_number: int, vit_result) -> None:
        """Create inspectable padded previews from the same frame ViT used."""
        image = Image.open(io.BytesIO(jpeg)).convert("RGB")
        width, height = image.size
        crops: list[CandidateCrop] = []
        for candidate, ball in enumerate(vit_result.balls, start=1):
            side = max(40, round(max(ball.right - ball.left, ball.bottom - ball.top) * 1.5))
            center_x, center_y = (ball.left + ball.right) / 2, (ball.top + ball.bottom) / 2
            left = max(0, min(width - 1, round(center_x - side / 2)))
            top = max(0, min(height - 1, round(center_y - side / 2)))
            right, bottom = min(width, left + side), min(height, top + side)
            left, top = max(0, right - side), max(0, bottom - side)
            encoded = io.BytesIO()
            image.crop((left, top, right, bottom)).save(encoded, format="JPEG", quality=88)
            crops.append(CandidateCrop(frame_number, candidate, ball.confidence_percent, encoded.getvalue()))
        with self.lock:
            self.candidate_crops = tuple(crops)
            self.candidate_crop_frame = frame_number

    def store_opencv_crops(self, frame_number: int, vit_result, predictions) -> None:
        """Expose the exact post-OpenCV tiles used by OCV1 and OCV2 for QA."""
        crops: list[CandidateCrop] = []
        for prediction in predictions:
            candidate = next(
                (
                    index for index, ball in enumerate(vit_result.balls, start=1)
                    if (ball.left, ball.top, ball.right, ball.bottom) == prediction.bbox
                ),
                None,
            )
            if candidate is None or prediction.processed_crop.size == 0:
                continue
            encoded = io.BytesIO()
            Image.fromarray(prediction.processed_crop).save(encoded, format="JPEG", quality=90)
            crops.append(CandidateCrop(
                frame_number,
                candidate,
                vit_result.balls[candidate - 1].confidence_percent,
                encoded.getvalue(),
            ))
        with self.lock:
            self.opencv_crops = tuple(crops)
            self.opencv_crop_frame = frame_number

    def submit_openai_with_vit(self, jpeg: bytes, vit_result) -> None:
        """Pass same-frame ViT proposals as context, never as ground truth."""
        proposals = "; ".join(
            f"candidate {index + 1}: center ({ball.x}, {ball.y}), {ball.confidence_percent}%"
            for index, ball in enumerate(vit_result.balls)
        )
        context = (
            "Local ViT proposal from this exact frame: "
            + (f"{len(vit_result.balls)} candidate ball(s), left-to-right: {proposals}. " if proposals else "no candidate balls. ")
            + "Use these as location/count hints, and inspect the image. Reject a false proposal if you deem necessary "
            "and return each real visible ball exactly once in left-to-right order. For every returned ball, "
            "choose only one of: green, purple, pink, yellow, orange, red, blue. A vertically aligned lower "
            "copy is a water reflection, not another ball. Distinguish neon orange, red, and pink from the "
            "broad body/rim, not a small glare core. Focus on color only."
        )
        self.hybrid_runner.submit(jpeg, context)

    def submit_meta_v2(self, jpeg: bytes, frame_number: int, vit_result) -> None:
        """Classify the exact ViT close-ups with GPT, retaining ViT as a hint."""
        self.store_candidate_crops(jpeg, frame_number, vit_result)
        if not vit_result.balls:
            self.meta_v2_runner.record_no_ball_detected()
            return

        # Candidate crops are already square/padded and derived from this exact
        # ViT result. A single vertical sheet preserves their left-to-right
        # source ordering while making one vision request rather than one per ball.
        with self.lock:
            crops = self.candidate_crops
        tiles = [Image.open(io.BytesIO(crop.image)).convert("RGB") for crop in crops]
        side = max(tile.width for tile in tiles)
        gap = 6
        sheet = Image.new("RGB", (side, side * len(tiles) + gap * (len(tiles) - 1)), "black")
        for index, tile in enumerate(tiles):
            sheet.paste(tile.resize((side, side)), (0, index * (side + gap)))
        encoded = io.BytesIO()
        sheet.save(encoded, format="JPEG", quality=90)

        proposals = "; ".join(
            f"tile {index + 1}: center ({ball.x}, {ball.y}), {ball.confidence_percent}%"
            for index, ball in enumerate(vit_result.balls)
        )
        context = (
            "ViT handoff for this exact contact sheet, in its top-to-bottom tile order: "
            f"{proposals}. These locations and scores are useful evidence, but validate each tile "
            "visually before returning a color."
        )
        self.meta_v2_runner.submit(encoded.getvalue(), context)

    @staticmethod
    def _ocv_predictions(jpeg: bytes, vit_result):
        """Turn ViT boxes into bottom-half-only color crops."""
        image = Image.open(io.BytesIO(jpeg)).convert("RGB")
        detections = [
            {"label": "ball", "bbox": (ball.left, ball.top, ball.right, ball.bottom)}
            for ball in vit_result.balls
        ]
        return run_pipeline(image, detections)

    def submit_ocv1(self, jpeg: bytes, frame_number: int, vit_result) -> None:
        """OCV1: Meta-v2-style ViT handoff after OpenCV crop enhancement."""
        self.store_candidate_crops(jpeg, frame_number, vit_result)
        predictions = self._ocv_predictions(jpeg, vit_result)
        self.store_opencv_crops(frame_number, vit_result, predictions)
        sheet = make_contact_sheet(predictions)
        if not sheet:
            self.ocv1_runner.record_no_ball_detected()
            return
        proposals = "; ".join(
            f"tile {index + 1}: center ({ball.x}, {ball.y}), {ball.confidence_percent}%"
            for index, ball in enumerate(vit_result.balls)
        )
        self.ocv1_runner.submit(
            sheet,
            "ViT handoff for this exact OpenCV contact sheet, in its top-to-bottom tile order: "
            f"{proposals}. Use each location and confidence as helpful evidence, but visually "
            "validate the processed tile before returning its base color.",
        )

    def submit_ocv2(self, jpeg: bytes, frame_number: int, vit_result) -> None:
        """OCV2: the same preprocessing followed by local KMeans color mapping."""
        self.store_candidate_crops(jpeg, frame_number, vit_result)
        predictions = self._ocv_predictions(jpeg, vit_result)
        self.store_opencv_crops(frame_number, vit_result, predictions)
        self.ocv2_runner.complete(predictions)


STATE = CameraState()


# The original dashboard animation remains in the page template for layout
# compatibility. This small direct renderer owns the visible terminal content,
# so sweep text and completed results appear immediately instead of scrambling
# character-by-character.
DIRECT_TERMINAL_SCRIPT = r"""
(() => {
  const legacyTerminal = document.querySelector('#ai-result');
  // Keep the legacy animated element hidden and render into a separate terminal
  // that the old script cannot overwrite.
  const terminal = document.createElement('strong');
  terminal.id = 'direct-ai-result';
  legacyTerminal.insertAdjacentElement('afterend', terminal);
  legacyTerminal.style.display = 'none';
  const candidatePanel = document.querySelector('#candidate-panel');
  const candidateLabel = document.querySelector('#candidate-label');
  const candidateList = document.querySelector('#candidate-list');
  let cropSignature = '';
  const renderCandidates = (data) => {
    const enabled = ['vit', 'meta_v2', 'ocv1', 'ocv2'].includes(data.vision_mode);
    candidatePanel.hidden = !enabled;
    if (!enabled) return;
    const isOpenCv = ['ocv1', 'ocv2'].includes(data.vision_mode);
    const crops = isOpenCv ? data.opencv_crops : data.vit_crops;
    const cropFrame = isOpenCv ? data.opencv_crop_frame : data.vit_crop_frame;
    candidateLabel.textContent = isOpenCv ? 'OpenCV Output' : 'Candidate Close-Up';
    const signature = JSON.stringify([data.vision_mode, cropFrame, crops]);
    if (signature === cropSignature) return;
    cropSignature = signature;
    candidateList.replaceChildren();
    if (!crops.length) {
      const empty = document.createElement('div');
      empty.className = 'candidate-empty';
      empty.textContent = isOpenCv ? 'WAITING FOR OPENCV OUTPUT...' : 'WAITING FOR A ViT RESULT...';
      candidateList.append(empty);
      return;
    }
    crops.forEach((crop) => {
      const item = document.createElement('div');
      item.className = 'candidate-item';
      item.textContent = `CANDIDATE ${crop.candidate} · ${crop.confidence_percent}%`;
      const image = document.createElement('img');
      image.alt = `${isOpenCv ? 'OpenCV output' : 'ViT candidate'} ${crop.candidate}`;
      image.src = `${isOpenCv ? '/opencv-crop.jpg' : '/candidate-crop.jpg'}?frame=${crop.frame}&candidate=${crop.candidate}`;
      item.append(image);
      candidateList.append(item);
    });
  };
  const render = async () => {
    try {
      const response = await fetch('/api/status');
      const data = await response.json();
      renderCandidates(data);
      const vision = data.ai;
      if (!vision.has_started) terminal.textContent = '';
      else if (vision.is_analyzing) terminal.textContent = 'DEEPWATER OBJECTIVE\nSWEEP...';
      else if (!vision.balls.length) terminal.textContent = vision.status;
      else terminal.textContent = vision.balls.map((ball, index) => ball.color
        ? `Ball ${index + 1}: ${ball.color}\n${ball.confidence_percent}% confidence`
        : `Ball ${index + 1}: ${ball.confidence_percent}% confidence · (${ball.x}, ${ball.y})`
      ).join('\n\n');
    } catch (_) { /* The primary dashboard poll handles connection errors. */ }
  };
  render();
  setInterval(render, 250);
})();
"""


PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Levithian Command Center</title>
  <style>
    :root { --ink:#071311; --panel:#0d211d; --line:#1c5c4b; --mint:#53f4ad; --soft:#9ccbb9; --module-height:430px; }
    * { box-sizing:border-box; } body { margin:0; min-height:100vh; color:#e9fff6; overflow-x:hidden;
      font-family:"Apple Chancery", "Snell Roundhand", "URW Chancery L", cursive; background:#06100e; }
    body::before { content:""; position:fixed; inset:-16px; z-index:0; background:url('/assets/levithian-deep-sea.png') center/cover no-repeat;
      filter:blur(2px) saturate(.95) brightness(.82); transform:scale(1.025); }
    body::after { content:""; position:fixed; inset:0; z-index:0; pointer-events:none;
      background:linear-gradient(120deg,rgba(2,14,13,.30),rgba(3,14,12,.54) 54%,rgba(4,22,18,.42)),repeating-linear-gradient(102deg,transparent 0,transparent 66px,rgba(91,255,183,.05) 67px,transparent 69px); }
    .shell { position:relative; z-index:1; max-width:1200px; margin:auto; padding:30px; }
    header { display:flex; align-items:center; gap:17px; border-bottom:1px solid var(--line); padding-bottom:22px; }
    .mark { width:58px; height:58px; border:2px solid var(--mint); border-radius:50%; overflow:hidden; display:grid; place-items:center; background:#06100e; box-shadow:0 0 24px #31ef9a66; }
    .mark img { width:100%; height:100%; object-fit:cover; transform:scale(1.12); }
    h1 { margin:0; color:#c8ffe6; font-size:28px; font-weight:600; letter-spacing:.04em; text-shadow:0 0 16px #3af0a066; } .sub { color:var(--soft); margin-top:4px; font-size:13px; }
    .grid { display:flex; justify-content:flex-end; align-items:stretch; gap:14px; margin-top:96px; }
    /* Candidate Close-Up becomes the left column of the camera module when it
       is visible, rather than looking like an unrelated floating card. */
    .grid:has(.candidate-panel:not([hidden])) { gap:0; }
    .grid:has(.candidate-panel:not([hidden])) .candidate-panel { border-radius:18px 0 0 18px; border-right:0; }
    .grid:has(.candidate-panel:not([hidden])) .camera-module { border-radius:0 18px 18px 0; }
    .panel { background:linear-gradient(145deg,rgba(10,34,29,.82),rgba(5,20,17,.74)); border:1px solid var(--line); border-radius:18px; box-shadow:0 20px 55px #0008; overflow:hidden; backdrop-filter:blur(8px); }
    .label { display:flex; align-items:center; gap:10px; color:var(--mint); font-family:Menlo, Monaco, Consolas, "Courier New", monospace; font-size:14px; font-weight:600; letter-spacing:.13em; text-transform:uppercase; padding:15px 17px 14px; border-bottom:1px solid var(--line); text-shadow:0 0 12px #35e89a66; }
    .vision > .label { font-size:14px; letter-spacing:.10em; white-space:nowrap; }
    .feed { display:grid; flex:1; min-height:0; place-items:center; padding:16px; background:repeating-linear-gradient(0deg,#06100e 0,#06100e 2px,#071511 3px); }
    .feed img { width:100%; max-width:100%; height:auto; max-height:100%; object-fit:contain; image-rendering:pixelated; border:1px solid #2b8069; box-shadow:0 0 30px #35e89a77; }
    .camera-module { display:grid; grid-template-columns:370px 320px; height:var(--module-height); overflow:hidden; border:1px solid var(--line); border-radius:18px; background:linear-gradient(145deg,rgba(10,34,29,.82),rgba(5,20,17,.74)); box-shadow:0 20px 55px #0008; backdrop-filter:blur(8px); }
    .camera-module > section, .camera-module > aside { display:flex; min-width:0; flex-direction:column; }
    .camera-module .vision { border-left:1px solid var(--line); }
    .camera-module .feed { border-bottom-left-radius:17px; }
    .side { flex:1; padding:18px; }
    .dot { width:10px; height:10px; flex:0 0 10px; border-radius:50%; background:var(--mint); box-shadow:0 0 14px var(--mint); } .dot.offline { background:#ff5964; box-shadow:0 0 14px #ff5964; }
    .metric { margin-top:0; padding:17px 0; min-height:96px; overflow:hidden; } .metric strong { display:block; max-width:100%; margin-top:0; min-height:58px; color:#d4ffe7; font-family:Menlo, Monaco, Consolas, "Courier New", monospace; font-size:16px; font-weight:500; line-height:1.55; letter-spacing:.02em; overflow-wrap:break-word; word-break:normal; text-shadow:0 0 10px #46f6aa55; white-space:pre-line; }
    .mode-picker { display:flex; flex-wrap:wrap; gap:7px; padding:13px 16px 0; } .mode-picker button { border:1px solid #28735f; background:#0a201b; color:#9ccbb9; border-radius:5px; cursor:pointer; font:600 11px Menlo, Monaco, Consolas, monospace; letter-spacing:.08em; padding:7px 9px; } .mode-picker button.active { background:#144b3c; color:#53f4ad; box-shadow:0 0 12px #35e89a55; }
    .candidate-panel { display:flex; width:220px; height:var(--module-height); flex-direction:column; overflow:hidden; } .candidate-panel[hidden] { display:none; }
    .candidate-panel .label { white-space:nowrap; } .candidate-list { flex:1; min-height:0; overflow-y:auto; padding:14px; } .candidate-empty { color:var(--soft); font:500 12px Menlo, Monaco, Consolas, monospace; line-height:1.5; }
    .candidate-item { margin:0 0 16px; color:#c8ffe6; font:600 12px Menlo, Monaco, Consolas, monospace; } .candidate-item:last-child { margin-bottom:0; } .candidate-item img { display:block; width:100%; margin-top:8px; border:1px solid #2b8069; image-rendering:pixelated; box-shadow:0 0 16px #35e89a55; }
    @media(max-width:900px) { .grid{justify-content:stretch}.candidate-panel{width:190px}.camera-module{grid-template-columns:minmax(0,1fr) 290px} }
    @media(max-width:760px) { .grid{display:block}.candidate-panel{width:100%;height:auto;min-height:160px;margin-bottom:14px}.camera-module{height:auto;grid-template-columns:1fr}.camera-module > section,.camera-module > aside{min-height:0}.feed{min-height:250px}.camera-module .vision{border-left:0;border-top:1px solid var(--line)}.shell{padding:18px} }
  </style>
</head>
<body><main class="shell">
  <header><div class="mark"><img src="/assets/levithian-emblem-source.png" alt="Levithian sea monster emblem" /></div><div><h1>Levithian Command Center</h1></div></header>
  <section class="grid"><aside class="panel candidate-panel" id="candidate-panel" hidden><div class="label" id="candidate-label">Candidate Close-Up</div><div class="candidate-list" id="candidate-list"><div class="candidate-empty">WAITING FOR A ViT RESULT...</div></div></aside><section class="camera-module"><section><div class="label"><i class="dot offline" id="feed-dot"></i>Live camera feed</div><div class="feed"><img id="camera-frame" src="/frame.jpg" alt="Live Arduino camera feed" /></div></section>
  <aside class="vision"><div class="label" id="vision-label">Detection &amp; Classifier</div><div class="mode-picker"><button data-mode="vit" class="active">ViT</button><button data-mode="openai">GPT 5.4</button><button data-mode="vit_openai">Meta</button><button data-mode="meta_v2">Meta v2</button><button data-mode="ocv1">OCV1</button><button data-mode="ocv2">OCV2 (FULLY LOCAL)</button></div><div class="side">
    <div class="metric"><strong id="ai-result"></strong></div>
  </div></aside></section></section>
</main><script>const terminal=document.querySelector('#ai-result');const feedDot=document.querySelector('#feed-dot');const cameraFrame=document.querySelector('#camera-frame');const visionLabel=document.querySelector('#vision-label');const buttons=[...document.querySelectorAll('[data-mode]')];const labels={vit:'ViT Zero-Shot Detection',yolo:'Sports Ball Segmentation',openai:'ViT-Guided Color Classification'};const glyphs='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@#$%&*';const sweep='DEEPWATER OBJECTIVE\\nSWEEP...';let mode='';let displayed='';let animation;let booting=true;let shownFrame=-1;function clearTerminal(){if(mode==='empty')return;mode='empty';displayed='';clearInterval(animation);terminal.textContent=''}function showSweep(){if(mode==='sweep-idle'||mode==='sweeping')return;mode='sweeping';displayed='';clearInterval(animation);let tick=0;animation=setInterval(()=>{const resolved=Math.min(sweep.length,Math.floor(tick/2)+1);terminal.textContent=[...sweep].map((char,index)=>char==='\\n'?'\\n':index<resolved?char:glyphs[(index*11+tick*7)%glyphs.length]).join('');if(resolved===sweep.length){clearInterval(animation);mode='sweep-idle'}tick++},18)}function resolveTerminal(text){if(mode==='resolved'&&text===displayed)return;mode='resolved';displayed=text;clearInterval(animation);let tick=0;animation=setInterval(()=>{const resolved=Math.min(text.length,Math.floor(tick/2)+1);terminal.textContent=[...text].map((char,index)=>char==='\\n'?'\\n':index<resolved?char:glyphs[(index*11+tick*7)%glyphs.length]).join('');if(resolved===text.length)clearInterval(animation);tick++},18)}function setMode(mode){visionLabel.textContent=labels[mode];buttons.forEach(button=>button.classList.toggle('active',button.dataset.mode===mode))}buttons.forEach(button=>button.onclick=async()=>{await fetch('/api/mode',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode:button.dataset.mode})});mode='';displayed='';clearTerminal();await status()});async function status(){if(booting){feedDot.classList.add('offline');clearTerminal();return}try{const r=await fetch('/api/status');const d=await r.json();setMode(d.vision_mode);if(d.frames!==shownFrame&&d.frames>0){shownFrame=d.frames;cameraFrame.src='/frame.jpg?frame='+d.frames}feedDot.classList.toggle('offline',!d.ai.has_started);if(!d.ai.has_started){clearTerminal();return}const completed=d.ai.result_age_seconds!==null;if(completed&&!d.ai.is_analyzing){resolveTerminal(d.ai.balls.length?d.ai.balls.map((ball,index)=>ball.color?`Ball ${index+1}: ${ball.color}\\n${ball.confidence_percent}% confidence`:`Ball ${index+1}: ball\\n${ball.confidence_percent}% confidence, (${ball.x}, ${ball.y})`).join('\\n\\n'):d.ai.status)}else if(d.ai.is_analyzing){showSweep()}else{clearTerminal()}}catch(e){feedDot.classList.add('offline');clearTerminal()}}status();setTimeout(()=>{booting=false;status()},700);setInterval(status,1000);</script></body></html>"""

# Keep the compact in-page client readable at runtime and expose only the three
# supported modes: standalone ViT, standalone OpenAI, and their hybrid.
PAGE = PAGE.replace(
    "function setMode(mode){visionLabel.textContent=labels[mode];buttons.forEach(button=>button.classList.toggle('active',button.dataset.mode===mode))}",
    "function setMode(mode){buttons.forEach(button=>button.classList.toggle('active',button.dataset.mode===mode))}",
)


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.end_headers()
            page = PAGE.replace("</body>", '<script src="/direct-terminal.js"></script></body>')
            self.wfile.write(page.encode())
        elif self.path == "/direct-terminal.js":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/javascript; charset=utf-8")
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.end_headers()
            self.wfile.write(DIRECT_TERMINAL_SCRIPT.encode())
        elif self.path == "/api/status":
            with STATE.lock:
                ai = STATE.ball_vision.snapshot()
                result_age_seconds = None if not ai.completed_at else max(0.0, time.time() - ai.completed_at)
                payload = json.dumps({
                    "status": STATE.status,
                    "frames": STATE.frame_number,
                    "frame_rate_fps": STATE.frame_rate_fps,
                    "color_profile": STATE.color_profile,
                    "vision_mode": STATE.vision_mode,
                    "vit_crop_frame": STATE.candidate_crop_frame,
                    "vit_crops": [
                        {
                            "frame": crop.frame,
                            "candidate": crop.candidate,
                            "confidence_percent": crop.confidence_percent,
                        }
                        for crop in STATE.candidate_crops
                    ],
                    "opencv_crop_frame": STATE.opencv_crop_frame,
                    "opencv_crops": [
                        {
                            "frame": crop.frame,
                            "candidate": crop.candidate,
                            "confidence_percent": crop.confidence_percent,
                        }
                        for crop in STATE.opencv_crops
                    ],
                    "ai": {
                        "status": ai.status,
                        "balls": [
                            ({"color": ball[0], "confidence_percent": ball[1]} if isinstance(ball, tuple) else {
                                "confidence_percent": ball.confidence_percent, "x": ball.x, "y": ball.y,
                                "mask_points": ball.mask_points,
                            }) for ball in ai.balls
                        ],
                        "result_age_seconds": result_age_seconds,
                        "latency_seconds": ai.latency_seconds,
                        "is_analyzing": ai.is_analyzing,
                        "has_started": ai.has_started,
                        "seconds_until_next_start": STATE.ball_vision.seconds_until_next_start(),
                    },
                }).encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.end_headers()
            self.wfile.write(payload)
        elif self.path.split("?", 1)[0] == "/candidate-crop.jpg":
            query = parse_qs(urlparse(self.path).query)
            try:
                frame = int(query.get("frame", [""])[0])
                candidate = int(query.get("candidate", [""])[0])
            except ValueError:
                self.send_error(HTTPStatus.BAD_REQUEST, "Expected frame and candidate")
                return
            with STATE.lock:
                image = next(
                    (crop.image for crop in STATE.candidate_crops
                     if crop.frame == frame and crop.candidate == candidate),
                    None,
                )
            if image is None:
                self.send_error(HTTPStatus.NOT_FOUND, "Candidate crop unavailable")
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(image)))
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.end_headers()
            self.wfile.write(image)
        elif self.path.split("?", 1)[0] == "/opencv-crop.jpg":
            query = parse_qs(urlparse(self.path).query)
            try:
                frame = int(query.get("frame", [""])[0])
                candidate = int(query.get("candidate", [""])[0])
            except ValueError:
                self.send_error(HTTPStatus.BAD_REQUEST, "Expected frame and candidate")
                return
            with STATE.lock:
                image = next(
                    (crop.image for crop in STATE.opencv_crops
                     if crop.frame == frame and crop.candidate == candidate),
                    None,
                )
            if image is None:
                self.send_error(HTTPStatus.NOT_FOUND, "OpenCV crop unavailable")
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(image)))
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.end_headers()
            self.wfile.write(image)
        elif self.path == "/api/mode":
            self.send_error(HTTPStatus.METHOD_NOT_ALLOWED)
        elif self.path.split("?", 1)[0] == "/frame.jpg":
            with STATE.lock:
                image = STATE.jpeg
            if image is None:
                self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "Waiting for camera frame")
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(image)))
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.end_headers()
            self.wfile.write(image)
        elif self.path == "/assets/levithian-deep-sea.png":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/png")
            self.send_header("Cache-Control", "public, max-age=3600")
            self.end_headers()
            self.wfile.write(BACKGROUND_IMAGE.read_bytes())
        elif self.path == "/assets/levithian-monster-mark.png":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/png")
            self.send_header("Cache-Control", "public, max-age=3600")
            self.end_headers()
            self.wfile.write((BACKGROUND_IMAGE.parent / "levithian-monster-mark.png").read_bytes())
        elif self.path == "/assets/levithian-emblem-source.png":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/png")
            self.send_header("Cache-Control", "public, max-age=3600")
            self.end_headers()
            self.wfile.write((BACKGROUND_IMAGE.parent / "levithian-emblem-source.png").read_bytes())
        elif self.path == "/stream.mjpg":
            self.stream()
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def stream(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        last_frame = -1
        try:
            while True:
                with STATE.condition:
                    STATE.condition.wait_for(lambda: STATE.jpeg is not None and STATE.frame_number != last_frame, timeout=2)
                    image, frame = STATE.jpeg, STATE.frame_number
                if image is None or frame == last_frame:
                    continue
                last_frame = frame
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(image)}\r\n\r\n".encode())
                self.wfile.write(image + b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/mode":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            mode = json.loads(self.rfile.read(length)).get("mode")
        except (ValueError, json.JSONDecodeError):
            self.send_error(HTTPStatus.BAD_REQUEST, "Expected JSON mode")
            return
        if not isinstance(mode, str) or not STATE.set_vision_mode(mode):
            self.send_error(HTTPStatus.BAD_REQUEST, "Unknown vision mode")
            return
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()


if __name__ == "__main__":
    STATE.start()
    print(f"Levithian Camera View: http://{HOST}:{PORT}")
    ThreadingHTTPServer((HOST, PORT), DashboardHandler).serve_forever()
