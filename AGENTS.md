# CRAB: OV7670 → Arduino Uno → OpenCV

## Goal

This phase is a camera and computer-vision proof of concept. Capture frames from
an OV7670 camera, stream them to the Mac over the Arduino USB serial connection,
then use Python/OpenCV to identify non-blue colored balls and report their
approximate `(x, y)` position and contour area. Motor control, navigation, and
autonomy are explicitly out of scope.

```text
OV7670 (no FIFO) → Arduino Uno-compatible board → USB serial → Mac
                                                       → Python/OpenCV
```

## Confirmed Hardware and Connection State

- PlatformIO target: classic Arduino Uno (`atmelavr`, `board = uno`).
- The Mac currently detects the board at `/dev/cu.usbserial-130`. The numeric
  suffix can change after a USB reconnect; always use the current
  `/dev/cu.usbserial-*` entry from `pio device list`.
- USB serial adapter: VID:PID `1A86:7523` (CH340).
- Firmware upload and verification have succeeded over the USB serial port.
- During upload, the board reported signature `0x1e950f` (`m328p`), which is
  the processor used by a classic Uno-compatible board. This is compatible with
  the Uno-focused tutorial used here.
- A breadboard and its physical wiring cannot be detected by the computer.
- The OV7670 capture path has been verified with the camera's internal color-bar
  pattern: stable bars reached the Mac viewer. This proves the USB serial link
  and the camera's `VSYNC`, `PCLK`, and D0–D7 capture path are functioning.
- On 2026-07-23, a fresh internal color-bar run in `UART_MODE 3` was stable at
  about 0.9 FPS. Mode 5 (320×240 RGB) produced missing horizontal bands on this
  no-FIFO Uno setup, so mode 3 is the current practical color configuration.
- `CameraOV7670::init()` now waits 500 ms before its first SCCB transaction.
  This avoids a reset-timing failure caused by the dashboard's intentional DTR
  reconnect reset; the former 10 ms delay could yield a false SCCB failure.
- A loose physical camera connection caused the earlier static/tearing. A stable
  live color image was obtained after reseating that wire; preserve the working
  breadboard setup and do not disturb it before CV work.
- The dashboard serial baud must match `UART_MODE`. On 2026-07-22, mode 1
  firmware at 115200 baud was mistakenly paired with a 500000-baud dashboard;
  correcting that produced one valid live frame. The temporary timing
  diagnostic then confirmed SCCB configuration, PCLK activity, and pixel-data
  changes but only one VSYNC event before VSYNC stayed low. The stream firmware
  waits for every VSYNC, so it necessarily freezes after that one frame.
  Restore `CAMERA_TIMING_DIAGNOSTIC` to `0` after using the diagnostic.
- Never enable `OV7670_INIT_TRACE` in `lib/CameraOV7670/CameraOV7670.cpp` for
  normal use: text emitted on `Serial` corrupts the binary camera protocol and
  appears as static. The 2026-07-22 RGB and grayscale internal-color-bar tests
  produced no completed frames even after that text was removed. A subsequent
  timing read reported SCCB setup failure and zero VSYNC/PCLK/data transitions;
  that condition is below the viewer and capture-stream software layers.

## Camera Hardware

- Camera: HiLetgo OV7670, no FIFO.
- Camera power and all camera-control pull-ups use **3.3 V only**. Never put
  camera signals or pull-ups on the Uno's 5 V rail.
- `HREF` is intentionally left unconnected for this tutorial.

### Tutorial Wiring Map

Use the labels printed on the camera, not a remembered physical pin position.

| OV7670 | Arduino Uno connection |
|---|---|
| `3.3V` / `VCC` | 3.3 V rail |
| `GND` | GND rail |
| `RESET` | 3.3 V rail |
| `PWDN` | GND rail |
| `SIOD` / `SDA` | A4, with 4.7 kΩ pull-up to 3.3 V |
| `SIOC` / `SCL` | A5, with 4.7 kΩ pull-up to 3.3 V |
| `VSYNC` | D2 |
| `PCLK` | D12 |
| `XCLK` / `MCLK` | D3 through the divider below |
| camera `D0`–`D3` | A0–A3 respectively |
| camera `D4`–`D7` | D4–D7 respectively |
| `HREF` / `HS` | no connection |

XCLK divider:

```text
Uno D3 → 3.3 kΩ → camera XCLK
                   |
                 4.7 kΩ
                   |
                  GND
```

For breadboard tracing: a terminal-strip group such as `F17` through `J17` is
one electrical node. Adjacent numbered rows (for example `F17` and `F16`) are
not connected. Power rails may be split in the middle; bridge matching `+` to
`+` and `-` to `-` only when necessary.

## Current Implementation

- `src/main.cpp` wraps the provided `LiteOV7670` sketch:
  `initializeScreenAndCamera()` in `setup()` and `processFrame()` in `loop()`.
- The supplied tutorial dependencies are now local to this project:
  - `lib/LiteOV7670/`
  - `lib/CameraOV7670/`
- The supplied Mac viewer is local at `tools/ArduImageCapture/`.
- `tools/do_we_know_camera.py` is a minimal direct serial viewer named
  **Camera View**. It reads the same stream directly and runs with
  `python3 tools/do_we_know_camera.py` after installing
  `tools/requirements-camera-ui.txt`. Its serial reader explicitly toggles DTR
  on connection to reset the Uno; this prevents a reconnect from inheriting a
  capture loop stalled before the next camera frame header.
- The UART firmware uses RGB565 byte parity so the supplied desktop viewer can
  detect serial corruption. `rgb565_to_rgb()` intentionally mirrors the
  supplied ArduImageCapture conversion; do not strip or reinterpret its parity
  marker bits while debugging color, because that changes the viewer's colors.
- The green-themed local web dashboard is `tools/levithian_dashboard.py`.
  It is the **Levithian Command Center**, reads the same serial stream directly,
  and serves at `http://127.0.0.1:8765`. It contains the generated deep-sea
  background and sea-monster emblem in `tools/assets/`, plus a right-aligned
  three-panel module: **Candidate Close-Up**, **Live Camera Feed**, and
  **Detection & Classifier**. The three panels share a fixed-height row;
  Candidate Close-Up appears automatically in ViT, Meta v2, OCV1, and OCV2
  modes, but not GPT 5.4 or normal Meta. In OCV1/OCV2 it becomes **OpenCV
  Output** and displays the exact processed tiles used for the next inference;
  this is the development QA surface for crop preprocessing.
  Close all other viewers before running it because only one process can own the
  serial port. Launch with `python3 tools/levithian_dashboard.py`, or use the
  globally installed `levi` launcher from any terminal directory. The dashboard
  waits for a `/dev/cu.usbserial-*` port if macOS recreates it after startup,
  rather than requiring a manual restart.
- The Command Center is currently intentionally displaying raw RGB values from
  the tutorial-compatible stream: no white balance, gamma, contrast, or other
  color profile is applied. Local ViT and all GPT methods run on these raw
  frames; GPT prompt reasoning, rather than a color-correction transform, is
  the current color-classification strategy. `tools/color_correction.py` remains
  available for later, measured calibration experiments but is not in the
  active display path.
- The Command Center has six selectable modes. **ViT** runs local OWLv2 using
  “sports ball”, “tennis ball”, and “floating ball” queries, yielding candidate
  boxes, confidence, and approximate `(x, y)`. **GPT 5.4** sends the full frame
  directly to Azure GPT. **Meta** is the recommended path: GPT receives the
  same full frame plus ViT’s same-frame count/coordinate handoff as primary
  spatial evidence, but may correct a clear detector error. **Meta v2** sends
  GPT a vertical sheet of same-frame, padded ViT candidate crops plus each
  candidate's coordinate and detector confidence. It is a crop-based color
  experiment, not the normal Meta path. **OCV1** applies bottom-half masking,
  dark-background suppression, denoising, and LAB-lightness normalization to
  the same ViT crop sheet, then sends it with the same coordinate/confidence
  handoff and GPT guardrails. **OCV2** sends the same enhanced crops to a fully
  local seven-class KMeans dominant-color plus LAB-prototype classifier; it
  makes no cloud call. Treat both OCV modes as pool-test experiments. OWLv2
  uses a 50% threshold, 30% overlap suppression, and an
  18px minimum box; tune this only with documented pool evidence.
- Candidate Close-Up displays the exact padded crop from each ViT proposal in
  the same frame. Crops are square, 1.5× the larger source-box side (minimum
  40px), and clamped to the image edge. It is an inspection panel shown in
  ViT, Meta v2, OCV1, and OCV2 modes; GPT 5.4 and normal Meta still receive
  the full image and do not show this panel. OCV modes display the post-OpenCV
  tile through `/opencv-crop.jpg`, while other crop modes display raw ViT crops.
- `tools/openai_ball_vision.py` is the Azure GPT-5.4 integration and must be
  preserved as Plan B. It holds at most one cloud request in flight. `PROMPT`
  is the shared full-frame instruction set; `META_V2_PROMPT` adds the
  close-up contact-sheet rules for Meta v2; and `CROP_PROMPT` is the OCV1
  preprocessed-crop instruction set. All require seven permitted colors,
  upper physical-ball-only figure-8 handling, broad-body/rim color reasoning,
  glare handling, left-to-right output, and JSON-only results. Meta’s dynamic
  ViT handoff is in `CameraState.submit_openai_with_vit()` and Meta v2’s
  contact-sheet handoff is in `CameraState.submit_meta_v2()`.
  The runner reads only `AZURE_OPENAI_API_KEY`,
  `CLINICAL_DATA_AZURE_OPENAI_ENDPOINT`, and
  `CLINICAL_DATA_AZURE_OPENAI_API_VERSION` from
  `/Users/hjoshi/Documents/PlatformIO/.env.local`. Never place an API key in
  the repository, `AGENTS.md`, or committed environment files. The gateway
  accepts `gpt-5.4` directly and the runner uses macOS trust with
  `httpx.Client(trust_env=False)` because copied CA variables are stale.
- UI timing is frame-synchronized in every mode: show `DEEPWATER OBJECTIVE
  SWEEP...` while the fresh frame is being inferred, then keep that result until
  the next fresh frame begins inference. Camera preview remains uncapped, but
  all perception modes are single-flight and start at most one inference every
  **5 seconds**, always using the newest available frame. A measured live average was ~0.133 FPS
  (about one frame per 7.5 seconds); ViT was ~0.63 s, GPT ~3 s, and normal Meta
  ~3.3 s after frame arrival. Treat these as measurements, not guarantees.
- `tools/vision.py` is the first laptop-side OpenCV layer. It converts the
  corrected RGB frame to HSV, deliberately excludes blue, and finds saturated,
  round color candidates with contour area and circularity filters. It publishes
  a candidate color and `(x, y)` to the Command Center. The default hue ranges
  are only a starting profile; calibrate them against the actual balls and tank
  lighting before treating a detection as authoritative.
- The current test firmware uses `UART_MODE 3`: 160×120 RGB565 at 500000 baud.
  `tools/do_we_know_camera.py` and the dashboard use the matching 500000-baud
  speed. Change all serial-reader settings together before selecting a different
  `UART_MODE`; mode 4 is the 160×120 grayscale fallback. Optional in-stream debug
  text packets are
  disabled because they caused the older viewer to lose packet alignment.
- Homebrew OpenJDK is installed at `/opt/homebrew/opt/openjdk/bin/java`.
- The viewer initially bundled Intel-only `jSerialComm-2.6.2.jar`. On Apple
  silicon, the active `jSerialComm-2.6.2.jar` has been replaced with the
  ARM-compatible 2.11.4 release; the original is retained as
  `jSerialComm-2.6.2-x86_64.jar`.
- Launch the viewer with:

  ```sh
  /opt/homebrew/opt/openjdk/bin/java --enable-native-access=ALL-UNNAMED \
    -jar tool/ArduImageCapture.jar
  ```

  Run that command from `tools/ArduImageCapture/`.

## Commands

PlatformIO's `pio` command is not on the shell PATH. Use:

```sh
/Users/hjoshi/.platformio/penv/bin/pio device list
/Users/hjoshi/.platformio/penv/bin/pio run
/Users/hjoshi/.platformio/penv/bin/pio run --target upload --upload-port /dev/cu.usbserial-110
```

The current frame stream is binary data at **500000 baud**. Do not expect a useful text
output in the ordinary serial monitor once the stream sketch is uploaded.

## Resume Plan

Before changing any prompt, ViT threshold, or crop setting after a pool session,
record baseline evidence and the proposed adjustment in `GUARDRAIL.md`. The
source-of-truth prompts remain in `tools/openai_ball_vision.py`; the guardrail
file records evidence, current methods, reference images, and the test table.

`opencv.md` records the proposed hybrid local OpenCV plus GPT-5.4 design. Treat
its feasibility estimates as unverified until pool images are captured and
logged.

1. Leave the now-working camera/breadboard wiring unchanged. Confirm the current
   `/dev/cu.usbserial-*` port with `pio device list` after any reconnect.
2. Start `python3 tools/levithian_dashboard.py`, open `http://127.0.0.1:8765`,
   and verify the live color feed before beginning CV. Close it before uploads.
3. For the boat-mounted, higher-angle baseline, capture screenshots with no
   balls; one ball near/mid/far; multiple balls; balls near a wall/boat/target;
   and stationary versus moving camera.
4. Record actual count/color plus ViT, Meta, and optional Meta v2 outcomes in
   `GUARDRAIL.md`. Inspect whether Candidate Close-Up includes the upper ball
   cleanly or mixes reflection/neighboring balls.
5. Confirm blue tank/wall rejection and other-object false positives before
   changing prompts, ViT threshold, or crop padding.

## Safety and Scope

- Never rewire camera/breadboard connections while USB power is connected.
- Do not connect `A4`/`A5`, `PCLK`, or `XCLK` to power rails.
- Close Camera View, ArduImageCapture, and the Command Center before uploading;
  the serial port can only be used by one process at a time.
- Keep generated `.pio/` artifacts out of version control (already covered by
  `.gitignore`).
