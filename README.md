# CRAB CV Module - OV7670 Ball-Color Vision

<div align="center">

**Streams an Arduino OV7670 camera to a laptop dashboard that spots non-blue floating balls and names their color**

[![PlatformIO](https://img.shields.io/badge/PlatformIO-Arduino%20Uno-orange.svg)](https://platformio.org/)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Local%20CV-green.svg)](https://opencv.org/)
[![OWLv2](https://img.shields.io/badge/OWLv2-Zero%20Shot%20ViT-purple.svg)](https://huggingface.co/docs/transformers/model_doc/owlv2)

</div>

---

## The Problem

The CRAB boat project needs to find plain solid-color balls floating in a competition water tank and report their color and approximate position. The capture hardware makes that hard:

- A HiLetgo OV7670 has no FIFO, so an Arduino Uno must bit-bang every pixel.
- The only link to the laptop is USB serial, which caps the usable frame rate to roughly one frame every 7-8 seconds.
- OV7670 frames arrive dark, low-resolution, and heavily magenta/green shifted, and each real ball casts a near-identical water reflection directly below it.
- The tank, walls, and shadows are the same blue/green as one of the valid ball colors.

**CRAB CV Module** solves this by streaming the camera over USB serial to a local dashboard that runs six interchangeable perception modes over the same frame, from a fully offline OpenCV color classifier to Azure GPT-5.4 vision with a local detector handoff.

This phase is a camera and computer-vision proof of concept only. Motor control, navigation, and autonomy are explicitly out of scope.

Runs locally at `http://127.0.0.1:8765`.

---

## Key Features

- **Single-file Arduino firmware** wrapping the LiteOV7670 sketch: 160x120 RGB565 at 500000 baud, internal color bars available for link validation.
- **Levithian Command Center dashboard** that reads the raw serial stream directly and serves a green deep-sea themed web UI with live preview, detection panel, and candidate close-ups.
- **Six perception modes** selectable at runtime, spanning local-only and cloud paths with a shared result interface.
- **Local OWLv2 zero-shot detector** for candidate ball boxes and approximate `(x, y)` with no training data.
- **Azure GPT-5.4 color reasoning** with pool-tuned prompts for reflection handling, glare caps, and seven allowed colors, held to one request in flight.
- **Fully offline OCV2 mode** using KMeans dominant color plus LAB prototype matching, so a demo can run with no network.
- **Deliberate 5-second control loop**: the preview stays live while each perception decision uses only the newest frame, at most once every five seconds.
- **Reset-safe serial reconnect** that toggles DTR on connect and waits for `/dev/cu.usbserial-*` to reappear after a USB bounce.
- **Standalone viewers** retained as fallbacks: a Python Tk viewer (`do_we_know_camera.py`) and the bundled ArduImageCapture Java tool.

---

## Perception Modes

The dashboard's Detection and Classifier panel switches between six modes over one shared runner interface. All model work is single-flight and rate-limited to one start every five seconds.

| Mode | Image sent | Where it runs | Role |
|---|---|---|---|
| **ViT** | none | local (OWLv2) | Zero-shot candidate boxes, count, and approximate `(x, y)`. |
| **GPT 5.4** | full frame | Azure GPT-5.4 | Standalone full-scene ball and color judgment. |
| **Meta** | full frame + same-frame ViT count/coordinates | local + cloud | Recommended path: ViT spatial handoff plus GPT full-scene color reasoning, GPT may override a clear detector error. |
| **Meta v2** | vertical contact sheet of padded ViT crops | local + cloud | Experimental crop-based comparison, only as good as the ViT boxes. |
| **OCV1** | OpenCV-enhanced ViT crop sheet | local + cloud | Bottom-half masking, dark-pixel suppression, denoise, and LAB-lightness normalization before GPT color reasoning. |
| **OCV2** | OpenCV-enhanced ViT crop sheet | local only | Seven-class KMeans dominant color plus LAB prototype classifier, no cloud call. |

**Candidate Close-Up** is an inspection panel shown in ViT, Meta v2, OCV1, and OCV2. It displays the exact padded crop from each ViT proposal (square, 1.5x the larger box side, minimum 40px). In OCV modes it becomes **OpenCV Output** and shows the processed tiles fed to the next inference. GPT 5.4 and normal Meta receive the whole frame and hide this panel.

OWLv2 uses the queries `a sports ball`, `a tennis ball`, `a floating ball`, a 50% confidence threshold, 30% IoU suppression, an 18px minimum box side, and returns at most five balls. Allowed colors are green, purple, pink, yellow, orange, red, and blue; blue is valid only for a clearly spherical cobalt body, never tank water.

---

## Tech Stack

### Firmware

- PlatformIO with the Arduino framework, target `atmelavr` / `board = uno`
- LiteOV7670 UART sketch and the CameraOV7670 register library, vendored into `lib/`
- RGB565 stream with per-byte parity marking so the desktop viewer can detect serial corruption
- `src/main.cpp` wrapper plus an optional compile-time SCCB/I2C bus scanner diagnostic

### Serial and Dashboard

- Python 3.11+ standard library `http.server` (`ThreadingHTTPServer`), no web framework
- `pyserial` stream reader that decodes the LiteOV7670 command/pixel protocol
- Pillow for frame decode, rotation, and JPEG encoding
- `levi` zsh launcher for one-command start from any directory

### Local CV

- OpenCV (`opencv-python-headless`) HSV masking, morphology, and contour filtering
- OWLv2 (`google/owlv2-base-patch16-ensemble`) via `transformers` and `safetensors`, running on Apple MPS when available, else CPU
- NumPy and SciPy for the KMeans and LAB prototype color path
- `color_correction.py` white-patch auto white balance, kept for later measured calibration, not in the active display path

### Cloud Vision

- Azure OpenAI GPT-5.4 (`openai` SDK), one request in flight, `httpx.Client(trust_env=False)` to use macOS trust
- Prompt set kept in code (`PROMPT`, `META_V2_PROMPT`, `OPENCV_CROP_PROMPT`, `CROP_PROMPT`) so runtime and documentation cannot drift

---

## Hardware

- Camera: HiLetgo OV7670, no FIFO. Camera power and all control-line pull-ups use **3.3 V only**.
- Board: classic Arduino Uno-compatible (`m328p`), CH340 USB serial (`1A86:7523`), enumerating as `/dev/cu.usbserial-*`.
- `HREF` is intentionally left unconnected for this tutorial wiring.

### Wiring Map

| OV7670 | Arduino Uno |
|---|---|
| `3.3V` / `VCC`, `RESET` | 3.3 V rail |
| `GND`, `PWDN` | GND rail |
| `SIOD` / `SDA` | A4, 4.7 kΩ pull-up to 3.3 V |
| `SIOC` / `SCL` | A5, 4.7 kΩ pull-up to 3.3 V |
| `VSYNC` | D2 |
| `PCLK` | D12 |
| `XCLK` / `MCLK` | D3 through the divider below |
| `D0`-`D3` | A0-A3 |
| `D4`-`D7` | D4-D7 |
| `HREF` / `HS` | no connection |

XCLK divider:

```text
Uno D3 -> 3.3 kΩ -> camera XCLK
                     |
                   4.7 kΩ
                     |
                    GND
```

Never rewire the camera or breadboard while USB power is connected, and never put `A4`/`A5`, `PCLK`, or `XCLK` on a power rail.

---

## Setup

### Prerequisites

- macOS with a `/dev/cu.usbserial-*` device for the Uno
- Python 3.11+
- PlatformIO Core (`pio` is not assumed on `PATH`; the commands below use the penv path)
- Homebrew OpenJDK only if you want the bundled ArduImageCapture Java viewer
- Optional: an Azure OpenAI GPT-5.4 deployment for the GPT, Meta, Meta v2, and OCV1 modes

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/HarryJ12/CRAB_CV-Module.git
cd CRAB_CV-Module
```

2. **Install Python dependencies**
```bash
python3 -m pip install --user -r tools/requirements-camera-ui.txt
```

3. **Build and upload the firmware**
```bash
/Users/hjoshi/.platformio/penv/bin/pio device list
/Users/hjoshi/.platformio/penv/bin/pio run
/Users/hjoshi/.platformio/penv/bin/pio run --target upload --upload-port /dev/cu.usbserial-XXX
```
Close every serial viewer first; only one process can own the port. The active firmware is `UART_MODE 3` in `lib/LiteOV7670/OV7670_UART.h`.

4. **Launch the dashboard**
```bash
python3 tools/levithian_dashboard.py
```
Or use the `levi` launcher from any directory, which kills a previous instance, starts a detached process, and opens the browser.

5. **Open the UI**
```text
http://127.0.0.1:8765
```
Verify the live color feed before selecting a perception mode. Close the dashboard before any firmware upload.

### Standalone Viewers

```bash
# Python Tk viewer
python3 tools/do_we_know_camera.py

# Bundled Java viewer, run from tools/ArduImageCapture/
/opt/homebrew/opt/openjdk/bin/java --enable-native-access=ALL-UNNAMED -jar tool/ArduImageCapture.jar
```

---

## Environment Variables

The GPT-5.4, Meta, Meta v2, and OCV1 modes read three values from `~/Documents/PlatformIO/.env.local` (override the path with `AZURE_OPENAI_ENV_FILE`):

```bash
AZURE_OPENAI_API_KEY=
CLINICAL_DATA_AZURE_OPENAI_ENDPOINT=
CLINICAL_DATA_AZURE_OPENAI_API_VERSION=
```

The gateway accepts the model id `gpt-5.4` directly. The ViT and OCV2 modes are fully local and need none of these. Never commit an API key to the repository or to `AGENTS.md`.

---

## Camera Modes

`UART_MODE` in `lib/LiteOV7670/OV7670_UART.h` selects resolution, pixel format, and baud together. The serial readers must match.

| Mode | Resolution | Format | Baud | Use |
|---|---|---|---|---|
| 1 | 160x120 | RGB565 | 115200 | legacy RGB |
| **3** | 160x120 | RGB565 | 500000 | **active live dashboard setting** |
| 4 | 160x120 | grayscale | 500000 | grayscale fallback |
| 5 | 320x240 | RGB565 | 500000 | color-bar validation; drops bands on this no-FIFO Uno |

Set `OV7670_SENSOR_TEST_BARS` to `1` to emit the sensor's internal color-bar pattern for link validation. Never enable `OV7670_INIT_TRACE` in `lib/CameraOV7670/CameraOV7670.cpp`; text on `Serial` corrupts the binary protocol.

---

## Project Structure

```text
CRAB_CV-Module/
├── platformio.ini                     (Uno / atmelavr / arduino env)
├── src/
│   └── main.cpp                       (setup/loop wrapper + SCCB scanner diagnostic)
├── lib/
│   ├── LiteOV7670/                    (UART sketch, UART_MODE + frame protocol)
│   └── CameraOV7670/                  (OV7670 register profiles, vendored)
├── tools/
│   ├── levithian_dashboard.py         (web dashboard, mode routing, HTML/JS UI)
│   ├── do_we_know_camera.py           (serial protocol decoder + Tk viewer)
│   ├── owlv2_ball_vision.py           (local OWLv2 zero-shot detector)
│   ├── openai_ball_vision.py          (Azure GPT-5.4 runner + in-code prompts)
│   ├── opencv_experiments.py          (OCV1/OCV2 crop prep + local color classifier)
│   ├── demo_opencv_experiments.py     (run the crop experiment on a saved frame)
│   ├── vision.py                      (standalone HSV + contour ball finder)
│   ├── color_correction.py            (white-patch AWB, not in active path)
│   ├── levi                           (zsh launcher for the dashboard)
│   ├── requirements-camera-ui.txt     (Python deps)
│   ├── assets/                        (deep-sea background + emblem art)
│   └── ArduImageCapture/              (bundled Java viewer, ARM jSerialComm)
├── AGENTS.md                          (hardware, wiring, run state, resume plan)
├── GUARDRAIL.md                       (pool-test log + color methodology)
└── opencv.md                          (proposed hybrid local + GPT design)
```

---

## Data Flow

### Live Preview

```text
OV7670 -> Uno bit-bangs pixels -> RGB565 + parity over USB serial (500000 baud)
    ↓
ArduinoStreamReader decodes command/frame headers
    ↓
Pillow: rotate, JPEG encode
    ↓
GET /frame.jpg  (preview never throttled)
```

### Meta Inference

```text
Newest frame (>= 5 s since last start)
    ↓
OWLv2 local: candidate boxes, count, approximate (x, y)
    ↓
POST to Azure GPT-5.4: full frame + same-frame ViT handoff as spatial evidence
    ↓
JSON balls[] {color, confidence_percent}
    ↓
Detection panel, held until the next fresh frame begins inference
```

### OCV2 (Fully Local)

```text
Newest frame -> OWLv2 boxes
    ↓
OpenCV: bottom-half mask, dark-pixel suppression, denoise, LAB lightness normalize
    ↓
Per-crop KMeans dominant color -> LAB prototype match (7 classes)
    ↓
Detection panel + OpenCV Output tiles, no network call
```

---

## Measured Performance

Live measurements on the current Uno + OV7670 setup in `UART_MODE 3`. Treat these as observations, not guarantees.

- **~0.133 FPS** end to end, roughly one usable frame every 7.5 seconds
- **~0.63 s** local OWLv2 inference after a frame arrives
- **~3 s** GPT-5.4 full-frame classification
- **~3.3 s** combined Meta path (ViT plus GPT)
- **5 s** minimum interval between perception starts; preview stays uncapped
- **~0.9 FPS** internal color-bar run, used only to validate the serial and capture path

---

## Why a Hybrid Local + Cloud Pipeline

The design keeps both a local and a cloud path instead of committing to one:

- Local OpenCV and OWLv2 run in well under a second after a frame arrives, need no network, and can return location, size, and shape as well as color.
- GPT-5.4 is the strongest option for interpreting a distorted, dark, color-shifted OV7670 frame and applying pool context, but it is slower and network-dependent.
- The OV7670 is the real bottleneck. Image processing cannot recover detail the sensor never captured, and an aggressive color transform can make a crop look cleaner while making its true color less reliable, so frames are shown raw and reasoning is done in the prompt.

Pool frames are for calibration, not immediate model training: roughly 20-50 labeled frames to tune blue-background rejection, HSV ranges, ball size limits, and the confidence threshold that escalates a candidate to GPT-5.4. `GUARDRAIL.md` is the evidence log; record baseline results there before changing any prompt, ViT threshold, or crop setting.

The firmware and the ArduImageCapture viewer derive from Khaled Magdy's DeepBlueMbedded OV7670 tutorial and the CameraOV7670 / LiteOV7670 library, vendored here and adapted for this project.

---

## License

Copyright © 2026 CRAB CV Module. All rights reserved.

---

<div align="center">

**Built for the CRAB boat project - camera and computer-vision proof of concept**

</div>
