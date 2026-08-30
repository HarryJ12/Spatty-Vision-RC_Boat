# OpenCV and Cloud Vision Options

## OpenCV Experiments

The experiment code is in `tools/opencv_experiments.py`. It consumes an RGB
frame plus ViT/OWL detections (`label`, `bbox`) and only processes proposals
whose label contains `ball`. Every such crop is deliberately masked so only
its bottom half is available to the color stage. This is specific to the pool
camera: the top arc commonly carries a clipped lamp-glare cap, while the lower
half of the detected *upper physical sphere* is better color evidence. The
detector's box must exclude the lower water reflection.

### OpenCV 1: processed crops to GPT

1. ViT supplies ball boxes.
2. OpenCV crops, bottom-half-masks, removes near-black tank pixels, blurs
   lightly, and resizes each crop.
3. A vertical contact sheet of those processed crops goes to GPT-5.4.
4. GPT performs the final allowed-color reasoning using the existing pool
   guardrails.

This is a controlled comparison against raw Meta v2 crops. It does not claim
that preprocessing restores missing camera information.

### OpenCV 2: processed crops to local color extraction

1. The same ViT boxes and same OpenCV preprocessing are used.
2. `adikuma/Colour-Extract` is conceptually a KMeans dominant-color extractor,
   not a trained ball classifier. Its dominant RGB/HEX output is mapped in LAB
   space to the allowed labels: red, orange, yellow, green, blue, purple, pink,
   brown, gray, black, white.
3. The implementation includes an equivalent local OpenCV KMeans path so this
   experiment can run offline and the extractor can be swapped later without
   changing crop/mask behavior.

Run the isolated demo with any saved test frame and explicit ViT-style boxes:

```sh
python3 tools/demo_opencv_experiments.py path/to/test-frame.jpg \
  --detection ball,20,30,90,100 --detection ball,110,35,180,105
```

## Decision to Validate

Use a hybrid ball-detection pipeline rather than choosing only one method:

```text
OV7670 frame
    -> OpenCV proposes round ball candidates locally
    -> OpenCV classifies obvious colors quickly
    -> GPT-5.4 checks only ambiguous candidates or frames
```

The existing GPT-5.4 dashboard path remains the working baseline. Do not remove
it until pool tests show that local detection works reliably.

## Proposed First Hybrid Implementation

This is the concrete plan to review with a vision/physics specialist:

1. The Mac receives the existing OV7670 frame from Arduino serial.
2. **OpenCV runs locally**. It performs all color masks, blue-background
   rejection, image cleanup, round-shape filtering, and candidate location
   calculation. OpenAI is not used for these pixel-level operations.
3. OpenCV assigns a local color and confidence to each candidate ball.
4. If the local confidence is high, the dashboard uses that local result
   immediately, including its `(x, y)` position.
5. If the local confidence is low, the Mac sends only the cropped candidate
   region, or the full frame when necessary, to GPT-5.4 for a contextual second
   opinion.
6. The dashboard displays the selected final result and records whether it came
   from local OpenCV or GPT-5.4.

This avoids using a cloud call for every frame while preserving the current
GPT-5.4 fallback for distorted or ambiguous images.

## What Pool Frames Are For

Pool frames are for **calibration**, not immediate custom-model training. Use
roughly 20 to 50 representative labeled frames to tune:

- Blue water and wall rejection ranges
- Ball HSV color ranges
- Minimum and maximum ball size at usable distances
- Circle/contour roundness thresholds
- Lighting and reflection handling
- The confidence threshold that routes a candidate to GPT-5.4

For each saved frame, record true ball color, approximate size, distance,
lighting, whether the boat was moving, and the actual number of balls.

## Why Consider OpenCV

- It runs locally on the Mac, usually in milliseconds after a frame arrives.
- It does not send every frame to an API.
- It can return candidate location, size, and shape as well as color.
- It could run every received frame, while cloud inference is intentionally
  slower and network-dependent.

## Current Limits

- The camera is an OV7670 at 160 x 120 RGB565 over serial, so frame quality and
  transfer rate are the primary bottlenecks.
- Current home images have strong magenta, green, pink, and orange casts plus
  glare. A simple fixed color threshold may fail outside the lighting where it
  was tuned.
- Motion can corrupt or blur frames. A stationary capture before collection is
  the realistic first demo behavior.
- Water, reflections, and blue walls can produce false round/color candidates.
- Detecting bubbles is not a strong first target: reflections and water texture
  can look bubble-like. Focus first on solid-color floating balls.

## Feasibility Estimate

| Question | Current estimate | Why |
|---|---:|---|
| Local OpenCV speed after frame arrival | 9 / 10 | Color math and contour finding are fast on a Mac. |
| Local OpenCV color accuracy in current home lighting | 4 / 10 | Severe color cast and glare make fixed thresholds fragile. |
| Local OpenCV in a controlled pool setup | 7 / 10 | Fixed lighting, known colors, and blue-background rejection improve it. |
| OpenCV as the only system today | 6 / 10 | Promising, but needs real pool calibration data. |
| Bubble detection as a primary signal | 3 / 10 | Too many reflection and texture false positives. |

These are working estimates, not pool-verified conclusions.

## Suggested Local OpenCV Pipeline

1. Decode the existing serial RGB565 frame on the Mac.
2. Rotate it into dashboard orientation.
3. Convert RGB to HSV.
4. Build masks for candidate non-background colors. Explicitly reject the
   calibrated blue tank/background range.
5. Clean masks with small morphological operations to remove isolated pixel
   noise and fill small holes.
6. Find connected regions, called contours.
7. Reject contours that are too small, too large, too thin, or not round enough.
8. For each round candidate, calculate its center `(x, y)`, area, radius, and
   dominant color from its non-highlighted interior pixels.
9. Assign local confidence from size, circularity, color consistency, and how
   far the candidate is from the blue-background model.
10. Send only low-confidence or conflicting candidates to GPT-5.4 as a second
    opinion. Keep local results when they are strong.

## Classifier Alternatives

### Rule-based OpenCV, recommended first

HSV masks plus round-contour filters. It is explainable, fast, and does not need
training data. Its weakness is sensitivity to color cast and lighting changes.

### Small trained classifier, later option

Collect labeled pool images and train a small model to classify cropped ball
patches, for example blue, green, yellow, red, or unknown. It can learn some
camera-specific distortion, but it requires enough representative images from
the real tank and can still fail on new lighting/backgrounds.

### Object detector, later option

A compact detector can return a bounding box and ball class directly. It needs
many labeled examples and is more work than needed for the first demo. It is not
the recommended first step with the current camera.

### GPT-5.4 vision, current fallback

Best at interpreting distorted low-quality frames and applying context from the
prompt. It is slower, network-dependent, and should not be the only control
loop signal once the boat must react quickly.

## Questions for Pool and Work Review

1. What are the exact official ball colors, sizes, and surface finishes?
2. Are balls always solid-color and approximately spherical?
3. What lighting is present above and below the tank?
4. How blue are the water and walls under competition lighting?
5. At what distance must the boat recognize a ball to act successfully?
6. Can the boat pause briefly before a classification/capture decision?
7. Can we collect and label 50 to 200 representative pool frames for calibration?
8. Is a local OpenCV result allowed/preferred for the competition demo?
9. What false positive is worse: missing a ball or chasing a reflection?
10. Is HSV plus contours sufficient for this camera, or is another color space
    such as Lab or a different simple feature approach more robust?
11. What local confidence features would they use to decide whether to escalate
    a crop to GPT-5.4?
12. Should GPT-5.4 receive the full frame, a ball crop, or both for ambiguous
    cases?

## Recommended Next Step

Do not implement a final classifier from home images. First capture a small pool
dataset with known ball color, distance, lighting, and ground-truth result.
Record observations in `GUARDRAIL.md`, then tune HSV ranges and contour limits
against those images. Compare the local OpenCV result with GPT-5.4 on the same
frames before deciding the final hybrid threshold.
