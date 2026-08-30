# Ball-Vision Guardrails and Boat-Test Log

## Purpose

This is the evidence log for color classification in the Levithian Command
Center. The active prompts live in code so they cannot drift from runtime:

- GPT 5.4 full-frame prompt: `tools/openai_ball_vision.py` → `PROMPT`
- Meta v2 crop-sheet prompt: `tools/openai_ball_vision.py` → `CROP_PROMPT`
- Meta ViT handoff text: `tools/levithian_dashboard.py` →
  `submit_openai_with_vit()`

The pool reference images informed the guardrails; they did **not** train any
model or change GPT/ViT weights.

## Active Methods

| Dashboard mode | Image sent to GPT | Intended role |
|---|---|---|
| ViT | None | Local OWLv2 proposal count, boxes, and approximate coordinates. |
| GPT 5.4 | Full camera frame | Standalone full-scene ball/color judgment. |
| Meta | Full frame + same-frame ViT candidate count/coordinates | Default recommended method: ViT spatial handoff plus GPT full-scene color reasoning. |
| Meta v2 | One vertical sheet of ViT padded close-up crops | Experimental comparison mode. It is only as good as ViT’s candidate boxes. |

The `Candidate Close-Up` panel is visible only in ViT and Meta v2. It displays
the exact padded crops from the frame that ViT analyzed; it does not change
normal Meta inference.

## Evidence So Far

- Firmware is currently `UART_MODE 3`: 160×120 RGB565 at 500000 baud, with internal color bars disabled for the dashboard.
- A live measurement was about **0.133 FPS** (roughly one usable frame every
  7.5 seconds). This is an average, not a guarantee; expect frame timing to
  vary with serial/camera behavior.
- Typical measured inference latency: local ViT ~0.63 s; GPT 5.4 ~3 s; Meta
  combines ViT plus GPT work at roughly ~3.3 s after a fresh frame is available.
- Each physical target ball can form a vertical figure-8 with its water
  reflection. Count/classify the upper physical sphere once; ignore the lower
  reflected copy.
- A bright top semicircle is common. It may be the ball’s color or pink/red/
  orange/white glare. Infer color primarily from the broad middle/lower body
  and rim of the upper sphere.
- Pool light is dim; balls can move; OV7670 frames can be dark, neon, blurred,
  or magenta/green shifted. Prompting cannot recover color/detail absent from a
  frame.

## Labeled Pool References

The earlier local pool-reference screenshots were used to shape the guardrails,
but they are not kept in the cleaned GitHub repo. Future pool screenshots should
be saved only when they are useful evidence, named clearly, and documented in the
test table below.

## Tomorrow: Mounted-Boat Baseline

The next test changes the camera viewpoint to roughly 1.5 feet above the
water. That can change apparent ball size, overlap, reflection geometry, and
background coverage. Do not edit thresholds or prompts before logging a short
baseline.

1. Confirm live feed and note the current `/dev/cu.usbserial-*` port.
2. Capture screenshots for no balls; one ball near/mid/far; multiple balls;
   balls near a wall/boat/target; and moving versus stationary boat.
3. Record actual count/color, ViT proposals, Meta result, and Meta v2 result.
4. Inspect whether Candidate Close-Up contains the upper ball cleanly or mixes
   reflection/neighboring balls.
5. Only after repeatable evidence, record a prompt, threshold, crop-padding, or
   layout change in the decision log.

## Next Experiment: OpenCV Close-Up Comparison

Do **not** replace the raw ViT crop before establishing a baseline. The OV7670
can lose detail through blur, clipping, and color distortion; image processing
cannot recover pixels that were never captured and can accidentally change the
color evidence that GPT needs.

After collecting mounted-boat frames, create a side-by-side inspection view for
each ViT candidate:

```text
raw candidate crop | mild OpenCV brightness/detail variant
```

Initial safe operations to compare:

- mild denoising for pixel/static noise;
- local contrast enhancement on brightness only (for example CLAHE on the Lab
  `L` channel);
- gentle upscale for manual inspection.

Do not initially use automatic white balance, hue/saturation shifts, aggressive
gamma, strong sharpening, or glare removal/inpainting. Those can make a crop
look clearer while making its real-world color less reliable.

For each representative pool frame, compare raw versus enhanced crops using the
known ball color. Only send an enhanced crop to GPT as an additional view if it
repeatedly improves color accuracy without increasing false colors. The raw
crop remains the source-of-truth fallback.

## Pool-Test Results

| Date | Mode | Camera height / distance | Lighting and context | Actual balls (left → right) | Result | Correct? | Screenshot path / notes |
|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |

## Change Decision Log

| Date | Repeatable observation | Proposed prompt/code change | Why it may help | Retest result |
|---|---|---|---|---|
|  |  |  |  |  |
