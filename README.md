# Levithian Command Center

OV7670 camera frames travel through an Uno-compatible Arduino over USB serial
to a local dashboard at `http://127.0.0.1:8765`.

## Run

```sh
python3 tools/levithian_dashboard.py
```

Close other serial viewers first: only one process can own the Arduino port.
The dashboard automatically waits for `/dev/cu.usbserial-*` after reconnects.

## Perception modes

- **ViT**: local OWLv2 candidate boxes and approximate `(x, y)` locations.
- **GPT 5.4**: full-frame cloud color classification.
- **Meta**: recommended full-frame GPT classification with ViT count/location
  handoff from the same frame.
- **Meta v2**: experimental GPT classification of ViT close-up crops only.

`Candidate Close-Up` is a collapsible inspection panel in ViT and Meta v2. It
does not affect standard Meta.

## Current camera mode

The active firmware is `UART_MODE 3`: 160x120 RGB565 at 500000 baud, with
internal sensor color bars disabled. Mode 5 is useful for color-bar validation,
but mode 3 is the safer live dashboard setting for the Uno.

See [AGENTS.md](AGENTS.md) for hardware/run details and
[GUARDRAIL.md](GUARDRAIL.md) for pool-test evidence and prompt methodology.
