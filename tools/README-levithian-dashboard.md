# Levithian Camera View

Local green-themed dashboard for the live OV7670 feed. It reads the Arduino's
serial stream directly; it does not embed the tutorial's Java window, because
desktop Java windows cannot be embedded in a web page.

Close Camera View and ArduImageCapture first, then run:

```sh
python3 tools/levithian_dashboard.py
```

Open `http://127.0.0.1:8765` in a browser. The page includes a ready-to-fill
vision panel for the ball color and `(x, y)` position pipeline.
