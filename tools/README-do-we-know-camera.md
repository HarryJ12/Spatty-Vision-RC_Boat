# Camera View

Small desktop viewer for the OV7670 stream emitted by this project. It replaces
the tutorial viewer with a project-specific UI and rotates the sideways camera
image clockwise by default.

## Run

Close `ArduImageCapture` first; only one program can own the Arduino serial
port at a time. From the repository root:

```sh
python3 -m pip install --user -r tools/requirements-camera-ui.txt
python3 tools/do_we_know_camera.py
```

Choose the current `/dev/cu.usbserial-*` port and click **Start**. The current
Arduino RGB profile uses 115200 baud, which is fixed in the UI to match the
firmware. Use **Rotate left** or **Rotate right** if the default orientation is
not correct.
