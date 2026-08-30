#!/usr/bin/env python3
"""Camera View — a tiny viewer for the project's OV7670 serial stream."""

from __future__ import annotations

import glob
import queue
import termios
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk

from PIL import Image, ImageTk
from PIL import ImageStat
import serial


# Must match lib/LiteOV7670/OV7670_UART.h (UART_MODE 3).
BAUD_RATE = 500000
COMMAND_START = 0x00
COMMAND_NEW_FRAME = 0x11
PIXEL_FORMAT_RGB565 = 1
PIXEL_FORMAT_GRAYSCALE = 2


@dataclass
class FrameSpec:
    width: int
    height: int
    pixel_format: int


def rgb565_to_rgb(high: int, low: int) -> tuple[int, int, int]:
    """Decode a LiteOV7670 RGB565 pixel exactly as ArduImageCapture does."""
    value = (high << 8) | low
    red = ((value >> 11) & 0x1F) * 255 // 31
    green = ((value >> 5) & 0x3F) * 255 // 63
    blue = (value & 0x1F) * 255 // 31
    return red, green, blue


class ArduinoStreamReader(threading.Thread):
    """Decode the command/pixel format emitted by LiteOV7670's UART sketch."""

    def __init__(self, port: str, frames: queue.Queue[Image.Image], status: queue.Queue[str]):
        super().__init__(daemon=True)
        self.port = port
        self.frames = frames
        self.status = status
        self.stop_event = threading.Event()
        self.frame_spec: FrameSpec | None = None
        self.pixel_bytes = bytearray()
        self.command: bytearray | None = None
        self.command_expected: int | None = None
        # Once a valid frame header has been received, every following byte is
        # pixel data until the advertised frame size is complete.  RGB565 image
        # data may legitimately contain 0x00, so it must not be mistaken for a
        # command delimiter while a frame is in progress.
        self.collecting_pixels = False
        self.received_live_frame = False
        self.camera_init_failed = False

    def stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        try:
            with serial.Serial(self.port, BAUD_RATE, timeout=0.25) as device:
                # A viewer may reconnect while the Uno is blocked waiting for a
                # camera edge. Toggle DTR to restart the sketch and guarantee a
                # fresh frame-stream header for this reader.
                device.dtr = False
                time.sleep(0.10)
                device.reset_input_buffer()
                device.dtr = True
                time.sleep(0.25)
                self.status.put(f"Listening on {self.port} at {BAUD_RATE} baud")
                while not self.stop_event.is_set():
                    data = device.read(device.in_waiting or 1)
                    for byte in data:
                        self.consume(byte)
        except (serial.SerialException, OSError, termios.error) as exc:
            self.status.put(f"Serial error: {exc}")
        finally:
            self.status.put("Stopped")

    def consume(self, byte: int) -> None:
        if self.command is not None:
            self.command.append(byte)
            if self.command_expected is None and self.command:
                # Stored bytes include the length byte itself, followed by the
                # payload and its checksum.
                self.command_expected = self.command[0] + 2
            if self.command_expected is not None and len(self.command) >= self.command_expected:
                self.process_command(self.command)
                self.command = None
                self.command_expected = None
            return

        if self.collecting_pixels:
            self.pixel_bytes.append(byte)
            if self.frame_spec:
                bytes_per_pixel = {
                    PIXEL_FORMAT_RGB565: 2,
                    PIXEL_FORMAT_GRAYSCALE: 1,
                }.get(self.frame_spec.pixel_format)
                if bytes_per_pixel and len(self.pixel_bytes) >= self.frame_spec.width * self.frame_spec.height * bytes_per_pixel:
                    self.publish_frame(bytes_per_pixel)
                    self.collecting_pixels = False
            return

        if byte == COMMAND_START:
            self.command = bytearray()
            return

    def process_command(self, command: bytearray) -> None:
        if len(command) < 2:
            return
        length = command[0]
        payload_and_checksum = command[1:]
        if len(payload_and_checksum) != length + 1:
            return
        payload, checksum = payload_and_checksum[:-1], payload_and_checksum[-1]
        calculated = 0
        for value in payload:
            calculated ^= value
        if calculated != checksum or not payload:
            return
        if payload[0] != COMMAND_NEW_FRAME or len(payload) != 4:
            return

        packed = payload[3]
        self.frame_spec = FrameSpec(
            width=payload[1] | ((packed & 0x03) << 8),
            height=payload[2] | ((packed & 0x0C) << 6),
            pixel_format=(packed >> 4) & 0x0F,
        )
        self.pixel_bytes.clear()
        self.collecting_pixels = True

    def publish_frame(self, bytes_per_pixel: int) -> None:
        assert self.frame_spec is not None
        raw = self.pixel_bytes[: self.frame_spec.width * self.frame_spec.height * bytes_per_pixel]
        self.pixel_bytes.clear()
        # The firmware emits a solid-green RGB565 placeholder after a successful
        # camera init. Detect it before undoing the transport marker bits.
        # (After decoding it is no longer mathematically pure #00ff00.)
        # Only the known LiteOV7670 RGB startup/error markers are placeholders.
        # Do not discard every uniform frame: a sensor test pattern or a real
        # solid-color scene can legitimately be uniform.
        if raw and raw == b"\x0f\xc1" * (len(raw) // 2):
            self.camera_init_failed = False
            if not self.received_live_frame:
                self.status.put("Camera connected; waiting for the first live frame…")
            return
        if raw and raw == b"\xf0\x01" * (len(raw) // 2):
            self.camera_init_failed = True
            self.status.put("Camera initialization failed; SCCB/I²C is not responding")
            return
        if self.frame_spec.pixel_format == PIXEL_FORMAT_RGB565:
            pixels = bytearray()
            for index in range(0, len(raw), 2):
                pixels.extend(rgb565_to_rgb(raw[index], raw[index + 1]))
            image = Image.frombytes("RGB", (self.frame_spec.width, self.frame_spec.height), bytes(pixels))
        else:
            # LiteOV7670 reserves the low bit of alternating grayscale bytes
            # as a transport marker. Dropping it restores the visible luma.
            pixels = bytes(value & 0xFE for value in raw)
            image = Image.frombytes("L", (self.frame_spec.width, self.frame_spec.height), pixels).convert("RGB")
        stat = ImageStat.Stat(image)
        if stat.mean[0] > 150 and stat.mean[0] > stat.mean[1] * 3 and stat.mean[0] > stat.mean[2] * 2.5:
            self.camera_init_failed = True
            self.status.put("Camera initialization failed; SCCB/I²C is not responding")
            return
        if self.camera_init_failed:
            return
        self.received_live_frame = True
        if self.frames.full():
            try:
                self.frames.get_nowait()
            except queue.Empty:
                pass
        self.frames.put_nowait(image)


class CameraApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Camera View")
        self.resizable(False, False)
        self.frames: queue.Queue[Image.Image] = queue.Queue(maxsize=1)
        self.status_messages: queue.Queue[str] = queue.Queue()
        self.reader: ArduinoStreamReader | None = None
        self.rotation = 90  # Camera is mounted sideways; clockwise is the default.
        self.photo: ImageTk.PhotoImage | None = None

        available_ports = glob.glob("/dev/cu.usbserial-*")
        default_port = available_ports[0] if available_ports else ""
        self.port = tk.StringVar(value=default_port)
        self.status = tk.StringVar(value="Choose the Arduino port and press Start.")

        controls = ttk.Frame(self, padding=12)
        controls.grid(row=0, column=0, sticky="ew")
        ttk.Label(controls, text="Camera View", font=("Helvetica", 20, "bold")).grid(
            row=0, column=0, columnspan=4, sticky="w"
        )
        ttk.Label(controls, text="Arduino port:").grid(row=1, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(controls, textvariable=self.port, width=28).grid(row=1, column=1, sticky="ew", pady=(10, 0))
        ttk.Button(controls, text="Start", command=self.start).grid(row=1, column=2, padx=(8, 0), pady=(10, 0))
        ttk.Button(controls, text="Stop", command=self.stop).grid(row=1, column=3, padx=(8, 0), pady=(10, 0))
        ttk.Button(controls, text="Rotate left", command=lambda: self.rotate(-90)).grid(row=2, column=1, pady=(8, 0))
        ttk.Button(controls, text="Rotate right", command=lambda: self.rotate(90)).grid(row=2, column=2, pady=(8, 0))
        ttk.Label(controls, textvariable=self.status).grid(row=3, column=0, columnspan=4, sticky="w", pady=(8, 0))

        self.preview = ttk.Label(self, text="Waiting for a camera frame…", anchor="center", padding=12)
        self.preview.grid(row=1, column=0, padx=12, pady=(0, 12))
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.after(30, self.poll)
        if default_port:
            self.after(200, self.start)

    def start(self) -> None:
        self.stop()
        if not self.port.get().strip():
            self.status.set("No Arduino port found. Reconnect the board, then enter its /dev/cu.usbserial-* port.")
            return
        self.reader = ArduinoStreamReader(self.port.get().strip(), self.frames, self.status_messages)
        self.reader.start()

    def stop(self) -> None:
        if self.reader:
            self.reader.stop()
            self.reader = None

    def rotate(self, degrees: int) -> None:
        self.rotation = (self.rotation + degrees) % 360
        self.status.set(f"Display rotation: {self.rotation}°")

    def poll(self) -> None:
        try:
            while True:
                self.status.set(self.status_messages.get_nowait())
        except queue.Empty:
            pass
        try:
            image = self.frames.get_nowait()
            if self.rotation:
                image = image.rotate(-self.rotation, expand=True)
            image = image.resize((image.width * 2, image.height * 2), Image.Resampling.NEAREST)
            self.photo = ImageTk.PhotoImage(image)
            self.preview.configure(image=self.photo, text="")
        except queue.Empty:
            pass
        self.after(30, self.poll)

    def close(self) -> None:
        self.stop()
        self.destroy()


if __name__ == "__main__":
    CameraApp().mainloop()
