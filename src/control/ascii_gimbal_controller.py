"""ASCII serial output for the ESP32 gimbal firmware."""
from __future__ import annotations

import time
from typing import Optional


class AsciiGimbalController:
    """Send ESP32 gimbal ASCII commands with rate limiting and reconnects."""

    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        dry_run: bool = False,
        max_hz: float = 30.0,
        reconnect_interval_s: float = 1.0,
    ) -> None:
        self.port = port
        self.baudrate = int(baudrate)
        self.dry_run = bool(dry_run)
        self.min_interval_s = 1.0 / max(1.0, float(max_hz))
        self.reconnect_interval_s = max(0.1, float(reconnect_interval_s))
        self._serial: Optional[object] = None
        self._last_send_s = 0.0
        self._last_connect_attempt_s = 0.0
        self._last_error = ""
        self._last_dry_run_message = ""

    def send(self, pan_speed: float, tilt_speed: float, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_send_s < self.min_interval_s:
            return
        self._last_send_s = now

        pan = int(round(pan_speed))
        tilt = int(round(tilt_speed))
        self._send_message(f"V:{pan},{tilt}\n", force=force)

    def set_enabled(self, enabled: bool, repeat: int = 3) -> None:
        message = f"E:{1 if enabled else 0}\n"
        for _ in range(max(1, int(repeat))):
            self._send_message(message, force=True)

    def hold(self, repeat: int = 1, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_send_s < self.min_interval_s:
            return
        self._last_send_s = now

        for _ in range(max(1, int(repeat))):
            self._send_message("H:1\n", force=force)

    def _send_message(self, message: str, force: bool = False) -> None:
        if self.dry_run:
            if force or message != self._last_dry_run_message:
                print(f"[GIMBAL] {message.strip()}")
            self._last_dry_run_message = message
            return

        serial_port = self._ensure_connected(time.monotonic())
        if serial_port is None:
            return

        try:
            serial_port.write(message.encode("ascii"))  # type: ignore[attr-defined]
            serial_port.flush()  # type: ignore[attr-defined]
        except Exception as exc:
            self._report_error(f"serial write failed: {exc}")
            self._close_serial()

    def stop(self, repeat: int = 3) -> None:
        for _ in range(max(1, int(repeat))):
            self.send(0, 0, force=True)

    def close(self) -> None:
        self._close_serial()

    def _ensure_connected(self, now: float) -> Optional[object]:
        if self._serial is not None and getattr(self._serial, "is_open", True):
            return self._serial
        if now - self._last_connect_attempt_s < self.reconnect_interval_s:
            return None
        self._last_connect_attempt_s = now
        try:
            import serial  # type: ignore[import-untyped]

            self._serial = serial.Serial(port=self.port, baudrate=self.baudrate, timeout=0.1)
            self._last_error = ""
            print(f"[GIMBAL] serial connected: {self.port}")
            return self._serial
        except Exception as exc:
            self._report_error(f"serial unavailable: {exc}")
            self._serial = None
            return None

    def _close_serial(self) -> None:
        if self._serial is None:
            return
        try:
            if getattr(self._serial, "is_open", False):
                self._serial.close()  # type: ignore[attr-defined]
        except Exception:
            pass
        self._serial = None

    def _report_error(self, message: str) -> None:
        if message != self._last_error:
            print(f"[GIMBAL] {message}")
            self._last_error = message
