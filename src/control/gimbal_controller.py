from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from typing import Optional


class GimbalBase(ABC):
    """Abstract interface for gimbal control backends.

    Concrete backends (dry-run, serial, future simulator) must implement
    :meth:`send` and :meth:`close`.  The context-manager protocol is provided
    here so all subclasses can be used in a ``with`` statement.
    """

    @abstractmethod
    def send(self, pan_speed: float, tilt_speed: float) -> None:
        """Send pan and tilt speed commands to the gimbal.

        Args:
            pan_speed: Pan angular velocity (arbitrary units, signed).
            tilt_speed: Tilt angular velocity (arbitrary units, signed).
        """

    @abstractmethod
    def close(self) -> None:
        """Release any hardware resources held by this backend."""

    def __enter__(self) -> "GimbalBase":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class GimbalController(GimbalBase):
    """PID output serializer supporting dry-run and serial-port backends.

    In *dry-run* mode no hardware is required; commands are printed to stdout
    as JSON lines.  In *serial* mode, commands are JSON-encoded and sent over
    the configured UART port.

    Args:
        port: Serial port identifier (e.g. ``"COM3"`` or ``"/dev/ttyUSB0"``).
        baudrate: UART baud rate. Defaults to ``115200``.
        timeout: Serial read timeout in seconds. Defaults to ``0.1``.
        dry_run: When ``True``, commands are printed rather than transmitted.

    Raises:
        ImportError: If *pyserial* is not installed and *dry_run* is ``False``.
    """

    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        timeout: float = 0.1,
        dry_run: bool = False,
    ) -> None:
        self.dry_run = dry_run
        self._serial: Optional[object] = None
        if not dry_run:
            try:
                import serial  # type: ignore[import-untyped]
            except ImportError as exc:
                raise ImportError("缺少 pyserial，相依套件列於 requirements.txt。") from exc
            self._serial = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)
        self.last_sent = time.time()

    def close(self) -> None:
        """Close the underlying serial connection if open."""
        if self._serial is not None and self._serial.is_open:  # type: ignore[union-attr]
            self._serial.close()  # type: ignore[union-attr]

    def send(self, pan_speed: float, tilt_speed: float) -> None:
        """Encode and transmit a gimbal speed command.

        Args:
            pan_speed: Pan axis speed.
            tilt_speed: Tilt axis speed.
        """
        payload = {
            "pan": float(pan_speed),
            "tilt": float(tilt_speed),
            "timestamp": time.time(),
        }
        message = json.dumps(payload) + "\n"
        if self.dry_run or self._serial is None:
            print(f"[GIMBAL] {message.strip()}")
        else:
            encoded = message.encode("utf-8")
            self._serial.write(encoded)  # type: ignore[union-attr]
            self._serial.flush()  # type: ignore[union-attr]
        self.last_sent = time.time()
