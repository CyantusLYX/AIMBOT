from __future__ import annotations

import json
import time
from typing import Optional

try:
    import serial  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError("缺少 pyserial，相依套件列於 requirements.txt。") from exc


class GimbalController:
    """Serialize PID輸出為簡單的 JSON 命令。"""

    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        timeout: float = 0.1,
        dry_run: bool = False,
    ) -> None:
        self.dry_run = dry_run
        self.serial: Optional[serial.Serial] = None
        if not dry_run:
            self.serial = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)
        self.last_sent = time.time()

    def close(self) -> None:
        if self.serial is not None and self.serial.is_open:
            self.serial.close()

    def send(self, pan_speed: float, tilt_speed: float) -> None:
        payload = {
            "pan": float(pan_speed),
            "tilt": float(tilt_speed),
            "timestamp": time.time(),
        }
        message = json.dumps(payload) + "\n"
        if self.dry_run or self.serial is None:
            print(f"[GIMBAL] {message.strip()}")
        else:
            encoded = message.encode("utf-8")
            self.serial.write(encoded)
            self.serial.flush()
        self.last_sent = time.time()

    def __enter__(self) -> "GimbalController":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
