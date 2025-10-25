from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np


class OpenCVViewer:
    def __init__(self, window_name: str = "AIMBOT") -> None:
        self.window_name = window_name
        self._clicks: Deque[Tuple[int, int]] = deque(maxlen=5)
        self._closed = False
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._on_mouse)

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param: None) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self._clicks.append((x, y))

    def poll_click(self) -> Optional[Tuple[int, int]]:
        if not self._clicks:
            return None
        return self._clicks.pop()

    def render(
        self,
        frame: np.ndarray,
        tracks: List[dict],
        target_id: Optional[int],
        fps: Optional[float] = None,
    ) -> None:
        if not self.is_open():
            return

        output = frame.copy()
        for track in tracks:
            age = int(track.get("time_since_update", 0))
            is_target = track["track_id"] == target_id
            if age > 0 and not is_target:
                continue
            bbox = np.array(track.get("bbox", []), dtype=float).reshape(-1)
            if bbox.size != 4:
                continue
            x1, y1, x2, y2 = bbox.astype(int).tolist()
            if is_target:
                color = (0, 255, 0) if age == 0 else (0, 165, 255)
            else:
                color = (255, 0, 0)
            thickness = 2 if age == 0 else 1
            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
            if is_target and age > 0:
                label = f"TARGET {track['track_id']} (LOST {age})"
            elif is_target:
                label = f"TARGET {track['track_id']}"
            else:
                label = f"ID {track['track_id']}"
            cv2.putText(output, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        if fps is not None and fps > 0:
            cv2.putText(
                output,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        cv2.imshow(self.window_name, output)

    @staticmethod
    def wait_key(delay: int = 1) -> int:
        return cv2.waitKey(delay) & 0xFF

    def is_open(self) -> bool:
        if self._closed:
            return False
        try:
            visible = cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE)
        except cv2.error:
            self._closed = True
            return False
        if visible <= 0:
            self._closed = True
            return False
        return True

    def close(self) -> None:
        if not self._closed:
            cv2.destroyWindow(self.window_name)
            self._closed = True
