from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional, Set, Tuple

import cv2
import numpy as np


class OpenCVViewer:
    """OpenCV-based display window with left-click event tracking.

    Each call to :meth:`render` draws a fresh overlay on a copy of the frame
    and calls ``cv2.imshow``.  Mouse events are buffered in a small deque so
    the caller can consume them with :meth:`poll_click`.

    Args:
        window_name: Title of the OS window. Defaults to ``"AIMBOT"``.
    """

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
        """Return and consume the most recent left-click, or ``None``."""
        if not self._clicks:
            return None
        return self._clicks.pop()

    def render(
        self,
        frame: np.ndarray,
        tracks: List[dict],
        target_id: Optional[int],
        fps: Optional[float] = None,
        secondary_target_ids: Optional[Set[int]] = None,
        lifecycle_state: Optional[str] = None,
    ) -> None:
        """Draw tracking overlays and display the frame.

        Colour scheme:
        - **Green** (age=0) / **Orange** (age>0): primary target.
        - **Yellow**: Re-ID match candidates.
        - **Blue**: all other confirmed tracks.

        Args:
            frame: BGR source frame; a copy is used for drawing.
            tracks: Track dicts with keys ``track_id``, ``bbox``,
                and ``time_since_update``.
            target_id: Primary target track ID to highlight.
            fps: Optional FPS value to overlay in the top-left corner.
            secondary_target_ids: Additional IDs to highlight as Re-ID matches.
            lifecycle_state: Optional target lifecycle state string to render.
        """
        if not self.is_open():
            return

        output = frame.copy()
        for track in tracks:
            tid = track["track_id"]
            age = int(track.get("time_since_update", 0))
            is_primary = tid == target_id
            is_secondary = secondary_target_ids is not None and tid in secondary_target_ids

            # Skip stale non-target tracks to reduce visual clutter.
            if age > 0 and not (is_primary or is_secondary):
                continue

            bbox = np.array(track.get("bbox", []), dtype=float).reshape(-1)
            if bbox.size != 4:
                continue
            x1, y1, x2, y2 = bbox.astype(int).tolist()

            if is_primary:
                color = (0, 255, 0) if age == 0 else (0, 165, 255)  # green / orange
                thickness = 3
                label = f"TARGET {tid}"
            elif is_secondary:
                color = (0, 255, 255)  # yellow
                thickness = 2
                label = f"MATCH {tid}"
            else:
                color = (255, 0, 0)  # blue
                thickness = 1
                label = f"ID {tid}"

            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)

            if is_primary and age > 0:
                label += f" (LOST {age})"

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

        if lifecycle_state:
            cv2.putText(
                output,
                f"STATE: {lifecycle_state}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        cv2.imshow(self.window_name, output)

    @staticmethod
    def wait_key(delay: int = 1) -> int:
        """Wrapper around ``cv2.waitKey`` that masks to the low 8 bits.

        Args:
            delay: Milliseconds to wait; ``0`` blocks indefinitely.

        Returns:
            Key code in ``[0, 255]``, or ``255`` if no key was pressed.
        """
        return cv2.waitKey(delay) & 0xFF

    def is_open(self) -> bool:
        """Return ``True`` if the window is still visible.

        Returns:
            ``False`` once the window has been closed by any means.
        """
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
        """Destroy the OpenCV window and mark the viewer as closed."""
        if not self._closed:
            cv2.destroyWindow(self.window_name)
            self._closed = True
