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

    def __init__(self, window_name: str = "AIMBOT", display_width: int = 640) -> None:
        self.window_name = window_name
        self.display_width = max(160, int(display_width))
        self._clicks: Deque[Tuple[int, int]] = deque(maxlen=5)
        self._closed = False
        self._has_rendered = False
        self._window_sized = False
        self._display_to_frame = (1.0, 1.0)
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._on_mouse)
        try:
            cv2.startWindowThread()
        except cv2.error:
            pass

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param: None) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self._clicks.append((x, y))

    def poll_click(self) -> Optional[Tuple[int, int]]:
        """Return and consume the most recent left-click, or ``None``."""
        if not self._clicks:
            return None
        x, y = self._clicks.pop()
        scale_x, scale_y = self._display_to_frame
        return int(round(x * scale_x)), int(round(y * scale_y))

    def render(
        self,
        frame: np.ndarray,
        tracks: List[dict],
        target_id: Optional[int],
        fps: Optional[float] = None,
        secondary_target_ids: Optional[Set[int]] = None,
        lifecycle_state: Optional[str] = None,
    ) -> bool:
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
            return False

        output = frame.copy()
        height, width = output.shape[:2]
        font_scale = max(0.75, min(1.4, min(width, height) / 720.0))
        label_scale = max(0.65, font_scale * 0.75)
        text_thickness = max(2, int(round(font_scale * 2.0)))
        box_thickness = max(2, int(round(font_scale * 2.5)))
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
                thickness = box_thickness + 1
                label = f"TARGET {tid}"
            elif is_secondary:
                color = (0, 255, 255)  # yellow
                thickness = box_thickness
                label = f"MATCH {tid}"
            else:
                color = (255, 0, 0)  # blue
                thickness = max(1, box_thickness - 1)
                label = f"ID {tid}"

            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)

            if is_primary and age > 0:
                label += f" (LOST {age})"

            label_y = max(24, y1 - 8)
            cv2.putText(
                output,
                label,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                label_scale,
                color,
                max(1, text_thickness - 1),
                cv2.LINE_AA,
            )

        status_lines = []
        if fps is not None and fps > 0:
            status_lines.append(f"FPS {fps:.1f}")
        if lifecycle_state:
            status_lines.append(f"STATE {lifecycle_state}")

        if status_lines:
            line_height = int(round(32 * font_scale))
            panel_width = min(width - 20, int(round(360 * font_scale)))
            panel_height = 16 + line_height * len(status_lines)
            overlay = output.copy()
            cv2.rectangle(overlay, (8, 8), (8 + panel_width, 8 + panel_height), (0, 0, 0), -1)
            output = cv2.addWeighted(overlay, 0.45, output, 0.55, 0)
            y = 8 + int(round(26 * font_scale))
            for line in status_lines:
                color = (0, 255, 255) if line.startswith("FPS") else (255, 255, 255)
                cv2.putText(
                    output,
                    line,
                    (18, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    color,
                    text_thickness,
                    cv2.LINE_AA,
                )
                y += line_height

        display_width = min(width, self.display_width)
        display_height = max(1, int(round(height * (display_width / float(max(width, 1))))))
        if display_width != width or display_height != height:
            output = cv2.resize(output, (display_width, display_height), interpolation=cv2.INTER_AREA)
        self._display_to_frame = (
            float(width) / float(max(display_width, 1)),
            float(height) / float(max(display_height, 1)),
        )

        if not self._window_sized:
            try:
                cv2.resizeWindow(self.window_name, display_width, display_height)
            except cv2.error:
                pass
            self._window_sized = True

        try:
            cv2.imshow(self.window_name, output)
            self._has_rendered = True
        except cv2.error:
            self._closed = True
            return False
        return True

    def wait_key(self, delay: int = 1) -> int:
        """Wrapper around ``cv2.waitKey`` that masks to the low 8 bits.

        Args:
            delay: Milliseconds to wait; ``0`` blocks indefinitely.

        Returns:
            Key code in ``[0, 255]``, or ``255`` if no key was pressed.
        """
        try:
            return cv2.waitKey(delay) & 0xFF
        except cv2.error:
            self._closed = True
            return 27

    def is_open(self) -> bool:
        """Return ``True`` if the window is still visible.

        Returns:
            ``False`` once the window has been closed by any means.
        """
        return not self._closed

    def close(self) -> None:
        """Destroy the OpenCV window and mark the viewer as closed."""
        if not self._closed:
            try:
                cv2.destroyWindow(self.window_name)
            except cv2.error:
                pass
            self._closed = True
