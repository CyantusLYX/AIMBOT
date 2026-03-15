"""Video source adapter helpers."""
from __future__ import annotations

import cv2


def create_capture(source: str) -> cv2.VideoCapture:
    """Create and validate an OpenCV video capture.

    Args:
        source: Camera index string (e.g. ``"0"``) or file path.

    Returns:
        Opened ``cv2.VideoCapture`` object with a small input buffer.

    Raises:
        RuntimeError: If the source cannot be opened.
    """
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟來源: {source}")
    return cap
