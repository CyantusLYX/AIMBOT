"""Video source adapter helpers."""

from typing import Optional, Tuple

import cv2


def _open_capture(source: str, backend: str) -> cv2.VideoCapture:
    if not source.isdigit():
        return cv2.VideoCapture(source)

    index = int(source)
    if backend == "v4l2":
        return cv2.VideoCapture(index, cv2.CAP_V4L2)
    if backend in ("gstreamer", "gst"):
        return cv2.VideoCapture(index, cv2.CAP_GSTREAMER)
    return cv2.VideoCapture(index)


def _probe_capture(cap: cv2.VideoCapture) -> Tuple[bool, Optional[object]]:
    if not cap.isOpened():
        return False, None
    ret, frame = cap.read()
    if not ret:
        return False, None
    return True, frame


def create_capture(source: str, camera_backend: str = "auto") -> cv2.VideoCapture:
    """Create and validate an OpenCV video capture.

    Args:
        source: Camera index string (e.g. ``"0"``) or file path.
        camera_backend: Camera backend for numeric sources: ``auto``,
            ``default``, ``v4l2``, or ``gstreamer``.

    Returns:
        Opened ``cv2.VideoCapture`` object with a small input buffer.

    Raises:
        RuntimeError: If the source cannot be opened.
    """
    backend = camera_backend.lower()
    if backend not in ("auto", "default", "v4l2", "gstreamer", "gst"):
        raise ValueError("不支援的 camera backend: {}".format(camera_backend))

    if source.isdigit() and backend == "auto":
        candidates = ("v4l2", "default")
    elif source.isdigit() and backend == "default":
        candidates = ("default",)
    else:
        candidates = (backend,)

    last_cap = None
    for candidate in candidates:
        cap = _open_capture(source, candidate)
        ok, _ = _probe_capture(cap)
        if ok:
            cap.release()
            cap = _open_capture(source, candidate)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            return cap
        cap.release()
        last_cap = cap

    cap = last_cap if last_cap is not None else _open_capture(source, "default")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    if not cap.isOpened():
        raise RuntimeError("無法開啟來源: {}".format(source))
    return cap
