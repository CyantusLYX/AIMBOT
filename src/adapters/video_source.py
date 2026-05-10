"""Video source adapter helpers."""

from typing import Optional, Tuple

import cv2


def _argus_pipeline(sensor_id: int, width: int, height: int, fps: int) -> str:
    return (
        "nvarguscamerasrc sensor-id={sensor_id} ! "
        "video/x-raw(memory:NVMM), width=(int){width}, height=(int){height}, "
        "format=(string)NV12, framerate=(fraction){fps}/1 ! "
        "nvvidconv ! video/x-raw, format=(string)BGRx ! "
        "videoconvert ! video/x-raw, format=(string)BGR ! "
        "appsink drop=true sync=false max-buffers=1"
    ).format(sensor_id=sensor_id, width=width, height=height, fps=fps)


def _v4l2_pipeline(device: str, width: int, height: int, fps: int) -> str:
    return (
        "v4l2src device={device} ! "
        "video/x-raw, width=(int){width}, height=(int){height}, framerate=(fraction){fps}/1 ! "
        "videoconvert ! video/x-raw, format=(string)BGR ! "
        "appsink drop=true sync=false max-buffers=1"
    ).format(device=device, width=width, height=height, fps=fps)


def _open_capture(source: str, backend: str, width: int, height: int, fps: int) -> cv2.VideoCapture:
    if "!" in source:
        return cv2.VideoCapture(source, cv2.CAP_GSTREAMER)

    if not source.isdigit():
        return cv2.VideoCapture(source)

    index = int(source)
    if backend == "argus":
        return cv2.VideoCapture(_argus_pipeline(index, width, height, fps), cv2.CAP_GSTREAMER)
    if backend == "v4l2":
        return cv2.VideoCapture(index, cv2.CAP_V4L2)
    if backend in ("gstreamer", "gst"):
        return cv2.VideoCapture(_v4l2_pipeline("/dev/video{}".format(index), width, height, fps), cv2.CAP_GSTREAMER)
    return cv2.VideoCapture(index)


def _probe_capture(cap: cv2.VideoCapture) -> Tuple[bool, Optional[object]]:
    if not cap.isOpened():
        return False, None
    ret, frame = cap.read()
    if not ret:
        return False, None
    return True, frame


def create_capture(
    source: str,
    camera_backend: str = "auto",
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
) -> cv2.VideoCapture:
    """Create and validate an OpenCV video capture.

    Args:
        source: Camera index string (e.g. ``"0"``) or file path.
        camera_backend: Camera backend for numeric sources: ``auto``,
            ``default``, ``v4l2``, ``gstreamer``, or ``argus``.
        width: Requested camera width for GStreamer backends.
        height: Requested camera height for GStreamer backends.
        fps: Requested camera frame rate for GStreamer backends.

    Returns:
        Opened ``cv2.VideoCapture`` object with a small input buffer.

    Raises:
        RuntimeError: If the source cannot be opened.
    """
    backend = camera_backend.lower()
    if backend not in ("auto", "default", "v4l2", "gstreamer", "gst", "argus"):
        raise ValueError("不支援的 camera backend: {}".format(camera_backend))

    if "!" in source:
        candidates = ("gstreamer",)
    elif source.isdigit() and backend == "auto":
        candidates = ("argus", "v4l2", "default")
    elif source.isdigit() and backend == "default":
        candidates = ("default",)
    else:
        candidates = (backend,)

    last_cap = None
    for candidate in candidates:
        cap = _open_capture(source, candidate, width, height, fps)
        ok, _ = _probe_capture(cap)
        if ok:
            cap.release()
            cap = _open_capture(source, candidate, width, height, fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            print("使用影像來源 backend: {}".format(candidate))
            return cap
        cap.release()
        last_cap = cap

    cap = last_cap if last_cap is not None else _open_capture(source, "default", width, height, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    if not cap.isOpened():
        raise RuntimeError("無法開啟來源: {}".format(source))
    return cap
