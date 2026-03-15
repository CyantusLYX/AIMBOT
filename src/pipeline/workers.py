"""Async pipeline workers for frame capture, preprocessing, detection, and Re-ID.

This module provides the concurrency primitives that decouple IO-bound camera
reading from GPU-bound detection inference:

- ``FramePrefetcher`` — background thread that drains ``cv2.VideoCapture``.
- ``GpuPreprocessor`` — optional CUDA-accelerated frame resize.
- ``AsyncDetector`` — single-worker thread-pool for YOLOv7 inference.
- ``ReIDHelper`` — schedules OSNet embedding extraction for top-K detections.
"""
from __future__ import annotations

import concurrent.futures
import queue
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Iterable, Optional

import cv2
import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from detection.detector import YoloV7Detector
    from reid.osnet import OSNetEmbedder


@dataclass
class DetectionResult:
    """Result produced by :class:`AsyncDetector` for a single frame.

    Attributes:
        frame: Original BGR frame at full resolution (uint8).
        detections: Detection array of shape ``(N, 6)`` — columns are
            ``x1, y1, x2, y2, confidence, class_id``.
    """

    frame: np.ndarray
    detections: np.ndarray


class FramePrefetcher:
    """Background-thread wrapper around ``cv2.VideoCapture``.

    Continuously fills an internal queue so the main thread never blocks on
    ``cap.read()``.  A ``None`` sentinel is enqueued when the source is
    exhausted or an error occurs.

    Args:
        capture: An already-opened ``cv2.VideoCapture`` instance.
        queue_size: Maximum number of frames to buffer.  Must be >= 1.
    """

    def __init__(self, capture: cv2.VideoCapture, queue_size: int = 4) -> None:
        self._capture = capture
        self._queue: "queue.Queue[Optional[np.ndarray]]" = queue.Queue(maxsize=max(1, queue_size))
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        while not self._stop.is_set():
            if self._queue.full():
                time.sleep(0.002)
                continue
            ret, frame = self._capture.read()
            if not ret:
                self._queue.put(None)
                break
            self._queue.put(frame)

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        """Block until a frame is available and return it.

        Returns:
            A ``(success, frame)`` pair.  ``success`` is ``False`` when the
            source has been exhausted; ``frame`` is ``None`` in that case.
        """
        frame = self._queue.get()
        if frame is None:
            return False, None
        return True, frame

    def stop(self) -> None:
        """Signal the reader thread to stop and wait for it to exit."""
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=0.5)


class GpuPreprocessor:
    """Optional CUDA-accelerated frame resizer.

    If OpenCV was built with CUDA support and at least one CUDA device is
    available, frames are resized on the GPU.  Otherwise CPU ``cv2.resize``
    with ``INTER_AREA`` is used transparently.

    Args:
        scale: Target scale factor in the range ``[0.3, 1.0]``.  Values
            outside this range are clamped.
    """

    def __init__(self, scale: float) -> None:
        self.scale = float(max(min(scale, 1.0), 0.3))
        self._use_cuda = self.scale < 0.999 and hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0
        self._stream = cv2.cuda_Stream() if self._use_cuda else None

    @staticmethod
    def auto_scale(width: int, height: int) -> float:
        """Suggest a scale factor based on the frame resolution.

        Args:
            width: Frame width in pixels.
            height: Frame height in pixels.

        Returns:
            A scale factor in ``(0, 1]`` tuned to keep the detector input
            at a reasonable size without excessive downsampling.
        """
        if max(width, height) <= 1280:
            return 1.0
        if max(width, height) <= 1920:
            return 0.85
        if max(width, height) <= 2560:
            return 0.75
        return 0.6

    def resize(self, frame: np.ndarray) -> np.ndarray:
        """Resize *frame* by the configured scale factor.

        Args:
            frame: Input BGR frame (uint8).

        Returns:
            Resized frame.  Returns the same array unchanged when
            ``scale >= 1.0``.
        """
        if self.scale >= 0.999:
            return frame
        h, w = frame.shape[:2]
        target_w = max(1, int(round(w * self.scale)))
        target_h = max(1, int(round(h * self.scale)))
        if not self._use_cuda:
            return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        gpu_src = cv2.cuda_GpuMat()
        gpu_src.upload(frame)
        gpu_dst = cv2.cuda.resize(gpu_src, (target_w, target_h), stream=self._stream)
        if self._stream is not None:
            self._stream.waitForCompletion()
        return gpu_dst.download()


class AsyncDetector:
    """One-frame-deep asynchronous wrapper around :class:`YoloV7Detector`.

    Inference runs in a single-worker ``ThreadPoolExecutor``.  ``submit()``
    enqueues the *current* frame and returns the *previous* frame's result,
    keeping latency at exactly one frame.  ``flush()`` drains the last
    pending result after the video source is exhausted.

    Args:
        detector: Initialised detector instance.
        preprocessor: Frame resizer applied before inference.
        class_filter: Optional callable that filters the raw detection array.
            Receives an ``(N, 6)`` array and must return a filtered array.
    """

    def __init__(
        self,
        detector: "YoloV7Detector",
        preprocessor: GpuPreprocessor,
        class_filter: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        self.detector = detector
        self.preprocessor = preprocessor
        self.class_filter = class_filter
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._pending: Optional[concurrent.futures.Future[DetectionResult]] = None

    def _run_inference(self, frame: np.ndarray) -> DetectionResult:
        proc_frame = self.preprocessor.resize(frame)
        detections = self.detector.detect(proc_frame)
        if detections.size:
            scale_x = proc_frame.shape[1] / frame.shape[1]
            scale_y = proc_frame.shape[0] / frame.shape[0]
            if abs(scale_x - 1.0) > 1e-3 or abs(scale_y - 1.0) > 1e-3:
                detections = detections.copy()
                detections[:, [0, 2]] /= max(scale_x, 1e-6)
                detections[:, [1, 3]] /= max(scale_y, 1e-6)
                detections[:, [0, 2]] = np.clip(detections[:, [0, 2]], 0, frame.shape[1] - 1)
                detections[:, [1, 3]] = np.clip(detections[:, [1, 3]], 0, frame.shape[0] - 1)
        if self.class_filter is not None and detections.size:
            detections = self.class_filter(detections)
        return DetectionResult(frame=frame, detections=detections)

    def submit(self, frame: np.ndarray) -> Optional[DetectionResult]:
        """Submit *frame* for inference and return the previous result.

        Args:
            frame: BGR frame to detect on.

        Returns:
            The :class:`DetectionResult` for the *previous* frame, or
            ``None`` if this is the first call.
        """
        future = self._executor.submit(self._run_inference, frame)
        previous = self._pending
        self._pending = future
        if previous is None:
            return None
        return previous.result()

    def flush(self) -> Optional[DetectionResult]:
        """Block until the last pending inference completes and return it.

        Returns:
            The final :class:`DetectionResult`, or ``None`` if no inference
            is pending.
        """
        if self._pending is None:
            return None
        result = self._pending.result()
        self._pending = None
        return result

    def shutdown(self) -> None:
        """Shut down the thread-pool executor, waiting for in-flight work."""
        self._executor.shutdown(wait=True)


class ReIDHelper:
    """Schedules OSNet Re-ID embedding extraction for top-K detections.

    Only a subset of detections receive embeddings each frame to limit GPU
    load: the ``max_candidates`` highest-confidence detections plus any
    detections that overlap the current target bounding-box by at least
    ``target_iou``.

    Args:
        embedder: Initialised :class:`OSNetEmbedder`, or ``None`` to disable
            Re-ID (``build_embeddings`` will always return ``None``).
        max_candidates: Maximum number of detections to embed per frame.
        target_iou: IoU overlap threshold for including low-confidence
            detections that may be near the tracked target.
    """

    def __init__(
        self,
        embedder: Optional["OSNetEmbedder"],
        max_candidates: int = 6,
        target_iou: float = 0.1,
    ) -> None:
        self.embedder = embedder
        self.max_candidates = max(1, max_candidates)
        self.target_iou = target_iou

    def build_embeddings(
        self,
        frame: np.ndarray,
        detections: np.ndarray,
        target_bbox: Optional[np.ndarray],
    ) -> Optional[list[Optional[np.ndarray]]]:
        """Extract Re-ID features for a subset of *detections*.

        Args:
            frame: Full-resolution BGR frame used to crop detection regions.
            detections: Array of shape ``(N, 6)`` from the detector.
            target_bbox: Current tracked target bounding-box ``(x1,y1,x2,y2)``
                used to prioritise nearby detections.  Pass ``None`` when no
                target is currently locked.

        Returns:
            A list of length ``N`` where each element is either a 1-D
            float32 feature vector or ``None`` (detection was not embedded).
            Returns ``None`` entirely when Re-ID is disabled or *detections*
            is empty.
        """
        if self.embedder is None or detections is None or detections.size == 0:
            return None
        num_dets = len(detections)
        order = np.argsort(-detections[:, 4])[: self.max_candidates]
        if target_bbox is not None:
            overlaps = []
            for idx in range(num_dets):
                if self._iou(detections[idx, :4], target_bbox) >= self.target_iou:
                    overlaps.append(idx)
            if overlaps:
                order = np.unique(np.concatenate([order, np.array(overlaps, dtype=np.int32)]))
        boxes = detections[order, :4]
        feats = self.embedder.extract_from_frame(frame, boxes)
        result: list[Optional[np.ndarray]] = [None] * num_dets
        for det_idx, feat in zip(order, feats):
            result[int(det_idx)] = feat
        return result

    @staticmethod
    def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
        x1 = max(float(box_a[0]), float(box_b[0]))
        y1 = max(float(box_a[1]), float(box_b[1]))
        x2 = min(float(box_a[2]), float(box_b[2]))
        y2 = min(float(box_a[3]), float(box_b[3]))
        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        inter = inter_w * inter_h
        area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
        area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
        union = area_a + area_b - inter + 1e-12
        return inter / union
