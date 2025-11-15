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
    frame: np.ndarray
    detections: np.ndarray


class FramePrefetcher:
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
        frame = self._queue.get()
        if frame is None:
            return False, None
        return True, frame

    def stop(self) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=0.5)


class GpuPreprocessor:
    def __init__(self, scale: float) -> None:
        self.scale = float(max(min(scale, 1.0), 0.3))
        self._use_cuda = self.scale < 0.999 and hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0
        self._stream = cv2.cuda_Stream() if self._use_cuda else None

    @staticmethod
    def auto_scale(width: int, height: int) -> float:
        if max(width, height) <= 1280:
            return 1.0
        if max(width, height) <= 1920:
            return 0.85
        if max(width, height) <= 2560:
            return 0.75
        return 0.6

    def resize(self, frame: np.ndarray) -> np.ndarray:
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
        future = self._executor.submit(self._run_inference, frame)
        previous = self._pending
        self._pending = future
        if previous is None:
            return None
        return previous.result()

    def flush(self) -> Optional[DetectionResult]:
        if self._pending is None:
            return None
        result = self._pending.result()
        self._pending = None
        return result

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)


class ReIDHelper:
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
