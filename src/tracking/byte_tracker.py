from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_a = max(0.0, (a[2] - a[0])) * max(0.0, (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1]))
    union = area_a + area_b - inter + 1e-12
    return inter / union


def _iou_matrix(tracks: List["_Track"], detections: np.ndarray) -> np.ndarray:
    if not tracks or len(detections) == 0:
        return np.zeros((len(tracks), len(detections)), dtype=np.float32)
    matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    for i, track in enumerate(tracks):
        for j, det in enumerate(detections):
            matrix[i, j] = _iou(track.bbox, det[:4])
    return matrix


@dataclass
class _Track:
    track_id: int
    bbox: np.ndarray
    score: float
    class_id: int
    hits: int = 1
    time_since_update: int = 0

    def update(self, bbox: np.ndarray, score: float, class_id: int) -> None:
        self.bbox = bbox.astype(np.float32)
        self.score = float(score)
        self.class_id = int(class_id)
        self.hits += 1
        self.time_since_update = 0

    def step(self) -> None:
        self.time_since_update += 1


class ByteTrack:
    """簡化版多目標追蹤器 (基於 IoU 匹配)。"""

    def __init__(
        self,
        track_thresh: float = 0.5,
        match_iou_thresh: float = 0.3,
        max_age: int = 30,
        min_hits: int = 3,
    ) -> None:
        self.track_thresh = track_thresh
        self.match_iou_thresh = match_iou_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self._next_id = 1
        self._tracks: List[_Track] = []

    def update(
        self,
        detections: Optional[np.ndarray],
        img_size: Iterable[int],
    ) -> List[dict]:
        del img_size
        dets = (
            detections[:, :6] if detections is not None else np.empty((0, 6), dtype=np.float32)
        )
        if dets.size == 0:
            dets = np.empty((0, 6), dtype=np.float32)

        for track in self._tracks:
            track.step()

        confirmed_tracks = [t for t in self._tracks if t.time_since_update <= self.max_age]
        iou_matrix = _iou_matrix(confirmed_tracks, dets)

        matches: List[Tuple[int, int]] = []
        unmatched_track_indices = set(range(len(confirmed_tracks)))
        unmatched_det_indices = set(range(len(dets)))

        if confirmed_tracks and len(dets) > 0:
            cost = 1.0 - iou_matrix
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= self.match_iou_thresh:
                    matches.append((r, c))
                    unmatched_track_indices.discard(r)
                    unmatched_det_indices.discard(c)

        for track_idx, det_idx in matches:
            track = confirmed_tracks[track_idx]
            det = dets[det_idx]
            track.update(det[:4], det[4], det[5] if det.shape[0] > 5 else -1)

        for track_idx in unmatched_track_indices:
            track = confirmed_tracks[track_idx]
            track.time_since_update += 1

        for det_idx in unmatched_det_indices:
            det = dets[det_idx]
            if float(det[4]) < self.track_thresh:
                continue
            new_track = _Track(
                track_id=self._next_id,
                bbox=det[:4].astype(np.float32),
                score=float(det[4]),
                class_id=int(det[5]) if det.shape[0] > 5 else -1,
            )
            self._tracks.append(new_track)
            self._next_id += 1

        self._tracks = [track for track in self._tracks if track.time_since_update <= self.max_age]

        results: List[dict] = []
        for track in self._tracks:
            if track.hits >= self.min_hits or track.time_since_update == 0:
                results.append(
                    {
                        "track_id": track.track_id,
                        "bbox": track.bbox.copy(),
                        "score": track.score,
                        "class_id": track.class_id,
                        "is_confirmed": track.hits >= self.min_hits,
                    }
                )
        return results

    @staticmethod
    def as_xyxy(tracks: List[dict]) -> np.ndarray:
        if not tracks:
            return np.empty((0, 6), dtype=np.float32)
        rows = []
        for track in tracks:
            x1, y1, x2, y2 = track["bbox"]
            rows.append([x1, y1, x2, y2, track["score"], track["track_id"]])
        return np.array(rows, dtype=np.float32)
