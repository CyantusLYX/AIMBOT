from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

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
    feature: Optional[np.ndarray] = None

    def update(
        self,
        bbox: np.ndarray,
        score: float,
        class_id: int,
        feature: Optional[np.ndarray],
        momentum: float,
        min_similarity: float,
        similarity: float,
    ) -> None:
        self.bbox = bbox.astype(np.float32)
        self.score = float(score)
        self.class_id = int(class_id)
        self.hits += 1
        self.time_since_update = 0
        if feature is not None:
            feature_norm = np.linalg.norm(feature) + 1e-12
            normalized = feature.astype(np.float32) / feature_norm
            if self.feature is None:
                if similarity >= min_similarity:
                    self.feature = normalized
            else:
                if similarity >= min_similarity:
                    blended = momentum * self.feature + (1.0 - momentum) * normalized
                    norm = np.linalg.norm(blended) + 1e-12
                    self.feature = (blended / norm).astype(np.float32)

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
        enable_reid: bool = False,
        reid_match_thresh: float = 0.45,
        feature_momentum: float = 0.9,
        feature_min_similarity: float = 0.55,
        reid_max_center_dist: float = 0.35,
    ) -> None:
        self.track_thresh = track_thresh
        self.match_iou_thresh = match_iou_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.enable_reid = enable_reid
        self.reid_match_thresh = reid_match_thresh
        self.feature_momentum = feature_momentum
        self.feature_min_similarity = feature_min_similarity
        self.reid_max_center_dist = max(0.0, float(reid_max_center_dist))
        self._next_id = 1
        self._tracks: List[_Track] = []

    def update(
        self,
        detections: Optional[np.ndarray],
        img_size: Iterable[int],
        embeddings: Optional[Sequence[Optional[np.ndarray]]] = None,
    ) -> List[dict]:
        height = width = 1
        if img_size is not None:
            size_seq = list(img_size)
            if len(size_seq) >= 2:
                height, width = int(size_seq[0]), int(size_seq[1])
        diag = float(np.hypot(width, height)) + 1e-6
        max_center_dist = self.reid_max_center_dist * diag
        dets = (
            detections[:, :6] if detections is not None else np.empty((0, 6), dtype=np.float32)
        )
        if dets.size == 0:
            dets = np.empty((0, 6), dtype=np.float32)

        features: List[Optional[np.ndarray]]
        if embeddings is None:
            features = [None] * len(dets)
        else:
            features = [None] * len(dets)
            for idx, feature in enumerate(embeddings):
                if idx >= len(dets):
                    break
                if feature is None:
                    continue
                features[idx] = feature.astype(np.float32, copy=False)

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
            track.update(
                det[:4],
                det[4],
                det[5] if det.shape[0] > 5 else -1,
                features[det_idx],
                self.feature_momentum,
                self.feature_min_similarity,
                1.0,
            )

        # Re-ID matching for remaining tracks and detections
        if (
            self.enable_reid
            and self.reid_match_thresh > 0
            and confirmed_tracks
            and len(dets) > 0
        ):
            track_candidates = [
                (idx, confirmed_tracks[idx])
                for idx in sorted(unmatched_track_indices)
                if confirmed_tracks[idx].feature is not None
            ]
            det_candidates = [
                (idx, features[idx])
                for idx in sorted(unmatched_det_indices)
                if features[idx] is not None
            ]
            if track_candidates and det_candidates:
                cost_matrix = np.ones((len(track_candidates), len(det_candidates)), dtype=np.float32)
                for i, (_, track) in enumerate(track_candidates):
                    track_feat = track.feature
                    for j, (det_idx, det_feat) in enumerate(det_candidates):
                        if det_feat is None or track_feat is None:
                            cost_matrix[i, j] = 1.0
                            continue
                        det_bbox = dets[det_idx][:4]
                        track_bbox = track.bbox
                        if track.class_id >= 0 and dets[det_idx].shape[0] > 5:
                            det_class = int(dets[det_idx][5])
                            if det_class != track.class_id:
                                cost_matrix[i, j] = 1.0
                                continue
                        if self.reid_max_center_dist > 0:
                            cx_t = float((track_bbox[0] + track_bbox[2]) * 0.5)
                            cy_t = float((track_bbox[1] + track_bbox[3]) * 0.5)
                            cx_d = float((det_bbox[0] + det_bbox[2]) * 0.5)
                            cy_d = float((det_bbox[1] + det_bbox[3]) * 0.5)
                            dist = float(np.hypot(cx_t - cx_d, cy_t - cy_d))
                            if dist > max_center_dist:
                                cost_matrix[i, j] = 1.0
                                continue
                        similarity = float(np.dot(track_feat, det_feat)) if track_feat is not None else 0.0
                        cost_matrix[i, j] = 1.0 - similarity
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                for r, c in zip(row_ind, col_ind):
                    track_idx, track_obj = track_candidates[r]
                    det_idx, det_feat = det_candidates[c]
                    similarity = 1.0 - cost_matrix[r, c]
                    if similarity < self.reid_match_thresh:
                        continue
                    det = dets[det_idx]
                    track_obj.update(
                        det[:4],
                        det[4],
                        det[5] if det.shape[0] > 5 else -1,
                        det_feat,
                        self.feature_momentum,
                        self.feature_min_similarity,
                        similarity,
                    )
                    unmatched_track_indices.discard(track_idx)
                    unmatched_det_indices.discard(det_idx)

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
            feature = features[det_idx]
            if feature is not None:
                new_track.feature = feature.astype(np.float32)
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
                        "feature": track.feature.copy() if track.feature is not None else None,
                        "time_since_update": track.time_since_update,
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
