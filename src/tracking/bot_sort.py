"""Local BoT-SORT style tracker with optional Re-ID association.

This module implements the project-facing subset of BoT-SORT without pulling
in BoxMOT or FastReID dependencies. It keeps the existing AIMBOT track-dict
contract so callers can swap it with ByteTrack backends through
``TrackingService``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


def _xyxy_to_xywh(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5, max(1e-3, x2 - x1), max(1e-3, y2 - y1)], dtype=np.float32)


def _xywh_to_xyxy(box: np.ndarray) -> np.ndarray:
    cx, cy, w, h = [float(v) for v in box[:4]]
    return np.array([cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5], dtype=np.float32)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return inter / (area_a + area_b - inter + 1e-12)


def _normalize_feature(feature: np.ndarray) -> np.ndarray:
    feature = feature.astype(np.float32, copy=False)
    return feature / (np.linalg.norm(feature) + 1e-12)


class KalmanFilterXYWH:
    """Eight-dimensional constant-velocity Kalman filter for ``cx,cy,w,h``."""

    def __init__(self) -> None:
        self._motion = np.eye(8, dtype=np.float32)
        self._motion[0, 4] = 1.0
        self._motion[1, 5] = 1.0
        self._motion[2, 6] = 1.0
        self._motion[3, 7] = 1.0
        self._update = np.eye(4, 8, dtype=np.float32)

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.zeros(8, dtype=np.float32)
        mean[:4] = measurement.astype(np.float32)
        covariance = np.eye(8, dtype=np.float32)
        covariance[:4, :4] *= 10.0
        covariance[4:, 4:] *= 100.0
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        process_noise = np.eye(8, dtype=np.float32)
        process_noise[:4, :4] *= 1.0
        process_noise[4:, 4:] *= 0.05
        mean = self._motion @ mean
        covariance = self._motion @ covariance @ self._motion.T + process_noise
        return mean.astype(np.float32), covariance.astype(np.float32)

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        measurement_noise = np.eye(4, dtype=np.float32)
        measurement_noise[:2, :2] *= 1.0
        measurement_noise[2:, 2:] *= 4.0
        projected_mean = self._update @ mean
        projected_cov = self._update @ covariance @ self._update.T + measurement_noise
        kalman_gain = covariance @ self._update.T @ np.linalg.inv(projected_cov)
        innovation = measurement.astype(np.float32) - projected_mean
        mean = mean + kalman_gain @ innovation
        covariance = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return mean.astype(np.float32), covariance.astype(np.float32)


@dataclass
class _BoTTrack:
    track_id: int
    mean: np.ndarray
    covariance: np.ndarray
    score: float
    class_id: int
    hits: int = 1
    time_since_update: int = 0
    state: str = "tracked"
    feature: Optional[np.ndarray] = None

    @property
    def bbox(self) -> np.ndarray:
        return _xywh_to_xyxy(self.mean[:4])

    def predict(self, kalman_filter: KalmanFilterXYWH) -> None:
        self.mean, self.covariance = kalman_filter.predict(self.mean, self.covariance)
        self.time_since_update += 1

    def update(
        self,
        kalman_filter: KalmanFilterXYWH,
        detection: np.ndarray,
        feature: Optional[np.ndarray],
        feature_momentum: float,
        feature_min_similarity: float,
    ) -> None:
        self.mean, self.covariance = kalman_filter.update(self.mean, self.covariance, _xyxy_to_xywh(detection[:4]))
        self.score = float(detection[4])
        self.class_id = int(detection[5]) if detection.shape[0] > 5 else -1
        self.hits += 1
        self.time_since_update = 0
        self.state = "tracked"
        self._update_feature(feature, feature_momentum, feature_min_similarity)

    def mark_lost(self) -> None:
        if self.state != "removed":
            self.state = "lost"

    def mark_removed(self) -> None:
        self.state = "removed"

    def _update_feature(
        self,
        feature: Optional[np.ndarray],
        momentum: float,
        min_similarity: float,
    ) -> None:
        if feature is None:
            return
        normalized = _normalize_feature(feature)
        if self.feature is None:
            self.feature = normalized
            return
        similarity = float(np.dot(self.feature, normalized))
        if similarity < min_similarity:
            return
        blended = momentum * self.feature + (1.0 - momentum) * normalized
        self.feature = _normalize_feature(blended)


class BoTSort:
    """BoT-SORT style multi-object tracker using motion and optional Re-ID."""

    kind = "botsort"

    def __init__(
        self,
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        track_buffer: int = 30,
        max_age: Optional[int] = None,
        min_hits: int = 3,
        match_thresh: float = 0.8,
        proximity_thresh: float = 0.5,
        appearance_thresh: float = 0.25,
        second_match_thresh: float = 0.5,
        min_box_area: float = 64.0,
        max_aspect_ratio: float = 8.0,
        enable_reid: bool = False,
        feature_momentum: float = 0.9,
        feature_min_similarity: float = 0.5,
    ) -> None:
        self.track_high_thresh = float(track_high_thresh)
        self.track_low_thresh = float(track_low_thresh)
        self.new_track_thresh = float(new_track_thresh)
        self.max_age = max(1, int(max_age if max_age is not None else track_buffer))
        self.min_hits = max(1, int(min_hits))
        self.match_thresh = float(match_thresh)
        self.proximity_thresh = float(proximity_thresh)
        self.appearance_thresh = float(appearance_thresh)
        self.second_match_thresh = float(second_match_thresh)
        self.min_box_area = max(0.0, float(min_box_area))
        self.max_aspect_ratio = max(1.0, float(max_aspect_ratio))
        self.enable_reid = bool(enable_reid)
        self.feature_momentum = float(feature_momentum)
        self.feature_min_similarity = float(feature_min_similarity)
        self._next_id = 1
        self._tracks: list[_BoTTrack] = []
        self._kalman_filter = KalmanFilterXYWH()

    def update(
        self,
        detections: Optional[np.ndarray],
        img_size: Iterable[int],
        embeddings: Optional[Sequence[Optional[np.ndarray]]] = None,
    ) -> list[dict]:
        """Run one BoT-SORT update and return track dictionaries."""
        height, width = self._parse_img_size(img_size)
        dets = detections[:, :6] if detections is not None else np.empty((0, 6), dtype=np.float32)
        dets, keep_indices = self._filter_detections(dets.astype(np.float32, copy=False), width, height)
        features = self._align_features(embeddings, keep_indices, len(dets))

        for track in self._tracks:
            if track.state != "removed":
                track.predict(self._kalman_filter)

        high_indices = np.flatnonzero(dets[:, 4] >= self.track_high_thresh) if len(dets) else np.empty((0,), dtype=np.int32)
        low_indices = (
            np.flatnonzero((dets[:, 4] >= self.track_low_thresh) & (dets[:, 4] < self.track_high_thresh))
            if len(dets)
            else np.empty((0,), dtype=np.int32)
        )

        confirmed_pool = [track for track in self._tracks if track.state in {"tracked", "lost"} and track.hits >= self.min_hits]
        unconfirmed = [track for track in self._tracks if track.state == "tracked" and track.hits < self.min_hits]

        unmatched_pool, unmatched_high = self._associate(
            confirmed_pool,
            dets,
            features,
            high_indices,
            self.match_thresh,
            use_reid=self.enable_reid,
        )

        still_tracked = [track for track in unmatched_pool if track.state == "tracked"]
        unmatched_tracked, _unmatched_low = self._associate(
            still_tracked,
            dets,
            features,
            low_indices,
            self.second_match_thresh,
            use_reid=False,
        )

        unmatched_high_after_unconfirmed = self._associate_unconfirmed(unconfirmed, dets, features, unmatched_high)

        unmatched_confirmed = {track.track_id: track for track in unmatched_pool}
        unmatched_confirmed.update({track.track_id: track for track in unmatched_tracked})
        for track in unmatched_confirmed.values():
            if track.state == "tracked":
                track.mark_lost()

        for track in unconfirmed:
            if track.time_since_update > 0:
                track.mark_removed()

        self._init_new_tracks(dets, features, unmatched_high_after_unconfirmed)
        self._tracks = [
            track
            for track in self._tracks
            if track.state != "removed" and track.time_since_update <= self.max_age
        ]
        return self._results()

    def _associate(
        self,
        tracks: list[_BoTTrack],
        detections: np.ndarray,
        features: list[Optional[np.ndarray]],
        det_indices: np.ndarray,
        threshold: float,
        use_reid: bool,
    ) -> tuple[list[_BoTTrack], np.ndarray]:
        if not tracks or len(det_indices) == 0:
            return list(tracks), det_indices.copy()
        dists = self._distance_matrix(tracks, detections, features, det_indices, use_reid=use_reid)
        row_ind, col_ind = linear_sum_assignment(dists)
        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        for row, col in zip(row_ind, col_ind):
            if float(dists[row, col]) > threshold:
                continue
            track = tracks[row]
            det_idx = int(det_indices[col])
            track.update(
                self._kalman_filter,
                detections[det_idx],
                features[det_idx],
                self.feature_momentum,
                self.feature_min_similarity,
            )
            matched_tracks.add(row)
            matched_dets.add(col)
        unmatched_tracks = [track for idx, track in enumerate(tracks) if idx not in matched_tracks]
        unmatched_dets = np.array(
            [int(det_indices[idx]) for idx in range(len(det_indices)) if idx not in matched_dets],
            dtype=np.int32,
        )
        return unmatched_tracks, unmatched_dets

    def _associate_unconfirmed(
        self,
        tracks: list[_BoTTrack],
        detections: np.ndarray,
        features: list[Optional[np.ndarray]],
        det_indices: np.ndarray,
    ) -> np.ndarray:
        if not tracks or len(det_indices) == 0:
            return det_indices.copy()
        dists = self._iou_distance_matrix(tracks, detections, det_indices)
        row_ind, col_ind = linear_sum_assignment(dists)
        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        for row, col in zip(row_ind, col_ind):
            if float(dists[row, col]) > self.match_thresh:
                continue
            det_idx = int(det_indices[col])
            tracks[row].update(
                self._kalman_filter,
                detections[det_idx],
                features[det_idx],
                self.feature_momentum,
                self.feature_min_similarity,
            )
            matched_tracks.add(row)
            matched_dets.add(col)
        for idx, track in enumerate(tracks):
            if idx not in matched_tracks:
                track.mark_removed()
        return np.array(
            [int(det_indices[idx]) for idx in range(len(det_indices)) if idx not in matched_dets],
            dtype=np.int32,
        )

    def _distance_matrix(
        self,
        tracks: list[_BoTTrack],
        detections: np.ndarray,
        features: list[Optional[np.ndarray]],
        det_indices: np.ndarray,
        use_reid: bool,
    ) -> np.ndarray:
        iou_dists = self._iou_distance_matrix(tracks, detections, det_indices)
        if not use_reid:
            return iou_dists
        appearance = np.ones_like(iou_dists, dtype=np.float32)
        for row, track in enumerate(tracks):
            if track.feature is None:
                continue
            for col, det_idx in enumerate(det_indices):
                det_idx = int(det_idx)
                feature = features[det_idx]
                if feature is None or not self._class_compatible(track, detections[det_idx]):
                    continue
                distance = (1.0 - float(np.dot(track.feature, feature))) * 0.5
                if distance > self.appearance_thresh:
                    continue
                if track.state != "lost" and iou_dists[row, col] > self.proximity_thresh:
                    continue
                appearance[row, col] = distance
        return np.minimum(iou_dists, appearance)

    def _iou_distance_matrix(self, tracks: list[_BoTTrack], detections: np.ndarray, det_indices: np.ndarray) -> np.ndarray:
        matrix = np.ones((len(tracks), len(det_indices)), dtype=np.float32)
        for row, track in enumerate(tracks):
            for col, det_idx in enumerate(det_indices):
                det = detections[int(det_idx)]
                if not self._class_compatible(track, det):
                    continue
                matrix[row, col] = 1.0 - _iou(track.bbox, det[:4])
        return matrix

    @staticmethod
    def _class_compatible(track: _BoTTrack, detection: np.ndarray) -> bool:
        if track.class_id < 0 or detection.shape[0] <= 5:
            return True
        det_class = int(detection[5])
        return det_class < 0 or det_class == track.class_id

    def _init_new_tracks(
        self,
        detections: np.ndarray,
        features: list[Optional[np.ndarray]],
        det_indices: np.ndarray,
    ) -> None:
        for det_idx in det_indices:
            det = detections[int(det_idx)]
            if float(det[4]) < self.new_track_thresh:
                continue
            mean, covariance = self._kalman_filter.initiate(_xyxy_to_xywh(det[:4]))
            feature = features[int(det_idx)]
            track = _BoTTrack(
                track_id=self._next_id,
                mean=mean,
                covariance=covariance,
                score=float(det[4]),
                class_id=int(det[5]) if det.shape[0] > 5 else -1,
                feature=_normalize_feature(feature) if feature is not None else None,
            )
            self._tracks.append(track)
            self._next_id += 1

    def _results(self) -> list[dict]:
        results: list[dict] = []
        for track in self._tracks:
            if track.hits < self.min_hits:
                continue
            results.append(
                {
                    "track_id": track.track_id,
                    "bbox": track.bbox.copy(),
                    "score": track.score,
                    "class_id": track.class_id,
                    "is_confirmed": True,
                    "feature": track.feature.copy() if track.feature is not None else None,
                    "time_since_update": track.time_since_update,
                }
            )
        return results

    def _align_features(
        self,
        embeddings: Optional[Sequence[Optional[np.ndarray]]],
        keep_indices: np.ndarray,
        detection_count: int,
    ) -> list[Optional[np.ndarray]]:
        if embeddings is None:
            return [None] * detection_count
        features: list[Optional[np.ndarray]] = [None] * detection_count
        for filtered_idx, original_idx in enumerate(keep_indices):
            if int(original_idx) >= len(embeddings):
                break
            feature = embeddings[int(original_idx)]
            if feature is not None:
                features[filtered_idx] = _normalize_feature(feature)
        return features

    def _filter_detections(self, detections: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
        if detections.size == 0:
            return np.empty((0, 6), dtype=np.float32), np.empty((0,), dtype=np.int32)
        boxes = detections[:, :4]
        scores = detections[:, 4]
        finite = np.isfinite(detections[:, :6]).all(axis=1)
        x1 = np.clip(np.minimum(boxes[:, 0], boxes[:, 2]), 0, max(1, width) - 1)
        y1 = np.clip(np.minimum(boxes[:, 1], boxes[:, 3]), 0, max(1, height) - 1)
        x2 = np.clip(np.maximum(boxes[:, 0], boxes[:, 2]), 0, max(1, width) - 1)
        y2 = np.clip(np.maximum(boxes[:, 1], boxes[:, 3]), 0, max(1, height) - 1)
        box_w = x2 - x1
        box_h = y2 - y1
        area = box_w * box_h
        aspect = np.maximum(box_w / np.maximum(box_h, 1e-6), box_h / np.maximum(box_w, 1e-6))
        valid = (
            finite
            & (scores >= 0.0)
            & (box_w >= 2.0)
            & (box_h >= 2.0)
            & (area >= self.min_box_area)
            & (aspect <= self.max_aspect_ratio)
        )
        filtered = detections[valid].copy()
        if filtered.size == 0:
            return np.empty((0, 6), dtype=np.float32), np.empty((0,), dtype=np.int32)
        filtered[:, 0] = x1[valid]
        filtered[:, 1] = y1[valid]
        filtered[:, 2] = x2[valid]
        filtered[:, 3] = y2[valid]
        return filtered, np.flatnonzero(valid).astype(np.int32)

    @staticmethod
    def _parse_img_size(img_size: Iterable[int]) -> tuple[int, int]:
        height = width = 1
        if img_size is not None:
            size_seq = list(img_size)
            if len(size_seq) >= 2:
                height, width = int(size_seq[0]), int(size_seq[1])
        return max(1, height), max(1, width)
