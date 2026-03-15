"""Simplified multi-object tracker based on IoU and optional Re-ID matching.

Implements a two-stage assignment strategy inspired by ByteTrack:

1. **IoU matching** — high-confidence detections are matched to existing
   tracks using the Hungarian algorithm on an IoU cost matrix.
2. **Re-ID matching** — unmatched tracks and detections are matched using
   cosine similarity of OSNet feature vectors, subject to a maximum centre
   distance gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from tracking.reid_strategy import (
    CosineReIDMatchingStrategy,
    ReIDMatchingStrategy,
    ReIDStrategyConfig,
)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the intersection-over-union between two axis-aligned boxes.

    Args:
        a: Box in ``(x1, y1, x2, y2)`` format.
        b: Box in ``(x1, y1, x2, y2)`` format.

    Returns:
        IoU score in ``[0.0, 1.0]``.
    """
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
    """Build a pairwise IoU matrix between *tracks* and *detections*.

    Args:
        tracks: Active track objects.
        detections: Detection array of shape ``(M, 6+)``.

    Returns:
        Float32 array of shape ``(len(tracks), M)`` where entry ``[i, j]``
        is the IoU score between track ``i`` and detection ``j``.
    """
    if not tracks or len(detections) == 0:
        return np.zeros((len(tracks), len(detections)), dtype=np.float32)
    matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    for i, track in enumerate(tracks):
        for j, det in enumerate(detections):
            matrix[i, j] = _iou(track.bbox, det[:4])
    return matrix


@dataclass
class _Track:
    """Internal per-track state maintained by :class:`ByteTrack`.

    Attributes:
        track_id: Unique integer identifier assigned at track creation.
        bbox: Latest bounding-box in ``(x1, y1, x2, y2)`` format (float32).
        score: Detection confidence at the last successful match.
        class_id: Detector class index (``-1`` if unavailable).
        hits: Total number of successful matches since creation.
        time_since_update: Frames elapsed since the last match.
        feature: EMA-smoothed L2-normalised Re-ID embedding, or ``None``.
    """

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
        """Update the track with a new matched detection.

        Args:
            bbox: New bounding-box ``(x1, y1, x2, y2)``.
            score: Detection confidence.
            class_id: Detector class index.
            feature: L2-normalised Re-ID embedding for this detection, or
                ``None`` if Re-ID is disabled or extraction failed.
            momentum: EMA momentum for feature blending (``[0, 1)``).
            min_similarity: Minimum cosine similarity required to accept an
                incoming feature into the EMA.
            similarity: Cosine similarity between the incoming feature and the
                stored feature; used to gate the EMA update.
        """
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
        """Increment ``time_since_update`` by one (called every frame)."""
        self.time_since_update += 1


class ByteTrack:
    """Simplified multi-object tracker based on IoU and optional Re-ID matching.

    Args:
        track_thresh: Minimum detection confidence to create a new track.
        match_iou_thresh: Minimum IoU required for an IoU-stage match.
        max_age: Frames a track may survive without a match before removal.
        min_hits: Minimum successful matches before a track is "confirmed".
        enable_reid: Whether to use Re-ID embeddings in the second matching
            stage.
        reid_match_thresh: Minimum cosine similarity for a Re-ID match.
        feature_momentum: EMA momentum for track feature updates.
        feature_min_similarity: Minimum similarity to accept an incoming
            feature into the EMA.
        reid_max_center_dist: Maximum normalised centre distance (relative to
            the frame diagonal) allowed for a Re-ID match candidate.
        reid_strategy: Optional pluggable Re-ID matching strategy. Defaults
            to :class:`CosineReIDMatchingStrategy`.
    """

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
        reid_strategy: Optional[ReIDMatchingStrategy] = None,
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
        self.reid_strategy: ReIDMatchingStrategy = reid_strategy or CosineReIDMatchingStrategy(
            ReIDStrategyConfig(
                match_thresh=self.reid_match_thresh,
                max_center_dist=self.reid_max_center_dist,
            )
        )
        self._next_id = 1
        self._tracks: List[_Track] = []

    def update(
        self,
        detections: Optional[np.ndarray],
        img_size: Iterable[int],
        embeddings: Optional[Sequence[Optional[np.ndarray]]] = None,
    ) -> List[dict]:
        """Run one tracking step given new detections.

        Args:
            detections: Array of shape ``(N, 6)`` — columns are
                ``x1, y1, x2, y2, confidence, class_id``.  Pass ``None``
                or an empty array for frames with no detections.
            img_size: ``(height, width)`` of the frame; used to normalise
                positions for the spatial gate in Re-ID matching.
            embeddings: Per-detection Re-ID feature list of length ``N``.
                Each element is either a float32 feature vector or ``None``.
                Pass ``None`` to disable Re-ID for this step.

        Returns:
            List of track dicts with keys: ``track_id``, ``bbox``,
            ``score``, ``class_id``, ``is_confirmed``, ``feature``,
            ``time_since_update``.
        """
        height = width = 1
        if img_size is not None:
            size_seq = list(img_size)
            if len(size_seq) >= 2:
                height, width = int(size_seq[0]), int(size_seq[1])
        diag = float(np.hypot(width, height)) + 1e-6
        max_center_dist = self.reid_max_center_dist * diag
        if isinstance(self.reid_strategy, CosineReIDMatchingStrategy):
            self.reid_strategy = CosineReIDMatchingStrategy(
                ReIDStrategyConfig(
                    match_thresh=self.reid_match_thresh,
                    max_center_dist=max_center_dist,
                )
            )
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
                strategy_matches = self.reid_strategy.match(track_candidates, det_candidates, dets)
                for track_idx, det_idx, similarity in strategy_matches:
                    track_obj = confirmed_tracks[track_idx]
                    det_feat = features[det_idx]
                    if det_feat is None:
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
        """Convert a list of track dicts to a dense ``(N, 6)`` array.

        Args:
            tracks: Track dicts as returned by :meth:`update`.

        Returns:
            Float32 array of shape ``(N, 6)`` with columns
            ``x1, y1, x2, y2, score, track_id``.  Returns an empty
            ``(0, 6)`` array when *tracks* is empty.
        """
        if not tracks:
            return np.empty((0, 6), dtype=np.float32)
        rows = []
        for track in tracks:
            x1, y1, x2, y2 = track["bbox"]
            rows.append([x1, y1, x2, y2, track["score"], track["track_id"]])
        return np.array(rows, dtype=np.float32)
