"""Re-ID matching strategy interfaces and default implementation.

This module isolates Re-ID cost computation and assignment policy from
``ByteTrack`` so different matching strategies can be plugged in without
modifying tracker control flow.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class ReIDStrategyConfig:
    """Configuration for the default cosine Re-ID strategy.

    Attributes:
        match_thresh: Minimum cosine similarity required for a match.
        max_center_dist: Maximum absolute pixel distance between detection and
            track centres for candidate gating. A non-positive value disables
            distance gating.
    """

    match_thresh: float
    max_center_dist: float


class ReIDMatchingStrategy(Protocol):
    """Protocol for Re-ID matching strategies.

    Implementations receive unmatched track and detection candidate sets and
    return accepted match index pairs with their cosine similarities.
    """

    def match(
        self,
        track_candidates: List[Tuple[int, object]],
        det_candidates: List[Tuple[int, np.ndarray]],
        detections: np.ndarray,
    ) -> List[Tuple[int, int, float]]:
        """Match tracks to detections using Re-ID signals.

        Args:
            track_candidates: ``(local_track_idx, track_obj)`` tuples.
            det_candidates: ``(det_idx, det_feature)`` tuples.
            detections: Detection array of shape ``(N, 6+)``.

        Returns:
            List of accepted matches as ``(local_track_idx, det_idx,
            similarity)`` tuples.
        """


class CosineReIDMatchingStrategy:
    """Default cosine-similarity Re-ID matching strategy.

    Matching pipeline:
    1. Build cost matrix ``1 - cosine_similarity``.
    2. Apply class-consistency gate when class labels are available.
    3. Apply centre-distance gate using ``max_center_dist``.
    4. Solve assignment with Hungarian algorithm.
    5. Keep assignments whose similarity >= ``match_thresh``.
    """

    def __init__(self, config: ReIDStrategyConfig) -> None:
        self.config = config

    def match(
        self,
        track_candidates: List[Tuple[int, object]],
        det_candidates: List[Tuple[int, np.ndarray]],
        detections: np.ndarray,
    ) -> List[Tuple[int, int, float]]:
        if not track_candidates or not det_candidates:
            return []

        cost_matrix = np.ones((len(track_candidates), len(det_candidates)), dtype=np.float32)
        for i, (_, track) in enumerate(track_candidates):
            track_feat = getattr(track, "feature", None)
            track_bbox = getattr(track, "bbox", None)
            track_class_id = int(getattr(track, "class_id", -1))

            if track_feat is None or track_bbox is None:
                continue

            for j, (det_idx, det_feat) in enumerate(det_candidates):
                if det_feat is None:
                    continue

                det = detections[det_idx]
                det_bbox = det[:4]

                if track_class_id >= 0 and det.shape[0] > 5:
                    det_class = int(det[5])
                    if det_class != track_class_id:
                        continue

                if self.config.max_center_dist > 0:
                    cx_t = float((track_bbox[0] + track_bbox[2]) * 0.5)
                    cy_t = float((track_bbox[1] + track_bbox[3]) * 0.5)
                    cx_d = float((det_bbox[0] + det_bbox[2]) * 0.5)
                    cy_d = float((det_bbox[1] + det_bbox[3]) * 0.5)
                    if float(np.hypot(cx_t - cx_d, cy_t - cy_d)) > self.config.max_center_dist:
                        continue

                similarity = float(np.dot(track_feat, det_feat))
                cost_matrix[i, j] = 1.0 - similarity

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matches: List[Tuple[int, int, float]] = []
        for r, c in zip(row_ind, col_ind):
            similarity = 1.0 - float(cost_matrix[r, c])
            if similarity < self.config.match_thresh:
                continue
            local_track_idx, _ = track_candidates[r]
            det_idx, _ = det_candidates[c]
            matches.append((local_track_idx, det_idx, similarity))
        return matches
