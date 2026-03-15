"""Tracking domain service composing ByteTrack and optional Re-ID helper."""
from __future__ import annotations

from typing import Optional

import numpy as np

from pipeline.workers import ReIDHelper
from tracking.byte_tracker import ByteTrack


class TrackingService:
    """Service that produces track results from detections and frame context.

    This service centralizes the interaction between ByteTrack and Re-ID
    embedding scheduling so callers only need one update call per frame.
    """

    def __init__(self, tracker: ByteTrack, reid_helper: Optional[ReIDHelper]) -> None:
        self._tracker = tracker
        self._reid_helper = reid_helper

    def update(
        self,
        frame: np.ndarray,
        detections: np.ndarray,
        target_bbox: Optional[np.ndarray],
    ) -> list[dict]:
        """Compute current-frame tracks.

        Args:
            frame: Full-resolution BGR frame.
            detections: Detector output array of shape ``(N, 6)``.
            target_bbox: Current target bbox used to prioritize Re-ID crops.

        Returns:
            Track dictionary list from :class:`ByteTrack`.
        """
        embeddings = None
        if self._reid_helper is not None and detections.size:
            embeddings = self._reid_helper.build_embeddings(frame, detections, target_bbox)
        return self._tracker.update(detections, frame.shape, embeddings=embeddings)
