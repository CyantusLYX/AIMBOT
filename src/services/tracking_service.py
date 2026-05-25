"""Tracking domain service composing ByteTrack and optional Re-ID helper."""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional, Protocol

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from pipeline.workers import ReIDHelper


class TrackerBackend(Protocol):
    """Minimal tracker contract consumed by :class:`TrackingService`."""

    def update(
        self,
        detections: Optional[np.ndarray],
        img_size: Iterable[int],
        embeddings: Optional[list[Optional[np.ndarray]]] = None,
    ) -> list[dict]:
        """Update tracker state and return current tracks."""


class TrackingService:
    """Service that produces track results from detections and frame context.

    This service centralizes the interaction between ByteTrack and Re-ID
    embedding scheduling so callers only need one update call per frame.
    """

    def __init__(self, tracker: TrackerBackend, reid_helper: Optional["ReIDHelper"]) -> None:
        self._tracker = tracker
        self._reid_helper = reid_helper
        self.last_embedding_count = 0

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
            Track dictionary list from the configured tracker backend.
        """
        embeddings = None
        self.last_embedding_count = 0
        if self._reid_helper is not None and detections.size:
            embeddings = self._reid_helper.build_embeddings(frame, detections, target_bbox)
            self.last_embedding_count = sum(1 for feature in embeddings or [] if feature is not None)
        return self._tracker.update(detections, frame.shape, embeddings=embeddings)
