from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np


@dataclass
class TargetState:
    track_id: int
    bbox: np.ndarray
    score: float


class TargetController:
    def __init__(
        self,
        max_lost_frames: int = 60,
        reacquire_thresh: float = 0.6,
    ) -> None:
        self.target_id: Optional[int] = None
        self.cached_feature: Optional[np.ndarray] = None
        self._lost_frames: int = 0
        self.max_lost_frames = max(1, int(max_lost_frames))
        self.reacquire_thresh = float(reacquire_thresh)
        self.last_bbox: Optional[np.ndarray] = None

    def select_by_point(self, tracks: List[dict], point: Tuple[int, int]) -> Optional[int]:
        if point is None:
            return self.target_id
        x, y = point
        for track in tracks:
            x1, y1, x2, y2 = track["bbox"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.target_id = track["track_id"]
                feature = track.get("feature")
                if feature is not None:
                    self.cached_feature = feature.astype(np.float32, copy=True)
                self.last_bbox = track["bbox"].astype(np.float32, copy=True)
                self._lost_frames = 0
                return self.target_id
        return self.target_id

    def update_by_reid(self, track_id: int) -> None:
        self.target_id = track_id
        self._lost_frames = 0

    def maintain(self, tracks: List[dict]) -> None:
        if self.target_id is None:
            return
        for track in tracks:
            if track["track_id"] == self.target_id:
                feature = track.get("feature")
                if feature is not None:
                    self.cached_feature = feature.astype(np.float32, copy=True)
                self.last_bbox = track["bbox"].astype(np.float32, copy=True)
                self._lost_frames = 0
                return
        self._lost_frames += 1
        if self.cached_feature is None:
            if self._lost_frames >= self.max_lost_frames:
                self.reset()
            return
        best_track_id: Optional[int] = None
        best_feature: Optional[np.ndarray] = None
        best_score = 0.0
        for track in tracks:
            feature = track.get("feature")
            if feature is None:
                continue
            denom = (np.linalg.norm(self.cached_feature) * np.linalg.norm(feature)) + 1e-12
            similarity = float(np.dot(self.cached_feature, feature) / denom)
            if similarity > best_score:
                best_score = similarity
                best_track_id = track["track_id"]
                best_feature = feature
        if best_track_id is not None and best_score >= self.reacquire_thresh:
            self.target_id = best_track_id
            if best_feature is not None:
                self.cached_feature = best_feature.astype(np.float32, copy=True)
            for track in tracks:
                if track["track_id"] == best_track_id:
                    self.last_bbox = track["bbox"].astype(np.float32, copy=True)
                    break
            self._lost_frames = 0
            return
        if self._lost_frames >= self.max_lost_frames:
            self.reset()

    def current_state(self, tracks: List[dict]) -> Optional[TargetState]:
        if self.target_id is None:
            return None
        for track in tracks:
            if track["track_id"] == self.target_id:
                return TargetState(
                    track_id=self.target_id,
                    bbox=track["bbox"],
                    score=track["score"],
                )
        return None

    def reset(self) -> None:
        self.target_id = None
        self.cached_feature = None
        self._lost_frames = 0
        self.last_bbox = None

    @staticmethod
    def compute_error(state: TargetState, frame_shape: Iterable[int]) -> Tuple[float, float]:
        height, width = frame_shape[:2]
        x1, y1, x2, y2 = state.bbox
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        error_x = cx - (width * 0.5)
        error_y = cy - (height * 0.5)
        return error_x, error_y
