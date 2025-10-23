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
    def __init__(self) -> None:
        self.target_id: Optional[int] = None
        self.cached_feature: Optional[np.ndarray] = None

    def select_by_point(self, tracks: List[dict], point: Tuple[int, int]) -> Optional[int]:
        if point is None:
            return self.target_id
        x, y = point
        for track in tracks:
            x1, y1, x2, y2 = track["bbox"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.target_id = track["track_id"]
                return self.target_id
        return self.target_id

    def update_by_reid(self, track_id: int) -> None:
        self.target_id = track_id

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

    @staticmethod
    def compute_error(state: TargetState, frame_shape: Iterable[int]) -> Tuple[float, float]:
        height, width = frame_shape[:2]
        x1, y1, x2, y2 = state.bbox
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        error_x = cx - (width * 0.5)
        error_y = cy - (height * 0.5)
        return error_x, error_y
