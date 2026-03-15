"""Target lifecycle management: lock, lost-frame counting, and reacquisition.

Provides :class:`TargetController` which sits between the tracker output and
the PID loop.  It maintains a primary target ID, counts lost frames, and
uses a combined Re-ID / colour / ORB score to re-acquire the target after
transient occlusions.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np


class TargetLifecycleState(str, Enum):
    """High-level lifecycle states for target tracking."""

    IDLE = "idle"
    LOCKED = "locked"
    SEARCHING = "searching"
    REACQUIRED = "reacquired"
    LOST = "lost"


@dataclass
class TargetState:
    """Immutable snapshot of the locked target for the current frame.

    Attributes:
        track_id: ByteTrack identifier of the locked target.
        bbox: Bounding box in ``(x1, y1, x2, y2)`` format (float32).
        score: Detection / track confidence score.
    """

    track_id: int
    bbox: np.ndarray
    score: float


class TargetController:
    """Manages the target lock lifecycle across frames.

    Responsibilities:
    - Click-based target selection (Mode 1).
    - Reference-image-based initial lock and reacquisition (Mode 2).
    - Lost-frame counting; automatic :meth:`reset` when the target has been
      absent for ``max_lost_frames`` consecutive frames.
    - Exposing the current :class:`TargetState` and pixel error for the PID.

    Args:
        max_lost_frames: Consecutive frames without a match before the lock
            is cleared.  The cached Re-ID feature is preserved so automatic
            reacquisition can restart immediately.
        reacquire_thresh: Minimum combined score (Re-ID + colour + ORB)
            required to (re-)acquire a target in :meth:`search_and_lock`.
        debug: If ``True``, print per-candidate fusion scores in
            :meth:`search_and_lock` for tuning.
    """

    def __init__(
        self,
        max_lost_frames: int = 60,
        reacquire_thresh: float = 0.6,
        debug: bool = False,
    ) -> None:
        self.target_ids: set[int] = set()
        self.primary_target_id: Optional[int] = None
        self.cached_feature: Optional[np.ndarray] = None
        self._lost_frames: int = 0
        self._lifecycle_state = TargetLifecycleState.IDLE
        self.max_lost_frames = max(1, int(max_lost_frames))
        self.reacquire_thresh = float(reacquire_thresh)
        self.debug = bool(debug)
        self.last_bbox: Optional[np.ndarray] = None
        self.ref_histogram: Optional[np.ndarray] = None
        # ORB is robust to rotation and lightweight enough for real-time matching.
        self.orb = cv2.ORB_create(nfeatures=500, scoreType=cv2.ORB_FAST_SCORE)
        self.ref_keypoints = None
        self.ref_descriptors = None
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    @property
    def lifecycle_state(self) -> TargetLifecycleState:
        """Return the current target lifecycle state."""
        return self._lifecycle_state

    def _set_state(self, state: TargetLifecycleState) -> None:
        self._lifecycle_state = state

    @staticmethod
    def compute_histogram(image: np.ndarray) -> Optional[np.ndarray]:
        """Compute a normalised 2-D hue-saturation histogram for *image*.

        Uses only the centre 50 % crop to reduce background influence, and
        ignores the value channel to improve robustness to illumination
        changes.

        Args:
            image: BGR uint8 image crop.

        Returns:
            Normalised ``(30, 32)`` float32 histogram, or ``None`` when the
            image is empty or an error occurs during conversion.
        """
        if image is None or image.size == 0:
            return None
        try:
            # Use the center 50% crop to suppress background influence.
            h, w = image.shape[:2]
            cy, cx = h // 2, w // 2
            h_crop, w_crop = int(h * 0.5), int(w * 0.5)
            y1 = max(0, cy - h_crop // 2)
            y2 = min(h, cy + h_crop // 2)
            x1 = max(0, cx - w_crop // 2)
            x2 = min(w, cx + w_crop // 2)
            crop = image[y1:y2, x1:x2]

            if crop.size == 0:
                return None

            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            # Use HS channels only to be less sensitive to illumination changes.
            hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
            cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            return hist
        except Exception:
            return None

    def set_reference_image(self, image: np.ndarray) -> None:
        """Pre-compute reference features from a target image crop.

        Stores both a colour histogram and ORB keypoints/descriptors which
        are later used by :meth:`search_and_lock` to score candidates.

        Args:
            image: BGR uint8 image crop of the reference target.
        """
        self.ref_histogram = self.compute_histogram(image)
        # Extract ORB features from the reference image for geometric matching.
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            self.ref_keypoints, self.ref_descriptors = self.orb.detectAndCompute(gray, None)
            if self.ref_keypoints:
                print(f"已提取參考圖片特徵點: {len(self.ref_keypoints)} 點")
        except Exception as e:
            print(f"ORB 特徵提取失敗: {e}")

    def select_by_point(self, tracks: List[dict], point: Tuple[int, int]) -> Optional[int]:
        """Lock onto the track whose bounding box contains *point*.

        Args:
            tracks: Current frame's track list from :class:`ByteTrack`.
            point: ``(x, y)`` pixel coordinate, typically from a mouse click.

        Returns:
            The ``track_id`` of the newly locked target, or the previously
            locked ``primary_target_id`` if no track contains *point*.
        """
        if point is None:
            return self.primary_target_id
        x, y = point
        for track in tracks:
            x1, y1, x2, y2 = track["bbox"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                tid = track["track_id"]
                self.target_ids = {tid}
                self.primary_target_id = tid
                feature = track.get("feature")
                if feature is not None:
                    self.cached_feature = feature.astype(np.float32, copy=True)
                self.last_bbox = track["bbox"].astype(np.float32, copy=True)
                self._lost_frames = 0
                self._set_state(TargetLifecycleState.LOCKED)
                return tid
        return self.primary_target_id

    def update_by_reid(self, track_id: int) -> None:
        """Directly lock onto *track_id* (used after an external Re-ID match).

        Args:
            track_id: The track identifier to lock onto.
        """
        self.target_ids = {track_id}
        self.primary_target_id = track_id
        self._lost_frames = 0
        self._set_state(TargetLifecycleState.REACQUIRED)

    def set_reference_feature(self, feature: np.ndarray) -> None:
        """Store an OSNet Re-ID feature as the search reference.

        Clears any existing track lock so the next call to
        :meth:`search_and_lock` starts fresh.

        Args:
            feature: L2-normalised float32 feature vector.
        """
        if feature is not None:
            self.cached_feature = feature.astype(np.float32, copy=True)
            self.target_ids = set()
            self.primary_target_id = None
            self._lost_frames = 0
            self._set_state(TargetLifecycleState.SEARCHING)

    def search_and_lock(self, tracks: List[dict], frame: Optional[np.ndarray] = None) -> bool:
        """Score all tracks and lock onto the best-matching candidate.

        The combined score is a weighted sum of:

        - **Re-ID** (weight 0.2): cosine similarity to ``cached_feature``.
        - **Colour** (weight 0.5): histogram correlation between the
          reference image and the candidate crop.
        - **ORB** (weight 0.3): ratio of matching keypoints to reference
          keypoints.

        A candidate is accepted when its final score exceeds
        ``reacquire_thresh``.

        Args:
            tracks: Current frame's track list from :class:`ByteTrack`.
            frame: Full-resolution BGR frame used to crop candidates for
                colour / ORB scoring.  Pass ``None`` to use Re-ID only.

        Returns:
            ``True`` if at least one candidate exceeded the threshold.
        """
        if self.cached_feature is None:
            if self.primary_target_id is None:
                self._set_state(TargetLifecycleState.IDLE)
            return False

        found_any = False
        best_score = 0.0
        best_id: Optional[int] = None
        best_bbox: Optional[np.ndarray] = None

        # Score every active track.
        for track in tracks:
            feature = track.get("feature")
            if feature is None:
                continue

            # 1) Re-ID score (base signal).
            denom = (np.linalg.norm(self.cached_feature) * np.linalg.norm(feature)) + 1e-12
            reid_score = float(np.dot(self.cached_feature, feature) / denom)

            # 2) Optional color and ORB cues from cropped image evidence.
            color_score = 0.0
            orb_score = 0.0
            has_color = False
            has_orb = False

            if frame is not None:
                bbox = track["bbox"].astype(int)
                x1, y1, x2, y2 = bbox
                h, w = frame.shape[:2]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                if x2 > x1 and y2 > y1:
                    crop = frame[y1:y2, x1:x2]

                    # A) Color histogram correlation.
                    if self.ref_histogram is not None:
                        track_hist = self.compute_histogram(crop)
                        if track_hist is not None:
                            color_score = cv2.compareHist(self.ref_histogram, track_hist, cv2.HISTCMP_CORREL)
                            color_score = max(0.0, color_score)
                            has_color = True

                    # B) ORB keypoint consistency.
                    if self.ref_descriptors is not None and crop.size > 0:
                        try:
                            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                            _, des = self.orb.detectAndCompute(gray_crop, None)
                            if des is not None and len(des) > 0:
                                matches = self.bf_matcher.match(self.ref_descriptors, des)
                                good_matches = [m for m in matches if m.distance < 60]
                                if len(self.ref_keypoints) > 0:
                                    orb_score = min(1.0, len(good_matches) / (len(self.ref_keypoints) * 0.2 + 1e-5))
                                    has_orb = True
                        except Exception:
                            pass

            # 3) Weighted fusion.
            final_score = 0.0
            weights = 0.0

            # Re-ID (base)
            final_score += 0.2 * reid_score
            weights += 0.2

            # Color
            if has_color:
                final_score += 0.5 * color_score
                weights += 0.5

            # ORB detail score
            if has_orb:
                final_score += 0.3 * orb_score
                weights += 0.3

            if weights > 0:
                final_score /= weights

            # Debug visibility for promising candidates.
            if self.debug and final_score > 0.4:
                print(f"ID {track['track_id']}: ReID={reid_score:.2f}, Color={color_score:.2f}, ORB={orb_score:.2f}, Final={final_score:.2f}")

            # Accept candidate if threshold is met.
            if final_score >= self.reacquire_thresh:
                self.target_ids.add(track["track_id"])
                found_any = True
                if final_score > best_score:
                    best_score = final_score
                    best_id = track["track_id"]
                    best_bbox = track["bbox"].astype(np.float32, copy=True)

        if found_any:
            was_tracking = self.primary_target_id is not None
            if self.primary_target_id is None or (best_id is not None and best_score > 0.6):
                if best_id is not None:
                    self.primary_target_id = best_id

            if best_bbox is not None and self.primary_target_id is not None:
                self.last_bbox = best_bbox

            self._lost_frames = 0
            if was_tracking:
                self._set_state(TargetLifecycleState.LOCKED)
            else:
                self._set_state(TargetLifecycleState.REACQUIRED)
            return True

        if self.primary_target_id is None:
            self._set_state(TargetLifecycleState.SEARCHING)
        return False

    def maintain(self, tracks: List[dict]) -> None:
        """Update lost-frame counter and clear stale locks.

        Called every frame after :meth:`search_and_lock` / track update.
        Increments the lost-frame counter when the primary target is absent;
        calls :meth:`reset` when the counter reaches ``max_lost_frames``.

        Args:
            tracks: Current frame's track list from :class:`ByteTrack`.
        """
        if not self.target_ids and self.primary_target_id is None:
            if self.cached_feature is not None:
                self._set_state(TargetLifecycleState.SEARCHING)
            else:
                self._set_state(TargetLifecycleState.IDLE)
            return

        current_track_ids = {t["track_id"] for t in tracks}
        self.target_ids &= current_track_ids
        if self.primary_target_id is not None:
            self.target_ids.add(self.primary_target_id)

        primary_found = False
        for track in tracks:
            tid = track["track_id"]
            if tid == self.primary_target_id:
                primary_found = True
                self.last_bbox = track["bbox"].astype(np.float32, copy=True)
                break

        if primary_found:
            self._lost_frames = 0
            if self._lifecycle_state in (TargetLifecycleState.LOST, TargetLifecycleState.REACQUIRED):
                self._set_state(TargetLifecycleState.LOCKED)
        else:
            self._lost_frames += 1
            if self.primary_target_id is not None:
                self._set_state(TargetLifecycleState.LOST)

        if self._lost_frames >= self.max_lost_frames:
            self.reset()

    @property
    def target_id(self) -> Optional[int]:
        return self.primary_target_id

    @target_id.setter
    def target_id(self, value: Optional[int]) -> None:
        self.primary_target_id = value
        if value is None:
            self.target_ids = set()
            if self.cached_feature is not None:
                self._set_state(TargetLifecycleState.SEARCHING)
            else:
                self._set_state(TargetLifecycleState.IDLE)
        else:
            self.target_ids = {value}
            self._set_state(TargetLifecycleState.LOCKED)

    def current_state(self, tracks: List[dict]) -> Optional[TargetState]:
        """Return the :class:`TargetState` for the current primary target.

        Args:
            tracks: Current frame's track list from :class:`ByteTrack`.

        Returns:
            A :class:`TargetState` snapshot if the primary target is
            present in *tracks*, otherwise ``None``.
        """
        if self.primary_target_id is None:
            return None
        for track in tracks:
            if track["track_id"] == self.primary_target_id:
                return TargetState(
                    track_id=self.primary_target_id,
                    bbox=track["bbox"],
                    score=track["score"],
                )
        return None

    def reset(self) -> None:
        """Clear the track lock while preserving the reference features.

        After ``reset()``, :meth:`search_and_lock` immediately becomes
        active again using the preserved ``cached_feature``,
        ``ref_histogram``, and ORB descriptors to re-acquire the target.
        """
        self.primary_target_id = None
        self.target_ids = set()
        self._lost_frames = 0
        self.last_bbox = None
        if self.cached_feature is not None:
            self._set_state(TargetLifecycleState.SEARCHING)
        else:
            self._set_state(TargetLifecycleState.IDLE)

    def clear_reference(self) -> None:
        """Fully clear all state including the reference features.

        Call this only when switching to a new reference target; it is more
        destructive than :meth:`reset`.
        """
        self.reset()
        self.cached_feature = None
        self.ref_histogram = None
        self.ref_keypoints = None
        self.ref_descriptors = None
        self._set_state(TargetLifecycleState.IDLE)

    @staticmethod
    def compute_error(state: TargetState, frame_shape: Iterable[int]) -> Tuple[float, float]:
        """Compute the pixel error from the frame centre to the target centre.

        Args:
            state: Current :class:`TargetState` snapshot.
            frame_shape: Frame dimensions as ``(height, width, ...)``.  Only
                the first two elements are used.

        Returns:
            ``(error_x, error_y)`` in pixels.  Positive ``error_x`` means the
            target is to the right of centre; positive ``error_y`` means it
            is below centre.
        """
        height, width = frame_shape[:2]
        x1, y1, x2, y2 = state.bbox
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        error_x = cx - (width * 0.5)
        error_y = cy - (height * 0.5)
        return error_x, error_y
