import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tracking.bot_sort import BoTSort


def det(x1: float, y1: float, x2: float, y2: float, score: float = 0.9, class_id: int = 0) -> np.ndarray:
    return np.array([[x1, y1, x2, y2, score, class_id]], dtype=np.float32)


class BoTSortTest(unittest.TestCase):
    def test_hides_new_tracks_until_min_hits(self) -> None:
        tracker = BoTSort(min_hits=3, track_high_thresh=0.5, new_track_thresh=0.6)
        detection = det(10, 20, 70, 100)

        self.assertEqual([], tracker.update(detection, (120, 160, 3)))
        self.assertEqual([], tracker.update(detection, (120, 160, 3)))

        tracks = tracker.update(detection, (120, 160, 3))
        self.assertEqual(1, len(tracks))
        self.assertEqual(1, tracks[0]["track_id"])
        self.assertTrue(tracks[0]["is_confirmed"])

    def test_low_score_detection_does_not_create_new_track(self) -> None:
        tracker = BoTSort(min_hits=1, track_high_thresh=0.5, track_low_thresh=0.1, new_track_thresh=0.6)

        self.assertEqual([], tracker.update(det(10, 20, 70, 100, score=0.4), (120, 160, 3)))

    def test_reacquires_same_id_within_max_age(self) -> None:
        tracker = BoTSort(min_hits=1, max_age=2, track_high_thresh=0.5, new_track_thresh=0.6)

        first = tracker.update(det(10, 20, 70, 100), (120, 160, 3))
        self.assertEqual(1, first[0]["track_id"])
        tracker.update(np.empty((0, 6), dtype=np.float32), (120, 160, 3))

        reacquired = tracker.update(det(10, 20, 70, 100), (120, 160, 3))
        live = [track for track in reacquired if int(track["time_since_update"]) == 0]
        self.assertEqual(1, live[0]["track_id"])

    def test_reid_can_match_when_iou_distance_is_too_high(self) -> None:
        tracker = BoTSort(
            min_hits=1,
            enable_reid=True,
            match_thresh=0.2,
            proximity_thresh=1.0,
            appearance_thresh=0.25,
        )
        feature = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        first = tracker.update(det(10, 20, 70, 100), (160, 220, 3), embeddings=[feature])
        self.assertEqual(1, first[0]["track_id"])

        shifted = tracker.update(det(80, 20, 140, 100), (160, 220, 3), embeddings=[feature])
        live = [track for track in shifted if int(track["time_since_update"]) == 0]
        self.assertEqual(1, live[0]["track_id"])

    def test_lost_track_can_reenter_without_iou_when_reid_matches(self) -> None:
        tracker = BoTSort(
            min_hits=1,
            max_age=3,
            enable_reid=True,
            match_thresh=0.8,
            proximity_thresh=0.95,
            appearance_thresh=0.18,
        )
        feature = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        first = tracker.update(det(10, 20, 70, 100), (160, 240, 3), embeddings=[feature])
        self.assertEqual(1, first[0]["track_id"])

        tracker.update(np.empty((0, 6), dtype=np.float32), (160, 240, 3), embeddings=[])
        reentered = tracker.update(det(150, 20, 210, 100), (160, 240, 3), embeddings=[feature])

        live = [track for track in reentered if int(track["time_since_update"]) == 0]
        self.assertEqual(1, live[0]["track_id"])

    def test_appearance_gate_blocks_bad_reid_match(self) -> None:
        tracker = BoTSort(
            min_hits=1,
            enable_reid=True,
            match_thresh=0.2,
            proximity_thresh=1.0,
            appearance_thresh=0.25,
        )
        first_feature = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        bad_feature = np.array([-1.0, 0.0, 0.0], dtype=np.float32)

        first = tracker.update(det(10, 20, 70, 100), (160, 220, 3), embeddings=[first_feature])
        self.assertEqual(1, first[0]["track_id"])

        shifted = tracker.update(det(80, 20, 140, 100), (160, 220, 3), embeddings=[bad_feature])
        live = [track for track in shifted if int(track["time_since_update"]) == 0]
        self.assertEqual(2, live[0]["track_id"])


if __name__ == "__main__":
    unittest.main()
