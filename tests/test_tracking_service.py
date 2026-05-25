import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from services.tracking_service import TrackingService


class FakeTracker:
    def __init__(self) -> None:
        self.embeddings = None
        self.img_size = None

    def update(self, detections, img_size, embeddings=None):
        self.embeddings = embeddings
        self.img_size = img_size
        return [{"track_id": 7, "bbox": detections[0, :4], "time_since_update": 0}]


class FakeReIDHelper:
    def __init__(self) -> None:
        self.target_bbox = None

    def build_embeddings(self, frame, detections, target_bbox):
        self.target_bbox = target_bbox
        return [np.ones(4, dtype=np.float32), None]


class TrackingServiceTest(unittest.TestCase):
    def test_accepts_tracker_backend_and_counts_embeddings(self) -> None:
        tracker = FakeTracker()
        reid_helper = FakeReIDHelper()
        service = TrackingService(tracker=tracker, reid_helper=reid_helper)
        frame = np.zeros((40, 60, 3), dtype=np.uint8)
        detections = np.array(
            [
                [1, 2, 10, 20, 0.9, 0],
                [3, 4, 12, 24, 0.7, 0],
            ],
            dtype=np.float32,
        )
        target_bbox = np.array([1, 2, 10, 20], dtype=np.float32)

        tracks = service.update(frame, detections, target_bbox)

        self.assertEqual(1, len(tracks))
        self.assertEqual(7, tracks[0]["track_id"])
        self.assertEqual((40, 60, 3), tracker.img_size)
        self.assertIsNotNone(tracker.embeddings)
        self.assertEqual(1, service.last_embedding_count)
        np.testing.assert_array_equal(target_bbox, reid_helper.target_bbox)


if __name__ == "__main__":
    unittest.main()
