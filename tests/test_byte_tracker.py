import unittest

import numpy as np

from tracking.byte_tracker import ByteTrack


class ByteTrackStabilityTest(unittest.TestCase):
    def test_hides_new_tracks_until_min_hits(self) -> None:
        tracker = ByteTrack(track_thresh=0.5, match_iou_thresh=0.2, min_hits=3)
        det = np.array([[10, 20, 70, 100, 0.9, 0]], dtype=np.float32)

        self.assertEqual([], tracker.update(det, (120, 160, 3)))
        self.assertEqual([], tracker.update(det, (120, 160, 3)))

        tracks = tracker.update(det, (120, 160, 3))
        self.assertEqual(1, len(tracks))
        self.assertEqual(1, tracks[0]["track_id"])
        self.assertTrue(tracks[0]["is_confirmed"])

    def test_filters_tiny_one_frame_detections(self) -> None:
        tracker = ByteTrack(track_thresh=0.5, match_iou_thresh=0.2, min_hits=1, min_box_area=64)
        tiny = np.array([[10, 20, 13, 23, 0.95, 0]], dtype=np.float32)

        self.assertEqual([], tracker.update(tiny, (120, 160, 3)))


if __name__ == "__main__":
    unittest.main()
