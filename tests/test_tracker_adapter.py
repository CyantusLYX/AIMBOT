import unittest

from tracking.tracker_adapter import ByteTrackAdapter


class ByteTrackAdapterTest(unittest.TestCase):
    def test_reid_uses_python_tracker(self) -> None:
        tracker = ByteTrackAdapter(enable_reid=True)

        self.assertEqual("python", tracker.kind)

    def test_reid_rejects_required_cpp_tracker(self) -> None:
        with self.assertRaises(RuntimeError):
            ByteTrackAdapter(enable_reid=True, require_cpp=True)

    def test_reid_max_age_is_applied_to_python_tracker(self) -> None:
        tracker = ByteTrackAdapter(enable_reid=True, max_age=90)

        self.assertEqual(90, tracker._impl.max_age)  # pylint: disable=protected-access


if __name__ == "__main__":
    unittest.main()
