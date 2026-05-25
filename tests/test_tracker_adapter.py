import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tracking.bot_sort import BoTSort
from tracking.tracker_adapter import ByteTrackAdapter, create_tracker_backend


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

    def test_factory_defaults_to_botsort(self) -> None:
        tracker = create_tracker_backend()

        self.assertIsInstance(tracker, BoTSort)

    def test_factory_rejects_cpp_requirement_for_botsort(self) -> None:
        with self.assertRaises(RuntimeError):
            create_tracker_backend(tracker_backend="botsort", require_cpp=True)

    def test_factory_keeps_bytetrack_backend_available(self) -> None:
        tracker = create_tracker_backend(tracker_backend="bytetrack", enable_reid=True, max_age=90)

        self.assertIsInstance(tracker, ByteTrackAdapter)
        self.assertEqual("python", tracker.kind)


if __name__ == "__main__":
    unittest.main()
