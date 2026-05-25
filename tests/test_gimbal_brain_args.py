import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from app.gimbal_brain_pc import parse_args


class GimbalBrainArgsTest(unittest.TestCase):
    def test_control_mode_defaults_to_pid(self) -> None:
        with patch.object(sys, "argv", ["gimbal_brain_pc.py"]):
            args = parse_args()

        self.assertEqual("pid", args.control_mode)
        self.assertEqual(25.0, args.pid_deadband)
        self.assertEqual(250.0, args.pid_min_speed)
        self.assertEqual(15.0, args.pid_kp_pan)
        self.assertEqual(0.0, args.pid_ki_pan)
        self.assertEqual(0.8, args.pid_kd_pan)
        self.assertEqual(10.0, args.pid_kp_tilt)
        self.assertEqual(0.0, args.pid_ki_tilt)
        self.assertEqual(0.8, args.pid_kd_tilt)
        self.assertEqual(20000.0, args.max_speed)

    def test_accepts_p_control_mode(self) -> None:
        with patch.object(sys, "argv", ["gimbal_brain_pc.py", "--control-mode", "p"]):
            args = parse_args()

        self.assertEqual("p", args.control_mode)


if __name__ == "__main__":
    unittest.main()
