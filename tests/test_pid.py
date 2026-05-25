import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from control.pid import PIDGains, VelocityPIDAxisController


class VelocityPIDAxisControllerTest(unittest.TestCase):
    def test_preserves_error_sign(self) -> None:
        controller = VelocityPIDAxisController(PIDGains(kp=2.0, ki=0.0, kd=0.0))

        self.assertEqual(20.0, controller.update(10.0, 0.02))
        self.assertEqual(-10.0, controller.update(-5.0, 0.02))

    def test_deadband_outputs_zero_and_resets_state(self) -> None:
        controller = VelocityPIDAxisController(
            PIDGains(kp=1.0, ki=10.0, kd=0.0),
            deadband=5.0,
            integral_limit=100.0,
        )

        self.assertGreater(controller.update(10.0, 1.0), 0.0)
        self.assertNotEqual(0.0, controller.integral)
        self.assertEqual(0.0, controller.update(2.0, 0.02))
        self.assertEqual(0.0, controller.integral)
        self.assertIsNone(controller.previous_error)

    def test_min_speed_boosts_small_outputs(self) -> None:
        controller = VelocityPIDAxisController(
            PIDGains(kp=1.0, ki=0.0, kd=0.0),
            deadband=1.0,
            min_speed=10.0,
        )

        self.assertEqual(10.0, controller.update(3.0, 0.02))
        self.assertEqual(-10.0, controller.update(-3.0, 0.02))

    def test_integral_is_clamped(self) -> None:
        controller = VelocityPIDAxisController(
            PIDGains(kp=0.0, ki=1.0, kd=0.0),
            integral_limit=5.0,
            max_dt=2.0,
        )

        self.assertEqual(5.0, controller.update(100.0, 1.0))
        self.assertEqual(5.0, controller.integral)

    def test_output_is_clamped(self) -> None:
        controller = VelocityPIDAxisController(
            PIDGains(kp=1000.0, ki=0.0, kd=0.0),
            output_limit=100.0,
        )

        self.assertEqual(100.0, controller.update(1.0, 0.02))
        self.assertEqual(-100.0, controller.update(-1.0, 0.02))

    def test_slew_rate_limits_command_jumps(self) -> None:
        controller = VelocityPIDAxisController(
            PIDGains(kp=1000.0, ki=0.0, kd=0.0),
            output_limit=1000.0,
            output_slew_rate=100.0,
        )

        self.assertAlmostEqual(10.0, controller.update(1.0, 0.1))
        self.assertAlmostEqual(20.0, controller.update(1.0, 0.1))

    def test_large_dt_skips_derivative_spike(self) -> None:
        controller = VelocityPIDAxisController(
            PIDGains(kp=1.0, ki=0.0, kd=1000.0),
            max_dt=0.25,
            fallback_dt=0.1,
        )

        self.assertEqual(10.0, controller.update(10.0, 0.1))
        self.assertEqual(20.0, controller.update(20.0, 1.0))


if __name__ == "__main__":
    unittest.main()
