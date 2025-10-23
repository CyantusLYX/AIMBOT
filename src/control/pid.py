from dataclasses import dataclass
from typing import Tuple


@dataclass
class PIDGains:
    kp: float
    ki: float
    kd: float


class PIDController:
    def __init__(self, gains: PIDGains, setpoint: float = 0.0, clamp: Tuple[float, float] | None = None) -> None:
        self.gains = gains
        self.setpoint = setpoint
        self.integral = 0.0
        self.previous_error: float | None = None
        self.clamp = clamp

    def reset(self) -> None:
        self.integral = 0.0
        self.previous_error = None

    def update(self, measurement: float, dt: float) -> float:
        error = self.setpoint - measurement
        self.integral += error * dt
        derivative = 0.0
        if self.previous_error is not None and dt > 0.0:
            derivative = (error - self.previous_error) / dt
        self.previous_error = error
        output = (
            self.gains.kp * error
            + self.gains.ki * self.integral
            + self.gains.kd * derivative
        )
        if self.clamp is not None:
            lo, hi = self.clamp
            output = max(lo, min(hi, output))
        return output
