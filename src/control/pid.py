from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class PIDGains:
    """PID gain coefficients.

    Attributes:
        kp: Proportional gain — scales the instantaneous error directly.
        ki: Integral gain — eliminates steady-state offset over time.
        kd: Derivative gain — dampens overshoot (empirically tuned;
            see docs/REFRACTOR_DECISIONS.md §PID Gains).
    """

    kp: float
    ki: float
    kd: float


class PIDController:
    """Discrete-time PID controller with optional output clamping.

    The setpoint defaults to ``0.0`` because both gimbal axes aim to drive
    the pixel-centre error to zero.

    Args:
        gains: Proportional, integral, and derivative coefficients.
        setpoint: Target process value. Defaults to ``0.0``.
        clamp: Optional ``(lo, hi)`` tuple to saturate the output.
    """

    def __init__(
        self,
        gains: PIDGains,
        setpoint: float = 0.0,
        clamp: Optional[Tuple[float, float]] = None,
    ) -> None:
        self.gains = gains
        self.setpoint = setpoint
        self.integral: float = 0.0
        self.previous_error: Optional[float] = None
        self.clamp = clamp

    def reset(self) -> None:
        """Reset the integral accumulator and clear the previous-error memory."""
        self.integral = 0.0
        self.previous_error = None

    def update(self, measurement: float, dt: float) -> float:
        """Compute the PID output for the current timestep.

        Args:
            measurement: Current process variable (e.g. pixel-centre error).
            dt: Elapsed seconds since the last call; must be positive.

        Returns:
            PID output, clamped to ``self.clamp`` when configured.
        """
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
