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


class VelocityPIDAxisController:
    """Velocity-oriented PID controller for one gimbal axis.

    Unlike :class:`PIDController`, this controller accepts pixel error directly
    so the output sign matches the current proportional controller used by the
    PC brain: positive error produces positive velocity.
    """

    def __init__(
        self,
        gains: PIDGains,
        *,
        deadband: float = 0.0,
        min_speed: float = 0.0,
        output_limit: float = 0.0,
        integral_limit: float = 0.0,
        derivative_alpha: float = 0.25,
        output_slew_rate: float = 0.0,
        max_dt: float = 0.25,
        fallback_dt: float = 1.0 / 30.0,
    ) -> None:
        self.gains = gains
        self.deadband = abs(float(deadband))
        self.min_speed = abs(float(min_speed))
        self.output_limit = abs(float(output_limit))
        self.integral_limit = abs(float(integral_limit))
        self.derivative_alpha = max(0.0, min(1.0, float(derivative_alpha)))
        self.output_slew_rate = max(0.0, float(output_slew_rate))
        self.max_dt = max(float(fallback_dt), float(max_dt))
        self.fallback_dt = max(1e-6, float(fallback_dt))
        self.integral = 0.0
        self.previous_error: Optional[float] = None
        self.filtered_derivative = 0.0
        self.previous_output = 0.0

    def reset(self) -> None:
        self.integral = 0.0
        self.previous_error = None
        self.filtered_derivative = 0.0
        self.previous_output = 0.0

    def update(self, error: float, dt: float) -> float:
        error = float(error)
        if abs(error) <= self.deadband:
            self.reset()
            return 0.0

        effective_dt = float(dt)
        derivative = 0.0
        valid_dt = 0.0 < effective_dt <= self.max_dt
        if not valid_dt:
            effective_dt = self.fallback_dt
            self.previous_error = error
            self.filtered_derivative = 0.0
        elif self.previous_error is not None:
            raw_derivative = (error - self.previous_error) / effective_dt
            self.filtered_derivative = (
                self.derivative_alpha * raw_derivative
                + (1.0 - self.derivative_alpha) * self.filtered_derivative
            )
            derivative = self.filtered_derivative
        self.previous_error = error

        self.integral += error * effective_dt
        if self.integral_limit > 0.0:
            self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral))

        output = (
            self.gains.kp * error
            + self.gains.ki * self.integral
            + self.gains.kd * derivative
        )
        if output != 0.0 and self.min_speed > 0.0 and abs(output) < self.min_speed:
            output = self.min_speed if output > 0.0 else -self.min_speed
        if self.output_limit > 0.0:
            output = max(-self.output_limit, min(self.output_limit, output))
        if self.output_slew_rate > 0.0:
            max_delta = self.output_slew_rate * effective_dt
            delta = max(-max_delta, min(max_delta, output - self.previous_output))
            output = self.previous_output + delta
        self.previous_output = output
        return output
