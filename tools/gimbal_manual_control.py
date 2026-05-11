#!/usr/bin/env python3
"""Manual gimbal tester for the AIMBOT ESP32 velocity firmware.

Controls:
  Arrow keys / WASD: pan and tilt velocity
  Joystick left stick: pan on X axis, tilt on Y axis
  Space: emergency stop while held/pressed
  +/-: adjust max speed
  [ / ]: adjust TMC2209 microsteps
  R: request ESP32/TMC status
  Q or Esc: quit
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from typing import Optional

pygame = None
serial = None
list_ports = None


DEFAULT_BAUD = 115200
DEFAULT_MAX_SPEED = 4000
DEFAULT_SEND_HZ = 50
DEFAULT_KEY_RAMP = 12000.0
DEFAULT_JOYSTICK_DEADZONE = 0.12
DEFAULT_JOYSTICK_EXPO = 1.6
DEFAULT_MICROSTEPS = 16
MICROSTEP_OPTIONS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
MIN_MAX_SPEED = 100
HARD_MAX_SPEED = 80000
WINDOW_SIZE = (820, 380)


@dataclass
class CommandState:
    pan: float = 0.0
    tilt: float = 0.0
    max_speed: int = DEFAULT_MAX_SPEED
    microsteps: int = DEFAULT_MICROSTEPS
    pan_scale: float = 1.0
    tilt_scale: float = 1.0


class GimbalSerial:
    def __init__(
        self,
        port: Optional[str],
        baud: int,
        timeout: float,
        dry_run: bool,
    ) -> None:
        self.dry_run = dry_run
        self.serial = None
        self.rx_buffer = ""
        if not dry_run:
            load_serial()
            if not port:
                raise ValueError("Serial port is required unless --dry-run is used.")
            self.serial = serial.Serial(port=port, baudrate=baud, timeout=timeout)
            time.sleep(0.25)

    def send_velocity(self, pan_steps_s: int, tilt_steps_s: int) -> None:
        self._write_line(f"V:{pan_steps_s},{tilt_steps_s}")

    def send_max_speed(self, max_speed_steps_s: int) -> None:
        self._write_line(f"S:{max_speed_steps_s}")

    def send_microsteps(self, microsteps: int) -> None:
        self._write_line(f"M:{microsteps}")

    def request_status(self) -> None:
        self._write_line("?")

    def read_available_lines(self) -> list[str]:
        if self.dry_run or self.serial is None:
            return []

        waiting = self.serial.in_waiting
        if waiting <= 0:
            return []

        chunk = self.serial.read(waiting).decode("utf-8", errors="replace")
        self.rx_buffer += chunk
        lines: list[str] = []
        while "\n" in self.rx_buffer:
            line, self.rx_buffer = self.rx_buffer.split("\n", 1)
            line = line.strip()
            if line:
                lines.append(line)
        return lines

    def _write_line(self, line: str) -> None:
        message = f"{line}\n"
        if self.dry_run or self.serial is None:
            print(message.strip())
            return

        self.serial.write(message.encode("ascii"))
        self.serial.flush()

    def close(self) -> None:
        if self.serial is not None and self.serial.is_open:
            try:
                self.send_velocity(0, 0)
            finally:
                self.serial.close()


def list_serial_ports() -> None:
    load_serial()
    ports = list(list_ports.comports())
    if not ports:
        print("No serial ports found.")
        return

    for port in ports:
        description = f" - {port.description}" if port.description else ""
        print(f"{port.device}{description}")


def load_pygame() -> None:
    global pygame
    try:
        import pygame as pygame_module
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Missing pygame. Install it with: pip install pygame") from exc

    pygame = pygame_module


def load_serial() -> None:
    global serial, list_ports
    try:
        import serial as serial_module
        from serial.tools import list_ports as list_ports_module
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Missing pyserial. Install it with: pip install pyserial") from exc

    serial = serial_module
    list_ports = list_ports_module


def clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def clamp_max_speed(value: int) -> int:
    return max(MIN_MAX_SPEED, min(HARD_MAX_SPEED, value))


def normalize_microsteps(value: int) -> int:
    if value in MICROSTEP_OPTIONS:
        return value

    return min(MICROSTEP_OPTIONS, key=lambda option: abs(option - value))


def cycle_microsteps(current: int, direction: int) -> int:
    index = MICROSTEP_OPTIONS.index(normalize_microsteps(current))
    next_index = max(0, min(len(MICROSTEP_OPTIONS) - 1, index + direction))
    return MICROSTEP_OPTIONS[next_index]


def apply_deadzone(value: float, deadzone: float, expo: float) -> float:
    magnitude = abs(value)
    if magnitude <= deadzone:
        return 0.0

    scaled = (magnitude - deadzone) / (1.0 - deadzone)
    curved = math.pow(scaled, expo)
    return math.copysign(curved, value)


def ramp_toward(current: float, target: float, max_delta: float) -> float:
    delta = target - current
    if abs(delta) <= max_delta:
        return target
    return current + math.copysign(max_delta, delta)


def read_keyboard_target(max_speed: int) -> tuple[float, float]:
    keys = pygame.key.get_pressed()

    pan_axis = 0
    if keys[pygame.K_LEFT] or keys[pygame.K_a]:
        pan_axis -= 1
    if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        pan_axis += 1

    tilt_axis = 0
    if keys[pygame.K_UP] or keys[pygame.K_w]:
        tilt_axis += 1
    if keys[pygame.K_DOWN] or keys[pygame.K_s]:
        tilt_axis -= 1

    return pan_axis * max_speed, tilt_axis * max_speed


def apply_axis_scale(target: tuple[float, float], state: CommandState) -> tuple[float, float]:
    pan = clamp(target[0] * state.pan_scale, state.max_speed)
    tilt = clamp(target[1] * state.tilt_scale, state.max_speed)
    return pan, tilt


def read_joystick_target(
    joystick: Optional[pygame.joystick.Joystick],
    max_speed: int,
    deadzone: float,
    expo: float,
    invert_tilt: bool,
) -> tuple[float, float]:
    if joystick is None:
        return 0.0, 0.0

    pan_axis = apply_deadzone(joystick.get_axis(0), deadzone, expo)
    tilt_axis = apply_deadzone(joystick.get_axis(1), deadzone, expo)
    if invert_tilt:
        tilt_axis = -tilt_axis

    return pan_axis * max_speed, tilt_axis * max_speed


def choose_command_target(
    keyboard_target: tuple[float, float],
    joystick_target: tuple[float, float],
) -> tuple[float, float]:
    joystick_active = abs(joystick_target[0]) > 0 or abs(joystick_target[1]) > 0
    if joystick_active:
        return joystick_target
    return keyboard_target


def draw_status(
    screen: pygame.Surface,
    font: pygame.font.Font,
    state: CommandState,
    joystick_name: str,
    dry_run: bool,
    port: Optional[str],
    last_messages: list[str],
) -> None:
    screen.fill((24, 27, 31))

    lines = [
        "AIMBOT Gimbal Manual Control",
        f"Serial: {'dry-run' if dry_run else port}",
        f"Joystick: {joystick_name}",
        f"Pan: {int(round(state.pan)):>6} step/s",
        f"Tilt: {int(round(state.tilt)):>5} step/s",
        f"Max speed: {state.max_speed} step/s",
        f"Microsteps: {state.microsteps}x",
        f"Axis scale: pan {state.pan_scale:.2f}x / tilt {state.tilt_scale:.2f}x",
        "Arrows/WASD/left stick move | Space stop | +/- speed | [/] microsteps | R status | Q/Esc quit",
    ]

    y = 24
    for index, line in enumerate(lines):
        color = (235, 239, 244) if index == 0 else (190, 199, 208)
        surface = font.render(line, True, color)
        screen.blit(surface, (28, y))
        y += 34 if index == 0 else 30

    y += 6
    for message in last_messages[-4:]:
        surface = font.render(message[-92:], True, (132, 207, 185))
        screen.blit(surface, (28, y))
        y += 24

    pygame.display.flip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Control/test the ESP32 gimbal with joystick or arrow keys."
    )
    parser.add_argument("--port", help="Serial port connected to ESP32, e.g. /dev/ttyUSB0.")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--dry-run", action="store_true", help="Print commands instead of opening serial.")
    parser.add_argument("--list-ports", action="store_true", help="List serial ports and exit.")
    parser.add_argument("--max-speed", type=int, default=DEFAULT_MAX_SPEED, help="Initial max speed in step/s.")
    parser.add_argument("--microsteps", type=int, default=DEFAULT_MICROSTEPS, help="Initial TMC2209 microsteps.")
    parser.add_argument("--pan-scale", type=float, default=1.0, help="Host-side pan velocity multiplier.")
    parser.add_argument("--tilt-scale", type=float, default=1.0, help="Host-side tilt velocity multiplier.")
    parser.add_argument("--send-hz", type=float, default=DEFAULT_SEND_HZ, help="Command send rate.")
    parser.add_argument("--key-ramp", type=float, default=DEFAULT_KEY_RAMP, help="Keyboard ramp in step/s^2.")
    parser.add_argument("--deadzone", type=float, default=DEFAULT_JOYSTICK_DEADZONE)
    parser.add_argument("--expo", type=float, default=DEFAULT_JOYSTICK_EXPO, help="Joystick response curve.")
    parser.add_argument(
        "--no-invert-tilt",
        action="store_true",
        help="Do not invert joystick Y axis. Default: stick up is positive tilt.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_ports:
        list_serial_ports()
        return 0

    if args.send_hz <= 0:
        raise SystemExit("--send-hz must be greater than 0.")

    load_pygame()
    pygame.init()
    pygame.joystick.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("AIMBOT Gimbal Manual Control")
    font = pygame.font.Font(None, 26)
    clock = pygame.time.Clock()

    joystick: Optional[pygame.joystick.Joystick] = None
    joystick_name = "none"
    if pygame.joystick.get_count() > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        joystick_name = joystick.get_name()

    state = CommandState(
        max_speed=clamp_max_speed(args.max_speed),
        microsteps=normalize_microsteps(args.microsteps),
        pan_scale=max(0.05, args.pan_scale),
        tilt_scale=max(0.05, args.tilt_scale),
    )
    gimbal = GimbalSerial(args.port, args.baud, timeout=0.05, dry_run=args.dry_run)
    gimbal.send_max_speed(state.max_speed)
    gimbal.send_microsteps(state.microsteps)
    gimbal.request_status()
    send_interval_s = 1.0 / args.send_hz
    last_send = 0.0
    last_frame = time.monotonic()
    last_messages: list[str] = []

    try:
        running = True
        while running:
            now = time.monotonic()
            dt = max(0.001, now - last_frame)
            last_frame = now

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key in (pygame.K_SPACE,):
                        state.pan = 0.0
                        state.tilt = 0.0
                    elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                        state.max_speed = clamp_max_speed(state.max_speed + 500)
                        gimbal.send_max_speed(state.max_speed)
                        gimbal.request_status()
                    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        state.max_speed = clamp_max_speed(state.max_speed - 500)
                        gimbal.send_max_speed(state.max_speed)
                        gimbal.request_status()
                    elif event.key == pygame.K_LEFTBRACKET:
                        state.microsteps = cycle_microsteps(state.microsteps, -1)
                        state.pan = 0.0
                        state.tilt = 0.0
                        gimbal.send_velocity(0, 0)
                        gimbal.send_microsteps(state.microsteps)
                        gimbal.request_status()
                    elif event.key == pygame.K_RIGHTBRACKET:
                        state.microsteps = cycle_microsteps(state.microsteps, 1)
                        state.pan = 0.0
                        state.tilt = 0.0
                        gimbal.send_velocity(0, 0)
                        gimbal.send_microsteps(state.microsteps)
                        gimbal.request_status()
                    elif event.key == pygame.K_r:
                        gimbal.request_status()
                elif event.type == pygame.JOYDEVICEADDED and joystick is None:
                    joystick = pygame.joystick.Joystick(event.device_index)
                    joystick.init()
                    joystick_name = joystick.get_name()
                elif event.type == pygame.JOYDEVICEREMOVED:
                    joystick = None
                    joystick_name = "none"

            keys = pygame.key.get_pressed()
            if keys[pygame.K_SPACE]:
                target_pan, target_tilt = 0.0, 0.0
            else:
                target_pan, target_tilt = choose_command_target(
                    read_keyboard_target(state.max_speed),
                    read_joystick_target(
                        joystick,
                        state.max_speed,
                        args.deadzone,
                        args.expo,
                        invert_tilt=not args.no_invert_tilt,
                    ),
                )
                target_pan, target_tilt = apply_axis_scale((target_pan, target_tilt), state)

            max_delta = args.key_ramp * dt
            state.pan = clamp(ramp_toward(state.pan, target_pan, max_delta), state.max_speed)
            state.tilt = clamp(ramp_toward(state.tilt, target_tilt, max_delta), state.max_speed)

            if now - last_send >= send_interval_s:
                gimbal.send_velocity(int(round(state.pan)), int(round(state.tilt)))
                last_send = now

            last_messages.extend(gimbal.read_available_lines())
            del last_messages[:-8]

            draw_status(screen, font, state, joystick_name, args.dry_run, args.port, last_messages)
            clock.tick(60)
    finally:
        gimbal.close()
        pygame.quit()

    return 0


if __name__ == "__main__":
    sys.exit(main())
