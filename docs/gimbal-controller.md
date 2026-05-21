# Gimbal Controller Notes

This document describes the ESP32 gimbal firmware and the host-side manual
control tool used to bring up and test the pan/tilt motors.

## Hardware Summary

- Controller: ESP32 Dev Module, Arduino framework through PlatformIO.
- Motor drivers: two TMC2209 drivers on a shared UART bus.
- STEP/DIR:
  - Pan: STEP `GPIO12`, DIR `GPIO14`, TMC2209 address `0`.
  - Tilt: STEP `GPIO27`, DIR `GPIO26`, TMC2209 address `1`.
- Shared enable: `GPIO13`, active low.
- TMC2209 UART: `Serial1` remapped to RX `GPIO16`, TX `GPIO17`.

GPIO12 is an ESP32 strapping pin. Keep external circuitry from pulling it to an
unsafe boot level during reset.

## Firmware Protocol

The ESP32 reads newline-terminated ASCII commands from `Serial` at `115200`
baud. Values are signed STEP pulse rates after the active microstep setting.

| Command | Meaning |
| --- | --- |
| `V:<pan>,<tilt>` | Set pan and tilt target velocity in step/s. |
| `H:1` | Hold: stop STEP pulses and keep the TMC2209 drivers enabled for holding torque. |
| `E:<0_or_1>` | Maintenance enable control for the shared TMC2209 driver enable line. `E:0` removes holding torque. |
| `S:<max_step_hz>` | Set the runtime speed clamp. Firmware clamps this to `100..80000`. |
| `M:<microsteps>` | Set both TMC2209 drivers to the requested microstep value. |

Valid microstep values are `1`, `2`, `4`, `8`, `16`, `32`, `64`, `128`, and
`256`.

Example:

```text
S:4000
M:16
H:1
V:1000,-500
H:1
```

If no valid velocity command is received for more than `500 ms`, the firmware
sets both target speeds to zero. Invalid commands do not refresh the fail-safe.

The PC brain uses `H:1` when tracking is paused or a target is lost. This keeps
basic damping/holding torque on the motors. `E:0` is reserved for maintenance
cases where holding torque must be removed.

## Manual Control Tool

The host tester lives at `tools/gimbal_manual_control.py`.

Install base dependencies:

```bash
pip install -r requirements.txt
```

List available serial ports:

```bash
python tools/gimbal_manual_control.py --list-ports
```

Run the tester:

```bash
python tools/gimbal_manual_control.py --port /dev/ttyUSB0 --max-speed 4000 --microsteps 16
```

Controls:

- Arrow keys or WASD: pan/tilt velocity.
- Joystick left stick: analog pan/tilt velocity.
- Space: stop.
- `+` / `-`: adjust max speed and send `S:<max_step_hz>`.
- `[` / `]`: adjust microsteps and send `M:<microsteps>`.
- `Q` or Esc: quit.

The tool sends velocity commands at `50 Hz` by default, which keeps the ESP32
fail-safe alive while the control window is open.

## Dependency Split

`requirements.txt` contains the base dependencies needed for control, tracking,
serial tools, and the manual gimbal tester.

CUDA/PyTorch-oriented dependencies live in `requirements-cuda.txt`. Install
them only on machines that will run YOLO/Re-ID inference:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-cuda.txt
```

Choose the PyTorch wheel index that matches the target machine's CUDA runtime.

## Safety Notes

- Send `H:1` before pausing tracking so the mechanism holds position without
  removing motor torque.
- Use `E:0` only when you intentionally need to remove holding torque for
  maintenance.
- Add a pull-up on the shared enable line so the TMC2209 drivers stay disabled
  while the ESP32 resets.
- If the gimbal relies on motor holding torque to support weight, reset or power
  loss can let the mechanism move freely.
- Start with low `--max-speed` values during bring-up, then increase only after
  verifying direction and travel limits.
