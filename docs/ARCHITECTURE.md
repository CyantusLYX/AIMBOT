# AIMBOT — Architecture Reference

> **Last updated**: 2026-05-23
> **Status**: Living document — update whenever a module boundary changes.

---

## System Overview

The repository now has two host-side application shapes:

- **Local pipeline**: reads from a local camera/video source and renders through
  `OpenCVViewer`.
- **Distributed PC brain**: receives Android video/IMU packets over UDP, renders
  an operator panel, and drives the ESP32 gimbal through the ASCII protocol.

Application assembly belongs in `src/app`. Files in `scripts/` should be thin
CLI compatibility wrappers only. The distributed PC brain implementation now
lives in `src/app/gimbal_brain_pc.py`; `scripts/gimbal_brain_pc.py` imports and
calls that app module for backwards-compatible command invocation.

```
Video Source / Camera
        │
        ▼
 FramePrefetcher          (pipeline/workers.py)
  dedicated read thread
        │  raw BGR frames
        ▼
 GpuPreprocessor           (pipeline/workers.py)
  optional CUDA resize
        │  resized frames
        ▼
 AsyncDetector             (pipeline/workers.py)
  YoloV7Detector in a     ◄── YoloV7Detector (detection/detector.py)
  single-worker thread pool
        │  DetectionResult(frame, detections[N,6])
        ▼
 ReIDHelper                (pipeline/workers.py)
  builds embeddings for    ◄── OSNetEmbedder (reid/osnet.py)
  top-K candidates
        │  embeddings[N] or None
        ▼
 Tracker backend           (tracking/tracker_adapter.py)
  BoT-SORT by default,
  ByteTrack optional
        │  tracks: List[dict]
        ▼
 TargetController          (control/target_controller.py)
  maintains lock, lost-    ─── feeds click events from OpenCVViewer
  frame counter, reacquire
        │  TargetState(track_id, bbox, score)
        ▼
 PIDController ×2          (control/pid.py)
  pan + tilt axes
        │  (pan_cmd, tilt_cmd)
        ▼
 GimbalController          (control/gimbal_controller.py)
  dry-run or serial JSON
        │
        ▼
     Gimbal Hardware

Side channel:
 OpenCVViewer              (ui/viewer.py)
  render overlays, emit
  click events
```

---

## Distributed PC Brain Flow

```
Android sensor node
        │
        ▼
 UdpFrameReceiver          (adapters/udp_stream.py)
  non-blocking UDP + JPEG
  reassembly + IMU samples
        │  DecodedFrame(frame, imu, frame_id)
        ▼
 AsyncDetector.submit_latest()
  drops stale frames to keep
  operator control responsive
        │  DetectionResult(frame, detections[N,6], context)
        ▼
 ReIDHelper                (pipeline/workers.py)
  optional OSNet features
        │
        ▼
 Tracker backend           (tracking/tracker_adapter.py)
  BoT-SORT by default,
  ByteTrack optional
        │
        ▼
 Target selection + control
  auto/manual lock, pixel
  error, command gating
        │
        ▼
 AsciiGimbalController     (control/ascii_gimbal_controller.py)
  V/H/E ASCII firmware
  commands with reconnect
        │
        ▼
 ESP32 gimbal firmware

Side channel:
 Brain UI                  (target: ui/brain_viewer.py or app-local)
  pygame operator panel,
  motor/track toggles,
  target selection
```

`gimbal_brain_pc.py` reuses the correct low-level building blocks:
`UdpFrameReceiver`, `YoloV7Detector`, `AsyncDetector`, `GpuPreprocessor`,
`ReIDHelper`, `TrackingService`, BoT-SORT/ByteTrack backends, and
`AsciiGimbalController`. The remaining app-local responsibilities are CLI
parsing, pygame rendering, target selection state, FPS metering, and
proportional velocity control.

Prefer extracting only reusable pieces:

| Current app-local responsibility | Preferred home                                | Notes |
| -------------------------------- | --------------------------------------------- | ----- |
| CLI parsing + application assembly | `src/app/gimbal_brain_pc.py`                | `scripts/` imports and calls `main()`. |
| Pygame operator panel            | `src/ui/brain_viewer.py` or app-local class   | Extract to `ui` only if it will be reused or tested separately. |
| FPS meter                        | `src/pipeline/metrics.py` or app-local helper | `scripts/run_pipeline.py` has a similar helper; avoid two permanent copies. |
| Detection/Re-ID/tracking composition | `src/services/tracking_service.py`        | Now reused by the PC brain and widened to the tracker-backend protocol. |
| Target lock / click selection    | `src/control/target_controller.py`            | Reuse where possible; add brain-specific auto-select behavior only if needed. |
| Pixel error and velocity control | `src/control/target_controller.py` + `pid.py` | Prefer `PIDController` or a small control service over app-local math. |
| UDP frame reception              | `src/adapters/udp_stream.py`                  | Already in the right layer. |
| ASCII firmware commands          | `src/control/ascii_gimbal_controller.py`      | Already in the right layer. |

---

## Module Responsibilities

| Module                 | Package     | Responsibility                                        |
| ---------------------- | ----------- | ----------------------------------------------------- |
| `gimbal_brain_pc.py`   | `app`       | Distributed PC brain assembly and run loop            |
| `detector.py`          | `detection` | YoloV7 inference + pre/post-processing                |
| `bot_sort.py`          | `tracking`  | Local BoT-SORT-ReID tracker backend                   |
| `byte_tracker.py`      | `tracking`  | Multi-object IoU tracking + Re-ID matching            |
| `tracker_adapter.py`   | `tracking`  | Tracker factory + C++ ByteTrack adapter fallback      |
| `osnet.py`             | `reid`      | OSNet feature extraction (batch, crop, encode)        |
| `workers.py`           | `pipeline`  | Async detect, CUDA resize, Re-ID scheduling           |
| `video_source.py`      | `adapters`  | Video source opening/validation adapter               |
| `udp_stream.py`        | `adapters`  | GBR1 UDP packet parsing and JPEG frame reassembly     |
| `tracking_service.py`  | `services`  | Tracking domain service (tracker backend + ReIDHelper) |
| `target_controller.py` | `control`   | Target lock lifecycle, reacquire, click-select        |
| `pid.py`               | `control`   | Discrete PID with optional clamping                   |
| `gimbal_controller.py` | `control`   | `GimbalBase` protocol + serial/dry-run implementation |
| `ascii_gimbal_controller.py` | `control` | ESP32 ASCII firmware command transport          |
| `viewer.py`            | `ui`        | OpenCV window, overlay rendering, mouse events        |
| `config.py`            | `core`      | Frozen dataclass config tree (`PipelineConfig`)       |
| `run_pipeline.py`      | `scripts`   | Current local-pipeline CLI; should become wrapper     |
| `gimbal_brain_pc.py`   | `scripts`   | Thin PC-brain compatibility wrapper                   |

---

## Data Structures

### `DetectionResult` (`pipeline/workers.py`)

```python
@dataclass
class DetectionResult:
    frame: np.ndarray           # BGR frame (original resolution)
    detections: np.ndarray      # shape (N, 6): x1 y1 x2 y2 conf class_id
```

### Track dict (`tracking/bot_sort.py`, `tracking/byte_tracker.py`)

```python
{
    "track_id": int,
    "bbox": np.ndarray,         # (4,) float32 — x1 y1 x2 y2
    "score": float,
    "class_id": int,
    "is_confirmed": bool,
    "feature": Optional[np.ndarray],    # L2-normalised OSNet embedding
    "time_since_update": int,
}
```

### `TargetState` (`control/target_controller.py`)

```python
@dataclass
class TargetState:
    track_id: int
    bbox: np.ndarray            # (4,) float32
    score: float
```

### `PipelineConfig` tree (`core/config.py`)

```
PipelineConfig
├── RuntimeConfig   — source, weights, device, reid flags, fps, max_frames …
├── TrackingConfig  — reid_similarity, reid_distance, feature_momentum …
└── ControlConfig
    └── PIDConfig   — kp, ki, kd
```

---

## Concurrency Model

```
Main thread         FramePrefetcher thread      AsyncDetector thread-pool
──────────          ──────────────────────      ────────────────────────
submit(frame) ─────────────────────────────────► _run_inference(frame)
    │                                                    │
    │◄── DetectionResult ◄─────────────────────────────┘
process_result()
  ├─ ReIDHelper.build_embeddings()   (main thread, GPU)
  ├─ ByteTrack.update()
  ├─ TargetController.maintain()
  ├─ PIDController.update() ×2
  ├─ GimbalController.send()
  └─ OpenCVViewer.render()
```

Key design choices:

- **One-frame pipeline depth**: `AsyncDetector` always holds at most one pending
  future. The main thread passes the _current_ frame to `submit()` and
  receives the _previous_ frame's result synchronously. Latency ≈ 1 frame.
- **Re-ID on main thread**: `OSNetEmbedder` runs on the main thread immediately
  after receiving the detection result, before tracking. This keeps GPU
  synchronisation simple.
- **`FramePrefetcher` is IO-bound**: its thread only calls `cap.read()` into a
  small queue; it never touches the GPU.

---

## Key Thresholds (see also `docs/REFRACTOR_DECISIONS.md`)

| Parameter          | Default                 | Location                      |
| ------------------ | ----------------------- | ----------------------------- |
| `reid_similarity`  | 0.60                    | `TrackingConfig`              |
| `reid_distance`    | 0.25                    | `TrackingConfig`              |
| `feature_momentum` | 0.90                    | `TrackingConfig`              |
| `reid_candidates`  | 6                       | `TrackingConfig`              |
| `target_iou`       | 0.10                    | `TrackingConfig`              |
| PID kp / ki / kd   | 0.005 / 0.0001 / 0.0005 | `PIDConfig`                   |
| `max_lost_frames`  | fps × 2                 | `main()` in `run_pipeline.py` |
