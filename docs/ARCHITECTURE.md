# AIMBOT — Architecture Reference

> **Last updated**: 2026-03-14
> **Status**: Living document — update whenever a module boundary changes.

---

## System Overview

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
 ByteTrack                 (tracking/byte_tracker.py)
  IoU + optional Re-ID
  cost-matrix matching
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

## Module Responsibilities

| Module                 | Package     | Responsibility                                        |
| ---------------------- | ----------- | ----------------------------------------------------- |
| `detector.py`          | `detection` | YoloV7 inference + pre/post-processing                |
| `byte_tracker.py`      | `tracking`  | Multi-object IoU tracking + Re-ID matching            |
| `osnet.py`             | `reid`      | OSNet feature extraction (batch, crop, encode)        |
| `workers.py`           | `pipeline`  | Async detect, CUDA resize, Re-ID scheduling           |
| `video_source.py`      | `adapters`  | Video source opening/validation adapter               |
| `tracking_service.py`  | `services`  | Tracking domain service (`ByteTrack` + `ReIDHelper`)  |
| `target_controller.py` | `control`   | Target lock lifecycle, reacquire, click-select        |
| `pid.py`               | `control`   | Discrete PID with optional clamping                   |
| `gimbal_controller.py` | `control`   | `GimbalBase` protocol + serial/dry-run implementation |
| `viewer.py`            | `ui`        | OpenCV window, overlay rendering, mouse events        |
| `config.py`            | `core`      | Frozen dataclass config tree (`PipelineConfig`)       |
| `run_pipeline.py`      | `scripts`   | CLI entry point — assembles and runs the pipeline     |

---

## Data Structures

### `DetectionResult` (`pipeline/workers.py`)

```python
@dataclass
class DetectionResult:
    frame: np.ndarray           # BGR frame (original resolution)
    detections: np.ndarray      # shape (N, 6): x1 y1 x2 y2 conf class_id
```

### Track dict (`tracking/byte_tracker.py`)

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
