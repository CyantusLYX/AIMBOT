from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime options for source/model/device and execution behavior."""

    weights: str = "models/epoch_149.pt"
    source: str = "0"
    device: str | None = None
    person_only: bool = False
    half: bool = False
    enable_reid: bool = False
    process_scale: float = 1.0
    reid_model: str = "osnet_x0_5"
    reid_weights: str | None = None
    dry_run: bool = False
    serial_port: str = "COM3"
    fps: int = 30
    max_frames: int | None = None


@dataclass(frozen=True)
class TrackingConfig:
    """Tracking/Re-ID strategy parameters."""

    tracker_backend: str = "botsort"
    track_high_thresh: float = 0.5
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.6
    track_match_iou: float = 0.2
    track_max_age: int = 30
    track_min_hits: int = 3
    botsort_match_thresh: float = 0.8
    botsort_proximity_thresh: float = 0.5
    botsort_appearance_thresh: float = 0.25
    botsort_second_match_thresh: float = 0.5
    reid_similarity: float = 0.6
    reid_distance: float = 0.25
    reid_candidates: int = 6
    target_iou: float = 0.1
    feature_momentum: float = 0.9


@dataclass(frozen=True)
class PIDConfig:
    """PID gains for gimbal control."""

    kp: float = 0.005
    ki: float = 0.0001
    kd: float = 0.0005


@dataclass(frozen=True)
class ControlConfig:
    """Control-loop and gimbal command parameters."""

    pid: PIDConfig = PIDConfig()


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level configuration for assembling the pipeline."""

    runtime: RuntimeConfig = RuntimeConfig()
    tracking: TrackingConfig = TrackingConfig()
    control: ControlConfig = ControlConfig()
