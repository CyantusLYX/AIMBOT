from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime options for source/model/device and execution behavior."""

    weights: str = "models/epoch_149.pt"
    source: str = "0"
    device: Optional[str] = None
    detector_backend: str = "auto"
    camera_backend: str = "auto"
    camera_width: int = 1280
    camera_height: int = 720
    camera_fps: int = 30
    no_display: bool = False
    debug_frame_dir: Optional[str] = None
    trt_input_shape: str = "640x640"
    trt_output_format: str = "auto"
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    person_only: bool = False
    half: bool = False
    enable_reid: bool = False
    process_scale: float = 1.0
    reid_model: str = "osnet_x0_5"
    reid_weights: Optional[str] = None
    dry_run: bool = False
    serial_port: str = "COM3"
    fps: int = 30
    max_frames: Optional[int] = None


@dataclass(frozen=True)
class TrackingConfig:
    """Tracking/Re-ID strategy parameters."""

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
