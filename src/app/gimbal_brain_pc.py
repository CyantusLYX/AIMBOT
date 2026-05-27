"""Distributed PC brain application for UDP video tracking and gimbal control."""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Deque, Optional

import numpy as np

if __package__ in (None, ""):
    src_dir = Path(__file__).resolve().parents[1]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

if TYPE_CHECKING:  # pragma: no cover
    import cv2
    import pygame

    from adapters.udp_stream import DecodedFrame

PANEL_WIDTH = 300
DEFAULT_VIDEO_WIDTH = 640
DEFAULT_VIDEO_HEIGHT = 480
MIN_PANEL_WIDTH = 280
MAX_PANEL_WIDTH = 360
MIN_VIDEO_WIDTH = 240
MIN_WINDOW_HEIGHT = 280


class FPSMeter:
    def __init__(self, window: int = 60) -> None:
        self.samples: Deque[float] = deque(maxlen=window)

    def update(self, dt: float) -> float:
        if dt > 0:
            self.samples.append(dt)
        return self.current()

    def current(self) -> float:
        if not self.samples:
            return 0.0
        avg = sum(self.samples) / len(self.samples)
        return 1.0 / avg if avg > 0 else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distributed AIMBOT PC brain")
    parser.add_argument("--listen-host", default="0.0.0.0", help="UDP bind address")
    parser.add_argument("--port", type=int, default=5005, help="UDP video/IMU port")
    parser.add_argument("--weights", default="models/ncv2_m.pt", help="YOLOv7 .pt weights")
    parser.add_argument("--device", default=None, help="torch device; default auto-selects CUDA, Intel XPU, then CPU")
    parser.add_argument("--half", action="store_true", help="Enable FP16 inference on CUDA")
    parser.add_argument("--serial-port", default="/dev/ttyUSB0", help="ESP32 serial port")
    parser.add_argument("--dry-run", action="store_true", help="Print gimbal commands instead of opening serial")
    parser.add_argument("--arm-motors", action="store_true", help="Start with tracking output enabled")
    parser.add_argument("--motors-off", action="store_true", help="Start with ESP32 driver enable output disabled")
    parser.add_argument("--control-timeout", type=float, default=0.20, help="Hold if no fresh detection drives control for this many seconds")
    parser.add_argument("--max-detection-age", type=float, default=0.35, help="Do not drive motors from detections older than this many seconds")
    parser.add_argument("--det-conf", type=float, default=0.35, help="YOLO confidence threshold")
    parser.add_argument("--det-iou", type=float, default=0.45, help="YOLO NMS IoU threshold")
    parser.add_argument("--kp-pan", type=float, default=16.0, help="Proportional pan gain in steps/s per pixel")
    parser.add_argument("--kp-tilt", type=float, default=8.0, help="Proportional tilt gain in steps/s per pixel")
    parser.add_argument("--deadband", type=float, default=20.0, help="Pixel error deadband")
    parser.add_argument("--max-speed", type=float, default=20000.0, help="Absolute velocity clamp in steps/s")
    parser.add_argument("--control-mode", choices=("p", "pid"), default="pid", help="Gimbal velocity controller")
    parser.add_argument("--pid-kp-pan", type=float, default=15.0, help="Experimental PID pan proportional gain")
    parser.add_argument("--pid-ki-pan", type=float, default=0.0, help="Experimental PID pan integral gain")
    parser.add_argument("--pid-kd-pan", type=float, default=0.8, help="Experimental PID pan derivative gain")
    parser.add_argument("--pid-kp-tilt", type=float, default=10.0, help="Experimental PID tilt proportional gain")
    parser.add_argument("--pid-ki-tilt", type=float, default=0.0, help="Experimental PID tilt integral gain")
    parser.add_argument("--pid-kd-tilt", type=float, default=0.8, help="Experimental PID tilt derivative gain")
    parser.add_argument("--pid-deadband", type=float, default=25.0, help="Experimental PID pixel deadband")
    parser.add_argument("--pid-min-speed", type=float, default=250.0, help="Experimental PID minimum nonzero velocity")
    parser.add_argument("--pid-integral-limit", type=float, default=2500.0, help="Experimental PID integral clamp")
    parser.add_argument("--pid-derivative-alpha", type=float, default=0.25, help="Experimental PID derivative smoothing alpha")
    parser.add_argument("--pid-output-slew-rate", type=float, default=60000.0, help="Experimental PID output slew rate in step/s^2")
    parser.add_argument("--tracker-backend", choices=("botsort", "bytetrack"), default="botsort", help="Tracking backend")
    parser.add_argument("--tracker-module", default="bytetrack_cpp", help="Preferred C++ ByteTrack module name")
    parser.add_argument("--require-cpp-tracker", action="store_true", help="Fail if C++ ByteTrack binding is unavailable")
    parser.add_argument("--track-thresh", type=float, default=0.35, help="BoT-SORT high threshold / ByteTrack new-track threshold")
    parser.add_argument("--track-low-thresh", type=float, default=0.10, help="BoT-SORT low detection threshold")
    parser.add_argument("--new-track-thresh", type=float, default=0.55, help="BoT-SORT new-track threshold")
    parser.add_argument("--track-match-iou", type=float, default=0.20, help="Minimum IoU to match an existing track")
    parser.add_argument(
        "--track-max-age",
        type=int,
        default=120,
        help="Frames to keep an unmatched track internally",
    )
    parser.add_argument("--track-min-hits", type=int, default=2, help="Consecutive hits before a track is exposed to UI/control")
    parser.add_argument("--botsort-match-thresh", type=float, default=0.8, help="BoT-SORT association distance threshold")
    parser.add_argument("--botsort-proximity-thresh", type=float, default=0.95, help="BoT-SORT IoU-distance gate before appearance matching")
    parser.add_argument("--botsort-appearance-thresh", type=float, default=0.18, help="BoT-SORT maximum embedding distance")
    parser.add_argument("--botsort-second-match-thresh", type=float, default=0.5, help="BoT-SORT low-score association threshold")
    reid_group = parser.add_mutually_exclusive_group()
    reid_group.add_argument("--enable-reid", dest="enable_reid", action="store_true", default=True, help="Enable OSNet Re-ID for ID stability (default)")
    reid_group.add_argument("--disable-reid", dest="enable_reid", action="store_false", help="Disable OSNet Re-ID")
    parser.add_argument("--reid-model", default="osnet_x0_5", help="torchreid OSNet model name")
    parser.add_argument("--reid-weights", default=None, help="Optional custom Re-ID weights path")
    parser.add_argument("--reid-candidates", type=int, default=8, help="Top detections to embed per processed frame")
    parser.add_argument("--reid-target-iou", type=float, default=0.10, help="Also embed detections overlapping the locked target")
    parser.add_argument("--reid-similarity", type=float, default=0.65, help="Minimum cosine similarity for Re-ID matching")
    parser.add_argument("--reid-distance", type=float, default=0.25, help="Max centre distance gate as a fraction of frame diagonal")
    parser.add_argument("--reid-memory-frames", type=int, default=120, help="Track feature memory in frames when Re-ID is enabled")
    parser.add_argument("--person-only", action="store_true", help="Keep only class 0 detections")
    return parser.parse_args()


def clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def p_control(error: float, gain: float, deadband: float, limit: float) -> float:
    if abs(error) <= deadband:
        return 0.0
    return clamp(error * gain, limit)


def compute_error(track: dict, frame_shape) -> tuple[float, float]:
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = np.asarray(track["bbox"], dtype=np.float32)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    return float(cx - width * 0.5), float(cy - height * 0.5)


def choose_primary_track(tracks: list[dict]) -> Optional[dict]:
    live_tracks = [track for track in tracks if int(track.get("time_since_update", 0)) == 0]
    if not live_tracks:
        return None
    return max(live_tracks, key=lambda track: _track_score(track))


def _track_score(track: dict) -> float:
    x1, y1, x2, y2 = np.asarray(track["bbox"], dtype=np.float32)
    area = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
    return area * max(0.01, float(track.get("score", 1.0)))


def live_confirmed_tracks(tracks: list[dict]) -> list[dict]:
    return [
        track
        for track in tracks
        if int(track.get("time_since_update", 0)) == 0 and bool(track.get("is_confirmed", True))
    ]


def pick_track_at(tracks: list[dict], point: tuple[int, int]) -> Optional[int]:
    x, y = point
    for track in tracks:
        x1, y1, x2, y2 = np.asarray(track["bbox"], dtype=np.float32)
        if x1 <= x <= x2 and y1 <= y <= y2:
            return int(track["track_id"])
    return None


def configure_sdl_for_wayland() -> None:
    if os.environ.get("SDL_VIDEODRIVER"):
        return
    if os.environ.get("WAYLAND_DISPLAY") or os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland":
        os.environ["SDL_VIDEODRIVER"] = "wayland,x11"
        os.environ.setdefault("SDL_VIDEO_WAYLAND_WMCLASS", "aimbot-pc-brain")


def rotate_decoded_frame_clockwise(decoded: DecodedFrame) -> DecodedFrame:
    frame = cv2.rotate(decoded.frame, cv2.ROTATE_90_CLOCKWISE)
    return replace(decoded, frame=frame, width=frame.shape[1], height=frame.shape[0])


@dataclass
class UiState:
    driver_enabled: bool = True
    tracking_enabled: bool = False
    auto_select: bool = True
    target_id: Optional[int] = None
    status: str = "AUTO"
    pan_cmd: float = 0.0
    tilt_cmd: float = 0.0
    detection_age_ms: float = 0.0
    detection_count: int = 0
    raw_track_count: int = 0
    reid_enabled: bool = False
    reid_feature_count: int = 0
    control_mode: str = "P"


class BrainUi:
    def __init__(
        self,
        panel_width: int = PANEL_WIDTH,
        initial_video_size: tuple[int, int] = (DEFAULT_VIDEO_WIDTH, DEFAULT_VIDEO_HEIGHT),
    ) -> None:
        self.panel_width = panel_width
        self.video_rect = pygame.Rect(0, 0, 0, 0)
        self.panel_rect = pygame.Rect(0, 0, 0, 0)
        self._buttons: dict[str, pygame.Rect] = {}
        self._screen: Optional[pygame.Surface] = None
        self._window_size = (0, 0)
        self._source_video_size = initial_video_size
        self._video_scale = 1.0
        self._panel_scroll = 0
        self._panel_content_height = 0
        self._reported_driver = False
        pygame.init()
        pygame.font.init()
        self._title_font = pygame.font.SysFont("Arial", 22, bold=True)
        self._font = pygame.font.SysFont("Arial", 17)
        self._small_font = pygame.font.SysFont("Arial", 14)
        self._clock = pygame.time.Clock()
        self._ensure_screen(*initial_video_size)

    def handle_click(self, x: int, y: int, tracks: list[dict]) -> Optional[tuple[str, Optional[int]]]:
        for action, rect in self._buttons.items():
            if rect.collidepoint(x, y):
                if action.startswith("target:"):
                    return "select_target", int(action.split(":", 1)[1])
                return action, None
        if self.video_rect.collidepoint(x, y):
            video_x = int((x - self.video_rect.x) / max(self._video_scale, 1e-6))
            video_y = int((y - self.video_rect.y) / max(self._video_scale, 1e-6))
            track_id = pick_track_at(tracks, (video_x, video_y))
            if track_id is not None:
                return "select_target", track_id
        return None

    def poll_actions(self, tracks: list[dict]) -> list[tuple[str, Optional[int]]]:
        actions: list[tuple[str, Optional[int]]] = []
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                actions.append(("quit", None))
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    actions.append(("quit", None))
                elif event.key == pygame.K_m:
                    actions.append(("toggle_motor", None))
                elif event.key == pygame.K_t:
                    actions.append(("toggle_tracking", None))
                elif event.key == pygame.K_a:
                    actions.append(("auto_target", None))
                elif event.key == pygame.K_c:
                    actions.append(("clear_target", None))
                elif event.key in (pygame.K_PAGEUP, pygame.K_PAGEDOWN):
                    step = -120 if event.key == pygame.K_PAGEUP else 120
                    self._scroll_panel(step)
            elif event.type == pygame.MOUSEWHEEL:
                self._scroll_panel(-event.y * 36)
            elif event.type == pygame.VIDEORESIZE:
                self._window_size = (max(1, event.w), max(1, event.h))
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                action = self.handle_click(event.pos[0], event.pos[1], tracks)
                if action is not None:
                    actions.append(action)
        return actions

    def render(
        self,
        frame: np.ndarray,
        tracks: list[dict],
        state: UiState,
        error: Optional[tuple[float, float]],
        fps: float,
        decoded: Optional[DecodedFrame],
    ) -> None:
        video = self._render_video(
            frame,
            tracks,
            state.target_id,
            error,
            state.driver_enabled,
            state.tracking_enabled,
        )
        height, width = video.shape[:2]
        self._ensure_screen(width, height)
        if self._screen is None:
            return
        self._update_layout(width, height)
        self._buttons.clear()
        self._screen.fill((12, 14, 18))

        rgb = cv2.cvtColor(video, cv2.COLOR_BGR2RGB)
        surface = pygame.surfarray.make_surface(np.swapaxes(rgb, 0, 1))
        if self.video_rect.size != (width, height):
            surface = pygame.transform.smoothscale(surface, self.video_rect.size)
        self._screen.blit(surface, self.video_rect)
        self._render_panel(tracks, state, error, fps, decoded)
        pygame.display.flip()
        self._clock.tick(60)

    def render_status(self, state: UiState, message: str) -> None:
        width, height = self._source_video_size
        if width <= 0 or height <= 0:
            width, height = DEFAULT_VIDEO_WIDTH, DEFAULT_VIDEO_HEIGHT
        self._ensure_screen(width, height)
        if self._screen is None:
            return
        self._update_layout(width, height)
        self._buttons.clear()
        self._screen.fill((12, 14, 18))
        self._screen.fill((14, 16, 20), self.video_rect)
        if message.startswith("Loading"):
            subtitle = "Initializing PyTorch / YOLOv7"
        elif message.startswith("Warming"):
            subtitle = "Preparing GPU inference"
        elif message.startswith("Opening"):
            subtitle = "Binding UDP socket"
        else:
            subtitle = "Start the Android sensor node on this port"
        self._center_text(message, self.video_rect.centerx, self.video_rect.centery - 16, (225, 225, 225), self._title_font)
        self._center_text(
            subtitle,
            self.video_rect.centerx,
            self.video_rect.centery + 18,
            (150, 160, 170),
            self._small_font,
        )
        self._render_panel([], state, None, 0.0, None)
        pygame.display.flip()
        self._clock.tick(60)

    def _render_video(
        self,
        frame: np.ndarray,
        tracks: list[dict],
        target_id: Optional[int],
        error: Optional[tuple[float, float]],
        driver_enabled: bool,
        tracking_enabled: bool,
    ) -> np.ndarray:
        output = frame.copy()
        height, width = output.shape[:2]
        center = (width // 2, height // 2)
        if not driver_enabled:
            crosshair_color = (0, 0, 255)
        elif tracking_enabled:
            crosshair_color = (0, 255, 0)
        else:
            crosshair_color = (0, 220, 255)
        cv2.line(output, (center[0] - 20, center[1]), (center[0] + 20, center[1]), crosshair_color, 1, cv2.LINE_AA)
        cv2.line(output, (center[0], center[1] - 20), (center[0], center[1] + 20), crosshair_color, 1, cv2.LINE_AA)
        cv2.circle(output, center, 4, crosshair_color, -1, cv2.LINE_AA)

        if error is not None:
            target_point = (int(center[0] + error[0]), int(center[1] + error[1]))
            cv2.line(output, center, target_point, (0, 255, 255), 1, cv2.LINE_AA)

        for track in tracks:
            bbox = np.asarray(track["bbox"], dtype=np.float32).astype(int)
            x1, y1, x2, y2 = bbox.tolist()
            is_target = int(track["track_id"]) == target_id
            color = (0, 255, 0) if is_target else (40, 40, 255)
            thickness = 3 if is_target else 1
            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
            label = f"ID {track['track_id']}"
            cv2.putText(output, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        return output

    def _render_panel(
        self,
        tracks: list[dict],
        state: UiState,
        error: Optional[tuple[float, float]],
        fps: float,
        decoded: Optional[DecodedFrame],
    ) -> None:
        if self._screen is None:
            return
        panel_rect = self.panel_rect
        height = panel_rect.height
        self._screen.fill((26, 28, 32), panel_rect)
        x = panel_rect.x + 16
        y = 24
        content_bottom = max(0, height - 58)
        old_clip = self._screen.get_clip()
        self._screen.set_clip(pygame.Rect(panel_rect.x, panel_rect.y, panel_rect.width, content_bottom))
        y -= self._panel_scroll
        button_w = max(1, panel_rect.width - 32)
        half_button_w = max(1, (button_w - 10) // 2)

        self._text("AIMBOT BRAIN", x, y, (245, 245, 245), self._title_font)
        y += 34
        frame_text = f"FPS {fps:.1f}   FRAME {decoded.frame_id}" if decoded is not None else "FPS --   FRAME --"
        self._text(frame_text, x, y, (180, 220, 255))
        y += 34

        motor_label = "MOTOR ENABLED" if state.driver_enabled else "MOTOR OFF"
        motor_color = (34, 130, 72) if state.driver_enabled else (92, 42, 50)
        self._button("toggle_motor", motor_label, pygame.Rect(x, y, button_w, 46), motor_color)
        y += 54
        track_label = "TRACK ARMED" if state.tracking_enabled else "TRACK HOLD"
        track_color = (56, 72, 96) if state.tracking_enabled else (52, 54, 62)
        self._button("toggle_tracking", track_label, pygame.Rect(x, y, button_w, 38), track_color)
        y += 54

        target_text = f"TARGET ID {state.target_id}" if state.target_id is not None else "NO TARGET"
        mode_text = "AUTO SELECT" if state.auto_select else "MANUAL LOCK"
        self._text(target_text, x, y, (40, 240, 110) if state.target_id is not None else (160, 160, 170))
        y += 24
        self._text(mode_text, x, y, (220, 220, 220))
        y += 34
        self._button("auto_target", "AUTO TARGET", pygame.Rect(x, y, half_button_w, 38), (56, 72, 96))
        self._button("clear_target", "CLEAR LOCK", pygame.Rect(x + half_button_w + 10, y, half_button_w, 38), (86, 56, 62))
        y += 58

        if error is None:
            self._text("ERR x=-- y=--", x, y, (160, 160, 170))
        else:
            self._text(f"ERR x={error[0]:.1f} y={error[1]:.1f}", x, y, (40, 240, 110))
        y += 24
        self._text(f"CMD pan={state.pan_cmd:.0f} tilt={state.tilt_cmd:.0f}", x, y, (40, 220, 255))
        y += 34
        self._text(f"CTRL {state.control_mode}", x, y, (190, 220, 255))
        y += 24
        self._text(f"DET age={state.detection_age_ms:.0f}ms", x, y, (190, 200, 210))
        y += 24
        reid_text = f"REID {state.reid_feature_count}" if state.reid_enabled else "REID OFF"
        self._text(reid_text, x, y, (190, 220, 255) if state.reid_enabled else (130, 135, 145))
        y += 24

        imu = decoded.imu if decoded is not None else None
        self._text(f"IMU roll={imu.roll:.2f}" if imu is not None else "IMU roll=--", x, y, (230, 230, 230))
        y += 22
        self._text(
            f"pitch={imu.pitch:.2f} yaw={imu.yaw:.2f}" if imu is not None else "pitch=-- yaw=--",
            x,
            y,
            (230, 230, 230),
        )
        y += 34
        self._text(f"DETS {state.detection_count}  TRACKS {len(tracks)}/{state.raw_track_count}", x, y, (235, 235, 235))
        y += 26

        for track in sorted(tracks, key=_track_score, reverse=True)[:5]:
            track_id = int(track["track_id"])
            selected = track_id == state.target_id
            label = f"ID {track_id}   score {float(track.get('score', 1.0)):.2f}"
            color = (38, 112, 62) if selected else (44, 48, 56)
            self._button(f"target:{track_id}", label, pygame.Rect(x, y, button_w, 32), color, font=self._small_font)
            y += 38

        if not tracks:
            self._text("No detections yet", x, y, (150, 150, 160), self._small_font)
            y += 28

        y += 12
        self._text(state.status, x, y, (255, 220, 120), self._small_font)
        self._panel_content_height = y + self._panel_scroll + 24
        self._clamp_panel_scroll()
        self._screen.set_clip(old_clip)

        footer_y = max(0, height - 54)
        pygame.draw.line(self._screen, (42, 45, 52), (panel_rect.x, footer_y - 10), (panel_rect.right, footer_y - 10), 1)
        self._text("M: motor   T: track   A: auto", x, footer_y, (150, 150, 160), self._small_font)
        self._text("C: clear   Q/Esc: quit", x, footer_y + 24, (150, 150, 160), self._small_font)
        if self._panel_content_height > content_bottom:
            track_h = max(24, content_bottom - 18)
            thumb_h = max(24, int(track_h * content_bottom / max(1, self._panel_content_height)))
            max_scroll = max(1, self._panel_content_height - content_bottom)
            thumb_y = 9 + int((track_h - thumb_h) * self._panel_scroll / max_scroll)
            pygame.draw.rect(self._screen, (60, 64, 74), pygame.Rect(panel_rect.right - 8, 9, 4, track_h), border_radius=2)
            pygame.draw.rect(self._screen, (130, 136, 150), pygame.Rect(panel_rect.right - 8, thumb_y, 4, thumb_h), border_radius=2)

    def _button(
        self,
        action: str,
        label: str,
        rect: pygame.Rect,
        color: tuple[int, int, int],
        fg: tuple[int, int, int] = (245, 245, 245),
        font: Optional[pygame.font.Font] = None,
    ) -> None:
        if self._screen is None:
            return
        clip = self._screen.get_clip()
        if clip is None or rect.colliderect(clip):
            self._buttons[action] = rect
        pygame.draw.rect(self._screen, color, rect, border_radius=4)
        pygame.draw.rect(self._screen, (100, 105, 112), rect, width=1, border_radius=4)
        font = font or self._font
        surface = font.render(label, True, fg)
        self._screen.blit(surface, surface.get_rect(center=rect.center))

    def _ensure_screen(self, video_width: int, video_height: int) -> None:
        self._source_video_size = (max(1, video_width), max(1, video_height))
        size = (
            max(MIN_VIDEO_WIDTH + MIN_PANEL_WIDTH, video_width + self.panel_width),
            max(MIN_WINDOW_HEIGHT, video_height),
        )
        if self._screen is None:
            flags = pygame.RESIZABLE
            self._screen = pygame.display.set_mode(size, flags)
            pygame.display.set_caption("AIMBOT PC Brain")
            self._window_size = self._screen.get_size()
            if not self._reported_driver:
                print(f"[UI] pygame video driver: {pygame.display.get_driver()}")
                self._reported_driver = True

    def _update_layout(self, video_width: int, video_height: int) -> None:
        if self._screen is None:
            return
        window_width, window_height = self._screen.get_size()
        self._window_size = (window_width, window_height)
        panel_width = min(MAX_PANEL_WIDTH, max(MIN_PANEL_WIDTH, int(window_width * 0.32)))
        if window_width - panel_width < MIN_VIDEO_WIDTH:
            panel_width = max(MIN_PANEL_WIDTH, window_width - MIN_VIDEO_WIDTH)
        panel_width = min(panel_width, max(1, window_width - 1))
        video_area = pygame.Rect(0, 0, max(1, window_width - panel_width), window_height)
        scale = min(video_area.width / max(1, video_width), video_area.height / max(1, video_height))
        scaled_w = max(1, int(video_width * scale))
        scaled_h = max(1, int(video_height * scale))
        self._video_scale = scale
        self.video_rect = pygame.Rect(
            video_area.x + (video_area.width - scaled_w) // 2,
            video_area.y + (video_area.height - scaled_h) // 2,
            scaled_w,
            scaled_h,
        )
        self.panel_rect = pygame.Rect(video_area.right, 0, panel_width, window_height)
        self._clamp_panel_scroll()

    def _scroll_panel(self, delta: int) -> None:
        self._panel_scroll += int(delta)
        self._clamp_panel_scroll()

    def _clamp_panel_scroll(self) -> None:
        content_height = max(0, self._panel_content_height)
        visible_height = max(0, self.panel_rect.height - 58)
        max_scroll = max(0, content_height - visible_height)
        self._panel_scroll = max(0, min(self._panel_scroll, max_scroll))

    def _text(
        self,
        text: str,
        x: int,
        y: int,
        color: tuple[int, int, int],
        font: Optional[pygame.font.Font] = None,
    ) -> None:
        if self._screen is None:
            return
        surface = (font or self._font).render(text, True, color)
        self._screen.blit(surface, (x, y))

    def _center_text(
        self,
        text: str,
        x: int,
        y: int,
        color: tuple[int, int, int],
        font: Optional[pygame.font.Font] = None,
    ) -> None:
        if self._screen is None:
            return
        surface = (font or self._font).render(text, True, color)
        self._screen.blit(surface, surface.get_rect(center=(x, y)))

    def close(self) -> None:
        pygame.quit()

    def idle(self) -> None:
        self._clock.tick(60)


def main() -> None:
    args = parse_args()

    global cv2, pygame

    configure_sdl_for_wayland()

    import cv2 as cv2_module
    import pygame as pygame_module

    from adapters.udp_stream import DecodedFrame, UdpFrameReceiver
    from control.ascii_gimbal_controller import AsciiGimbalController
    from control.pid import PIDGains, VelocityPIDAxisController
    from detection.detector import YoloV7Detector, filter_classes
    from pipeline.workers import AsyncDetector, GpuPreprocessor, ReIDHelper
    from services.tracking_service import TrackingService
    from tracking.tracker_adapter import create_tracker_backend

    cv2 = cv2_module
    pygame = pygame_module

    if args.tracker_backend == "bytetrack" and args.enable_reid and args.require_cpp_tracker:
        raise SystemExit(
            "Re-ID is incompatible with --require-cpp-tracker because C++ tracker bindings do not accept embeddings; "
            "use --disable-reid for C++ ByteTrack"
        )
    tracker_max_age = (
        max(1, int(args.track_max_age))
        if args.track_max_age is not None
        else max(1, int(args.reid_memory_frames if args.enable_reid else 15))
    )

    class_filter = (lambda dets: filter_classes(dets, [0]) if dets.size else dets) if args.person_only else None
    ui = BrainUi()
    ui_state = UiState(
        driver_enabled=not args.motors_off,
        tracking_enabled=args.arm_motors and not args.motors_off,
        reid_enabled=args.enable_reid,
        control_mode=args.control_mode.upper(),
    )
    ui_state.status = "STARTING"
    ui.render_status(ui_state, "Loading detector")

    detector = YoloV7Detector(
        weights_path=args.weights,
        device=args.device,
        use_half=args.half,
        conf_threshold=args.det_conf,
        iou_threshold=args.det_iou,
    )
    if detector.device.startswith(("cuda", "xpu")):
        ui.render_status(ui_state, f"Warming {detector.device.upper()} detector")
        detector.warmup(image_size=640)
    print(f"[DETECTOR] device={detector.device}{' fp16' if detector.using_half else ' fp32'}")

    detector_worker = AsyncDetector(detector=detector, preprocessor=GpuPreprocessor(1.0), class_filter=class_filter)
    reid_helper = None
    if args.enable_reid:
        ui.render_status(ui_state, "Loading Re-ID")
        try:
            from reid.osnet import OSNetEmbedder
        except ImportError as exc:
            raise SystemExit(
                "Re-ID requires torchreid. Install the reid extra, for example: "
                "uv sync --extra reid"
            ) from exc

        embedder = OSNetEmbedder(
            device=detector.device,
            use_half=args.half,
            model_name=args.reid_model,
            weights_path=args.reid_weights,
        )
        reid_helper = ReIDHelper(
            embedder,
            max_candidates=args.reid_candidates,
            target_iou=args.reid_target_iou,
        )
        print(
            f"[REID] enabled model={args.reid_model} "
            f"candidates={args.reid_candidates} similarity={args.reid_similarity:.2f}"
        )

    try:
        tracker = create_tracker_backend(
            tracker_backend=args.tracker_backend,
            cpp_module=args.tracker_module,
            track_thresh=args.track_thresh,
            match_iou_thresh=args.track_match_iou,
            max_age=tracker_max_age,
            min_hits=args.track_min_hits,
            enable_reid=args.enable_reid,
            reid_match_thresh=args.reid_similarity,
            reid_max_center_dist=args.reid_distance,
            require_cpp=args.require_cpp_tracker,
            track_low_thresh=args.track_low_thresh,
            new_track_thresh=args.new_track_thresh,
            botsort_match_thresh=args.botsort_match_thresh,
            botsort_proximity_thresh=args.botsort_proximity_thresh,
            botsort_appearance_thresh=args.botsort_appearance_thresh,
            botsort_second_match_thresh=args.botsort_second_match_thresh,
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    tracking_service = TrackingService(tracker=tracker, reid_helper=reid_helper)
    print(f"[TRACKER] backend={args.tracker_backend} max_age={tracker_max_age} min_hits={args.track_min_hits}")
    ui.render_status(ui_state, "Opening UDP receiver")
    receiver = UdpFrameReceiver(args.listen_host, args.port)
    gimbal = AsciiGimbalController(args.serial_port, dry_run=args.dry_run, max_hz=30.0)
    pid_pan = VelocityPIDAxisController(
        PIDGains(kp=args.pid_kp_pan, ki=args.pid_ki_pan, kd=args.pid_kd_pan),
        deadband=args.pid_deadband,
        min_speed=args.pid_min_speed,
        output_limit=args.max_speed,
        integral_limit=args.pid_integral_limit,
        derivative_alpha=args.pid_derivative_alpha,
        output_slew_rate=args.pid_output_slew_rate,
    )
    pid_tilt = VelocityPIDAxisController(
        PIDGains(kp=args.pid_kp_tilt, ki=args.pid_ki_tilt, kd=args.pid_kd_tilt),
        deadband=args.pid_deadband,
        min_speed=args.pid_min_speed,
        output_limit=args.max_speed,
        integral_limit=args.pid_integral_limit,
        derivative_alpha=args.pid_derivative_alpha,
        output_slew_rate=args.pid_output_slew_rate,
    )
    gimbal.set_enabled(ui_state.driver_enabled, repeat=3)
    if ui_state.driver_enabled:
        gimbal.hold(repeat=3, force=True)
    fps_meter = FPSMeter()
    current_tracks: list[dict] = []
    current_tracks_frame_id: Optional[int] = None
    last_frame_time: Optional[float] = None
    last_waiting_render = 0.0
    last_control_s = time.monotonic()
    print(f"[UDP] listening on {args.listen_host}:{args.port}")
    print(f"[CONTROL] mode={args.control_mode.upper()}")
    print("[UI] click a box or track-row to lock target; M toggles motor IO, T toggles tracking output")
    ui_state.status = "WAITING FOR UDP"
    ui.render_status(ui_state, f"Listening on UDP {args.port}")

    def reset_pid() -> None:
        pid_pan.reset()
        pid_tilt.reset()

    def compute_control(error: tuple[float, float], dt: float) -> tuple[float, float]:
        if args.control_mode == "pid":
            return pid_pan.update(error[0], dt), pid_tilt.update(error[1], dt)
        return (
            p_control(error[0], args.kp_pan, args.deadband, args.max_speed),
            p_control(error[1], args.kp_tilt, args.deadband, args.max_speed),
        )

    def apply_ui_action(action: str, track_id: Optional[int]) -> bool:
        nonlocal last_control_s
        if action == "quit":
            return False
        if action == "toggle_motor":
            ui_state.driver_enabled = not ui_state.driver_enabled
            if ui_state.driver_enabled:
                ui_state.status = "MOTOR ENABLED"
                gimbal.set_enabled(True, repeat=3)
                gimbal.hold(repeat=3, force=True)
                last_control_s = time.monotonic()
            else:
                ui_state.tracking_enabled = False
                ui_state.pan_cmd = 0.0
                ui_state.tilt_cmd = 0.0
                reset_pid()
                gimbal.hold(repeat=3, force=True)
                gimbal.set_enabled(False, repeat=3)
                ui_state.status = "MOTOR OFF"
                last_control_s = time.monotonic()
        elif action == "toggle_tracking":
            ui_state.tracking_enabled = not ui_state.tracking_enabled
            if ui_state.tracking_enabled:
                if not ui_state.driver_enabled:
                    ui_state.driver_enabled = True
                    gimbal.set_enabled(True, repeat=3)
                    gimbal.hold(repeat=3, force=True)
                ui_state.status = "TRACK ARMED"
            else:
                ui_state.pan_cmd = 0.0
                ui_state.tilt_cmd = 0.0
                reset_pid()
                ui_state.status = "TRACK HOLD"
                if ui_state.driver_enabled:
                    gimbal.hold(repeat=3, force=True)
                last_control_s = time.monotonic()
        elif action == "auto_target":
            reset_pid()
            ui_state.auto_select = True
            ui_state.target_id = None
            ui_state.status = "AUTO TARGET"
        elif action == "clear_target":
            reset_pid()
            ui_state.auto_select = False
            ui_state.target_id = None
            ui_state.pan_cmd = 0.0
            ui_state.tilt_cmd = 0.0
            ui_state.status = "LOCK CLEARED"
            if ui_state.driver_enabled:
                gimbal.hold(repeat=3, force=True)
                last_control_s = time.monotonic()
        elif action == "select_target" and track_id is not None:
            if ui_state.target_id != track_id:
                reset_pid()
            ui_state.auto_select = False
            ui_state.target_id = track_id
            ui_state.status = f"LOCKED ID {track_id}"
        return True

    def process(decoded: DecodedFrame, detections: np.ndarray, detection_age_s: float) -> None:
        nonlocal current_tracks, current_tracks_frame_id, last_control_s, last_frame_time
        process_start_s = time.monotonic()
        dt = 0.0 if last_frame_time is None else process_start_s - last_frame_time
        last_frame_time = process_start_s
        fps = fps_meter.update(dt)

        target_bbox = None
        if ui_state.target_id is not None:
            previous_target = next(
                (track for track in current_tracks if int(track["track_id"]) == ui_state.target_id),
                None,
            )
            if previous_target is not None:
                target_bbox = np.asarray(previous_target["bbox"], dtype=np.float32)

        all_tracks = tracking_service.update(decoded.frame, detections, target_bbox)
        if reid_helper is not None:
            ui_state.reid_feature_count = tracking_service.last_embedding_count
        now = time.monotonic()
        detection_age_s += max(0.0, now - process_start_s)
        ui_state.detection_age_ms = detection_age_s * 1000.0
        tracks = live_confirmed_tracks(all_tracks)
        ui_state.detection_count = int(len(detections))
        ui_state.raw_track_count = int(len(all_tracks))
        current_tracks = tracks
        current_tracks_frame_id = decoded.frame_id

        target_track = None
        previous_target_id = ui_state.target_id
        if ui_state.target_id is not None:
            target_track = next((track for track in tracks if int(track["track_id"]) == ui_state.target_id), None)
        if target_track is None and ui_state.auto_select:
            target_track = choose_primary_track(tracks)
            ui_state.target_id = int(target_track["track_id"]) if target_track is not None else None
        if ui_state.target_id != previous_target_id:
            reset_pid()

        error = None
        if target_track is None:
            ui_state.pan_cmd = 0.0
            ui_state.tilt_cmd = 0.0
            reset_pid()
            if ui_state.driver_enabled:
                gimbal.hold()
                last_control_s = now
            if ui_state.target_id is not None:
                ui_state.status = f"TARGET {ui_state.target_id} LOST"
            elif ui_state.auto_select:
                ui_state.status = "WAITING FOR TARGET"
        elif detection_age_s > args.max_detection_age:
            ui_state.pan_cmd = 0.0
            ui_state.tilt_cmd = 0.0
            error = compute_error(target_track, decoded.frame.shape)
            reset_pid()
            if ui_state.driver_enabled:
                gimbal.hold()
                last_control_s = now
            ui_state.status = f"STALE DET {detection_age_s * 1000.0:.0f}MS"
        else:
            error = compute_error(target_track, decoded.frame.shape)
            pan, tilt = compute_control(error, dt)
            if ui_state.driver_enabled and ui_state.tracking_enabled:
                ui_state.pan_cmd = pan
                ui_state.tilt_cmd = tilt
                gimbal.send(pan, tilt)
                last_control_s = now
            else:
                ui_state.pan_cmd = 0.0
                ui_state.tilt_cmd = 0.0
                reset_pid()
                ui_state.status = "MOTOR OFF" if not ui_state.driver_enabled else "TRACK HOLD"
            if ui_state.driver_enabled and ui_state.tracking_enabled:
                ui_state.status = f"TRACKING ID {ui_state.target_id}"

        ui.render(decoded.frame, tracks, ui_state, error, fps, decoded)

    def render_live(decoded: DecodedFrame) -> None:
        overlay_tracks = current_tracks if current_tracks_frame_id == decoded.frame_id else []
        ui.render(decoded.frame, overlay_tracks, ui_state, None, fps_meter.current(), decoded)

    try:
        running = True
        while running:
            for action, track_id in ui.poll_actions(current_tracks):
                running = apply_ui_action(action, track_id)
                if not running:
                    break
            if not running:
                break

            now = time.monotonic()
            if (
                ui_state.driver_enabled
                and ui_state.tracking_enabled
                and now - last_control_s > args.control_timeout
            ):
                ui_state.pan_cmd = 0.0
                ui_state.tilt_cmd = 0.0
                reset_pid()
                ui_state.status = "CONTROL TIMEOUT HOLD"
                gimbal.hold(force=True)
                last_control_s = now

            decoded = receiver.recv_latest(timeout_s=0.01)
            if decoded is not None:
                decoded = rotate_decoded_frame_clockwise(decoded)
                result, _accepted = detector_worker.submit_latest(decoded.frame, context=decoded)
                if result is not None and isinstance(result.context, DecodedFrame):
                    process(
                        decoded=result.context,
                        detections=result.detections,
                        detection_age_s=time.monotonic() - result.submitted_at_s,
                    )
                else:
                    render_live(decoded)
            else:
                now = time.monotonic()
                if current_tracks_frame_id is None and now - last_waiting_render > 0.25:
                    ui.render_status(ui_state, f"Listening on UDP {args.port}")
                    last_waiting_render = now
                ui.idle()
    except KeyboardInterrupt:
        pass
    finally:
        gimbal.hold(repeat=3, force=True)
        detector_worker.shutdown()
        gimbal.close()
        receiver.close()
        ui.close()


if __name__ == "__main__":
    main()
