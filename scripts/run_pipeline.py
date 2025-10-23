from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from detection.detector import YoloV7Detector, filter_classes
from tracking.byte_tracker import ByteTrack
from control.pid import PIDController, PIDGains
from control.target_controller import TargetController
from control.gimbal_controller import GimbalController
from ui.viewer import OpenCVViewer


class FPSMeter:
    def __init__(self, window: int = 60) -> None:
        self.window = window
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
    parser = argparse.ArgumentParser(description="AIMBOT pipeline")
    parser.add_argument("--weights", type=str, default="models/yolov7.pt", help="YOLOv7 權重路徑")
    parser.add_argument("--source", type=str, default="0", help="攝影機索引或影片路徑")
    parser.add_argument("--person-only", action="store_true", help="只追蹤 person 類別")
    parser.add_argument("--device", type=str, default=None, help="指定裝置，例如 cuda:0")
    parser.add_argument("--dry-run", action="store_true", help="僅輸出控制命令，不實際連線雲台")
    parser.add_argument("--serial-port", type=str, default="COM3", help="雲台序列埠")
    parser.add_argument("--fps", type=int, default=30, help="影像 FPS")
    parser.add_argument("--max-frames", type=int, default=None, help="限制處理的影格數量，用於快速測試")
    parser.add_argument("--half", action="store_true", help="啟用半精度推論 (僅限 CUDA)")
    parser.add_argument("--kp", type=float, default=0.005)
    parser.add_argument("--ki", type=float, default=0.0001)
    parser.add_argument("--kd", type=float, default=0.0005)
    return parser.parse_args()


def create_capture(source: str) -> cv2.VideoCapture:
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟來源: {source}")
    return cap


def main() -> None:
    args = parse_args()
    detector = YoloV7Detector(weights_path=args.weights, device=args.device, use_half=args.half)
    if detector.device.startswith("cuda"):
        detector.warmup()
    if args.half and not detector.using_half:
        print("警告: --half 選項僅在 CUDA 裝置上有效，已回退至 float32。")
    print(f"使用推論裝置: {detector.device}{' (FP16)' if detector.using_half else ''}")
    tracker = ByteTrack()
    target_ctrl = TargetController()
    pid_pan = PIDController(PIDGains(args.kp, args.ki, args.kd))
    pid_tilt = PIDController(PIDGains(args.kp, args.ki, args.kd))
    viewer = OpenCVViewer()
    cap = create_capture(args.source)
    try:
        with GimbalController(args.serial_port, dry_run=args.dry_run) as gimbal:
            fps_meter = FPSMeter(window=max(args.fps, 30))
            last_frame_time: Optional[float] = None
            last_control_time: Optional[float] = None
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if not viewer.is_open():
                    print("視窗已關閉，停止播放。")
                    break
                frame_count += 1
                now = time.time()
                frame_dt = 0.0 if last_frame_time is None else now - last_frame_time
                last_frame_time = now
                fps_value = fps_meter.update(frame_dt)
                detections = detector.detect(frame)
                if args.person_only:
                    detections = filter_classes(detections, [0])
                tracks = tracker.update(detections, frame.shape)
                click = viewer.poll_click()
                if click is not None:
                    target_ctrl.select_by_point(tracks, click)
                state = target_ctrl.current_state(tracks)
                if state is not None:
                    ctrl_dt = now - last_control_time if last_control_time is not None else 1.0 / args.fps
                    last_control_time = now
                    err_x, err_y = TargetController.compute_error(state, frame.shape)
                    pan_cmd = pid_pan.update(err_x, ctrl_dt)
                    tilt_cmd = pid_tilt.update(err_y, ctrl_dt)
                    gimbal.send(pan_cmd, tilt_cmd)
                viewer.render(frame, tracks, target_ctrl.target_id, fps=fps_value)
                key = viewer.wait_key(1)
                if key in (ord("q"), 27):
                    break
                if not viewer.is_open():
                    print("視窗已關閉，停止播放。")
                    break
                if args.max_frames is not None and frame_count >= args.max_frames:
                    print(f"已達影格上限 {args.max_frames}，提前結束。")
                    break
                if fps_value > 0 and frame_count % max(args.fps, 30) == 0:
                    print(f"近期平均 FPS: {fps_value:.1f}")
    finally:
        cap.release()
        viewer.close()


if __name__ == "__main__":
    main()
