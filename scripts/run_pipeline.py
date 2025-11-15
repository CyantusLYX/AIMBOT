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

from control.gimbal_controller import GimbalController
from control.pid import PIDController, PIDGains
from control.target_controller import TargetController
from detection.detector import YoloV7Detector, filter_classes
from pipeline.workers import AsyncDetector, FramePrefetcher, GpuPreprocessor, ReIDHelper
from tracking.byte_tracker import ByteTrack
from ui.viewer import OpenCVViewer


REID_SIMILARITY = 0.6
REID_DISTANCE = 0.25
REID_CANDIDATES = 6
TARGET_IOU = 0.1
FEATURE_MOMENTUM = 0.9
PID_GAINS = PIDGains(0.005, 0.0001, 0.0005)


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
    parser = argparse.ArgumentParser(description="AIMBOT pipeline")
    parser.add_argument("--weights", type=str, default="models/yolov7.pt", help="YOLOv7 權重路徑")
    parser.add_argument("--source", type=str, default="0", help="攝影機索引或影片路徑")
    parser.add_argument("--device", type=str, default=None, help="指定裝置，例如 cuda:0")
    parser.add_argument("--person-only", action="store_true", help="僅保留 person 類別")
    parser.add_argument("--half", action="store_true", help="啟用半精度推論 (僅限 CUDA)")
    parser.add_argument("--enable-reid", action="store_true", help="啟用 OSNet Re-ID")
    parser.add_argument(
        "--process-scale",
        type=float,
        default=1.0,
        help="偵測影像縮放比例，1.0 為原尺寸，0 表示自動依輸入大小調整",
    )
    parser.add_argument("--reid-model", type=str, default="osnet_x0_5", help="OSNet 模型名稱 (torchreid)")
    parser.add_argument("--reid-weights", type=str, default=None, help="自訂 Re-ID 權重路徑 (可選)")
    parser.add_argument("--dry-run", action="store_true", help="僅輸出控制命令，不實際連線雲台")
    parser.add_argument("--serial-port", type=str, default="COM3", help="雲台序列埠")
    parser.add_argument("--fps", type=int, default=30, help="控制迴圈目標 FPS")
    parser.add_argument("--max-frames", type=int, default=None, help="限制處理影格數量，用於快速測試")
    return parser.parse_args()


def create_capture(source: str) -> cv2.VideoCapture:
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟來源: {source}")
    return cap


def main() -> None:
    args = parse_args()
    detector = YoloV7Detector(weights_path=args.weights, device=args.device, use_half=args.half)
    if detector.device.startswith("cuda"):
        detector.warmup()
    if args.half and not detector.using_half:
        print("警告: --half 僅在 CUDA 裝置上有效，已回退至 float32")
    print(f"使用推論裝置: {detector.device}{' (FP16)' if detector.using_half else ''}")

    embedder = None
    if args.enable_reid:
        from reid.osnet import OSNetEmbedder

        embedder = OSNetEmbedder(
            device=detector.device,
            use_half=args.half,
            model_name=args.reid_model,
            weights_path=args.reid_weights,
        )

    tracker = ByteTrack(
        enable_reid=args.enable_reid,
        reid_match_thresh=REID_SIMILARITY,
        feature_momentum=FEATURE_MOMENTUM,
        feature_min_similarity=REID_SIMILARITY,
        reid_max_center_dist=REID_DISTANCE,
    )
    target_ctrl = TargetController(max_lost_frames=int(args.fps * 2), reacquire_thresh=REID_SIMILARITY)
    pid_pan = PIDController(PID_GAINS)
    pid_tilt = PIDController(PID_GAINS)
    viewer = OpenCVViewer()
    fps_meter = FPSMeter(window=max(args.fps, 30))

    cap = create_capture(args.source)
    is_camera = args.source.isdigit()
    playback_fps = cap.get(cv2.CAP_PROP_FPS) if not is_camera else 0.0
    if playback_fps and playback_fps < 0:
        playback_fps = 0.0
    target_period = 0.0
    if playback_fps and playback_fps >= 1.0:
        print(f"偵測到影片 FPS ≈ {playback_fps:.2f}，將同步播放速度。")
        target_period = 1.0 / playback_fps
    else:
        target_period = 0.0
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("來源沒有任何畫面")

    if args.process_scale <= 0:
        proc_scale = GpuPreprocessor.auto_scale(first_frame.shape[1], first_frame.shape[0])
        print(f"自動選擇偵測縮放比例 {proc_scale:.2f}")
    else:
        proc_scale = max(0.3, min(1.0, args.process_scale))
        if abs(proc_scale - args.process_scale) > 1e-3:
            print("偵測縮放比例已限制在 0.3 ~ 1.0 範圍內")
        print(f"使用手動偵測縮放比例 {proc_scale:.2f}")
    preprocessor = GpuPreprocessor(proc_scale)
    class_filter = None
    if args.person_only:
        class_filter = lambda dets: filter_classes(dets, [0]) if dets.size else dets
        print("僅追蹤 person 類別。若要偵測車輛請移除 --person-only。")

    detector_worker = AsyncDetector(
        detector=detector,
        preprocessor=preprocessor,
        class_filter=class_filter,
    )
    prefetcher = FramePrefetcher(cap)
    reid_helper = ReIDHelper(embedder, max_candidates=REID_CANDIDATES, target_iou=TARGET_IOU) if args.enable_reid else None

    running = True
    frame_count = 0
    last_frame_time: Optional[float] = None
    last_control_time: Optional[float] = None
    next_frame_time: Optional[float] = time.time() if target_period > 0 else None

    def process_result(result_frame: np.ndarray, detections: np.ndarray) -> bool:
        nonlocal frame_count, last_frame_time, last_control_time, next_frame_time
        if not viewer.is_open():
            print("視窗已關閉，停止播放。")
            return False
        frame_count += 1
        now = time.time()
        frame_dt = 0.0 if last_frame_time is None else now - last_frame_time
        last_frame_time = now
        fps_value = fps_meter.update(frame_dt)

        embeddings = None
        if reid_helper is not None and detections.size:
            embeddings = reid_helper.build_embeddings(result_frame, detections, target_ctrl.last_bbox)

        tracks = tracker.update(detections, result_frame.shape, embeddings=embeddings)
        target_ctrl.maintain(tracks)

        click = viewer.poll_click()
        if click is not None:
            target_ctrl.select_by_point(tracks, click)

        state = target_ctrl.current_state(tracks)
        if state is not None:
            ctrl_dt = now - last_control_time if last_control_time is not None else 1.0 / args.fps
            last_control_time = now
            err_x, err_y = TargetController.compute_error(state, result_frame.shape)
            pan_cmd = pid_pan.update(err_x, ctrl_dt)
            tilt_cmd = pid_tilt.update(err_y, ctrl_dt)
            gimbal.send(pan_cmd, tilt_cmd)

        viewer.render(result_frame, tracks, target_ctrl.target_id, fps=fps_value)
        key = viewer.wait_key(1)
        if key in (ord("q"), 27):
            return False
        if next_frame_time is not None:
            next_frame_time += target_period
            sleep_time = next_frame_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_frame_time = time.time()
        if args.max_frames is not None and frame_count >= args.max_frames:
            print(f"已達影格上限 {args.max_frames}，提前結束。")
            return False
        if fps_value > 0 and frame_count % max(args.fps, 30) == 0:
            print(f"近期平均 FPS: {fps_value:.1f}")
        return True

    detector_worker.submit(first_frame)

    try:
        with GimbalController(args.serial_port, dry_run=args.dry_run) as gimbal:
            while running:
                ret, frame = prefetcher.read()
                if not ret:
                    break
                ready = detector_worker.submit(frame)
                if ready is None:
                    continue
                running = process_result(ready.frame, ready.detections)
            if running:
                final_result = detector_worker.flush()
                if final_result is not None:
                    running = process_result(final_result.frame, final_result.detections)
    finally:
        detector_worker.shutdown()
        prefetcher.stop()
        cap.release()
        viewer.close()


if __name__ == "__main__":
    main()
