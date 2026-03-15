from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Deque, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adapters.video_source import create_capture
from control.pid import PIDController, PIDGains
from control.target_controller import TargetController
from core.config import PipelineConfig
from pipeline.workers import AsyncDetector, FramePrefetcher, GpuPreprocessor, ReIDHelper
from services.tracking_service import TrackingService
from tracking.byte_tracker import ByteTrack
from ui.viewer import OpenCVViewer

DEFAULT_CONFIG = PipelineConfig()


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
    runtime = DEFAULT_CONFIG.runtime
    parser = argparse.ArgumentParser(description="AIMBOT pipeline")
    parser.add_argument("--weights", type=str, default=runtime.weights, help="YOLOv7 權重路徑")
    parser.add_argument("--source", type=str, default=runtime.source, help="攝影機索引或影片路徑")
    parser.add_argument("--device", type=str, default=runtime.device, help="指定裝置，例如 cuda:0")
    parser.add_argument("--person-only", action="store_true", help="僅保留 person 類別")
    parser.add_argument("--half", action="store_true", help="啟用半精度推論 (僅限 CUDA)")
    parser.add_argument("--enable-reid", action="store_true", help="啟用 OSNet Re-ID")
    parser.add_argument(
        "--process-scale",
        type=float,
        default=runtime.process_scale,
        help="偵測影像縮放比例，1.0 為原尺寸，0 表示自動依輸入大小調整",
    )
    parser.add_argument("--reid-model", type=str, default=runtime.reid_model, help="OSNet 模型名稱 (torchreid)")
    parser.add_argument("--reid-weights", type=str, default=runtime.reid_weights, help="自訂 Re-ID 權重路徑 (可選)")
    parser.add_argument("--dry-run", action="store_true", help="僅輸出控制命令，不實際連線雲台")
    parser.add_argument("--serial-port", type=str, default=runtime.serial_port, help="雲台序列埠")
    parser.add_argument("--fps", type=int, default=runtime.fps, help="控制迴圈目標 FPS")
    parser.add_argument("--max-frames", type=int, default=runtime.max_frames, help="限制處理影格數量，用於快速測試")
    return parser.parse_args()


def build_runtime_config(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        runtime=replace(
            DEFAULT_CONFIG.runtime,
            weights=args.weights,
            source=args.source,
            device=args.device,
            person_only=args.person_only,
            half=args.half,
            enable_reid=args.enable_reid,
            process_scale=args.process_scale,
            reid_model=args.reid_model,
            reid_weights=args.reid_weights,
            dry_run=args.dry_run,
            serial_port=args.serial_port,
            fps=args.fps,
            max_frames=args.max_frames,
        ),
        tracking=DEFAULT_CONFIG.tracking,
        control=DEFAULT_CONFIG.control,
    )


def select_mode(runtime):
    mode = "1"
    print("請選擇模式:")
    print("1. 點擊追蹤 (預設)")
    print("2. 圖片特徵追蹤")
    user_input = input("請輸入選項 (1/2): ").strip()
    if user_input == "2":
        mode = "2"
        runtime = replace(runtime, enable_reid=True)
        print("已強制啟用 Re-ID 以支援特徵追蹤")

    reference_image_path = None
    if mode == "2":
        reference_image_path = input("請輸入圖片路徑: ").strip().strip('"')
        if not Path(reference_image_path).exists():
            raise FileNotFoundError(f"錯誤: 找不到檔案 {reference_image_path}")
    return mode, runtime, reference_image_path


def create_detector_and_embedder(runtime):
    try:
        from detection.detector import YoloV7Detector
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            print("錯誤: 偵測模組需要 torch，但目前環境未安裝。")
            print("建議修復:")
            print("  uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
            print("  或 CPU 版本: uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
            raise SystemExit(1) from exc
        raise

    detector = YoloV7Detector(weights_path=runtime.weights, device=runtime.device, use_half=runtime.half)
    if detector.device.startswith("cuda"):
        detector.warmup()
    if runtime.half and not detector.using_half:
        print("警告: --half 僅在 CUDA 裝置上有效，已回退至 float32")
    print(f"使用推論裝置: {detector.device}{' (FP16)' if detector.using_half else ''}")

    embedder = None
    if runtime.enable_reid:
        from reid.osnet import OSNetEmbedder

        embedder = OSNetEmbedder(
            device=detector.device,
            use_half=runtime.half,
            model_name=runtime.reid_model,
            weights_path=runtime.reid_weights,
        )
    return detector, embedder


def initialize_reference_target(mode: str, reference_image_path: Optional[str], embedder, target_ctrl: TargetController) -> bool:
    if mode != "2" or embedder is None or reference_image_path is None:
        return True
    ref_img = cv2.imread(reference_image_path)
    if ref_img is None:
        print("無法讀取參考圖片")
        return False
    try:
        ref_feature = embedder.encode(ref_img)
        target_ctrl.set_reference_feature(ref_feature)
        target_ctrl.set_reference_image(ref_img)  # 設定參考圖片以計算顏色直方圖
        print("已提取參考圖片特徵，等待目標出現...")
        return True
    except Exception as exc:
        print(f"特徵提取失敗: {exc}")
        return False


def resolve_process_scale(runtime, first_frame: np.ndarray) -> float:
    if runtime.process_scale <= 0:
        proc_scale = GpuPreprocessor.auto_scale(first_frame.shape[1], first_frame.shape[0])
        print(f"自動選擇偵測縮放比例 {proc_scale:.2f}")
        return proc_scale

    proc_scale = max(0.3, min(1.0, runtime.process_scale))
    if abs(proc_scale - runtime.process_scale) > 1e-3:
        print("偵測縮放比例已限制在 0.3 ~ 1.0 範圍內")
    print(f"使用手動偵測縮放比例 {proc_scale:.2f}")
    return proc_scale


def create_class_filter(person_only: bool):
    from detection.detector import filter_classes

    if person_only:
        print("僅追蹤 person 類別。若要偵測車輛請移除 --person-only。")
        return lambda dets: filter_classes(dets, [0]) if dets.size else dets
    return None


def _parse_major_minor(version_text: str) -> tuple[int, int] | None:
    raw = version_text.split("+", 1)[0]
    parts = raw.split(".")
    if len(parts) < 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def preflight_runtime_checks() -> None:
    ver = _parse_major_minor(np.__version__)
    if ver is not None and ver >= (1, 24):
        print(f"錯誤: 目前 numpy 版本為 {np.__version__}，YOLOv7/YOLOR 需要 numpy<1.24.0。")
        print("建議修復: uv pip install \"numpy>=1.21.0,<1.24.0\"")
        raise SystemExit(1)


def main() -> None:
    preflight_runtime_checks()

    args = parse_args()
    config = build_runtime_config(args)
    runtime = config.runtime
    tracking = config.tracking
    pid_cfg = config.control.pid
    pid_gains = PIDGains(pid_cfg.kp, pid_cfg.ki, pid_cfg.kd)

    try:
        mode, runtime, reference_image_path = select_mode(runtime)
    except FileNotFoundError as exc:
        print(exc)
        return

    detector, embedder = create_detector_and_embedder(runtime)
    tracker = ByteTrack(
        enable_reid=runtime.enable_reid,
        reid_match_thresh=tracking.reid_similarity,
        feature_momentum=tracking.feature_momentum,
        feature_min_similarity=tracking.reid_similarity,
        reid_max_center_dist=tracking.reid_distance,
    )
    target_ctrl = TargetController(max_lost_frames=int(runtime.fps * 2), reacquire_thresh=tracking.reid_similarity)
    if not initialize_reference_target(mode, reference_image_path, embedder, target_ctrl):
        return

    pid_pan = PIDController(pid_gains)
    pid_tilt = PIDController(pid_gains)
    viewer = OpenCVViewer()
    fps_meter = FPSMeter(window=max(runtime.fps, 30))

    cap = create_capture(runtime.source)
    is_camera = runtime.source.isdigit()
    playback_fps = cap.get(cv2.CAP_PROP_FPS) if not is_camera else 0.0
    if playback_fps and playback_fps < 0:
        playback_fps = 0.0
    target_period = 0.0
    if playback_fps and playback_fps >= 1.0:
        print(f"偵測到影片 FPS ~ {playback_fps:.2f}，將同步播放速度。")
        target_period = 1.0 / playback_fps

    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("來源沒有任何畫面")

    proc_scale = resolve_process_scale(runtime, first_frame)
    detector_worker = AsyncDetector(
        detector=detector,
        preprocessor=GpuPreprocessor(proc_scale),
        class_filter=create_class_filter(runtime.person_only),
    )
    prefetcher = FramePrefetcher(cap)
    reid_helper = (
        ReIDHelper(embedder, max_candidates=tracking.reid_candidates, target_iou=tracking.target_iou)
        if runtime.enable_reid
        else None
    )
    tracking_service = TrackingService(tracker=tracker, reid_helper=reid_helper)

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

        tracks = tracking_service.update(result_frame, detections, target_ctrl.last_bbox)

        if mode == "2" and target_ctrl.target_id is None:
            if target_ctrl.search_and_lock(tracks, frame=result_frame):
                print(f"已鎖定目標 ID: {target_ctrl.target_id} (共發現 {len(target_ctrl.target_ids)} 個相似目標)")
        elif mode == "2" and target_ctrl.target_id is not None:
            target_ctrl.search_and_lock(tracks, frame=result_frame)

        target_ctrl.maintain(tracks)

        click = viewer.poll_click()
        if click is not None:
            target_ctrl.select_by_point(tracks, click)

        state = target_ctrl.current_state(tracks)
        if state is not None:
            ctrl_dt = now - last_control_time if last_control_time is not None else 1.0 / runtime.fps
            last_control_time = now
            err_x, err_y = TargetController.compute_error(state, result_frame.shape)
            pan_cmd = pid_pan.update(err_x, ctrl_dt)
            tilt_cmd = pid_tilt.update(err_y, ctrl_dt)
            gimbal.send(pan_cmd, tilt_cmd)

        viewer.render(
            result_frame,
            tracks,
            target_ctrl.target_id,
            fps=fps_value,
            secondary_target_ids=target_ctrl.target_ids,
            lifecycle_state=target_ctrl.lifecycle_state.value.upper(),
        )
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

        if runtime.max_frames is not None and frame_count >= runtime.max_frames:
            print(f"已達影格上限 {runtime.max_frames}，提前結束。")
            return False

        if fps_value > 0 and frame_count % max(runtime.fps, 30) == 0:
            print(f"近期平均 FPS: {fps_value:.1f}")
        return True

    detector_worker.submit(first_frame)

    try:
        from control.gimbal_controller import GimbalController

        force_dry_run = runtime.dry_run or (not is_camera)
        if not runtime.dry_run and not is_camera:
            print("偵測到影片輸入，自動啟用 dry-run 模式 (不連線雲台)")

        with GimbalController(runtime.serial_port, dry_run=force_dry_run) as gimbal:
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
