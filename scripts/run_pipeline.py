from __future__ import annotations

import argparse
import queue
import sys
import threading
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
    parser.add_argument("--enable-reid", action="store_true", help="啟用 OSNet Re-ID 協助找回遺失目標")
    parser.add_argument("--reid-thresh", type=float, default=0.45, help="Re-ID 相似度門檻")
    parser.add_argument("--reid-momentum", type=float, default=0.9, help="Re-ID 特徵更新動量")
    parser.add_argument("--reid-topk", type=int, default=8, help="每幀僅對前 K 筆偵測執行 Re-ID，0 表示全部")
    parser.add_argument("--reid-max-dist", type=float, default=0.35, help="Re-ID 允許的中心距離 (以影像對角線比例表示)")
    parser.add_argument("--reid-min-sim", type=float, default=0.55, help="更新 Re-ID 特徵所需的最小相似度")
    parser.add_argument("--process-scale", type=float, default=1.0, help="偵測與追蹤使用的影像縮放比例 (0.3~1.0)")
    parser.add_argument("--kp", type=float, default=0.005)
    parser.add_argument("--ki", type=float, default=0.0001)
    parser.add_argument("--kd", type=float, default=0.0005)
    return parser.parse_args()


class FramePrefetcher:
    def __init__(self, cap: cv2.VideoCapture, queue_size: int = 4) -> None:
        self._cap = cap
        self._queue: "queue.Queue[Optional[np.ndarray]]" = queue.Queue(maxsize=max(1, queue_size))
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        while not self._stop_event.is_set():
            if self._queue.full():
                time.sleep(0.002)
                continue
            ret, frame = self._cap.read()
            if not ret:
                self._queue.put(None)
                break
            self._queue.put(frame)

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        frame = self._queue.get()
        if frame is None:
            return False, None
        return True, frame

    def stop(self) -> None:
        self._stop_event.set()
        # 推入 sentinel 確保讀取解除阻塞
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=0.5)


def create_capture(source: str) -> cv2.VideoCapture:
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟來源: {source}")
    return cap


def _bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    union = area_a + area_b - inter + 1e-12
    return inter / union


def main() -> None:
    args = parse_args()
    detector = YoloV7Detector(weights_path=args.weights, device=args.device, use_half=args.half)
    if detector.device.startswith("cuda"):
        detector.warmup()
    if args.half and not detector.using_half:
        print("警告: --half 選項僅在 CUDA 裝置上有效，已回退至 float32。")
    print(f"使用推論裝置: {detector.device}{' (FP16)' if detector.using_half else ''}")
    tracker = ByteTrack(
        enable_reid=args.enable_reid,
        reid_match_thresh=args.reid_thresh,
        feature_momentum=args.reid_momentum,
        feature_min_similarity=args.reid_min_sim,
        reid_max_center_dist=args.reid_max_dist,
    )
    target_ctrl = TargetController(
        max_lost_frames=int(args.fps * 2),
        reacquire_thresh=max(args.reid_thresh, args.reid_min_sim),
    )
    pid_pan = PIDController(PIDGains(args.kp, args.ki, args.kd))
    pid_tilt = PIDController(PIDGains(args.kp, args.ki, args.kd))
    viewer = OpenCVViewer()
    embedder: Optional["OSNetEmbedder"] = None
    if args.enable_reid:
        from reid.osnet import OSNetEmbedder

        embedder = OSNetEmbedder(device=detector.device, use_half=args.half)
    cap = create_capture(args.source)
    process_scale = float(args.process_scale)
    if process_scale <= 0:
        process_scale = 1.0
    process_scale = min(max(process_scale, 0.3), 1.0)
    prefetcher = FramePrefetcher(cap)
    try:
        with GimbalController(args.serial_port, dry_run=args.dry_run) as gimbal:
            fps_meter = FPSMeter(window=max(args.fps, 30))
            last_frame_time: Optional[float] = None
            last_control_time: Optional[float] = None
            frame_count = 0
            while True:
                ret, frame = prefetcher.read()
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
                proc_frame = frame
                if process_scale < 0.999:
                    target_w = max(1, int(round(frame.shape[1] * process_scale)))
                    target_h = max(1, int(round(frame.shape[0] * process_scale)))
                    proc_frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
                detections = detector.detect(proc_frame)
                if detections.size:
                    scale_x = proc_frame.shape[1] / frame.shape[1]
                    scale_y = proc_frame.shape[0] / frame.shape[0]
                    if abs(scale_x - 1.0) > 1e-3 or abs(scale_y - 1.0) > 1e-3:
                        detections = detections.copy()
                        detections[:, [0, 2]] /= max(scale_x, 1e-6)
                        detections[:, [1, 3]] /= max(scale_y, 1e-6)
                        detections[:, [0, 2]] = np.clip(detections[:, [0, 2]], 0, frame.shape[1] - 1)
                        detections[:, [1, 3]] = np.clip(detections[:, [1, 3]], 0, frame.shape[0] - 1)
                if args.person_only:
                    detections = filter_classes(detections, [0])
                reid_embeddings = None
                if embedder is not None and detections.size:
                    num_dets = len(detections)
                    if args.reid_topk > 0 and num_dets > args.reid_topk:
                        order = np.argsort(-detections[:, 4])[: args.reid_topk]
                    else:
                        order = np.arange(num_dets)
                    if target_ctrl.last_bbox is not None:
                        extra_indices = [
                            idx
                            for idx in range(num_dets)
                            if _bbox_iou(detections[idx, :4], target_ctrl.last_bbox) >= 0.1
                        ]
                        if extra_indices:
                            order = np.unique(
                                np.concatenate([order, np.array(extra_indices, dtype=np.int32)])
                            )
                    selected_boxes = detections[order, :4]
                    selected_embeddings = embedder.extract_from_frame(frame, selected_boxes)
                    reid_embeddings = [None] * num_dets
                    for idx, emb in zip(order, selected_embeddings):
                        reid_embeddings[idx] = emb
                tracks = tracker.update(detections, frame.shape, embeddings=reid_embeddings)
                target_ctrl.maintain(tracks)
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
        prefetcher.stop()
        cap.release()
        viewer.close()


if __name__ == "__main__":
    main()
