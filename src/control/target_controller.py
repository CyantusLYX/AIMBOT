from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class TargetState:
    track_id: int
    bbox: np.ndarray
    score: float


class TargetController:
    def __init__(
        self,
        max_lost_frames: int = 60,
        reacquire_thresh: float = 0.6,
    ) -> None:
        self.target_ids: set[int] = set()
        self.primary_target_id: Optional[int] = None
        self.cached_feature: Optional[np.ndarray] = None
        self._lost_frames: int = 0
        self.max_lost_frames = max(1, int(max_lost_frames))
        self.reacquire_thresh = float(reacquire_thresh)
        self.last_bbox: Optional[np.ndarray] = None
        self.ref_histogram: Optional[np.ndarray] = None
        # 初始化 ORB 特徵提取器 (快速且抗旋轉)
        self.orb = cv2.ORB_create(nfeatures=500, scoreType=cv2.ORB_FAST_SCORE)
        self.ref_keypoints = None
        self.ref_descriptors = None
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    @staticmethod
    def compute_histogram(image: np.ndarray) -> Optional[np.ndarray]:
        if image is None or image.size == 0:
            return None
        try:
            # 1. Center Crop: 只取中間 50% 區域，排除背景/馬路干擾
            h, w = image.shape[:2]
            cy, cx = h // 2, w // 2
            h_crop, w_crop = int(h * 0.5), int(w * 0.5)
            y1 = max(0, cy - h_crop // 2)
            y2 = min(h, cy + h_crop // 2)
            x1 = max(0, cx - w_crop // 2)
            x2 = min(w, cx + w_crop // 2)
            crop = image[y1:y2, x1:x2]
            
            if crop.size == 0: return None

            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            # 2. 只使用 H (Hue) 和 S (Saturation)，忽略 V (亮度) 以抵抗光影變化
            # H: 0-179 (30 bins), S: 0-255 (32 bins)
            hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
            cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            return hist
        except Exception:
            return None

    def set_reference_image(self, image: np.ndarray) -> None:
        self.ref_histogram = self.compute_histogram(image)
        # 計算參考圖片的 ORB 特徵點
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            self.ref_keypoints, self.ref_descriptors = self.orb.detectAndCompute(gray, None)
            if self.ref_keypoints:
                print(f"已提取參考圖片特徵點: {len(self.ref_keypoints)} 點")
        except Exception as e:
            print(f"ORB 特徵提取失敗: {e}")

    def select_by_point(self, tracks: List[dict], point: Tuple[int, int]) -> Optional[int]:
        if point is None:
            return self.primary_target_id
        x, y = point
        for track in tracks:
            x1, y1, x2, y2 = track["bbox"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                tid = track["track_id"]
                self.target_ids = {tid}
                self.primary_target_id = tid
                feature = track.get("feature")
                if feature is not None:
                    self.cached_feature = feature.astype(np.float32, copy=True)
                self.last_bbox = track["bbox"].astype(np.float32, copy=True)
                self._lost_frames = 0
                return tid
        return self.primary_target_id

    def update_by_reid(self, track_id: int) -> None:
        self.target_ids = {track_id}
        self.primary_target_id = track_id
        self._lost_frames = 0

    def set_reference_feature(self, feature: np.ndarray) -> None:
        if feature is not None:
            self.cached_feature = feature.astype(np.float32, copy=True)
            self.target_ids = set()
            self.primary_target_id = None
            self._lost_frames = 0

    def search_and_lock(self, tracks: List[dict], frame: Optional[np.ndarray] = None) -> bool:
        if self.cached_feature is None:
            return False

        found_any = False
        best_score = 0.0
        best_id = None

        # 掃描所有軌跡
        for track in tracks:
            feature = track.get("feature")
            if feature is None:
                continue
            
            # 1. ReID 特徵相似度 (基礎分數)
            denom = (np.linalg.norm(self.cached_feature) * np.linalg.norm(feature)) + 1e-12
            reid_score = float(np.dot(self.cached_feature, feature) / denom)
            
            # 2. 顏色直方圖相似度 (加分項)
            color_score = 0.0
            orb_score = 0.0
            has_color = False
            has_orb = False
            
            if frame is not None:
                bbox = track["bbox"].astype(int)
                x1, y1, x2, y2 = bbox
                h, w = frame.shape[:2]
                x1 = max(0, x1); y1 = max(0, y1); x2 = min(w, x2); y2 = min(h, y2)
                
                if x2 > x1 and y2 > y1:
                    crop = frame[y1:y2, x1:x2]
                    
                    # A. 顏色比對
                    if self.ref_histogram is not None:
                        track_hist = self.compute_histogram(crop)
                        if track_hist is not None:
                            color_score = cv2.compareHist(self.ref_histogram, track_hist, cv2.HISTCMP_CORREL)
                            color_score = max(0.0, color_score)
                            has_color = True
                    
                    # B. ORB 特徵點比對 (針對明顯特徵與角度變化)
                    if self.ref_descriptors is not None and crop.size > 0:
                        try:
                            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                            kp, des = self.orb.detectAndCompute(gray_crop, None)
                            if des is not None and len(des) > 0:
                                matches = self.bf_matcher.match(self.ref_descriptors, des)
                                # 篩選好的匹配點 (距離越小越好)
                                good_matches = [m for m in matches if m.distance < 60]
                                # 分數計算：匹配點數量 / 參考圖特徵點數量 (取 min 避免超過 1.0)
                                if len(self.ref_keypoints) > 0:
                                    orb_score = min(1.0, len(good_matches) / (len(self.ref_keypoints) * 0.2 + 1e-5))
                                    # 0.2 是一個經驗係數，假設只要有 20% 的特徵點對上就算很像了
                                    has_orb = True
                        except Exception:
                            pass

            # 3. 綜合評分策略
            final_score = 0.0
            weights = 0.0
            
            # ReID (基礎分)
            final_score += 0.2 * reid_score
            weights += 0.2
            
            # Color (顏色分)
            if has_color:
                final_score += 0.5 * color_score
                weights += 0.5
                
            # ORB (細節特徵分)
            if has_orb:
                final_score += 0.3 * orb_score
                weights += 0.3
            
            if weights > 0:
                final_score /= weights
            
            # Debug: 印出分數細節
            if final_score > 0.4:
                print(f"ID {track['track_id']}: ReID={reid_score:.2f}, Color={color_score:.2f}, ORB={orb_score:.2f}, Final={final_score:.2f}")

            # 門檻判斷
            if final_score >= self.reacquire_thresh:
                self.target_ids.add(track["track_id"])
                found_any = True
                if final_score > best_score:
                    best_score = final_score
                    best_id = track["track_id"]

        if found_any:
            if self.primary_target_id is None or (best_id is not None and best_score > 0.6): # 稍微降低搶奪主控權的門檻
                 if best_id is not None:
                    self.primary_target_id = best_id
            
            self._lost_frames = 0
            return True
            
        return False

    def maintain(self, tracks: List[dict]) -> None:
        if not self.target_ids and self.primary_target_id is None:
            return

        current_track_ids = {t["track_id"] for t in tracks}
        
        # 移除已經消失太久的目標 (這裡簡化處理，只要當前畫面沒有就視為 lost，若超過 max_lost_frames 則從集合移除)
        # 為了簡單起見，我們只對 primary_target 計算 lost_frames
        
        primary_found = False
        for track in tracks:
            tid = track["track_id"]
            if tid in self.target_ids:
                # 更新特徵 (如果是主要目標)
                if tid == self.primary_target_id:
                    primary_found = True
                    self.last_bbox = track["bbox"].astype(np.float32, copy=True)
                    # 選擇性更新 cached_feature? 這裡先不更新以免漂移
        
        if primary_found:
            self._lost_frames = 0
        else:
            self._lost_frames += 1
            
        # 如果主要目標丟失太久，清除鎖定狀態（保留基準特徵，等目標回來可重新鎖定）
        if self._lost_frames >= self.max_lost_frames:
            self.reset()
        # 另外，如果 target_ids 裡的目標在畫面中消失，是否要移除？
        # 這裡採取寬鬆策略：只負責加入符合條件的，不主動移除（除非 reset）
        # 或者可以每次 search_and_lock 都重新掃描

    @property
    def target_id(self) -> Optional[int]:
        return self.primary_target_id

    @target_id.setter
    def target_id(self, value: Optional[int]) -> None:
        self.primary_target_id = value
        if value is None:
            self.target_ids = set()
        else:
            self.target_ids = {value}

    def current_state(self, tracks: List[dict]) -> Optional[TargetState]:
        if self.primary_target_id is None:
            return None
        for track in tracks:
            if track["track_id"] == self.primary_target_id:
                return TargetState(
                    track_id=self.primary_target_id,
                    bbox=track["bbox"],
                    score=track["score"],
                )
        return None

    def reset(self) -> None:
        """重置追蹤狀態，但保留基準特徵，使目標回到畫面後仍可自動重新鎖定。"""
        self.primary_target_id = None
        self.target_ids = set()
        self._lost_frames = 0
        self.last_bbox = None

    def clear_reference(self) -> None:
        """完全清除所有狀態，包含基準特徵（僅在換參考目標時呼叫）。"""
        self.reset()
        self.cached_feature = None
        self.ref_histogram = None
        self.ref_keypoints = None
        self.ref_descriptors = None

    @staticmethod
    def compute_error(state: TargetState, frame_shape: Iterable[int]) -> Tuple[float, float]:
        height, width = frame_shape[:2]
        x1, y1, x2, y2 = state.bbox
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        error_x = cx - (width * 0.5)
        error_y = cy - (height * 0.5)
        return error_x, error_y
