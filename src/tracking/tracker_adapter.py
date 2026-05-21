"""Adapter that prefers a C++ ByteTrack binding and falls back to Python."""
from __future__ import annotations

import importlib
from typing import Iterable, Optional

import numpy as np

from tracking.byte_tracker import ByteTrack


class ByteTrackAdapter:
    """Normalise ByteTrack implementations to the repo's track-dict contract."""

    def __init__(
        self,
        cpp_module: str = "bytetrack_cpp",
        track_thresh: float = 0.65,
        match_iou_thresh: float = 0.2,
        max_age: int = 15,
        min_hits: int = 3,
        enable_reid: bool = False,
        reid_match_thresh: float = 0.65,
        reid_max_center_dist: float = 0.25,
        require_cpp: bool = False,
    ) -> None:
        if enable_reid and require_cpp:
            raise RuntimeError("Re-ID requires the Python ByteTrack path; --require-cpp-tracker is incompatible")

        self.kind = "python"
        self._impl = None
        self._min_hits = max(1, int(min_hits))
        self._cpp_seen_counts: dict[int, int] = {}
        if not enable_reid:
            self._try_load_cpp(cpp_module, track_thresh, match_iou_thresh, max_age)
        if self._impl is None:
            if require_cpp:
                raise RuntimeError(f"required C++ ByteTrack binding is unavailable: {cpp_module}")
            self._impl = ByteTrack(
                track_thresh=track_thresh,
                match_iou_thresh=match_iou_thresh,
                max_age=max_age,
                min_hits=self._min_hits,
                enable_reid=enable_reid,
                reid_match_thresh=reid_match_thresh,
                feature_min_similarity=reid_match_thresh,
                reid_max_center_dist=reid_max_center_dist,
            )
            suffix = " + Re-ID" if enable_reid else " fallback"
            print(f"[TRACKER] using Python ByteTrack{suffix}")

    def update(
        self,
        detections: Optional[np.ndarray],
        img_size: Iterable[int],
        embeddings: Optional[list[Optional[np.ndarray]]] = None,
    ) -> list[dict]:
        if self.kind == "python":
            return self._impl.update(detections, img_size, embeddings=embeddings)  # type: ignore[union-attr]
        return self._update_cpp(detections, img_size)

    def _try_load_cpp(self, module_name: str, track_thresh: float, match_iou_thresh: float, max_age: int) -> None:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            return
        cls = getattr(module, "BYTETracker", None) or getattr(module, "ByteTrack", None)
        if cls is None:
            return
        candidates = [
            {
                "track_thresh": track_thresh,
                "match_thresh": match_iou_thresh,
                "track_buffer": max_age,
                "frame_rate": 30,
            },
            {
                "track_thresh": track_thresh,
                "match_iou_thresh": match_iou_thresh,
                "max_age": max_age,
            },
            {},
        ]
        for kwargs in candidates:
            try:
                self._impl = cls(**kwargs)
                self.kind = "cpp"
                print(f"[TRACKER] using C++ ByteTrack binding: {module_name}.{cls.__name__}")
                return
            except TypeError:
                continue
            except Exception:
                return

    def _update_cpp(self, detections: Optional[np.ndarray], img_size: Iterable[int]) -> list[dict]:
        dets = detections if detections is not None else np.empty((0, 6), dtype=np.float32)
        for method_name in ("update", "update_tracks"):
            method = getattr(self._impl, method_name, None)
            if method is None:
                continue
            for args in ((dets, img_size), (dets,)):
                try:
                    return self._normalise_cpp_tracks(method(*args))
                except TypeError:
                    continue
        raise RuntimeError("C++ ByteTrack binding does not expose update/update_tracks")

    def _normalise_cpp_tracks(self, tracks) -> list[dict]:
        normalised = self._normalise_tracks(tracks)
        if self._min_hits <= 1:
            return normalised

        live_ids = {int(track["track_id"]) for track in normalised}
        for track_id in list(self._cpp_seen_counts):
            if track_id not in live_ids:
                self._cpp_seen_counts.pop(track_id, None)

        stable: list[dict] = []
        for track in normalised:
            track_id = int(track["track_id"])
            hits = self._cpp_seen_counts.get(track_id, 0) + 1
            self._cpp_seen_counts[track_id] = hits
            if hits >= self._min_hits:
                track["is_confirmed"] = True
                stable.append(track)
        return stable

    def _normalise_tracks(self, tracks) -> list[dict]:
        if tracks is None:
            return []
        normalised: list[dict] = []
        for track in tracks:
            item = self._normalise_one(track)
            if item is not None:
                normalised.append(item)
        return normalised

    def _normalise_one(self, track) -> Optional[dict]:
        if isinstance(track, dict):
            bbox = track.get("bbox")
            if bbox is None:
                bbox = track.get("tlbr")
            if bbox is None:
                bbox = track.get("xyxy")
            track_id = track.get("track_id", track.get("id", track.get("tracklet_id")))
            score = track.get("score", track.get("confidence", 1.0))
        else:
            bbox = getattr(track, "bbox", None)
            if bbox is None and hasattr(track, "tlbr"):
                bbox = track.tlbr
            if bbox is None and hasattr(track, "tlwh"):
                x, y, w, h = np.asarray(track.tlwh, dtype=np.float32).reshape(4)
                bbox = np.array([x, y, x + w, y + h], dtype=np.float32)
            track_id = getattr(track, "track_id", getattr(track, "id", None))
            score = getattr(track, "score", getattr(track, "confidence", 1.0))

        if bbox is None or track_id is None:
            arr = np.asarray(track, dtype=np.float32).reshape(-1)
            if arr.size >= 5:
                bbox = arr[:4]
                track_id = int(arr[4])
                score = float(arr[5]) if arr.size > 5 else 1.0
            else:
                return None

        bbox_arr = np.asarray(bbox, dtype=np.float32).reshape(-1)
        if bbox_arr.size != 4:
            return None
        return {
            "track_id": int(track_id),
            "bbox": bbox_arr.copy(),
            "score": float(score),
            "class_id": -1,
            "is_confirmed": True,
            "feature": None,
            "time_since_update": 0,
        }
