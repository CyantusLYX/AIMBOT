import pathlib
from typing import List, Optional

import numpy as np
import torch


class YoloV7Detector:
    """Thin wrapper around a YOLOv7 PyTorch model."""

    def __init__(
        self,
        weights_path: str = "models/yolov7.pt",
        device: Optional[str] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        use_half: bool = False,
    ) -> None:
        resolved = pathlib.Path(weights_path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"找不到權重檔案: {resolved}")
        default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if device is None:
            target_device = default_device
        else:
            target_device = device
            if target_device == "cuda":
                target_device = "cuda:0"
        if target_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("已要求使用 CUDA，但目前環境無法偵測到支援的 GPU 或 CUDA 驅動程式。")
        self.device = target_device
        if hasattr(np, "core") and hasattr(np.core, "multiarray"):
            reconstruct = getattr(np.core.multiarray, "_reconstruct", None)
            if reconstruct is not None:
                torch.serialization.add_safe_globals([reconstruct])

        original_load = torch.load

        def _patched_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original_load(*args, **kwargs)

        torch.load = _patched_load
        try:
            self.model = torch.hub.load(
                "WongKinYiu/yolov7",
                "custom",
                str(resolved),
                source="github",
                trust_repo=True,
            )
        finally:
            torch.load = original_load
        self.model.conf = conf_threshold
        self.model.iou = iou_threshold
        self.model.to(self.device)
        self.using_half = bool(use_half and self.device.startswith("cuda"))
        if self.using_half:
            self.model.half()
        self.model.eval()

    @torch.inference_mode()
    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Return detections in xyxy+score+class format."""
        if frame is None or frame.size == 0:
            return np.empty((0, 6), dtype=np.float32)
        results = self.model(frame)
        # results.xyxy[0] shape: (N, 6) -> x1, y1, x2, y2, score, class
        detections = results.xyxy[0]
        if detections is None or detections.nelement() == 0:
            return np.empty((0, 6), dtype=np.float32)
        return detections.detach().cpu().numpy()

    def warmup(self, image_size: int = 640, iterations: int = 1) -> None:
        dummy = torch.zeros(1, 3, image_size, image_size, device=self.device)
        if self.using_half:
            dummy = dummy.half()
        for _ in range(iterations):
            self.model(dummy)


def filter_classes(detections: np.ndarray, allowed: Optional[List[int]] = None) -> np.ndarray:
    if allowed is None or len(allowed) == 0:
        return detections
    mask = np.isin(detections[:, 5].astype(int), allowed)
    return detections[mask]
