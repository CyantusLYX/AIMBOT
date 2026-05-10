import pathlib
import sys
from typing import List, Optional

import numpy as np
import torch


class YoloV7Detector:
    """Thin wrapper around a YOLOv7 PyTorch model."""

    def __init__(
        self,
        weights_path: str = "models/epoch_149.pt",
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
            target_device = "cuda:0" if device == "cuda" else device
        if target_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("已要求使用 CUDA，但目前環境無法偵測到支援的 GPU 或 CUDA 驅動程式。")
        self.device = target_device

        original_load = torch.load

        def _patched_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original_load(*args, **kwargs)

        torch.load = _patched_load
        try:
            self.model = self._load_model(resolved)
        finally:
            torch.load = original_load

        self.model.conf = conf_threshold
        self.model.iou = iou_threshold
        self.model.to(self.device)
        self.using_half = bool(use_half and self.device.startswith("cuda"))
        if self.using_half:
            self.model.half()
        self.model.eval()

    def _load_model(self, resolved: pathlib.Path):
        """Load model from local hub cache (avoids pkg_resources / network dependency)."""
        hub_dir = pathlib.Path(torch.hub.get_dir())
        yolov7_dir = hub_dir / "WongKinYiu_yolov7_main"

        if not yolov7_dir.exists():
            raise RuntimeError(
                f"找不到 YOLOv7 本地快取目錄: {yolov7_dir}\n"
                "請先在有網路的環境下執行一次以下指令讓 torch.hub 下載 repo：\n"
                "  python -c \"import torch; torch.hub.load('WongKinYiu/yolov7', 'custom', 'models/epoch_149.pt', trust_repo=True)\""
            )

        if str(yolov7_dir) not in sys.path:
            sys.path.insert(0, str(yolov7_dir))

        # Use torch.hub.load with source='local' so it goes through hubconf.py (autoShape),
        # but skips the GitHub download and uses the local cache directory directly.
        model = torch.hub.load(
            str(yolov7_dir),
            "custom",
            str(resolved),
            source="local",
            trust_repo=True,
        )
        print("模型已從本地快取載入")
        return model

    @torch.inference_mode()
    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Return detections in xyxy+score+class format."""
        if frame is None or frame.size == 0:
            return np.empty((0, 6), dtype=np.float32)
        results = self.model(frame)
        # AutoShape returns an Inference object with .xyxy list
        if hasattr(results, "xyxy"):
            detections = results.xyxy[0]
            if detections is None or detections.nelement() == 0:
                return np.empty((0, 6), dtype=np.float32)
            return detections.detach().cpu().numpy()
        # Raw model returns a tuple; first element is (1, N, 85+) prediction tensor
        preds = results[0] if isinstance(results, (tuple, list)) else results
        if preds.ndim == 3:
            preds = preds[0]  # remove batch dim → (N, 85+)
        if preds.shape[0] == 0:
            return np.empty((0, 6), dtype=np.float32)
        boxes = preds[:, :4]
        scores = preds[:, 4] * preds[:, 5:].max(dim=1).values
        classes = preds[:, 5:].argmax(dim=1).float()
        out = torch.cat([boxes, scores.unsqueeze(1), classes.unsqueeze(1)], dim=1)
        return out.detach().cpu().numpy()

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
