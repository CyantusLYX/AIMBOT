from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import numpy as np
import torch
from torchvision import transforms

import cv2

try:
    import torchreid
except ImportError as exc:  # pragma: no cover
    raise ImportError("缺少 torchreid，相依套件列於 requirements.txt。") from exc


class OSNetEmbedder:
    def __init__(
        self,
        device: Optional[str] = None,
        use_half: bool = False,
        image_size: Sequence[int] = (256, 128),
    ) -> None:
        default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if device is None:
            target_device = default_device
        else:
            target_device = device
            if target_device == "cuda":
                target_device = "cuda:0"
        if target_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("要求使用 CUDA 作為 Re-ID 裝置，但未偵測到可用 GPU。")
        self.device = target_device
        self.model = torchreid.models.build_model("osnet_x0_25", num_classes=1, pretrained=True)
        self.model.to(self.device)
        self.use_half = bool(use_half and self.device.startswith("cuda"))
        if self.use_half:
            self.model.half()
        self.model.eval()
        height, width = int(image_size[0]), int(image_size[1])
        self.transforms = transforms.Compose(
            [
                transforms.ToPILImage(mode="RGB"),
                transforms.Resize((height, width)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 3, height, width, device=self.device)
            if self.use_half:
                dummy = dummy.half()
            feature_sample = self.model(dummy)
        self.feature_dim = int(feature_sample.shape[1])

    @torch.inference_mode()
    def encode(self, image: np.ndarray) -> np.ndarray:
        features = self.encode_batch([image])
        if not features:
            raise ValueError("Re-ID 輸入影像為空")
        first = features[0]
        if first is None:
            raise ValueError("Re-ID 輸入影像為空")
        return first

    @torch.inference_mode()
    def encode_batch(self, images: Sequence[np.ndarray]) -> List[Optional[np.ndarray]]:
        if images is None or len(images) == 0:
            return []
        tensors: List[Optional[torch.Tensor]] = []
        for img in images:
            if img is None or img.size == 0:
                tensors.append(None)
                continue
            if img.dtype != np.uint8:
                clipped = np.clip(img, 0, 255).astype(np.uint8)
            else:
                clipped = img
            rgb = cv2.cvtColor(clipped, cv2.COLOR_BGR2RGB)
            tensor = self.transforms(rgb)
            if self.use_half:
                tensor = tensor.half()
            tensors.append(tensor)

        valid_indices = [idx for idx, tensor in enumerate(tensors) if tensor is not None]
        if not valid_indices:
            return [None for _ in images]

        batch = torch.stack([tensors[idx] for idx in valid_indices], dim=0).to(self.device)
        features = self.model(batch)
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        feats_np = features.detach().cpu().float().numpy()

        result: List[Optional[np.ndarray]] = []
        feat_iter = iter(feats_np)
        for tensor in tensors:
            if tensor is None:
                result.append(None)
            else:
                result.append(next(feat_iter).astype(np.float32))
        return result

    def extract_from_frame(
        self,
        frame: np.ndarray,
        boxes: Iterable[Sequence[float]],
        padding: int = 4,
    ) -> List[Optional[np.ndarray]]:
        if frame is None or frame.size == 0:
            return []
        h, w = frame.shape[:2]
        crops: List[Optional[np.ndarray]] = []
        for box in boxes:
            if box is None:
                crops.append(None)
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            x1 -= padding
            y1 -= padding
            x2 += padding
            y2 += padding
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))
            if x2 <= x1 or y2 <= y1:
                crops.append(None)
                continue
            crop = frame[y1:y2, x1:x2]
            crops.append(crop)
        return self.encode_batch(crops)

    @staticmethod
    def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        denom = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)) + 1e-12
        return float(np.dot(vec_a, vec_b) / denom)
