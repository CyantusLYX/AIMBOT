from __future__ import annotations

import pathlib
from typing import Optional

import numpy as np
import torch

try:
    import torchreid
except ImportError as exc:  # pragma: no cover
    raise ImportError("缺少 torchreid，相依套件列於 requirements.txt。") from exc


class OSNetEmbedder:
    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torchreid.models.build_model("osnet_x0_25", num_classes=1, pretrained=True)
        self.model.to(self.device)
        self.model.eval()
        self.transforms = torchreid.data.transforms.build_transform(
            resize_h=256,
            resize_w=128,
            pixel_mean=[0.485, 0.456, 0.406],
            pixel_std=[0.229, 0.224, 0.225],
            interpolation=3,
        )

    @torch.inference_mode()
    def encode(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            raise ValueError("Re-ID 輸入影像為空")
        tensor = self.transforms(image).unsqueeze(0).to(self.device)
        features = self.model(tensor)
        vec = features[0].detach().cpu().numpy()
        norm = np.linalg.norm(vec) + 1e-12
        return vec / norm

    @staticmethod
    def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        denom = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)) + 1e-12
        return float(np.dot(vec_a, vec_b) / denom)
