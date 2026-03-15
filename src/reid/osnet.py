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
    """Thin wrapper around a torchreid OSNet model for Re-ID feature extraction.

    On construction the model is loaded, optionally initialised from a custom
    weight file, moved to the target device, and set to inference mode.  A
    warmup forward pass determines the feature vector dimension.

    Args:
        device: Torch device string (e.g. ``"cuda:0"`` or ``"cpu"``).
            Defaults to ``"cuda:0"`` when a GPU is available, else ``"cpu"``.
        use_half: Enable FP16 inference. Only effective on CUDA devices.
        image_size: ``(height, width)`` after resizing. Defaults to
            ``(256, 128)`` — the standard OSNet training resolution.
        model_name: torchreid model identifier (e.g. ``"osnet_x0_5"``).
        weights_path: Optional path to a custom pre-trained weight file.

    Raises:
        RuntimeError: If a CUDA device is requested but no GPU is available.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        use_half: bool = False,
        image_size: Sequence[int] = (256, 128),
        model_name: str = "osnet_x0_25",
        weights_path: Optional[str] = None,
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
        self.model_name = model_name
        self.model = torchreid.models.build_model(model_name, num_classes=1, pretrained=True)
        if weights_path:
            torchreid.utils.load_pretrained_weights(self.model, weights_path)
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
        # Warmup: determine feature vector dimensionality.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, height, width, device=self.device)
            if self.use_half:
                dummy = dummy.half()
            feature_sample = self.model(dummy)
        self.feature_dim = int(feature_sample.shape[1])

    @torch.inference_mode()
    def encode(self, image: np.ndarray) -> np.ndarray:
        """Extract a single L2-normalised feature vector from a BGR crop.

        Args:
            image: BGR uint8 crop of the target (any size; resized internally).

        Returns:
            1-D float32 array of length ``self.feature_dim``.

        Raises:
            ValueError: If *image* is empty or encoding fails.
        """
        features = self.encode_batch([image])
        if not features:
            raise ValueError("Re-ID 輸入影像為空")
        first = features[0]
        if first is None:
            raise ValueError("Re-ID 輸入影像為空")
        return first

    @torch.inference_mode()
    def encode_batch(self, images: Sequence[np.ndarray]) -> List[Optional[np.ndarray]]:
        """Extract L2-normalised feature vectors for a batch of BGR crops.

        ``None`` entries in *images* (or zero-size arrays) are passed through
        as ``None`` in the returned list so index alignment is preserved.

        Args:
            images: Sequence of BGR uint8 crops.

        Returns:
            List of float32 arrays (or ``None``) with the same length as
            *images*.
        """
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
        """Crop detections from *frame* and return their feature vectors.

        Args:
            frame: Full BGR frame from which crops are taken.
            boxes: Iterable of ``(x1, y1, x2, y2)`` bounding boxes.
            padding: Pixels of context to add around each box. Defaults to 4.

        Returns:
            Feature list aligned with *boxes*; entries are ``None`` for
            invalid or out-of-bounds crops.
        """
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
        """Compute cosine similarity between two feature vectors.

        Args:
            vec_a: First feature vector (need not be normalised).
            vec_b: Second feature vector.

        Returns:
            Scalar in ``[-1, 1]``.
        """
        denom = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)) + 1e-12
        return float(np.dot(vec_a, vec_b) / denom)
