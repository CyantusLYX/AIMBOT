import gc
import pathlib
import time
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


class _Binding:
    def __init__(self, index: int, name: str, dtype, shape: Tuple[int, ...], is_input: bool) -> None:
        self.index = index
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.is_input = is_input
        self.host = None
        self.device = None


class TensorRTYoloDetector:
    """YOLO detector backed by a serialized TensorRT engine.

    The detector accepts BGR OpenCV frames and returns the same detection
    contract as :class:`YoloV7Detector`: ``(N, 6)`` arrays with
    ``x1, y1, x2, y2, confidence, class_id``.
    """

    def __init__(
        self,
        engine_path: str,
        device: Optional[str] = None,
        input_shape: Optional[Sequence[int]] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        output_format: str = "auto",
        max_detections: int = 100,
        max_candidates: int = 1000,
        allowed_classes: Optional[Sequence[int]] = None,
        profile: bool = False,
    ) -> None:
        resolved = pathlib.Path(engine_path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError("找不到 TensorRT engine: {}".format(resolved))

        try:
            import pycuda.driver as cuda
            import tensorrt as trt
        except ImportError as exc:  # pragma: no cover - Jetson-only dependency
            raise ImportError(
                "TensorRT backend 需要 tensorrt 與 pycuda。JetPack 4.x 可用 apt 安裝 "
                "python3-libnvinfer、libnvinfer-bin、python3-pycuda。"
            ) from exc

        target_device = "cuda:0" if device in (None, "cuda") else device
        if target_device is None or not target_device.startswith("cuda"):
            raise RuntimeError("TensorRT backend 只能在 CUDA 裝置上執行，請使用 --device cuda 或 cuda:0。")
        try:
            device_id = int(target_device.split(":", 1)[1]) if ":" in target_device else 0
        except ValueError as exc:
            raise ValueError("無法解析 CUDA 裝置: {}".format(target_device)) from exc

        output_format = output_format.lower().strip()
        if output_format not in ("auto", "raw", "nms"):
            raise ValueError("--trt-output-format 必須是 auto、raw 或 nms")

        self.cuda = cuda
        self.trt = trt
        self.device = "cuda:{}".format(device_id)
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.output_format = output_format
        self.max_detections = max(1, int(max_detections))
        self.max_candidates = max(1, int(max_candidates))
        self.allowed_classes = None if allowed_classes is None else [int(cls) for cls in allowed_classes]
        self.profile = bool(profile)
        self._profile_totals = {"preprocess": 0.0, "infer": 0.0, "postprocess": 0.0, "count": 0}

        cuda.init()
        cuda_device = cuda.Device(device_id)
        if hasattr(cuda_device, "retain_primary_context"):
            self._cuda_context = cuda_device.retain_primary_context()
            self._cuda_context.push()
        else:
            self._cuda_context = cuda_device.make_context()
        self._closed = False
        try:
            self._load_engine(resolved, input_shape)
        finally:
            self._cuda_context.pop()

    def _load_engine(self, resolved: pathlib.Path, input_shape: Optional[Sequence[int]]) -> None:
        logger = self.trt.Logger(self.trt.Logger.WARNING)
        runtime = self.trt.Runtime(logger)
        with resolved.open("rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError("TensorRT engine 載入失敗: {}".format(resolved))

        self.context = self.engine.create_execution_context()
        self.stream = self.cuda.Stream()
        self.bindings: List[int] = [0] * int(self.engine.num_bindings)
        self.binding_info: List[_Binding] = []
        self.input_binding: Optional[_Binding] = None
        self.output_bindings: List[_Binding] = []

        for index in range(int(self.engine.num_bindings)):
            name = self.engine.get_binding_name(index)
            dtype = self.trt.nptype(self.engine.get_binding_dtype(index))
            shape = tuple(int(v) for v in self.engine.get_binding_shape(index))
            is_input = bool(self.engine.binding_is_input(index))
            binding = _Binding(index=index, name=name, dtype=dtype, shape=shape, is_input=is_input)
            self.binding_info.append(binding)
            if is_input:
                self.input_binding = binding
            else:
                self.output_bindings.append(binding)

        if self.input_binding is None:
            raise RuntimeError("TensorRT engine 沒有 input binding")

        requested_shape = self._resolve_input_shape(self.input_binding.shape, input_shape)
        if any(dim < 0 for dim in self.input_binding.shape):
            self.context.set_binding_shape(self.input_binding.index, requested_shape)
        self.input_binding.shape = tuple(int(v) for v in self.context.get_binding_shape(self.input_binding.index))
        if any(dim <= 0 for dim in self.input_binding.shape):
            self.input_binding.shape = requested_shape

        self._input_layout, self.input_height, self.input_width = self._parse_input_layout(self.input_binding.shape)
        self.using_half = self.input_binding.dtype == np.float16
        self._allocate_binding(self.input_binding)
        for binding in self.output_bindings:
            shape = tuple(int(v) for v in self.context.get_binding_shape(binding.index))
            if any(dim <= 0 for dim in shape):
                shape = binding.shape
            if any(dim <= 0 for dim in shape):
                raise RuntimeError(
                    "TensorRT output binding '{}' 是動態 shape {}，目前無法配置 buffer。".format(
                        binding.name, shape
                    )
                )
            binding.shape = shape
            self._allocate_binding(binding)

    def _resolve_input_shape(
        self,
        engine_shape: Tuple[int, ...],
        requested: Optional[Sequence[int]],
    ) -> Tuple[int, ...]:
        if requested is None:
            if len(engine_shape) == 4 and all(dim > 0 for dim in engine_shape):
                return engine_shape
            if len(engine_shape) == 3 and all(dim > 0 for dim in engine_shape):
                return engine_shape
            return (1, 3, 640, 640)

        dims = tuple(int(v) for v in requested)
        if len(dims) == 2:
            height, width = dims
            if len(engine_shape) == 3:
                return (3, height, width)
            return (1, 3, height, width)
        if len(dims) in (3, 4):
            return dims
        raise ValueError("--trt-input-shape 請使用 HxW、CxHxW 或 NxCxHxW")

    @staticmethod
    def _parse_input_layout(shape: Tuple[int, ...]) -> Tuple[str, int, int]:
        if len(shape) == 4:
            if shape[1] in (1, 3):
                return "nchw", int(shape[2]), int(shape[3])
            if shape[3] in (1, 3):
                return "nhwc", int(shape[1]), int(shape[2])
        if len(shape) == 3:
            if shape[0] in (1, 3):
                return "chw", int(shape[1]), int(shape[2])
            if shape[2] in (1, 3):
                return "hwc", int(shape[0]), int(shape[1])
        raise RuntimeError("不支援的 TensorRT input shape: {}".format(shape))

    def _allocate_binding(self, binding: _Binding) -> None:
        size = int(self.trt.volume(binding.shape))
        binding.host = self.cuda.pagelocked_empty(size, binding.dtype)
        binding.device = self.cuda.mem_alloc(binding.host.nbytes)
        self.bindings[binding.index] = int(binding.device)

    def detect(self, frame: np.ndarray) -> np.ndarray:
        if frame is None or frame.size == 0:
            return np.empty((0, 6), dtype=np.float32)

        self._cuda_context.push()
        try:
            start = time.time()
            tensor, ratio, pad = self._preprocess(frame)
            after_preprocess = time.time()
            outputs = self._infer(tensor)
            after_infer = time.time()
            detections = self._postprocess(outputs, ratio, pad, frame.shape)
            after_postprocess = time.time()
            if self.profile:
                self._profile_totals["preprocess"] += after_preprocess - start
                self._profile_totals["infer"] += after_infer - after_preprocess
                self._profile_totals["postprocess"] += after_postprocess - after_infer
                self._profile_totals["count"] += 1
                count = self._profile_totals["count"]
                if count % 30 == 0:
                    denom = float(count)
                    print(
                        "trt avg ms: preprocess={:.1f} infer={:.1f} postprocess={:.1f}".format(
                            self._profile_totals["preprocess"] * 1000.0 / denom,
                            self._profile_totals["infer"] * 1000.0 / denom,
                            self._profile_totals["postprocess"] * 1000.0 / denom,
                        )
                    )
            return detections
        finally:
            self._cuda_context.pop()

    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        image, ratio, pad = self._letterbox(frame, (self.input_height, self.input_width))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        if self._input_layout in ("nchw", "chw"):
            image = np.transpose(image, (2, 0, 1))
        if self._input_layout in ("nchw", "nhwc"):
            image = np.expand_dims(image, axis=0)
        dtype = np.float16 if self.using_half else np.float32
        return np.ascontiguousarray(image.astype(dtype, copy=False)), ratio, pad

    def _infer(self, tensor: np.ndarray) -> Dict[str, np.ndarray]:
        input_binding = self.input_binding
        if input_binding is None or input_binding.host is None or input_binding.device is None:
            raise RuntimeError("TensorRT input buffer 尚未初始化")

        np.copyto(input_binding.host, tensor.ravel())
        self.cuda.memcpy_htod_async(input_binding.device, input_binding.host, self.stream)
        ok = self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v2 執行失敗")

        outputs: Dict[str, np.ndarray] = {}
        for binding in self.output_bindings:
            if binding.host is None or binding.device is None:
                continue
            self.cuda.memcpy_dtoh_async(binding.host, binding.device, self.stream)
        self.stream.synchronize()

        for binding in self.output_bindings:
            if binding.host is None:
                continue
            outputs[binding.name] = np.array(binding.host, copy=True).reshape(binding.shape)
        return outputs

    @staticmethod
    def _letterbox(
        frame: np.ndarray,
        new_shape: Tuple[int, int],
        color: Tuple[int, int, int] = (114, 114, 114),
    ) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        src_h, src_w = frame.shape[:2]
        dst_h, dst_w = new_shape
        ratio = min(float(dst_w) / float(src_w), float(dst_h) / float(src_h))
        new_unpad_w = int(round(src_w * ratio))
        new_unpad_h = int(round(src_h * ratio))
        dw = (dst_w - new_unpad_w) / 2.0
        dh = (dst_h - new_unpad_h) / 2.0

        if (src_w, src_h) != (new_unpad_w, new_unpad_h):
            resized = cv2.resize(frame, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized = frame

        top = int(round(dh - 0.1))
        bottom = int(round(dh + 0.1))
        left = int(round(dw - 0.1))
        right = int(round(dw + 0.1))
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return padded, ratio, (dw, dh)

    def _postprocess(
        self,
        outputs: Dict[str, np.ndarray],
        ratio: float,
        pad: Tuple[float, float],
        frame_shape: Tuple[int, ...],
    ) -> np.ndarray:
        detections = None
        if self.output_format in ("auto", "nms"):
            detections = self._decode_nms_outputs(outputs)
        if detections is None and self.output_format in ("auto", "raw"):
            detections = self._decode_raw_outputs(outputs)
        if detections is None:
            raise RuntimeError("無法辨識 TensorRT 輸出格式，請調整 --trt-output-format。")
        if detections.size == 0:
            return np.empty((0, 6), dtype=np.float32)

        detections = detections.astype(np.float32, copy=False)
        detections = detections[detections[:, 4] >= self.conf_threshold]
        if detections.size == 0:
            return np.empty((0, 6), dtype=np.float32)
        detections[:, :4] = self._scale_boxes(detections[:, :4], ratio, pad, frame_shape)
        order = np.argsort(detections[:, 4])[::-1]
        return detections[order[: self.max_detections]].astype(np.float32, copy=False)

    def _decode_nms_outputs(self, outputs: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        if not outputs:
            return None

        values = list(outputs.values())
        if len(values) == 1:
            arr = np.asarray(values[0])
            arr = np.squeeze(arr, axis=0) if arr.ndim == 3 and arr.shape[0] == 1 else np.squeeze(arr)
            if arr.ndim == 1 and arr.shape[0] >= 6:
                arr = arr.reshape(1, -1)
            if arr.ndim == 2 and arr.shape[-1] >= 6:
                dets = arr[:, :6].astype(np.float32, copy=False)
                if dets.shape[0] == 0 or np.nanmax(dets[:, 4]) <= 1.5:
                    return dets
            return None

        named = {name.lower(): value for name, value in outputs.items()}
        boxes = self._find_output(named, ("box", "bbox"), last_dim=4)
        scores = self._find_output(named, ("score", "conf"))
        classes = self._find_output(named, ("class", "label"))
        num_dets = self._find_output(named, ("num", "count"))
        if boxes is None or scores is None or classes is None:
            return None

        boxes = np.asarray(boxes).reshape(-1, 4)
        scores = np.asarray(scores).reshape(-1)
        classes = np.asarray(classes).reshape(-1)
        count = boxes.shape[0]
        if num_dets is not None:
            count = min(count, int(np.asarray(num_dets).reshape(-1)[0]))
        count = min(count, scores.shape[0], classes.shape[0])
        if count <= 0:
            return np.empty((0, 6), dtype=np.float32)
        return np.column_stack([boxes[:count], scores[:count], classes[:count]]).astype(np.float32)

    @staticmethod
    def _find_output(
        named_outputs: Dict[str, np.ndarray],
        name_tokens: Tuple[str, ...],
        last_dim: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        for name, value in named_outputs.items():
            if not any(token in name for token in name_tokens):
                continue
            arr = np.asarray(value)
            if last_dim is not None and (arr.ndim == 0 or arr.shape[-1] != last_dim):
                continue
            return arr
        for value in named_outputs.values():
            arr = np.asarray(value)
            if last_dim is not None and arr.ndim > 0 and arr.shape[-1] == last_dim:
                return arr
        return None

    def _decode_raw_outputs(self, outputs: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        candidates = []
        for value in outputs.values():
            arr = np.asarray(value)
            if arr.ndim >= 2 and arr.shape[-1] >= 6:
                candidates.append(arr)
        if not candidates:
            return None

        raw = max(candidates, key=lambda item: int(np.prod(item.shape)))
        if raw.ndim == 3 and raw.shape[0] == 1:
            raw = raw[0]
        raw = raw.reshape(-1, raw.shape[-1]).astype(np.float32, copy=False)
        if raw.shape[0] == 0:
            return np.empty((0, 6), dtype=np.float32)

        boxes = raw[:, :4].copy()
        objectness = raw[:, 4]
        class_scores = raw[:, 5:]
        if self.allowed_classes:
            allowed = np.array([cls for cls in self.allowed_classes if 0 <= cls < class_scores.shape[1]], dtype=np.int32)
            if allowed.size == 0:
                return np.empty((0, 6), dtype=np.float32)
            selected_scores = class_scores[:, allowed]
            selected_indices = np.argmax(selected_scores, axis=1)
            class_ids = allowed[selected_indices].astype(np.float32)
            scores = objectness * selected_scores[np.arange(raw.shape[0]), selected_indices]
        else:
            class_ids = np.argmax(class_scores, axis=1).astype(np.float32)
            scores = objectness * class_scores[np.arange(raw.shape[0]), class_ids.astype(np.int32)]
        mask = scores >= self.conf_threshold
        if not np.any(mask):
            return np.empty((0, 6), dtype=np.float32)

        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]
        if scores.shape[0] > self.max_candidates:
            top_indices = np.argpartition(scores, -self.max_candidates)[-self.max_candidates:]
            boxes = boxes[top_indices]
            scores = scores[top_indices]
            class_ids = class_ids[top_indices]
        boxes = self._xywh_to_xyxy(boxes)
        if np.nanmax(boxes) <= 2.0:
            boxes[:, [0, 2]] *= float(self.input_width)
            boxes[:, [1, 3]] *= float(self.input_height)

        keep = self._classwise_nms(boxes, scores, class_ids, self.iou_threshold)
        if len(keep) == 0:
            return np.empty((0, 6), dtype=np.float32)
        return np.column_stack([boxes[keep], scores[keep], class_ids[keep]]).astype(np.float32)

    @staticmethod
    def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
        out = boxes.copy()
        out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
        return out

    def _scale_boxes(
        self,
        boxes: np.ndarray,
        ratio: float,
        pad: Tuple[float, float],
        frame_shape: Tuple[int, ...],
    ) -> np.ndarray:
        scaled = boxes.copy()
        if scaled.size and np.nanmax(scaled) <= 2.0:
            scaled[:, [0, 2]] *= float(self.input_width)
            scaled[:, [1, 3]] *= float(self.input_height)
        dw, dh = pad
        scaled[:, [0, 2]] -= dw
        scaled[:, [1, 3]] -= dh
        scaled[:, :4] /= max(ratio, 1e-9)
        height, width = frame_shape[:2]
        scaled[:, [0, 2]] = np.clip(scaled[:, [0, 2]], 0, width - 1)
        scaled[:, [1, 3]] = np.clip(scaled[:, [1, 3]], 0, height - 1)
        return scaled

    def _classwise_nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        iou_threshold: float,
    ) -> List[int]:
        keep: List[int] = []
        for class_id in np.unique(class_ids):
            indices = np.where(class_ids == class_id)[0]
            selected = self._nms_xyxy(boxes[indices], scores[indices], iou_threshold)
            keep.extend(indices[selected].tolist())
        keep.sort(key=lambda idx: float(scores[idx]), reverse=True)
        return keep[: self.max_detections]

    @staticmethod
    def _nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
        if boxes.shape[0] == 0:
            return np.empty((0,), dtype=np.int64)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break
            rest = order[1:]
            xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
            yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
            xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
            yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
            inter_w = np.maximum(0.0, xx2 - xx1)
            inter_h = np.maximum(0.0, yy2 - yy1)
            inter = inter_w * inter_h
            area_i = np.maximum(0.0, boxes[i, 2] - boxes[i, 0]) * np.maximum(0.0, boxes[i, 3] - boxes[i, 1])
            area_rest = np.maximum(0.0, boxes[rest, 2] - boxes[rest, 0]) * np.maximum(
                0.0, boxes[rest, 3] - boxes[rest, 1]
            )
            union = area_i + area_rest - inter + 1e-12
            iou = inter / union
            order = rest[iou <= iou_threshold]
        return np.array(keep, dtype=np.int64)

    def warmup(self, image_size: int = 640, iterations: int = 3) -> None:
        dummy = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        for _ in range(iterations):
            self.detect(dummy)

    def close(self) -> None:
        if not hasattr(self, "_closed") or self._closed:
            return
        cuda_context = getattr(self, "_cuda_context", None)
        if cuda_context is None:
            self._closed = True
            return

        pushed = False
        try:
            cuda_context.push()
            pushed = True
            stream = getattr(self, "stream", None)
            if stream is not None:
                stream.synchronize()

            for binding in getattr(self, "binding_info", []):
                if binding.device is not None:
                    try:
                        binding.device.free()
                    except Exception:
                        pass
                    binding.device = None
                binding.host = None

            self.bindings = []
            self.input_binding = None
            self.output_bindings = []
            self.context = None
            self.engine = None
            self.stream = None
            gc.collect()
        finally:
            if pushed:
                try:
                    cuda_context.pop()
                except Exception:
                    pass
            try:
                cuda_context.detach()
            except Exception:
                pass
            self._closed = True
            self._cuda_context = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
