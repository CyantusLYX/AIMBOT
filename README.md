# AIMBOT

> 即時影像伺服追蹤系統 — YOLOv7 + ByteTrack + OSNet Re-ID + PID 雲台控制  
> Real-time visual servo tracking — YOLOv7 + ByteTrack + OSNet Re-ID + PID gimbal control

---

## 專案目標 / Project Goals

**中文**  
打造一套即時的影像伺服系統：使用 YOLOv7 偵測目標，透過點擊或參考圖片選定目標後，驅動自製雲台鎖定並持續追蹤。目標短暫遮擋後，以 OSNet Re-ID 特徵自動重新辨識。

**English**  
Build a real-time visual servo system: detect targets with YOLOv7, lock onto them via mouse click or a reference image, then drive a custom gimbal to continuously track. After brief occlusions, OSNet Re-ID features allow automatic re-acquisition.

---

## 系統架構 / System Architecture

```
Video Source / Camera
        │
        ▼
 FramePrefetcher  ─── background read thread
        │
        ▼
 GpuPreprocessor  ─── optional CUDA resize
        │
        ▼
 AsyncDetector    ─── YOLOv7 in single thread-pool worker
        │  DetectionResult(frame, detections[N,6])
        ▼
 ReIDHelper       ─── OSNet embeddings for top-K candidates
        │
        ▼
 ByteTrack        ─── IoU + Re-ID two-stage assignment
        │  track list
        ▼
 TargetController ─── lock / lost-frame count / reacquire
        │  TargetState(track_id, bbox, score)
        ▼
 PIDController ×2 ─── pan + tilt error → speed command
        │
        ▼
 GimbalController ─── dry-run or serial JSON output
        │
        ▼
   Gimbal Hardware

Side channel:
 OpenCVViewer     ─── render overlays, emit click events
```

詳細架構說明見 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。  
See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for a detailed breakdown.

---

## 資料夾結構 / Repository Layout

```
scripts/
    run_pipeline.py       # CLI entry point / 主程式入口
src/
    adapters/
        video_source.py   # Video I/O adapter (capture open/validation)
    app/
        __init__.py       # Application-level assembly namespace
    core/
        config.py         # Frozen dataclass config tree (PipelineConfig)
    detection/
        detector.py       # YOLOv7 inference wrapper
    tracking/
        byte_tracker.py   # IoU + Re-ID multi-object tracker
    reid/
        osnet.py          # OSNet feature extractor (torchreid)
    pipeline/
        workers.py        # FramePrefetcher, GpuPreprocessor, AsyncDetector, ReIDHelper
    services/
        tracking_service.py  # ByteTrack + Re-ID composition service
    control/
        target_controller.py  # Lock / reacquire lifecycle
        pid.py                # Discrete PID controller
        gimbal_controller.py  # GimbalBase ABC + serial/dry-run implementation
    ui/
        viewer.py         # OpenCV window + mouse events
models/                   # YOLOv7 weights, OSNet checkpoints
hardware/                 # Gimbal firmware, wiring, calibration notes
data/                     # Test footage, demo assets
docs/
    ARCHITECTURE.md       # Detailed architecture reference
    STYLE_GUIDE.md        # Coding conventions
    REFRACTOR_DECISIONS.md  # Design decision log
```

---

## 完成進度 / Status

| 元件 / Component                                        | 狀態 / Status                |
| ------------------------------------------------------- | ---------------------------- |
| YOLOv7 偵測 / Detection (`detection/detector.py`)       | ✅ GPU / FP16                |
| ByteTrack 追蹤 / Tracking (`tracking/byte_tracker.py`)  | ✅ IoU + Re-ID               |
| OSNet Re-ID (`reid/osnet.py`)                           | ✅ batch encode, EMA feature |
| PID 控制 / PID control (`control/pid.py`)               | ✅                           |
| 雲台通訊 / Gimbal comm (`control/gimbal_controller.py`) | ✅ dry-run + serial          |
| 管線工人 / Pipeline workers (`pipeline/workers.py`)     | ✅ async inference           |
| OpenCV UI (`ui/viewer.py`)                              | ✅ overlays + click          |
| 集中設定 / Centralised config (`core/config.py`)        | ✅ frozen dataclasses        |
| TensorRT 加速 / TensorRT acceleration                   | ⏳ planned                   |
| 實機 PID 校調 / Hardware PID tuning                     | ⏳ in progress               |

---

## 安裝 / Installation

### 方法一：uv（推薦）/ Method 1: uv (recommended)

[uv](https://docs.astral.sh/uv/) manages the virtual environment and all
dependencies from `pyproject.toml`.

```powershell
# Install uv (once)
pip install uv

# Create venv and install CPU-only torch + all other deps
uv sync --extra reid
```

**CUDA PyTorch（GPU 推論必要）/ CUDA PyTorch (required for GPU inference)**

`pyproject.toml` cannot express the PyTorch CUDA index URL, so install it manually **before** `uv sync`.  
CUDA 13.0 driver is backward compatible with CUDA 12.x runtime wheels.

```powershell
# CUDA 12.1 wheels（目前安裝版本 2.5.1+cu121，相容 CUDA 13.0 驅動）
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# 或使用 CUDA 12.6 wheels 取得更新版本
# uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
uv sync --extra reid
```

驗證安裝 / Verify:

```powershell
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### 方法二：pip / Method 2: pip

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

# CUDA PyTorch（先安裝 CUDA 版本再裝其餘依賴 / install before requirements.txt）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## 準備模型 / Prepare Models

- 將 YOLOv7 權重放在 `models/`，例如 `models/epoch_149.pt`。  
  Place YOLOv7 weights in `models/`, e.g. `models/epoch_149.pt`.
- Re-ID 模型由 torchreid 自動下載至快取目錄。  
  Re-ID models are downloaded automatically by torchreid to its cache.

---

## 執行 / Running

### 方法一：uv run

```powershell
uv run python scripts/run_pipeline.py \
    --weights models/epoch_149.pt \
    --source data/demo.mp4 \
    --device cuda --half \
    --enable-reid --reid-model osnet_x0_75
```

### 方法二：直接呼叫 / Direct invocation

```powershell
python scripts/run_pipeline.py \
    --weights models/epoch_149.pt \
    --source data/demo.mp4 \
    --device cuda --half \
    --enable-reid --reid-model osnet_x0_75
```

### 常用參數 / Common Arguments

| 參數 / Argument   | 預設 / Default        | 說明 / Description                                    |
| ----------------- | --------------------- | ----------------------------------------------------- |
| `--weights`       | `models/epoch_149.pt` | YOLOv7 weight path                                    |
| `--source`        | `0`                   | Video file path or camera index                       |
| `--device`        | `cuda`                | Inference device (`cuda`, `cuda:0`, `cpu`)            |
| `--half`          | off                   | Enable FP16 inference (CUDA only)                     |
| `--person-only`   | off                   | Filter detections to `person` class only              |
| `--dry-run`       | off                   | Print gimbal commands without opening serial port     |
| `--serial-port`   | `COM3`                | Gimbal serial port                                    |
| `--fps`           | `30`                  | Target control-loop FPS; used for lost-frame timeout  |
| `--max-frames`    | unlimited             | Stop after N frames (smoke testing)                   |
| `--enable-reid`   | off                   | Enable OSNet Re-ID for reacquisition                  |
| `--reid-model`    | `osnet_x0_75`         | torchreid model name                                  |
| `--reid-weights`  | —                     | Path to custom Re-ID weights (optional)               |
| `--process-scale` | `0` (auto)            | Detector input scale factor; `0` = auto by resolution |

### 操作模式 / Operating Modes

啟動後系統會詢問操作模式 / After launch the system prompts for a mode:

- **Mode 1 — 點擊鎖定 / Click lock**: Click any bounding box in the viewer
  window to lock the PID onto that track.
- **Mode 2 — 參考圖片 / Reference image**: Provide a reference image path at
  startup. The system searches for the best-matching track each frame using
  combined Re-ID + colour histogram + ORB scoring.

### UI 操作 / UI Controls

- Click on a bounding box → lock target (Mode 1)
- Press `q` or `Esc`, or close the window → stop pipeline

---

## 文件 / Documentation

| 文件 / Document                                            | 內容 / Contents                      |
| ---------------------------------------------------------- | ------------------------------------ |
| [docs/INSTALL.md](docs/INSTALL.md)                         | CUDA 版本選擇、安裝順序、常見問題    |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)               | 詳細架構、資料流、並發模型           |
| [docs/STYLE_GUIDE.md](docs/STYLE_GUIDE.md)                 | 程式碼風格規範                       |
| [docs/REFRACTOR_DECISIONS.md](docs/REFRACTOR_DECISIONS.md) | 設計決策紀錄（版本依賴、閾值來源等） |
| [docs/BASELINE.md](docs/BASELINE.md)                       | 重構基線快照與對照輸入               |

---

## 開發 / Development

```powershell
# Install dev tools (ruff, mypy)
uv sync --extra dev

# Lint
uv run ruff check .
uv run ruff format --check .

# Type check
uv run mypy src scripts

# Syntax validation (no torch/serial required)
python -m compileall src scripts

# Pipeline smoke run（dry-run，不連線雲台）
python scripts/run_pipeline.py --dry-run --source data/DJI_20250422132606_0030_D.MP4 --max-frames 30
```
