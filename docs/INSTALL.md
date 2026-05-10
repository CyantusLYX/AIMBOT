# 安裝指南 / Installation Guide

## 環境需求 / Prerequisites

| 項目        | 需求                                   |
| ----------- | -------------------------------------- |
| Python      | 3.11+                                  |
| CUDA Driver | 12.x 或 13.x（執行 `nvidia-smi` 確認） |
| uv          | `pip install uv`                       |

---

## 快速安裝 / Quick Install

### 步驟 1：建立虛擬環境

```powershell
uv venv
.\.venv\Scripts\Activate.ps1
```

### 步驟 2：安裝 GPU 版 PyTorch（**必須先於其他套件**）

CUDA 驅動版本與 PyTorch wheel 的對應：

| nvidia-smi CUDA Version | 建議 wheel                  | Index URL                                |
| ----------------------- | --------------------------- | ---------------------------------------- |
| 13.x                    | cu121 ✅（目前安裝）/ cu126 | `https://download.pytorch.org/whl/cu121` |
| 12.4–12.6               | cu124                       | `https://download.pytorch.org/whl/cu124` |
| 12.1–12.3               | cu121                       | `https://download.pytorch.org/whl/cu121` |
| 11.8                    | cu118                       | `https://download.pytorch.org/whl/cu118` |
| CPU only                | cpu                         | `https://download.pytorch.org/whl/cpu`   |

> **注意**：`nvidia-smi` 顯示的是驅動程式支援的最高 CUDA 版本，PyTorch CUDA wheel 向下相容。
> 例如 CUDA 13.0 驅動可直接執行 cu126 runtime。

```powershell
# 以 CUDA 13.0 驅動（使用 cu126 wheel）為例
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

驗證 / Verify:

```powershell
python -c "import torch; print(torch.__version__, 'CUDA:', torch.cuda.is_available())"
```

### 步驟 3：安裝其餘依賴

```powershell
uv pip install -r requirements.txt
```

> **重要**：步驟 2 已安裝 CUDA 版 torch，此步驟不會覆蓋（uv/pip 發現版本已滿足 `>=2.0.0` 約束，跳過重新安裝）。

### 步驟 4（選用）：torchreid 源碼安裝

若 `requirements.txt` 安裝 torchreid 時失敗（PyPI 上的 0.2.5 wheel 有時不相容），改用源碼：

```powershell
uv pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
```

---

## 常見問題 / Troubleshooting

### `ModuleNotFoundError: No module named 'torch'`

torch 未安裝。執行步驟 2。

### `torch.cuda.is_available()` 回傳 `False`

- 確認安裝的是 CUDA wheel（不是 CPU-only）：`python -c "import torch; print(torch.version.cuda)"`
- 確認驅動已正確安裝：`nvidia-smi`
- 確認 CUDA Runtime 與 wheel 版本相容（見上表）

### `ImportError: DLL load failed` (Windows)

Visual C++ Redistributable 未安裝。從 Microsoft 下載安裝最新版本。

### `numpy` 版本衝突

本專案限制 `numpy>=1.24.0,<2.0.0`。numpy 2.x 與 YOLOv7 的 C-extension 不相容。

```powershell
uv pip install "numpy>=1.24.0,<2.0.0"
```

### `torchreid` / `protobuf` 版本衝突

torchreid 0.2.5 使用舊版 protobuf API，需要 `protobuf<4.21.3`（即 3.x 系列）。

```powershell
uv pip install "protobuf<4.21.3"
```

---

## 完整驗證 / Full Verification

```powershell
# 語法檢查
python -m compileall src scripts

# Pipeline 煙霧測試（dry-run，不連線雲台）
python scripts/run_pipeline.py \
    --dry-run \
    --source data/DJI_20250422132606_0030_D.MP4 \
    --max-frames 30
```
