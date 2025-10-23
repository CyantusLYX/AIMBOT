# AIMBOT

## 專案目標

- 打造即時的影像伺服系統：使用 YOLOv7 偵測人物，透過介面點擊目標後，驅動自製雲台鎖定並持續追蹤。

## 系統概觀

```
Camera Stream --> YOLOv7 偵測 --> ByteTrack 追蹤 --> Target Controller --> PID Servo --> 雲台
																					|                                         ^
																					+--> OSNet Re-ID (找回遺失 ID) -----------+
UI Layer <--------------------------------------------------------------- 已渲染畫面
```

- 偵測：沿用已訓練的 YOLOv7 模型產出每幀邊界框。
- 追蹤：整合 ByteTrack 以高速配對 ID，短暫遺失時可用緩存偵測結果補上。
- 重識別：僅在目標失聯時啟用 OSNet 特徵比對；若環境過於擁擠再考慮切換 TransReID。
- 控制迴圈：以每軸 PID 將像素誤差轉為雲台角速度，並利用追蹤器的卡爾曼預測降低延遲。
- 介面：即時渲染畫面與框線（OpenCV／Qt／Web），支援點擊設定新的 Target ID。

## 加速與精度重點

- 將 YOLOv7 與 OSNet 匯出為 TensorRT 引擎（FP16／INT8），在 NVIDIA 平台可帶來 2-5 倍推論速度。
- 拆分攝影機擷取、AI 推論、控制迴圈為獨立執行緒；即使推論僅 ~20 FPS，也能讓 PID 以高頻率保持平順。
- 重識別不必逐幀執行；在目標遺失時以固定頻率觸發即可節省 GPU 資源。
- 校正像素與雲台角度的換算比例，確保 PID 參數與實際動作對應一致。

## 開發路線圖

1. 追蹤整合：將 YOLOv7 輸出導入 ByteTrack，使用錄影檔驗證 ID 穩定性。
2. 伺服控制：寫好 PID 與雲台通訊，讓選定 ID 能被居中維持。
3. Re-ID 回復：擷取目標圖塊產生 OSNet 向量，遮擋後自動認出原目標。
4. UI 互動：顯示即時影像、繪製框線，並支援點擊切換追蹤對象。

## 資料夾結構

```
src/
	detection/   # YOLOv7 推論包裝與前處理
	tracking/    # ByteTrack 流程、卡爾曼狀態與 ID 管理
	control/     # PID 迴圈、運動計畫與雲台通訊
	reid/        # OSNet 特徵擷取與 ID 回復
	ui/          # 影像渲染與使用者互動介面
models/        # YOLOv7 權重、TensorRT 引擎、OSNet 檢查點
hardware/      # 雲台韌體、線材配置、校正紀錄
scripts/       # 工具腳本（訓練匯出、格式轉換、資料整理）
data/          # 測試錄影、示範資料、合成素材
```

## 完成進度

- ✅ YOLOv7 偵測模組 (`src/detection/detector.py`)：支援 GPU / FP16 推論。
- ✅ ByteTrack 追蹤器 (`src/tracking/byte_tracker.py`)：提供基本多目標追蹤與 ID 管理。
- ✅ PID 與雲台通訊介面 (`src/control/*`)：支援 dry-run 與實際序列埠控制。
- ✅ OpenCV UI (`src/ui/viewer.py`)：顯示追蹤框、FPS、滑鼠點選目標，視窗可正常關閉。
- ✅ 主流程腳本 (`scripts/run_pipeline.py`)：串連偵測 → 追蹤 → 控制 → UI，支援 CLI 參數、FPS 監控。

## 尚未完成

- ⏳ Re-ID 整合：`src/reid/` 目前僅有模型骨架，尚未串入主追蹤流程。
- ⏳ 多執行緒最佳化：仍使用單執行緒流程，未分離擷取 / 推論 / 控制。
- ⏳ TensorRT 加速：尚未將 YOLOv7 / OSNet 匯出為 TensorRT。
- ⏳ 控制模組實測：尚未於真實雲台硬體上做 PID 參數調校與封閉迴路測試。

## 安裝與使用說明

### 1. 建立虛擬環境

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. 安裝相依套件

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

> **注意**：為了使用 GPU 推論，請安裝對應 CUDA 版本的 PyTorch。例如 RTX 40 系列使用 CUDA 12.1：
>
> ```powershell
> pip uninstall -y torch torchvision torchaudio
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

安裝完成後確認：

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 3. 準備模型與資料

- 將訓練好的 YOLOv7 權重（例如 `epoch_149.pt`）放置於 `models/` 目錄。
- 將測試影片或攝影機來源路徑放在 `data/` 或直接指定攝影機 index。

### 4. 執行管線

```powershell
python scripts/run_pipeline.py --weights models/epoch_149.pt --source data/DJI_20250422132606_0030_D.MP4 --device cuda --half
```

常用參數：

- `--device`：指定推論裝置（如 `cuda`, `cuda:0`, `cpu`）。
- `--half`：啟用半精度推論（僅限 CUDA）。
- `--dry-run`：僅輸出控制命令，不連線雲台。
- `--person-only`：只保留 person 類別偵測。
- `--max-frames`：限制處理影格數，方便煙霧測試。
- `--serial-port`：雲台序列埠編號（預設 `COM3`）。

執行後畫面會顯示：

- 偵測框與追蹤 ID，若目標被使用者點選則高亮顯示。
- 即時 FPS 文字顯示於左上角。
- 終端會定期列印近期平均 FPS 與雲台 dry-run 控制命令。

### 5. 關閉程式

- 於視窗按 `X` 或按下鍵盤 `q` / `Esc` 即可停止播放並釋放資源。
