# Android Gimbal Brain 設計說明

這份文件整理 Android App 端的設計細節，以及它如何和 ESP32 雲臺控制器互動。

## App 角色

Android 手機是整套自動追蹤雲臺的「視覺與指令大腦」：

- 顯示全螢幕 CameraX 預覽，作為 HUD 背景。
- 對降解析度後的分析影格執行物件偵測。
- 將偵測框從影像座標映射回螢幕預覽座標。
- 計算目標中心相對螢幕中心的 X/Y 誤差。
- 使用簡單 P 控制器把誤差轉成 pan/tilt 速度。
- 透過 Bluetooth Classic SPP 傳送 ASCII 速度命令給 ESP32。
- 透過 App 控制馬達使能，必要時送出 `E:1\n` / `E:0\n`。

ESP32 仍負責馬達脈波輸出、TMC2209 設定、韌體端速度限制，以及收不到命令時的失效保護。

ESP32 韌體端已使用 Arduino `BluetoothSerial` 暴露 Bluetooth Classic SPP，
裝置名稱為：

```text
AIMBOT-Gimbal
```

USB Serial 與 Bluetooth SPP 共用同一套 ASCII 指令解析器，因此 Android 端
透過藍牙送出的 `V:<pan>,<tilt>\n` 和 host 工具透過 USB 送出的命令行為一致。

## 程式分層

目前 Android 程式位於 `src_android/app/src/main/java/com/cyantus/aimbot`。

| 層級 | 主要類別 | 職責 |
| --- | --- | --- |
| Compose HUD | `GimbalHudScreen.kt` | 權限請求、CameraX 預覽、HUD 繪製、Bluetooth 控制面板。 |
| ViewModel | `GimbalViewModel` | 持有 UI 狀態、偵測結果、追蹤開關、Bluetooth 狀態、命令節流。 |
| 偵測引擎 | `ObjectDetectorEngine`, `MLKitDetector` | 抽象化影格分析，未來可把 ML Kit 換成 TFLite。 |
| 控制數學 | `GimbalController` | 將目標誤差轉成 `V:<pan>,<tilt>\n`。 |
| Bluetooth | `BluetoothManager`, `AndroidBluetoothManager` | 與 UI 無關的 SPP 連線、已配對裝置列表、寫入與斷線。 |

UI 只和 `GimbalViewModel` 溝通。ViewModel 再組合三個可替換元件：

- `ObjectDetectorEngine` 隱藏具體偵測實作。
- `BluetoothManager` 隱藏 Android Bluetooth Socket 細節。
- `GimbalController` 是純 Kotlin 控制邏輯，可做單元測試。

## Camera 與偵測流程

CameraX 綁定兩個 use case 到 Activity lifecycle：

- `Preview`：驅動全螢幕 `PreviewView`。
- `ImageAnalysis`：把影格送進偵測引擎。

關鍵設定：

- `PreviewView.ScaleType.FILL_CENTER` 讓預覽鋪滿 HUD，避免黑邊。
- `ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST` 避免影格堆積。
- 分析解析度目標為 `640x480`，不影響螢幕預覽畫質。
- Analyzer 使用獨立 single-thread executor，不阻塞 Main Thread。

ML Kit 回傳的 bounding box 一開始是影像座標。`GimbalViewModel` 會用 CameraX transform API 對齊到螢幕：

1. 從 `ImageProxy` 建立 image-space transform。
2. 讀取目前 `PreviewView.outputTransform`。
3. 透過 `CoordinateTransform` 把每個 `RectF` 映射到可見預覽座標。

Compose `Canvas` 負責繪製：

- 螢幕正中央的半透明白色十字準心。
- 偵測物件的 bounding box：一般目標為霓虹綠，鎖定目標為紅色粗線。
- 觸控 hit-test：點擊框框鎖定目標，點擊空白處解除鎖定。

## 追蹤控制流程

追蹤命令只會在以下條件都成立時送出：

- Bluetooth 已連線。
- 使用者已在 HUD 面板開啟 `Motors: On`。
- 使用者按下 `Engage Tracking`。
- 預覽寬高有效。

每個分析影格的處理流程：

1. ML Kit 產生偵測結果。
2. 偵測框映射到螢幕預覽座標。
3. 依 Target Lock-On 狀態決定 primary target。
4. 計算目標中心和螢幕中心的 X/Y 誤差。
5. `GimbalController` 產生速度命令。
6. `GimbalViewModel` 透過 `BluetoothManager` 傳送命令。

命令傳送頻率目前限制在約 `30 Hz`，也就是 `commandIntervalMs = 33`。這樣能避免灌爆 ESP32 Serial buffer，同時仍遠低於韌體 `500 ms` 無命令停機保護的上限。

使用者解除追蹤或斷開 Bluetooth 時，Android 會送出停止命令：

```text
V:0,0
```

實際封包會包含結尾 newline：`V:0,0\n`。

在目前 ESP32 韌體中，`V:0,0\n` 只代表停止送 step pulse，不代表 disable
TMC2209。韌體初始化後會讓共用 EN 腳保持 enabled，因此追蹤 deadband、目標短暫消失、
或 fail-safe 將速度歸零時，tilt 軸仍保有 holding torque。

HUD 的 `Motors: On/Off` 是獨立於追蹤開關的馬達使能控制：

- `Motors: On`：送出 `E:1\n`，韌體會在 GPIO36 硬體停機腳未被拉低時啟用 TMC2209。
- `Motors: Off`：先送 `V:0,0\n`，再送 `E:0\n`，韌體會清除速度目標並關閉共用 EN。
- `Engage Tracking` 只有在 Bluetooth connected 且 `Motors: On` 時可用。
- App 斷線流程會主動送 `V:0,0\n` 和 `E:0\n`，降低無人看管時馬達保持通電的機率。

GPIO36 是韌體端硬體 interlock，低準位會覆蓋 App 的 `E:1` 請求。也就是說，
App 可能顯示使用者已請求 `Motors: On`，但 GPIO36 被拉低時 ESP32 仍會停止兩軸並關閉 driver。
GPIO36 在 ESP32 上沒有內建上拉，硬體必須提供外接 pull-up；按鍵或保護電路再把它拉到 GND 觸發停機。

## Target Lock-On

ML Kit 在 `STREAM_MODE` 下會替追蹤中的物件提供 `trackingId`。Android 會把這個 ID 存在 `DetectionResult.trackingId`，用來支援使用者手動鎖定。

Target selection 狀態機：

- 未鎖定：選擇面積最大的 bounding box 作為 primary target。
- 已鎖定：只追蹤 `trackingId == lockedTargetId` 的目標。
- 已鎖定但該 ID 暫時消失：不切換到其他物件，改送 `V:0,0\n` 讓雲臺暫停。
- 點擊空白處、Bluetooth 斷線或 disconnect：解除鎖定。

HUD 互動：

- 點擊有 `trackingId` 的偵測框：鎖定該目標。
- 點擊沒有 `trackingId` 的框或空白區域：解除鎖定。
- 鎖定框以紅色粗線顯示，其餘框維持霓虹綠。

## P 控制器

`GimbalController` 使用簡單 proportional controller：

```text
deltaX = objectCenterX - screenCenterX
deltaY = objectCenterY - screenCenterY
panSpeed = clamp(round(deltaX * kp * panSign), -maxAbsSpeed, maxAbsSpeed)
tiltSpeed = clamp(round(deltaY * kp * tiltSign), -maxAbsSpeed, maxAbsSpeed)
```

預設參數：

| 參數 | 預設值 | 意義 |
| --- | ---: | --- |
| `kp` | `4.0` | 像素誤差轉 step/s 的比例增益。 |
| `maxAbsSpeed` | `4000` | Android 端送出前的速度限制。 |
| `deadbandPx` | `12` | 誤差在此範圍內輸出 0，降低中心抖動。 |
| `panSign` | `1` | pan 方向反了就改成 `-1`。 |
| `tiltSign` | `1` | tilt 方向反了就改成 `-1`。 |

輸出格式固定為：

```text
V:<panSpeed>,<tiltSpeed>\n
```

範例：

```text
V:1500,-600
```

ESP32 會把數值解讀為目前 microstep 設定下的 signed step pulse rate。

## Bluetooth SPP 互動

Android 使用 Bluetooth Classic Serial Port Profile：

- SPP UUID：`00001101-0000-1000-8000-00805F9B34FB`
- 裝置來源：Android 系統內已配對的 Bluetooth 裝置。
- Socket API：`createRfcommSocketToServiceRecord`。
- 資料格式：ASCII 字串。
- 訊息邊界：newline `\n`。

目前 sprint 不做配對流程，也不主動掃描。預期使用方式：

1. 先到 Android 系統設定中和 ESP32 配對。
2. 開啟 App 並允許 Camera / Nearby Devices 權限。
3. 在底部 HUD 面板選擇已配對的 ESP32。
4. 按下 `Connect`。
5. 按下 `Motors: Off` / `Motors: On` 切換馬達使能；連線後預設會先送 `E:0\n`。
6. 按下 `Engage Tracking` 開始追蹤。

`AndroidBluetoothManager` 透過 `StateFlow` 暴露連線狀態：

- `Unavailable`
- `Disconnected`
- `Connecting`
- `Connected(device)`
- `Error(message)`

所有 socket connect / write / disconnect 操作都在 `Dispatchers.IO` 執行。連線或寫入失敗時會轉成 `Error` 狀態，不會讓 App crash。

## Android 權限

最低 API 是 31，因此使用 Android 12+ Bluetooth 權限模型：

- `android.permission.CAMERA`
- `android.permission.BLUETOOTH_CONNECT`
- `android.permission.BLUETOOTH_SCAN`

`BLUETOOTH_SCAN` 在 manifest 中加上 `neverForLocation`。目前 App 只把它納入 Nearby Devices 權限請求，尚未實作主動掃描。

Manifest feature：

- `android.hardware.camera`
- `android.hardware.bluetooth`

兩者目前都標記為 required。

## 與 ESP32 的協定契約

Android 目前會使用韌體協定中的速度與馬達使能命令：

| Command | Android 目前會送 | 意義 |
| --- | --- | --- |
| `V:<pan>,<tilt>\n` | 是 | 設定 pan/tilt 速度。 |
| `V:0,0\n` | 是 | 停止兩軸 step pulse，但韌體仍保持 driver enabled 以提供 holding torque。 |
| `E:1\n` | 是 | 請求啟用馬達 driver；若 GPIO36 被拉低，韌體仍會保持 disabled。 |
| `E:0\n` | 是 | 關閉馬達 driver，並清除兩軸速度目標。 |
| `S:<max_step_hz>\n` | 否 | 韌體支援的速度上限設定。 |
| `M:<microsteps>\n` | 否 | 韌體支援的 microstep 設定。 |
| `?\n` | 否 | 韌體支援的狀態查詢。 |

ESP32 韌體若超過 `500 ms` 沒收到有效速度命令，會自動把兩軸速度歸零。Android 仍應在解除追蹤、斷線、未來 App 退場流程中主動送 `V:0,0\n`，避免只依賴 fail-safe 造成可見的滑行時間。

韌體目前會同時接受兩個命令來源：

- USB `Serial`：保留給 PlatformIO monitor、host-side joystick 工具、bring-up debug。
- Bluetooth SPP `SerialBT`：保留給 Android App。

兩邊都支援 `V`、`S`、`M` 和 `?`。狀態回覆會寫回命令來源本身；例如 Android 若送 `?\n`，回覆會從 Bluetooth socket 回來。

## Bring-Up 檢查清單

1. 在 Android 系統設定中先配對 ESP32。
2. 硬體測試初期先用較低的韌體 speed clamp。
3. 測 pan 方向：
   - 把目標移到畫面中心右側。
   - 如果雲臺往遠離目標的方向轉，將 `panSign` 改成 `-1`。
4. 測 tilt 方向：
   - 把目標移到畫面中心下方。
   - 如果雲臺往遠離目標的方向轉，將 `tiltSign` 改成 `-1`。
5. 慢慢提高 `kp`，直到追蹤反應足夠快；若開始震盪就降低。
6. 若目標接近中心時雲臺抖動，增加 `deadbandPx`。

## 已知限制

- Target lock 依賴 ML Kit `trackingId`，目標長時間消失後仍需使用者重新點選。
- 控制器目前只有 P control，尚未加入 I/D 項。
- Bluetooth 配對與裝置發現交給 Android 系統設定。
- App 尚未提供 UI 來調整 `kp`、速度上限、軸向反轉、microstep 或 ESP32 狀態查詢。
