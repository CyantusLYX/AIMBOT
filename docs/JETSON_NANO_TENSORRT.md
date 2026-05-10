# Jetson Nano TensorRT Bring-up

Target: Jetson Nano on JetPack 4.6.6 / L4T 32.7.6.

NVIDIA documents L4T 32.7.6 as part of JetPack 4.6.6 and the final JetPack 4
release for the R32 line. The JetPack 4.6.x / R32.7.x stack is CUDA 10.2 /
TensorRT 8.2.x-era, so do not install the desktop `requirements-cuda.txt`
(`torch>=2.0.0`) on Nano.

## 1. Verify The Jetson Stack

```bash
cat /etc/nv_tegra_release
python3 --version
python3 - <<'PY'
import tensorrt as trt
print("TensorRT", trt.__version__)
PY
/usr/src/tensorrt/bin/trtexec -h
```

If `import tensorrt` fails, install the JetPack TensorRT Python package:

```bash
sudo apt update
sudo apt install python3-libnvinfer libnvinfer-bin python3-pycuda
```

For best camera/video compatibility on Jetson, use NVIDIA's apt OpenCV rather
than `opencv-python` from pip:

```bash
sudo apt install python3-opencv python3-numpy python3-scipy python3-venv
```

## 2. Create The Python Environment

Use `--system-site-packages` so the venv can see apt packages such as
`cv2`, `tensorrt`, and `pycuda`.

```bash
cd ~/AIMBOT
python3 -m venv --system-site-packages .venv-jetson
. .venv-jetson/bin/activate
python3 -m pip install --upgrade "pip<22"
pip3 install -r requirements-jetson.txt
```

Initial TensorRT testing should keep Re-ID disabled. Re-ID needs NVIDIA's
Jetson-specific PyTorch wheel and is separate from the TensorRT detector path.

## 3. Export YOLOv7 To ONNX

Export ONNX on a desktop or another machine with PyTorch installed. TensorRT
engines are not portable, but ONNX files are.

```bash
git clone https://github.com/WongKinYiu/yolov7
cd yolov7
python export.py \
  --weights ../AIMBOT/models/epoch_149.pt \
  --img-size 640 640 \
  --batch-size 1 \
  --grid \
  --simplify
```

Copy the generated `.onnx` file to `models/` on the Jetson.

If you export a YOLOv7 end-to-end/NMS ONNX, the runtime can also consume it;
start with the raw output path above first because it keeps NMS in this repo
and avoids plugin/version surprises.

## 4. Build The TensorRT Engine On Nano

Build the engine on the Jetson itself:

```bash
TRTEXEC=$(command -v trtexec || echo /usr/src/tensorrt/bin/trtexec)
$TRTEXEC \
  --onnx=models/epoch_149.onnx \
  --saveEngine=models/epoch_149_fp16.engine \
  --fp16 \
  --workspace=1024
```

Nano is memory-constrained. If `trtexec` is killed, close the desktop UI,
enable max power/clocks, and add swap before retrying:

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

## 5. Run AIMBOT With TensorRT

Raw YOLO output engine:

```bash
python3 scripts/run_pipeline.py \
  --backend tensorrt \
  --weights models/epoch_149_fp16.engine \
  --source 0 \
  --device cuda \
  --trt-input-shape 640x640 \
  --trt-output-format raw \
  --person-only \
  --process-scale 1.0 \
  --dry-run
```

Engine with built-in NMS:

```bash
python3 scripts/run_pipeline.py \
  --backend tensorrt \
  --weights models/epoch_149_fp16.engine \
  --source 0 \
  --device cuda \
  --trt-output-format nms \
  --dry-run
```

`--backend auto` also selects TensorRT automatically when `--weights` ends in
`.engine`, `.plan`, or `.trt`.

## SSH / X Forwarding Notes

When running through `ssh -Y`, OpenCV/GTK windows can behave differently from a
local desktop session.  For a quick inference smoke test without any window:

```bash
python3 scripts/run_pipeline.py \
  --backend tensorrt \
  --weights models/epoch_149_fp16.engine \
  --source 0 \
  --device cuda \
  --trt-output-format raw \
  --camera-backend v4l2 \
  --no-display \
  --dry-run \
  --max-frames 100
```

For interactive click tracking over X forwarding, keep the display enabled but
still force V4L2 for a USB camera:

```bash
python3 scripts/run_pipeline.py \
  --backend tensorrt \
  --weights models/epoch_149_fp16.engine \
  --source 0 \
  --device cuda \
  --trt-output-format raw \
  --camera-backend v4l2 \
  --dry-run
```

If the debug frame is pure green, the camera is likely a CSI camera being read
as raw Bayer through V4L2. Use the Argus backend instead:

```bash
python3 scripts/run_pipeline.py \
  --backend tensorrt \
  --weights models/epoch_149_fp16.engine \
  --source 0 \
  --device cuda \
  --trt-output-format raw \
  --camera-backend argus \
  --camera-width 1280 \
  --camera-height 720 \
  --camera-fps 30 \
  --debug-frame-dir /tmp/aimbot-debug \
  --dry-run
```

The `Gtk-Message: Failed to load module "canberra-gtk-module"` warning is
cosmetic.  It can be silenced with:

```bash
sudo apt install libcanberra-gtk-module libcanberra-gtk3-module
```

If X forwarding is not available, run locally on the Jetson desktop, use VNC,
or use `--no-display`.

## Troubleshooting

`ModuleNotFoundError: tensorrt`
: Install `python3-libnvinfer`, or recreate the venv with
`--system-site-packages`.

`ModuleNotFoundError: pycuda`
: Install `python3-pycuda`. The TensorRT detector uses a CUDA context that is
pushed inside the async detector worker thread.

No detections from a raw engine
: Confirm the ONNX output is YOLO-style `(1, N, 85+)`, then run with
`--trt-output-format raw`. If the engine already contains NMS, run with
`--trt-output-format nms`.

Bad box positions
: Make sure the ONNX was exported for the same input size you pass via
`--trt-input-shape`. The runtime letterboxes frames and rescales boxes back to
the original frame size.

Need Re-ID on Jetson
: Install NVIDIA's Jetson PyTorch wheel for your JetPack version, then install
`torchreid==0.2.5`. Do not use upstream `torch>=2.0.0` CUDA wheels on JetPack 4.

## References

- NVIDIA Jetson Linux R32.7.6:
  https://developer.nvidia.com/embedded/linux-tegra-r3276
- NVIDIA JetPack 4.6.1 release notes for the R32.7.x CUDA/TensorRT stack:
  https://docs.nvidia.com/jetson/jetpack/4.6.1/release-notes/index.html
- NVIDIA PyTorch for Jetson installation guide:
  https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html
