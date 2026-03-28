# 🚀 YOLO Edge Runner: WebGPU Optimization

A high-performance, edge-native inference engine for YOLO11, optimized for **WebGPU** and **Quantized (FP16)** models.

## 🛠️ Model Export Guide

To deploy your YOLO11 model to the edge with maximum performance, follow these export guidelines in your Google Colab or Local environment.

### 1. Standard Export (FP32)
For standard FP32 models (Higher accuracy, higher memory usage):
```python
from ultralytics import YOLO

model = YOLO("yolo11s.pt")
model.export(format="onnx", opset=12, dynamic=False, nms=False)
```

### 2. WebGPU-Safe Half-Precision (FP16) Export

Recommended for WebGPU. Reduces model size by 50% with faster inference on supported hardware.

> ⚠️ **Do NOT use `half=True` directly** — Ultralytics' `half=True` blindly converts all ops to float16, including `Resize` which WebGPU doesn't support. This causes `Invalid data type` errors at runtime.

**Step 1** — Export as FP32 first:
```python
from ultralytics import YOLO

model = YOLO("yolo11s.pt")
model.export(format="onnx", opset=12, dynamic=False, nms=False)
```

**Step 2** — Convert to FP16 using ONNX Runtime's converter, which keeps incompatible ops (like `Resize`) in FP32 automatically:
```python
import onnx
from onnxruntime.transformers.float16 import convert_float_to_float16

model = onnx.load("yolo11s.onnx")

model_fp16 = convert_float_to_float16(
    model,
    keep_io_types=True,
    op_block_list=["Resize", "GridSample"]  # These ops don't support float16 on WebGPU
)

onnx.save(model_fp16, "yolo11s_webgpu.onnx")
print("✅ WebGPU-safe FP16 model saved")
```

**Install dependencies:**
```bash
pip install onnx onnxruntime
```

---

## 🏗️ Getting Started

### Installation
Ensure you have Node.js 18+ and pnpm installed.

```bash
pnpm install
pnpm dev
```

### Browser Requirements
- **WebGPU Support**: Chrome 113+ or Edge 113+
- **FP16 Support**: Chrome 121+ (Requires `shader-f16` extension support)

## 💎 Features
- **Precision-Agnostic Pipeline**: Automatic detection of FP16 models with real-time bit-depth transformation.
- **Worker-Based Inference**: 100% Non-blocking UI using dedicated Web Workers for preprocessing and ONNX execution.
- **Dynamic Masking**: Hybrid CPU/GPU mask generation for Instance Segmentation.
- **Capability Aware**: Auto-tags models with `Q` (Quantized) or `I8` (Int8) indicators.
