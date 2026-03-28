# 🚀 YOLO Edge Runner: WebGPU Optimization

A high-performance, edge-native inference engine for YOLO11, optimized for **WebGPU** and **Quantized (FP16)** models.

## 🛠️ Model Export Guide

To deploy your YOLO11 model to the edge with maximum performance, follow these export guidelines in your Google Colab or Local environment.

### 1. Standard Export
For standard FP32 models (Higher accuracy, higher memory usage):
```python
from ultralytics import YOLO

model = YOLO("yolo11s.pt")
model.export(format="onnx", opset=12, dynamic=False, nms=False)
```

### 2. Half-Precision (FP16) Export
Recommended for WebGPU. Reduces model size by 50% with massive inference speedups on supported hardware.
```python
from ultralytics import YOLO

model = YOLO("yolo11s.pt")
# opset 12 is required for maximum browser compatibility
model.export(format="onnx", opset=12, half=True, nms=False)
```

---

## ⚡ WebGPU Sanitization (Crucial)

ONNX Runtime WebGPU has strict requirements for **Half-Precision** models. Standard exports often contain internal nodes (like `Resize` or `Cast`) that use incompatible data types (`INT64` or mismatched `FP32` constants).

**Before uploading to the app**, run this sanitization script to ensure the model structure is WebGPU-compliant:

```python
import onnx
from onnx import helper

def sanitize_for_webgpu(model_path, output_path):
    # Load the model and simplify it (Removes redundant INT64 layers)
    # !pip install onnx-simplifier
    from onnxsim import simplify
    
    model = onnx.load(model_path)
    model_simp, check = simplify(model)
    
    # Fix Resize nodes and Cast operations
    for node in model_simp.graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == 7:  # INT64
                    attr.i = 6 # Convert to INT32
                    
    onnx.save(model_simp, output_path)
    print(f"✅ WebGPU Ready: {output_path}")

sanitize_for_webgpu("yolo11s.onnx", "yolo11s_fixed.onnx")
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

