# Domain Context & Glossary

This document defines the core domain concepts, vocabulary, and algorithms used in the YOLO11 ONNX web application.

## 1. Domain Glossary

### Model Capabilities (`ModelCapability`)
YOLO models can be trained for different tasks. In the codebase, these are marked as:
*   **`D` (Detection)**: Predicts bounding boxes (`[x, y, w, h]`), confidence scores, and class labels.
*   **`S` (Segmentation)**: Predicts instance segmentation masks in addition to detection boxes.
*   **`P` (Pose)**: Predicts human body keypoints (e.g. coordinates for eyes, shoulders, elbows, knees).
*   **`Q` (Quantized)**: Models compiled with lower-precision numbers (e.g. Float16 or Int8) to run faster on client-side hardware.

### Execution Providers (EP)
ONNX Runtime Web executes models using specific backend providers:
*   **WebGPU (`webgpu`)**: Runs inference on the client's GPU via WebGPU. Offers the lowest latency but requires browser compatibility.
*   **WASM (`wasm`)**: Runs inference on the client's CPU via WebAssembly. Used as a universal fallback.

### Input Shape (`InputShape`)
The input dimensions required by the YOLO model. The default is `[1, 3, 640, 640]`:
*   `1`: Batch size (1 image at a time).
*   `3`: Color channels (Red, Green, Blue).
*   `640`: Image height in pixels.
*   `640`: Image width in pixels.

---

## 2. Mathematical & Post-Processing Algorithms

### Non-Maximum Suppression (NMS)
YOLO models often predict multiple overlapping bounding boxes for a single object. **NMS** filters out redundant boxes:
1.  Filter out all boxes with a confidence score below the `score_threshold`.
2.  Sort the remaining boxes by confidence score descending.
3.  Take the box with the highest score, save it, and calculate its **Intersection over Union (IoU)** overlap with all other boxes of the same class.
4.  Remove any boxes with an IoU greater than `iou_threshold` (default `0.25`).
5.  Repeat the process for the next highest-scoring box.

### Instance Segmentation Mask Processing
YOLO instance segmentation does not predict masks directly. Instead, it predicts:
1.  **Mask Weights**: A vector of 32 coefficients associated with each bounding box.
2.  **Prototype Masks**: A tensor of shape `[1, 32, 160, 160]` representing 32 base masks.

To compute the final overlay mask for a detected object:
$$\text{Mask} = \sigma \left( \sum_{i=1}^{32} w_i \times P_i \right)$$

Where:
*   $w_i$ is the mask weight coefficient for the box.
*   $P_i$ is the $i$-th channel of the prototype mask.
*   $\sigma$ is the Sigmoid activation function.

This computed mask is then cropped to the bounding box boundaries and scaled to match the original source image size.

### Pose Keypoint Estimation
For models with pose capability (`P`), the model output contains coordinates for keypoint coordinates and confidence scores:
*   Each keypoint is represented by $(x, y, c)$, where $c$ is the confidence score of the keypoint.
*   The coordinates are normalized relative to the model size and must be mapped back to the video/image resolution using scale ratios.

---

## 3. Pipeline Mechanics

### Back-pressure
The camera frame-rate runs at 30–60 frames per second (fps). However, GPU inference can take anywhere from 10ms to 100ms depending on the user's hardware. **Back-pressure** prevents the worker thread's queue from building up by dropping camera frames if the worker is currently busy running inference on a previous frame.

### Transferable Memory Ownership
Normally, transferring data between the main thread and a Web Worker copies the data, duplicating memory usage. **Transferable Objects** bypass this by transferring ownership of the underlying `ArrayBuffer`. Once transferred, the sender can no longer read/write to the buffer, eliminating serialization overhead and keeping performance smooth.
