# YOLO11 ONNX Web App Specification

Welcome to the technical specification directory for the YOLO11 ONNX Web Application. This project is a client-side React/Next.js application that runs YOLO11 instance segmentation and object detection model inference directly in the browser.

## Directory Map

To explore different aspects of the codebase's design, refer to the following documents:

*   [architecture.md](architecture.md): The structural design, showing how React hooks coordinate with a Web Worker to manage the ONNX session lifecycle and execution providers (WebGPU, WASM).
*   [domain_context.md](domain_context.md): The domain language and glossary mapping the terms used in the code to mathematical and deep-learning operations (e.g. Non-Maximum Suppression, Segment Masks, Back-pressure, GC reduction).

---

## High-Level Capabilities

1.  **Multi-Modal Inputs**: Supports processing static image files (uploaded by user) and real-time webcam streams.
2.  **Edge Processing**: ONNX Runtime Web runs model execution entirely client-side. WebGPU is used as the primary execution provider, falling back to WASM if WebGPU is unsupported or fails during worker execution.
3.  **Custom Model Uploads**: Users can upload custom `.onnx` models. The application persists the binary bytes in the browser's Cache API and saves model metadata (classes, capabilities) in `localStorage`.
4.  **Zero-Allocation Pipeline**: Employs transferrable objects (`ArrayBuffer` transfers) and pre-allocated pixel buffers between the main thread and the Web Worker to eliminate Garbage Collection (GC) pauses during real-time camera processing.
