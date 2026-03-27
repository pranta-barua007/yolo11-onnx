import * as ort from "onnxruntime-web";
import { workerInferencePipeline } from "./workerPipeline";
import { getModelFromCache, putModelInCache, deleteModelFromCache } from "../utils/model_cache";

// ── Worker-global state ──
let session: ort.InferenceSession | null = null;
let sessionKey: string | null = null;
/** Track blob URLs to revoke on session release (prevents memory leaks). */
let activeBlobUrl: string | null = null;

// ── Message types ──
export interface LoadModelMessage {
  type: "load-model";
  device: string;
  modelPath: string;
  config: {
    input_shape: number[];
    iou_threshold: number;
    score_threshold: number;
    classes?: string[];
  };
  forceReload?: boolean;
}

export interface RunInferenceMessage {
  type: "run-inference";
  pixels: Uint8ClampedArray;
  srcWidth: number;
  srcHeight: number;
  overlayWidth: number;
  overlayHeight: number;
  config: {
    input_shape: number[];
    iou_threshold: number;
    score_threshold: number;
    classes?: string[];
  };
}

export interface ReleaseMessage {
  type: "release";
}

export interface InvalidateCacheMessage {
  type: "invalidate-cache";
  modelPath: string;
}

export type WorkerInMessage =
  | LoadModelMessage
  | RunInferenceMessage
  | ReleaseMessage
  | InvalidateCacheMessage;

const ctx: Worker = self as unknown as Worker;

/** Revoke any active blob URL to free memory. */
function revokeActiveBlobUrl() {
  if (activeBlobUrl) {
    URL.revokeObjectURL(activeBlobUrl);
    activeBlobUrl = null;
  }
}

ctx.onmessage = async (e: MessageEvent<WorkerInMessage>) => {
  const msg = e.data;

  switch (msg.type) {
    // ── Load or switch model (with caching) ──
    case "load-model": {
      const key = `${msg.device}|${msg.modelPath}`;

      if (msg.forceReload) {
        // Clear cache and release session entirely before proceeding
        await deleteModelFromCache(msg.modelPath);
        if (session) {
          try { await session.release(); } catch { /* ignore */ }
          session = null;
          sessionKey = null;
          revokeActiveBlobUrl();
        }
      } else if (session && sessionKey === key) {
        // Reuse if same model and not a forced reload
        ctx.postMessage({
          type: "model-status",
          status: "Model loaded",
          warmUpTime: "0",
        });
        return;
      }

      // If we are switching models without forcing reload, we still need to release the old one
      if (session) {
        try { await session.release(); } catch { /* ignore */ }
        session = null;
        sessionKey = null;
        revokeActiveBlobUrl();
      }

      try {
        ort.env.logLevel = "error";
        const start = performance.now();

        // ── Resolve model bytes (from cache or network) ──
        let modelBuffer = await getModelFromCache(msg.modelPath);

        if (modelBuffer) {
          ctx.postMessage({ type: "model-status", status: "Loading from cache..." });
        } else {
          ctx.postMessage({ type: "model-status", status: "Downloading model..." });
          const response = await fetch(msg.modelPath);
          if (!response.ok) throw new Error(`Failed to fetch model: ${response.status}`);
          modelBuffer = await response.arrayBuffer();
          // Cache for next time (non-blocking)
          putModelInCache(msg.modelPath, modelBuffer.slice(0));
        }

        // ── Create session via blob URL (preserves ONNX RT's optimized URL path) ──
        ctx.postMessage({ type: "model-status", status: "Initializing model..." });
        const blob = new Blob([modelBuffer], { type: "application/octet-stream" });
        const blobUrl = URL.createObjectURL(blob);
        activeBlobUrl = blobUrl;

        session = await ort.InferenceSession.create(blobUrl, {
          executionProviders: [msg.device],
          graphOptimizationLevel: "all",
          logSeverityLevel: 3,
        });

        // Warm-up inference
        const shape = msg.config.input_shape;
        const dummySize = shape.reduce((a, b) => a * b, 1);
        const dummy = new ort.Tensor("float32", new Float32Array(dummySize), shape);
        const warmupOutput = await session.run({ images: dummy });
        
        // ── Dynamic Capability Parsing ──
        const outputNames = session.outputNames;
        const output0 = warmupOutput[outputNames[0]];
        const output1 = outputNames.length > 1 ? warmupOutput[outputNames[1]] : null;
        
        const capabilities: ("D" | "S" | "P")[] = ["D"]; // YOLO base
        
        if (output0) {
          const NUM_CHANNELS = output0.dims[1];
          const NUM_SCORES = msg.config.classes ? msg.config.classes.length : 80;
          const NUM_MASK_WEIGHTS = Math.max(0, NUM_CHANNELS - (4 + NUM_SCORES));

          if (output1 && output1.dims.length === 4 && NUM_MASK_WEIGHTS > 0) {
            capabilities.push("S");
          } else if (NUM_MASK_WEIGHTS > 0) {
            capabilities.push("P"); // Keypoints (future)
          }
        }
        
        for (const name of outputNames) {
          warmupOutput[name]?.dispose();
        }
        dummy.dispose();

        const warmUpTime = (performance.now() - start).toFixed(2);
        sessionKey = key;

        ctx.postMessage({
          type: "model-status",
          status: "Model loaded",
          warmUpTime,
          capabilities,
          modelPath: msg.modelPath,
        });
      } catch (error: unknown) {
        const errMsg = error instanceof Error ? error.message : "Unknown error";
        console.error("[Worker] Error loading model:", error);

        // If WebGPU failed, tell main thread to fallback
        if (msg.device === "webgpu") {
          ctx.postMessage({
            type: "model-status",
            status: "webgpu-failed",
            error: errMsg,
          });
        } else {
          ctx.postMessage({
            type: "model-status",
            status: "Model loading failed",
            error: errMsg,
          });
        }
      }
      break;
    }

    // ── Invalidate cached model ──
    case "invalidate-cache": {
      await deleteModelFromCache(msg.modelPath);
      // Also release session so next load-model re-downloads
      if (session) {
        try { await session.release(); } catch { /* ignore */ }
        session = null;
        sessionKey = null;
        revokeActiveBlobUrl();
      }
      break;
    }

    // ── Run inference on a frame ──
    case "run-inference": {
      if (!session) {
        ctx.postMessage({
          type: "inference-result",
          error: "No model loaded",
        });
        return;
      }

      try {
        const result = await workerInferencePipeline(
          msg.pixels,
          msg.srcWidth,
          msg.srcHeight,
          session,
          msg.config,
          msg.overlayWidth,
          msg.overlayHeight
        );

        // Transfer the mask pixel buffer for zero-copy (if it exists)
        const transfer: Transferable[] = [];
        if (result.maskPixels) {
          transfer.push(result.maskPixels.buffer);
        }

        ctx.postMessage(
          {
            type: "inference-result",
            boxes: result.boxes,
            inferenceTime: result.inferenceTime,
            maskPixels: result.maskPixels,
            maskWidth: result.maskWidth,
            maskHeight: result.maskHeight,
          },
          transfer
        );
      } catch (error: unknown) {
        const errMsg = error instanceof Error ? error.message : "Unknown error";
        console.error("[Worker] Inference error:", error);
        ctx.postMessage({ type: "inference-result", error: errMsg });
      }
      break;
    }

    // ── Release model ──
    case "release": {
      if (session) {
        try { await session.release(); } catch { /* ignore */ }
        session = null;
        sessionKey = null;
        revokeActiveBlobUrl();
      }
      break;
    }
  }
};
