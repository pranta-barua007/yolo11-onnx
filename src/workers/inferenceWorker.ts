import * as ort from "onnxruntime-web";
import { workerInferencePipeline } from "./workerPipeline";
import { getModelFromCache, putModelInCache, deleteModelFromCache } from "../utils/model_cache";
import { ensurePrecision } from "../utils/precision";
import { Config } from "../utils/model_config";

/**
 * Wraps a promise in a timeout.
 */
function withTimeout<T>(promise: Promise<T>, timeoutMs: number, errorMsg: string): Promise<T> {
  let timeoutId: ReturnType<typeof setTimeout> | undefined;
  const timeoutPromise = new Promise<never>((_, reject) => {
    timeoutId = setTimeout(() => {
      reject(new Error(errorMsg));
    }, timeoutMs);
  });
  return Promise.race([promise, timeoutPromise]).finally(() => {
    clearTimeout(timeoutId);
  });
}

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
  config: Config;
  forceReload?: boolean;
}

export interface RunInferenceMessage {
  type: "run-inference";
  pixels: Uint8ClampedArray;
  srcWidth: number;
  srcHeight: number;
  overlayWidth: number;
  overlayHeight: number;
  config: Config;
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

const queue: WorkerInMessage[] = [];
let processing = false;

ctx.onmessage = (e: MessageEvent<WorkerInMessage>) => {
  const msg = e.data;
  if (msg.type === "load-model") {
    queue.length = 0; // Cancel all pending loads/inferences
  }
  queue.push(msg);
  processQueue();
};

async function processQueue() {
  if (processing) return;
  processing = true;

  while (queue.length > 0) {
    const msg = queue.shift();
    if (msg) {
      try {
        await handleMessage(msg);
      } catch (err) {
        console.error("[Worker] Error processing message in queue:", err);
      }
    }
  }

  processing = false;
}

async function handleMessage(msg: WorkerInMessage) {
  switch (msg.type) {
    // ── Load or switch model (with caching) ──
    case "load-model": {
      const key = `${msg.device}|${msg.modelPath}`;
      
      const postStatus = (status: string, extra = {}) => {
        ctx.postMessage({
          type: "model-status",
          status,
          modelPath: msg.modelPath,
          device: msg.device,
          ...extra
        });
      };

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
        postStatus("Model loaded", { warmUpTime: "0" });
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
          postStatus("Loading from cache...");
        } else {
          postStatus("Downloading model... ");
          const response = await fetch(msg.modelPath);
          if (!response.ok) throw new Error(`Failed to fetch model: ${response.status}`);
          modelBuffer = await response.arrayBuffer();
          // Cache for next time (non-blocking)
          putModelInCache(msg.modelPath, modelBuffer.slice(0));
        }

        // ── Create session via blob URL (preserves ONNX RT's optimized URL path) ──
        postStatus("Initializing model...");
        const blob = new Blob([modelBuffer], { type: "application/octet-stream" });
        const blobUrl = URL.createObjectURL(blob);
        activeBlobUrl = blobUrl;

        // Wrap session creation and warmup in a timeout to catch WebGPU shader compilation/driver hangs
        const initializeSession = async () => {
          const ep = msg.device === "webgpu" ? {
            name: "webgpu",
            options: {
              forceCpuNodeNames: [
                "/model.11/Resize",
                "/model.11/Resize_input_cast0",
                "/model.11/Resize_input_cast_0",
                "/model.11/Resize_output_0",
                "/model.11/Resize_output_cast0",
                "/model.11/Resize_output_cast_0",
                "/model.14/Resize",
                "/model.14/Resize_input_cast0",
                "/model.14/Resize_input_cast_0",
                "/model.14/Resize_output_0",
                "/model.14/Resize_output_cast0",
                "/model.14/Resize_output_cast_0"
              ],
            }
          } : msg.device;

          const sess = await ort.InferenceSession.create(blobUrl, {
            executionProviders: [ep],
            graphOptimizationLevel: "all",
            logSeverityLevel: 3,
          });

          // ── Session Diagnostics ──
          const sessionWithMeta = sess as unknown as { 
            inputMetadata: Record<string, { type: string }>; 
            outputMetadata: Record<string, unknown>;
          };
          console.log(`[Worker] Session created on ${msg.device}`);
          console.log("[Worker] Input Metadata:", sessionWithMeta.inputMetadata);
          console.log("[Worker] Output Metadata:", sessionWithMeta.outputMetadata);

          // ── Precision-Aware Warm-up ──
          const inputName = sess.inputNames[0] || "images";
          const inputMeta = sessionWithMeta.inputMetadata?.[inputName];
          const inputType = inputMeta?.type || "float32";

          const shape = msg.config.input_shape;
          const dummySize = shape.reduce((a, b) => a * b, 1);
          
          // Use ensurePrecision to match model requirements
          const dummyData = ensurePrecision(new Float32Array(dummySize), inputType);
          const dummy = new ort.Tensor(inputType as "float32" | "float16", dummyData, shape);
          
          try {
            const warmupOutput = await sess.run({ [inputName]: dummy });
            Object.values(warmupOutput).forEach(t => t.dispose());
          } finally {
            dummy.dispose();
          }

          return sess;
        };

        session = await withTimeout(
          initializeSession(),
          30000,
          `Model initialization timed out after 30s on ${msg.device}`
        );

        const warmUpTime = (performance.now() - start).toFixed(2);
        sessionKey = key;

        postStatus("Model loaded", { warmUpTime });
      } catch (error: unknown) {
        const errMsg = error instanceof Error ? error.message : "Unknown error";
        console.error("[Worker] Error loading model:", error);

        if (msg.device === "webgpu") {
          postStatus("webgpu-failed", { error: errMsg });
        } else {
          postStatus("Model loading failed", { error: errMsg });
        }
      }
      break;
    }

    // ── Invalidate cached model ──
    case "invalidate-cache": {
      await deleteModelFromCache(msg.modelPath);
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
        const inputName = session.inputNames[0] || "images";
        const result = await workerInferencePipeline(
          msg.pixels,
          msg.srcWidth,
          msg.srcHeight,
          session,
          msg.config,
          msg.overlayWidth,
          msg.overlayHeight,
          inputName
        );

        const transfer: Transferable[] = [];
        if (result.maskPixels) {
          transfer.push(result.maskPixels.buffer);
        }
        if (result.originalBuffer) {
          transfer.push(result.originalBuffer as ArrayBuffer);
        }

        ctx.postMessage(
          {
            type: "inference-result",
            boxes: result.boxes,
            inferenceTime: result.inferenceTime,
            maskPixels: result.maskPixels,
            maskWidth: result.maskWidth,
            maskHeight: msg.type === "run-inference" ? result.maskHeight : 0, // safe check
            originalBuffer: result.originalBuffer,
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
}
