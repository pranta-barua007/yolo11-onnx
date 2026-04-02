"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { CustomModel } from "../utils/types";
import { isWebGPUSupported } from "../utils/gpu_check";
import {
  putModelInCache,
  deleteModelFromCache,
  getCustomModelsMetadata,
  addCustomModelMetadata,
  removeCustomModelMetadata,
} from "../utils/model_cache";
import {
  Config,
  ModelCapability,
  DEFAULT_INPUT_SHAPE,
  DEFAULT_IOU_THRESHOLD,
  DEFAULT_SCORE_THRESHOLD,
} from "../utils/model_config";
import defaultClasses from "../utils/yolo_classes.json";
import { BASE_PATH } from "../utils/paths";



/**
 * Manages the YOLO model lifecycle via Web Worker.
 *
 * The model is loaded ONLY in the worker (not on main thread),
 * keeping GPU/WASM memory usage to a single copy.
 * Both image mode and camera mode route through the same worker.
 *
 * Custom models are persisted:
 *   - Model bytes → Cache API (via model_cache.ts)
 *   - Metadata (name, classes) → localStorage
 */
export function useYoloModel(initialModelName: string = "yolo11n-seg") {
  // Hydrate custom models from localStorage on mount
  const [customModels, setCustomModels] = useState<CustomModel[]>(() => {
    const persisted = getCustomModelsMetadata();
    return persisted.map((m) => ({
      name: m.name,
      url: m.cacheKey,
      classes: m.classes,
      capabilities: m.capabilities,
    }));
  });

  const [isModelLoaded, setIsModelLoaded] = useState<boolean>(false);
  const [warmUpTime, setWarmUpTime] = useState<string>("0");
  const [device, setDevice] = useState<string>(isWebGPUSupported() ? "webgpu" : "wasm");
  const [modelName, setModelName] = useState<string>(initialModelName);
  const [modelStatus, setModelStatus] = useState<string>("Loading model...");
  const [scoreThreshold, setScoreThreshold] = useState<number>(DEFAULT_SCORE_THRESHOLD);

  // Web Worker — sole owner of the ONNX session
  const workerRef = useRef<Worker | null>(null);
  const workerReadyRef = useRef<boolean>(false);

  // Active classes for the currently selected model
  const activeClasses = (() => {
    const customModel = customModels.find((m) => m.url === modelName);
    if (customModel) return customModel.classes;
    if (modelName.includes("pose")) return ["person"];
    return defaultClasses;
  })();

  const activeCapabilities = (() => {
    const customModel = customModels.find((m) => m.url === modelName);
    if (customModel) return customModel.capabilities;
    if (modelName.includes("pose")) return ["D", "P"] as ModelCapability[];
    if (modelName.includes("seg")) return ["D", "S"] as ModelCapability[];
    if (modelName === "yolo11s") return ["D", "Q"] as ModelCapability[];
    return ["D"] as ModelCapability[];
  })();

  const config: Config = { input_shape: DEFAULT_INPUT_SHAPE, iou_threshold: DEFAULT_IOU_THRESHOLD, score_threshold: scoreThreshold, classes: activeClasses, capabilities: activeCapabilities };

  // Track whether a load is already in-flight
  const loadingRef = useRef<boolean>(false);

  /** Initialize the inference worker (call once on mount) */
  const initWorker = useCallback(() => {
    if (workerRef.current) return;

    const worker = new Worker(
      new URL("../workers/inferenceWorker.ts", import.meta.url),
      { type: "module" }
    );

    worker.onmessage = (e: MessageEvent) => {
      const msg = e.data;
      if (msg.type === "model-status") {
        if (msg.status === "Model loaded") {
          workerReadyRef.current = true;
          setWarmUpTime(msg.warmUpTime);
          setModelStatus("Model loaded");
          setIsModelLoaded(true);
          loadingRef.current = false;

        } else if (msg.status === "webgpu-failed") {
          console.warn("[useYoloModel] WebGPU failed in worker, falling back to WASM...");
          loadingRef.current = false;
          setDevice("wasm"); // triggers re-load via effect
        } else if (msg.status === "Model loading failed") {
          console.error("[useYoloModel] Worker model loading failed:", msg.error);
          setModelStatus("Model loading failed");
          loadingRef.current = false;
        } else {
          setModelStatus(msg.status);
        }
      }
    };

    worker.onerror = (error) => {
      console.error("[useYoloModel] Worker error:", error);
      loadingRef.current = false;
    };

    workerRef.current = worker;
  }, []);

  /** Resolve model path — built-in uses /models/ URL, custom uses cache key directly. */
  const resolveModelPath = useCallback((name: string): string => {
    const customModel = customModels.find((m) => m.url === name);
    return customModel ? customModel.url : `${BASE_PATH}/models/${name}.onnx`;
  }, [customModels]);

  /** Load model in worker only */
  const loadModel = useCallback(async () => {
    if (loadingRef.current) {
      console.log("[useYoloModel] Load already in progress, skipping.");
      return;
    }
    loadingRef.current = true;

    setModelStatus("Loading model...");
    setIsModelLoaded(false);
    workerReadyRef.current = false;

    const model_path = resolveModelPath(modelName);

    if (workerRef.current) {
      workerRef.current.postMessage({
        type: "load-model",
        device,
        modelPath: model_path,
        config,
      });
    } else {
      console.error("[useYoloModel] Worker not initialized");
      loadingRef.current = false;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [device, modelName, customModels, resolveModelPath]);

  /**
   * Invalidate cached model and re-download.
   * Directly manages state and worker messages (bypasses loadModel guard).
   */
  const reloadModel = useCallback(() => {
    const model_path = resolveModelPath(modelName);

    // Reset UI state
    loadingRef.current = true;
    setIsModelLoaded(false);
    setModelStatus("Loading model...");
    workerReadyRef.current = false;

    // Send a single load-model message with forceReload to prevent async race conditions
    workerRef.current?.postMessage({
      type: "load-model",
      device,
      modelPath: model_path,
      config,
      forceReload: true,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelName, device, resolveModelPath]);

  /**
   * Add a custom model — persists bytes to Cache API and metadata to localStorage.
   * Called from AddModelDialog with the file's ArrayBuffer.
   */
  const addCustomModel = useCallback(async (model: CustomModel & { buffer?: ArrayBuffer }) => {
    const cacheKey = `/custom-models/${model.name}`;

    // Persist bytes to Cache API if buffer provided
    if (model.buffer) {
      await putModelInCache(cacheKey, model.buffer);
    }

    // Persist metadata to localStorage
    addCustomModelMetadata({
      name: model.name,
      classes: model.classes,
      cacheKey,
      capabilities: model.capabilities,
    });

    // Register in React state
    const registeredModel: CustomModel = {
      name: model.name,
      url: cacheKey,
      classes: model.classes,
      capabilities: model.capabilities,
    };

    setCustomModels((prev) => {
      const filtered = prev.filter((m) => m.url !== cacheKey);
      return [...filtered, registeredModel];
    });
    setModelName(cacheKey);
  }, []);

  const removeCustomModel = useCallback((url: string) => {
    deleteModelFromCache(url);
    removeCustomModelMetadata(url);
    
    setCustomModels((prev) => prev.filter((m) => m.url !== url));
    
    setModelName((prev) => (prev === url ? "yolo11n-seg" : prev));
  }, []);

  // Initialize worker on mount
  useEffect(() => {
    initWorker();
    return () => {
      if (workerRef.current) {
        workerRef.current.postMessage({ type: "release" });
        workerRef.current.terminate();
        workerRef.current = null;
      }
    };
  }, [initWorker]);

  // Load model when device/modelName/customModels change
  useEffect(() => {
    loadModel();
  }, [loadModel]);

  return {
    customModels,
    isModelLoaded,
    warmUpTime,
    workerRef,
    workerReadyRef,
    modelStatus,
    device,
    setDevice,
    modelName,
    setModelName,
    config,
    loadModel,
    reloadModel,
    addCustomModel,
    removeCustomModel,
    activeClasses,
    scoreThreshold,
    setScoreThreshold,
  };
}
