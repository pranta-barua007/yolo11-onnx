/**
 * Worker-safe inference pipeline.
 *
 * Web Worker equivalent of `inference_pipeline.ts`.
 * Takes raw pixel data (Uint8ClampedArray) instead of HTMLCanvasElement,
 * returns pixel arrays instead of writing to canvas.
 *
 * All heavy logic is imported from shared modules (DRY):
 *   - extractDetections() from img_preprocess.ts
 *   - applyNMS() from img_preprocess.ts
 *   - generateMaskOverlay() from mask_processing.ts
 *
 * Only the preprocess function is unique (cv.matFromArray vs cv.imread).
 */
import * as ort from "onnxruntime-web";
import cv from "@techstark/opencv-js";
import { applyNMS, extractDetections, extractPoseDetections } from "../utils/img_preprocess";
import { generateMaskOverlay } from "../utils/mask_processing";

interface Config {
  input_shape: number[];
  iou_threshold: number;
  score_threshold: number;
  classes?: string[];
  capabilities?: ("D" | "S" | "P")[];
}

interface WorkerBox {
  bbox: number[];
  class_idx: number;
  score: number;
  keypoints?: { x: number; y: number; score: number }[];
}

export interface PipelineResult {
  boxes: WorkerBox[];
  inferenceTime: string;
  maskPixels: Uint8ClampedArray | null;
  maskWidth: number;
  maskHeight: number;
}

const DEFAULT_CLASSES = ["Crown", "Filling", "Periapical Lesion", "Root Canal Treatment"];

/**
 * Pre-process raw RGBA pixels into an ONNX-ready blob.
 * Worker-safe — uses cv.matFromArray instead of DOM-dependent cv.imread.
 */
function preProcessPixels(
  pixels: Uint8ClampedArray,
  srcWidth: number,
  srcHeight: number,
  modelW: number,
  modelH: number
): cv.Mat {
  const srcMat = cv.matFromArray(srcHeight, srcWidth, cv.CV_8UC4, pixels);
  cv.cvtColor(srcMat, srcMat, cv.COLOR_RGBA2RGB);
  cv.resize(srcMat, srcMat, new cv.Size(modelW, modelH));

  const blob = cv.blobFromImage(
    srcMat, 1 / 255.0,
    new cv.Size(modelW, modelH),
    new cv.Scalar(0, 0, 0),
    false, false
  );

  srcMat.delete();
  return blob;
}

/**
 * Full inference pipeline for the Web Worker.
 *
 * Flow: raw pixels → preprocess → ONNX session.run → post-process → NMS → masks
 */
export async function workerInferencePipeline(
  pixels: Uint8ClampedArray,
  srcWidth: number,
  srcHeight: number,
  session: ort.InferenceSession,
  config: Config,
  overlayW: number,
  overlayH: number,
  inputName: string
): Promise<PipelineResult> {
  const modelW = config.input_shape[3];
  const modelH = config.input_shape[2];

  const blob = preProcessPixels(pixels, srcWidth, srcHeight, modelW, modelH);
  const input_tensor = new ort.Tensor("float32", blob.data32F, [1, 3, modelH, modelW]);
  blob.delete();

  const start = performance.now();
  const output = await session.run({ [inputName]: input_tensor });
  const end = performance.now();
  input_tensor.dispose();

  const outputNames = session.outputNames;
  const output0 = output[outputNames[0]];
  const output1 = outputNames.length > 1 ? output[outputNames[1]] : null;

  if (!output0) {
    if (output1) output1.dispose();
    return { boxes: [], inferenceTime: "0", maskPixels: null, maskWidth: 0, maskHeight: 0 };
  }

  const NUM_PREDICTIONS = output0.dims[2];
  const activeClasses = config.classes ?? DEFAULT_CLASSES;
  const NUM_SCORES = activeClasses.length;
  const NUM_CHANNELS = output0.dims[1];
  const NUM_MASK_WEIGHTS = Math.max(0, NUM_CHANNELS - (4 + NUM_SCORES)); 
  const isSegmentation = config.capabilities ? config.capabilities.includes("S") : (output1 && output1.dims.length === 4 && NUM_MASK_WEIGHTS > 0);

  const predictionsData = output0.data as Float32Array;
  
  let proto_mask: Float32Array | null = null;
  let MASK_CHANNELS = 0, MASK_HEIGHT = 0, MASK_WIDTH = 0;
  
  if (isSegmentation && output1) {
    proto_mask = output1.data as Float32Array;
    MASK_CHANNELS = output1.dims[1];
    MASK_HEIGHT = output1.dims[2];
    MASK_WIDTH = output1.dims[3];
  }

  output0.dispose();
  if (output1) output1.dispose();

  const xRatio = overlayW / modelW;
  const yRatio = overlayH / modelH;

  // ── Post-process: shared functions ──
  const isPose = config.capabilities ? config.capabilities.includes("P") : false;

  let results: Array<{
    bbox: number[];
    class_idx: number;
    score: number;
    mask_weights: Float32Array;
    keypoints?: { x: number; y: number; score: number }[];
  }>;
  if (isPose) {
    results = extractPoseDetections(
      predictionsData, NUM_PREDICTIONS,
      config.score_threshold, xRatio, yRatio
    );
  } else {
    results = extractDetections(
      predictionsData, NUM_PREDICTIONS, NUM_SCORES,
      NUM_MASK_WEIGHTS, config.score_threshold, xRatio, yRatio
    );
  }

  const scoresArray = results.map((r) => r.score);
  const selected_indices = applyNMS(results, scoresArray, config.iou_threshold);
  const filtered = selected_indices.map((i) => results[i]);

  let maskResult = null;
  
  // Composition: only run mask generation if we detected a valid segmentation model
  if (isSegmentation && proto_mask) {
    maskResult = generateMaskOverlay(
      filtered, proto_mask,
      MASK_CHANNELS, MASK_HEIGHT, MASK_WIDTH,
      modelW, modelH, overlayW, overlayH,
      xRatio, yRatio
    );
  }

  const outputBoxes: WorkerBox[] = filtered.map((r) => ({
    bbox: r.bbox,
    class_idx: r.class_idx,
    score: r.score,
    keypoints: r.keypoints,
  }));

  return {
    boxes: outputBoxes,
    inferenceTime: (end - start).toFixed(2),
    maskPixels: maskResult?.pixels ?? null,
    maskWidth: overlayW,
    maskHeight: overlayH,
  };
}
