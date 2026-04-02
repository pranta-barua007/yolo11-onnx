/**
 * Centralized YOLO model configuration.
 *
 * Single source of truth for model input shape, inference thresholds,
 * and the shared Config interface. Imported by hooks, workers, and
 * utilities — keeping all pipeline consumers in sync.
 */

// ── Input Shape Types ──

/** Batch size for ONNX inference (always 1 for single-image). */
export type BatchSize = number;

/** Number of color channels (3 = RGB). */
export type ColorChannels = number;

/** Model input height in pixels. */
export type ModelHeight = number;

/** Model input width in pixels. */
export type ModelWidth = number;

/** ONNX model input tensor shape: [batch, channels, height, width]. */
export type InputShape = [BatchSize, ColorChannels, ModelHeight, ModelWidth];

// ── Model Capability Flags ──

/** Model capability identifiers: Detection, Segmentation, Pose, Quantized. */
export type ModelCapability = "D" | "S" | "P" | "Q";

// ── Pipeline Config Interface ──

/**
 * Runtime configuration passed through the inference pipeline.
 *
 * Shared by: useYoloModel → useImageProcessing → inferenceWorker → workerPipeline
 */
export interface Config {
  input_shape: InputShape;
  iou_threshold: number;
  score_threshold: number;
  classes?: string[];
  capabilities?: ModelCapability[];
}

// ── Default Values ──

/** Default ONNX input tensor shape for YOLO11 models. */
export const DEFAULT_INPUT_SHAPE: InputShape = [1, 3, 640, 640];

/** Default IoU threshold for Non-Maximum Suppression. */
export const DEFAULT_IOU_THRESHOLD = 0.25;

/** Default confidence score threshold for detections. */
export const DEFAULT_SCORE_THRESHOLD = 0.55;
