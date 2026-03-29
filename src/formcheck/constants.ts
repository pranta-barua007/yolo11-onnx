/**
 * COCO 17-Keypoint Index Map
 *
 * Readable aliases for the raw keypoint indices returned by the YOLO pose pipeline.
 * Reference: https://docs.ultralytics.com/datasets/pose/coco/#keypoints
 */

// ── Face ──
export const NOSE = 0;
export const LEFT_EYE = 1;
export const RIGHT_EYE = 2;
export const LEFT_EAR = 3;
export const RIGHT_EAR = 4;

// ── Upper Body ──
export const LEFT_SHOULDER = 5;
export const RIGHT_SHOULDER = 6;
export const LEFT_ELBOW = 7;
export const RIGHT_ELBOW = 8;
export const LEFT_WRIST = 9;
export const RIGHT_WRIST = 10;

// ── Lower Body ──
export const LEFT_HIP = 11;
export const RIGHT_HIP = 12;
export const LEFT_KNEE = 13;
export const RIGHT_KNEE = 14;
export const LEFT_ANKLE = 15;
export const RIGHT_ANKLE = 16;

/**
 * Keypoint labels for UI display.
 */
export const KEYPOINT_LABELS: Record<number, string> = {
  [NOSE]: "Nose",
  [LEFT_EYE]: "L Eye",
  [RIGHT_EYE]: "R Eye",
  [LEFT_EAR]: "L Ear",
  [RIGHT_EAR]: "R Ear",
  [LEFT_SHOULDER]: "L Shoulder",
  [RIGHT_SHOULDER]: "R Shoulder",
  [LEFT_ELBOW]: "L Elbow",
  [RIGHT_ELBOW]: "R Elbow",
  [LEFT_WRIST]: "L Wrist",
  [RIGHT_WRIST]: "R Wrist",
  [LEFT_HIP]: "L Hip",
  [RIGHT_HIP]: "R Hip",
  [LEFT_KNEE]: "L Knee",
  [RIGHT_KNEE]: "R Knee",
  [LEFT_ANKLE]: "L Ankle",
  [RIGHT_ANKLE]: "R Ankle",
};

/** Minimum keypoint confidence to consider it "visible" */
export const MIN_KEYPOINT_CONFIDENCE = 0.5;
