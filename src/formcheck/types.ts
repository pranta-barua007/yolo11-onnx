/** Joint angle definition used by exercises */
export interface JointAngleDef {
  /** Human-readable name, e.g. "left_knee" */
  name: string;
  /** COCO keypoint indices: [pointA, vertex, pointC] — angle measured at vertex */
  indices: [number, number, number];
}

/** A single form rule evaluated per frame */
export interface FormRule {
  /** Which joint angle to check */
  jointName: string;
  /** Condition: angle must be within this range to pass */
  min?: number;
  max?: number;
  /** Warning message when the rule is violated */
  message: string;
  /** Severity level */
  severity: "warning" | "bad";
}

/** Full exercise configuration */
export interface Exercise {
  id: string;
  name: string;
  description: string;
  icon: string;
  /** Primary joint angle used for rep counting */
  primaryJoint: JointAngleDef;
  /** Secondary angles to track (for display, not rep counting) */
  secondaryJoints?: JointAngleDef[];
  /** Angle thresholds that define the rep cycle */
  repThresholds: {
    /** Angle must go below this to enter "down" state */
    down: number;
    /** Angle must go above this to enter "up" state (completes rep) */
    up: number;
  };
  /** Form quality rules */
  formRules: FormRule[];
}

/** Rep counting state machine */
export type RepState = "idle" | "up" | "down";

/** Real-time form feedback */
export interface FormFeedbackData {
  quality: "good" | "warning" | "bad";
  message: string;
}

/** Accumulated session statistics */
export interface SessionStats {
  totalReps: number;
  goodFormReps: number;
  startTime: number;
  /** Duration in seconds */
  duration: number;
}

/** Keypoint with coordinates and confidence */
export interface Keypoint {
  x: number;
  y: number;
  score: number;
}
