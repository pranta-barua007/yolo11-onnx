import { Exercise } from "./types";
import * as KP from "./constants";

/**
 * Pre-configured exercise definitions.
 *
 * Each exercise defines:
 * - Which joint angle to track for rep counting
 * - The angle thresholds that define a complete rep cycle
 * - Form quality rules that trigger warnings
 */

export const SQUAT: Exercise = {
  id: "squat",
  name: "Squat",
  description: "Track knee flexion and hip depth for proper squatting form.",
  icon: "🏋️",
  primaryJoint: {
    name: "left_knee",
    indices: [KP.LEFT_HIP, KP.LEFT_KNEE, KP.LEFT_ANKLE],
  },
  secondaryJoints: [
    {
      name: "right_knee",
      indices: [KP.RIGHT_HIP, KP.RIGHT_KNEE, KP.RIGHT_ANKLE],
    },
    {
      name: "left_hip",
      indices: [KP.LEFT_SHOULDER, KP.LEFT_HIP, KP.LEFT_KNEE],
    },
  ],
  repThresholds: {
    down: 100, // Knee angle must drop below 100° to count as "down"
    up: 160,   // Knee angle must rise above 160° to count as "up" (standing)
  },
  formRules: [
    {
      jointName: "left_knee",
      max: 60,
      message: "Too deep — risk of knee strain",
      severity: "bad",
    },
    {
      jointName: "left_hip",
      max: 70,
      message: "Keep your chest up — leaning too far forward",
      severity: "warning",
    },
  ],
};

export const PUSHUP: Exercise = {
  id: "pushup",
  name: "Push-up",
  description: "Monitor elbow extension and body alignment during push-ups.",
  icon: "💪",
  primaryJoint: {
    name: "left_elbow",
    indices: [KP.LEFT_SHOULDER, KP.LEFT_ELBOW, KP.LEFT_WRIST],
  },
  secondaryJoints: [
    {
      name: "right_elbow",
      indices: [KP.RIGHT_SHOULDER, KP.RIGHT_ELBOW, KP.RIGHT_WRIST],
    },
  ],
  repThresholds: {
    down: 100, // Elbow angle must drop below 100° for "down"
    up: 155,   // Elbow angle must rise above 155° for "up" (arms extended)
  },
  formRules: [
    {
      jointName: "left_elbow",
      max: 50,
      message: "Elbows too tight — don't go below 90°",
      severity: "warning",
    },
  ],
};

export const SHOULDER_PRESS: Exercise = {
  id: "shoulder_press",
  name: "Shoulder Press",
  description: "Track overhead arm extension for proper pressing mechanics.",
  icon: "🙌",
  primaryJoint: {
    name: "left_elbow",
    indices: [KP.LEFT_SHOULDER, KP.LEFT_ELBOW, KP.LEFT_WRIST],
  },
  secondaryJoints: [
    {
      name: "right_elbow",
      indices: [KP.RIGHT_SHOULDER, KP.RIGHT_ELBOW, KP.RIGHT_WRIST],
    },
    {
      name: "left_shoulder",
      indices: [KP.LEFT_ELBOW, KP.LEFT_SHOULDER, KP.LEFT_HIP],
    },
  ],
  repThresholds: {
    down: 100, // Elbow angle below 100° = arms down
    up: 160,   // Elbow angle above 160° = arms fully extended overhead
  },
  formRules: [
    {
      jointName: "left_shoulder",
      max: 80,
      message: "Arms not reaching full extension overhead",
      severity: "warning",
    },
  ],
};

/** All available exercises */
export const EXERCISES: Exercise[] = [SQUAT, PUSHUP, SHOULDER_PRESS];
