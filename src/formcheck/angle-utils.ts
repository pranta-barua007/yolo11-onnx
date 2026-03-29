import { Keypoint, JointAngleDef } from "./types";
import { MIN_KEYPOINT_CONFIDENCE } from "./constants";

/**
 * Calculate the angle (in degrees) at point B formed by the line segments BA and BC.
 *
 * Uses atan2 for full 360° range, then normalizes to 0-180°.
 * Pure function — no side effects.
 *
 * @param a - First point
 * @param b - Vertex point (where the angle is measured)
 * @param c - Third point
 * @returns Angle in degrees (0-180), or null if any point is invalid.
 */
export function calculateAngle(
  a: { x: number; y: number },
  b: { x: number; y: number },
  c: { x: number; y: number }
): number {
  const radians =
    Math.atan2(c.y - b.y, c.x - b.x) - Math.atan2(a.y - b.y, a.x - b.x);

  let degrees = Math.abs(radians * (180 / Math.PI));

  // Normalize to 0-180 range
  if (degrees > 180) {
    degrees = 360 - degrees;
  }

  return Math.round(degrees);
}

/**
 * Get the angle for a named joint definition from a keypoints array.
 *
 * Returns null if any of the three keypoints have low confidence.
 *
 * @param keypoints - Array of 17 COCO keypoints from pose detection
 * @param joint - Joint angle definition specifying the 3 keypoint indices
 * @returns Angle in degrees, or null if keypoints aren't visible enough
 */
export function getJointAngle(
  keypoints: Keypoint[],
  joint: JointAngleDef
): number | null {
  const [idxA, idxB, idxC] = joint.indices;

  const a = keypoints[idxA];
  const b = keypoints[idxB];
  const c = keypoints[idxC];

  // All three keypoints must be visible
  if (
    !a || !b || !c ||
    a.score < MIN_KEYPOINT_CONFIDENCE ||
    b.score < MIN_KEYPOINT_CONFIDENCE ||
    c.score < MIN_KEYPOINT_CONFIDENCE
  ) {
    return null;
  }

  return calculateAngle(a, b, c);
}

/**
 * Compute all relevant angles for a set of joint definitions.
 *
 * @returns A map of joint name → angle (degrees), excluding invisible joints.
 */
export function computeAngles(
  keypoints: Keypoint[],
  joints: JointAngleDef[]
): Record<string, number> {
  const result: Record<string, number> = {};

  for (const joint of joints) {
    const angle = getJointAngle(keypoints, joint);
    if (angle !== null) {
      result[joint.name] = angle;
    }
  }

  return result;
}
