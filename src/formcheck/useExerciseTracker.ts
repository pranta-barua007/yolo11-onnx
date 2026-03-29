"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { Exercise, RepState, FormFeedbackData, SessionStats, Keypoint } from "./types";
import { getJointAngle, computeAngles } from "./angle-utils";
import { Box } from "../utils/types";

/**
 * useExerciseTracker — Core fitness tracking hook.
 *
 * Consumes the `details` (Box[]) array from useImageProcessing and
 * processes keypoints through a rep-counting state machine.
 *
 * Architecture:
 * - Receives detected boxes from the existing pipeline (no worker/model interaction)
 * - Extracts keypoints from the first detected person
 * - Calculates joint angles using pure math functions
 * - Runs a state machine: idle → down → up (1 rep) → down → up (2 reps) → ...
 * - Evaluates form rules on every frame
 *
 * @param exercise - The currently selected exercise config
 * @param details - Box[] from useImageProcessing (updated every frame)
 * @param isActive - Whether tracking is currently active (camera running)
 */
export function useExerciseTracker(
  exercise: Exercise | null,
  details: Box[],
  isActive: boolean
) {
  const [reps, setReps] = useState(0);
  const [repState, setRepState] = useState<RepState>("idle");
  const [currentAngle, setCurrentAngle] = useState<number | null>(null);
  const [allAngles, setAllAngles] = useState<Record<string, number>>({});
  const [formFeedback, setFormFeedback] = useState<FormFeedbackData>({
    quality: "good",
    message: "Ready to start",
  });
  const [sessionStats, setSessionStats] = useState<SessionStats>({
    totalReps: 0,
    goodFormReps: 0,
    startTime: 0,
    duration: 0,
  });

  // Refs for state machine (avoid stale closures in the hot loop)
  const repStateRef = useRef<RepState>("idle");
  const repsRef = useRef(0);
  const goodFormRepsRef = useRef(0);
  const startTimeRef = useRef(0);
  const lastFormRef = useRef<FormFeedbackData>({ quality: "good", message: "Ready to start" });

  // Throttle state updates to avoid re-render storms during real-time tracking
  const lastUpdateRef = useRef(0);
  const UPDATE_INTERVAL_MS = 80; // ~12 UI updates/sec (smooth enough, non-blocking)

  /**
   * Reset the tracker for a new session / exercise change.
   */
  const reset = useCallback(() => {
    setReps(0);
    setRepState("idle");
    setCurrentAngle(null);
    setAllAngles({});
    setFormFeedback({ quality: "good", message: "Ready to start" });
    setSessionStats({ totalReps: 0, goodFormReps: 0, startTime: 0, duration: 0 });

    repStateRef.current = "idle";
    repsRef.current = 0;
    goodFormRepsRef.current = 0;
    startTimeRef.current = 0;
    lastFormRef.current = { quality: "good", message: "Ready to start" };
  }, []);

  // Reset when exercise changes
  /* eslint-disable react-hooks/set-state-in-effect */
  useEffect(() => {
    reset();
  }, [exercise?.id, reset]);
  /* eslint-enable react-hooks/set-state-in-effect */

  /**
   * Process a frame's detections.
   * Called on every render when `details` changes and tracking is active.
   */
  /* eslint-disable react-hooks/set-state-in-effect */
  useEffect(() => {
    if (!exercise || !isActive || details.length === 0) return;

    // Find the first person with keypoints
    const person = details.find(
      (box) => box.keypoints && box.keypoints.length === 17
    );
    if (!person || !person.keypoints) return;

    const keypoints = person.keypoints as Keypoint[];

    // ── Calculate primary angle ──
    const primaryAngle = getJointAngle(keypoints, exercise.primaryJoint);
    if (primaryAngle === null) return; // Primary joint not visible

    // ── Calculate all tracked angles ──
    const allJoints = [exercise.primaryJoint, ...(exercise.secondaryJoints ?? [])];
    const angles = computeAngles(keypoints, allJoints);

    // ── Rep counting state machine ──
    const prevState = repStateRef.current;
    let newState = prevState;

    if (prevState === "idle") {
      // Start tracking once we see the person in "up" position
      if (primaryAngle >= exercise.repThresholds.up) {
        newState = "up";
        if (startTimeRef.current === 0) {
          startTimeRef.current = performance.now();
        }
      }
    } else if (prevState === "up") {
      // Transition to "down" when angle drops below threshold
      if (primaryAngle <= exercise.repThresholds.down) {
        newState = "down";
      }
    } else if (prevState === "down") {
      // Transition back to "up" = 1 complete rep
      if (primaryAngle >= exercise.repThresholds.up) {
        newState = "up";
        repsRef.current += 1;

        // Check if form was good during this rep
        if (lastFormRef.current.quality === "good") {
          goodFormRepsRef.current += 1;
        }
      }
    }

    repStateRef.current = newState;

    // ── Evaluate form rules ──
    let currentForm: FormFeedbackData = { quality: "good", message: "Great form! 💪" };

    for (const rule of exercise.formRules) {
      const ruleAngle = angles[rule.jointName];
      if (ruleAngle === undefined) continue;

      const violated =
        (rule.min !== undefined && ruleAngle < rule.min) ||
        (rule.max !== undefined && ruleAngle < rule.max && ruleAngle > 0);

      if (violated) {
        currentForm = {
          quality: rule.severity,
          message: rule.message,
        };
        break; // Show the first (most severe) violation
      }
    }

    lastFormRef.current = currentForm;

    // ── Throttled state updates ──
    const now = performance.now();
    if (now - lastUpdateRef.current >= UPDATE_INTERVAL_MS) {
      lastUpdateRef.current = now;

      // Batched state sync from external frame data — intentional
      setCurrentAngle(primaryAngle);
      setAllAngles(angles);
      setRepState(newState);
      setReps(repsRef.current);
      setFormFeedback(currentForm);

      const elapsed = startTimeRef.current > 0
        ? Math.floor((now - startTimeRef.current) / 1000)
        : 0;

      setSessionStats({
        totalReps: repsRef.current,
        goodFormReps: goodFormRepsRef.current,
        startTime: startTimeRef.current,
        duration: elapsed,
      });
    }
    // eslint-enable react-hooks/set-state-in-effect
  }, [details, exercise, isActive]);

  return {
    reps,
    repState,
    currentAngle,
    allAngles,
    formFeedback,
    sessionStats,
    reset,
  };
}
