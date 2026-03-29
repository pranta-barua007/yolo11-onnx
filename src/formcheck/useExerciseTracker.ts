"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { Exercise, RepState, FormFeedbackData, SessionStats, Keypoint } from "./types";
import { getJointAngle, computeAngles } from "./angle-utils";
import { Box } from "../utils/types";

/**
 * Snapshot of tracker state — written by the rAF-synchronized effect,
 * read by the component to trigger UI updates via the prevSnapshot pattern.
 */
interface TrackerSnapshot {
  reps: number;
  repState: RepState;
  currentAngle: number | null;
  allAngles: Record<string, number>;
  formFeedback: FormFeedbackData;
  sessionStats: SessionStats;
}

const INITIAL_SNAPSHOT: TrackerSnapshot = {
  reps: 0,
  repState: "idle",
  currentAngle: null,
  allAngles: {},
  formFeedback: { quality: "good", message: "Ready to start" },
  sessionStats: { totalReps: 0, goodFormReps: 0, startTime: 0, duration: 0 },
};

/**
 * useExerciseTracker — Core fitness tracking hook.
 *
 * Consumes the `details` (Box[]) array from useImageProcessing and
 * processes keypoints through a rep-counting state machine.
 *
 * Architecture:
 * - State machine runs in refs (no React state updates during the hot path)
 * - A useEffect synchronizes external frame data (details) into the state machine
 * - A ref-based snapshot is compared during render using the React 19
 *   "storing information from previous renders" pattern to batch UI updates
 *
 * React 19 patterns used:
 * - State reset: parent uses `key={exercise?.id}` to unmount/remount
 * - Frame processing: useEffect writes to a ref snapshot; render compares
 *   prev snapshot to detect changes and calls setState (pure, no impure calls)
 *
 * @see https://react.dev/learn/you-might-not-need-an-effect#resetting-all-state-when-a-prop-changes
 * @see https://react.dev/reference/react/useState#storing-information-from-previous-renders
 */
export function useExerciseTracker(
  exercise: Exercise | null,
  details: Box[],
  isActive: boolean
) {
  // ── UI state (what the component renders) ──
  const [snapshot, setSnapshot] = useState<TrackerSnapshot>(INITIAL_SNAPSHOT);

  // ── Refs for the state machine (mutated in effect, read during render) ──
  const repStateRef = useRef<RepState>("idle");
  const repsRef = useRef(0);
  const goodFormRepsRef = useRef(0);
  const startTimeRef = useRef(0);
  const lastFormRef = useRef<FormFeedbackData>({ quality: "good", message: "Ready to start" });

  // Throttle UI updates to ~12/sec
  const lastUpdateRef = useRef(0);
  const UPDATE_INTERVAL_MS = 80;

  // Snapshot ref for the render comparison
  const snapshotRef = useRef<TrackerSnapshot>(INITIAL_SNAPSHOT);

  /**
   * Reset the tracker for a new session (called by "Reset Session" button).
   * Note: Exercise change resets are handled by parent using key={exercise?.id}
   * which unmounts/remounts this hook, resetting all state automatically.
   */
  const reset = useCallback(() => {
    repStateRef.current = "idle";
    repsRef.current = 0;
    goodFormRepsRef.current = 0;
    startTimeRef.current = 0;
    lastFormRef.current = { quality: "good", message: "Ready to start" };
    lastUpdateRef.current = 0;
    snapshotRef.current = INITIAL_SNAPSHOT;
    setSnapshot(INITIAL_SNAPSHOT);
  }, []);

  /**
   * Process frame detections — runs as an effect that syncs external
   * inference data (details from the worker) into the internal state machine.
   * Only mutates refs (pure from React's perspective). UI updates are
   * batched into a snapshot ref and picked up during the next render.
   */
  useEffect(() => {
    if (!exercise || !isActive || details.length === 0) return;

    // Find the first person with keypoints
    const person = details.find(
      (box) => box.keypoints && box.keypoints.length === 17
    );
    if (!person?.keypoints) return;

    const keypoints = person.keypoints as Keypoint[];

    // ── Calculate primary angle ──
    const primaryAngle = getJointAngle(keypoints, exercise.primaryJoint);
    if (primaryAngle === null) return;

    // ── Calculate all tracked angles ──
    const allJoints = [exercise.primaryJoint, ...(exercise.secondaryJoints ?? [])];
    const angles = computeAngles(keypoints, allJoints);

    // ── Rep counting state machine (ref-only, no setState) ──
    const prevState = repStateRef.current;
    let newState = prevState;

    if (prevState === "idle") {
      if (primaryAngle >= exercise.repThresholds.up) {
        newState = "up";
        if (startTimeRef.current === 0) {
          startTimeRef.current = performance.now();
        }
      }
    } else if (prevState === "up") {
      if (primaryAngle <= exercise.repThresholds.down) {
        newState = "down";
      }
    } else if (prevState === "down") {
      if (primaryAngle >= exercise.repThresholds.up) {
        newState = "up";
        repsRef.current += 1;
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
        currentForm = { quality: rule.severity, message: rule.message };
        break;
      }
    }

    lastFormRef.current = currentForm;

    // ── Throttled snapshot update (ref only — no setState in effect) ──
    const now = performance.now();
    if (now - lastUpdateRef.current >= UPDATE_INTERVAL_MS) {
      lastUpdateRef.current = now;

      const elapsed = startTimeRef.current > 0
        ? Math.floor((now - startTimeRef.current) / 1000)
        : 0;

      // Write new snapshot to ref — will be picked up next render
      snapshotRef.current = {
        reps: repsRef.current,
        repState: newState,
        currentAngle: primaryAngle,
        allAngles: angles,
        formFeedback: currentForm,
        sessionStats: {
          totalReps: repsRef.current,
          goodFormReps: goodFormRepsRef.current,
          startTime: startTimeRef.current,
          duration: elapsed,
        },
      };

      // Schedule a re-render to pick up the new snapshot
      setSnapshot(snapshotRef.current);
    }
  }, [details, exercise, isActive]);

  return {
    reps: snapshot.reps,
    repState: snapshot.repState,
    currentAngle: snapshot.currentAngle,
    allAngles: snapshot.allAngles,
    formFeedback: snapshot.formFeedback,
    sessionStats: snapshot.sessionStats,
    reset,
  };
}
