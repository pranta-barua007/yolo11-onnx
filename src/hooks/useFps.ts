"use client";

import { useRef, useState, useCallback } from "react";

/**
 * Non-blocking FPS counter hook.
 *
 * Uses a rolling window of frame timestamps to compute
 * a smoothed FPS value. Zero allocations per tick — only
 * a single `performance.now()` call and array push.
 *
 * @param windowMs - Rolling window duration in ms (default 1000ms).
 * @returns {{ fps: number; tick: () => void; reset: () => void }}
 *
 * @example
 * const { fps, tick, reset } = useFps();
 * // Call tick() each time an inference frame completes
 * // fps updates reactively (max once per ~250ms to avoid re-render storms)
 */
export function useFps(windowMs: number = 1000) {
  const [fps, setFps] = useState<number>(0);
  const timestampsRef = useRef<number[]>([]);
  const lastUpdateRef = useRef<number>(0);

  /** Call once per completed frame. Non-blocking — O(1) amortized. */
  const tick = useCallback(() => {
    const now = performance.now();
    const timestamps = timestampsRef.current;

    timestamps.push(now);

    // Prune timestamps outside the rolling window
    const cutoff = now - windowMs;
    while (timestamps.length > 0 && timestamps[0] < cutoff) {
      timestamps.shift();
    }

    // Throttle React state updates to ~4Hz (every 250ms) to avoid re-render storms
    if (now - lastUpdateRef.current > 250) {
      lastUpdateRef.current = now;
      setFps(timestamps.length);
    }
  }, [windowMs]);

  /** Reset the counter (e.g., when switching from camera to image mode). */
  const reset = useCallback(() => {
    timestampsRef.current = [];
    lastUpdateRef.current = 0;
    setFps(0);
  }, []);

  return { fps, tick, reset };
}
