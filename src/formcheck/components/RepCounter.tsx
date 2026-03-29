"use client";

import { useEffect, useRef } from "react";
import { RepState } from "../types";

interface RepCounterProps {
  reps: number;
  repState: RepState;
  currentAngle: number | null;
  exerciseName: string;
}

/**
 * RepCounter — Large animated rep counter with a mini angle gauge.
 */
export default function RepCounter({
  reps,
  repState,
  currentAngle,
  exerciseName,
}: RepCounterProps) {
  const prevRepsRef = useRef(0);
  const pulseRef = useRef<HTMLDivElement>(null);

  // Trigger pulse animation on rep increment
  useEffect(() => {
    if (reps > prevRepsRef.current && pulseRef.current) {
      pulseRef.current.classList.remove("animate-rep-pulse");
      // Force reflow to restart animation
      void pulseRef.current.offsetWidth;
      pulseRef.current.classList.add("animate-rep-pulse");
    }
    prevRepsRef.current = reps;
  }, [reps]);

  const stateColor =
    repState === "down"
      ? "text-amber-400"
      : repState === "up"
        ? "text-emerald-400"
        : "text-muted-foreground";

  const stateLabel =
    repState === "down" ? "DOWN" : repState === "up" ? "UP" : "READY";

  return (
    <div className="flex flex-col items-center gap-3 py-4">
      {/* Rep count */}
      <div ref={pulseRef} className="relative">
        <span className="text-7xl font-black text-foreground tabular-nums tracking-tight leading-none">
          {reps}
        </span>
      </div>

      <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
        {exerciseName} Reps
      </span>

      {/* State + Angle */}
      <div className="flex items-center gap-3 mt-1">
        <span
          className={`text-[10px] font-bold uppercase tracking-widest px-2.5 py-1 rounded-full border ${stateColor} ${
            repState === "down"
              ? "border-amber-400/30 bg-amber-400/10"
              : repState === "up"
                ? "border-emerald-400/30 bg-emerald-400/10"
                : "border-border/40 bg-muted/50"
          }`}
        >
          {stateLabel}
        </span>

        {currentAngle !== null && (
          <span className="text-sm font-mono text-muted-foreground tabular-nums">
            {currentAngle}°
          </span>
        )}
      </div>
    </div>
  );
}
