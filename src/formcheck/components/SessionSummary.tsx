"use client";

import { SessionStats } from "../types";

interface SessionSummaryProps {
  stats: SessionStats;
  exerciseName: string;
  onReset: () => void;
}

/**
 * SessionSummary — Post-session stats card.
 */
export default function SessionSummary({
  stats,
  exerciseName,
  onReset,
}: SessionSummaryProps) {
  const formPercentage =
    stats.totalReps > 0
      ? Math.round((stats.goodFormReps / stats.totalReps) * 100)
      : 0;

  const minutes = Math.floor(stats.duration / 60);
  const seconds = stats.duration % 60;
  const timeStr = `${minutes}:${seconds.toString().padStart(2, "0")}`;

  const formColor =
    formPercentage >= 80
      ? "text-emerald-500"
      : formPercentage >= 50
        ? "text-amber-500"
        : "text-red-500";

  return (
    <div className="space-y-4 p-4 rounded-xl border border-border/40 bg-card">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-bold uppercase tracking-widest text-muted-foreground">
          Session Stats
        </h3>
        <span className="text-[10px] font-bold uppercase text-primary bg-primary/10 px-2 py-0.5 rounded-full tracking-wider">
          {exerciseName}
        </span>
      </div>

      <div className="grid grid-cols-3 gap-3">
        {/* Total Reps */}
        <div className="text-center space-y-1">
          <span className="text-2xl font-black text-foreground tabular-nums">
            {stats.totalReps}
          </span>
          <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">
            Reps
          </p>
        </div>

        {/* Form Quality */}
        <div className="text-center space-y-1">
          <span className={`text-2xl font-black tabular-nums ${formColor}`}>
            {formPercentage}%
          </span>
          <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">
            Good Form
          </p>
        </div>

        {/* Duration */}
        <div className="text-center space-y-1">
          <span className="text-2xl font-black text-foreground tabular-nums font-mono">
            {timeStr}
          </span>
          <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">
            Duration
          </p>
        </div>
      </div>

      <button
        onClick={onReset}
        className="w-full py-2 text-xs font-bold uppercase tracking-widest text-muted-foreground hover:text-foreground border border-border/40 rounded-lg hover:bg-muted/50 transition-colors duration-200"
      >
        Reset Session
      </button>
    </div>
  );
}
