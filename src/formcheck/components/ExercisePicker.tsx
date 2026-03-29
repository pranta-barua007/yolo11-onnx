"use client";

import { Exercise } from "../types";
import { EXERCISES } from "../exercises";

interface ExercisePickerProps {
  selectedExercise: Exercise | null;
  onSelect: (exercise: Exercise | null) => void;
  disabled?: boolean;
}

/**
 * ExercisePicker — Shows all exercises when none is selected.
 * When one is selected, collapses to show only that exercise + a back button.
 */
export default function ExercisePicker({
  selectedExercise,
  onSelect,
  disabled,
}: ExercisePickerProps) {
  // ── Focused view: only show the active exercise ──
  if (selectedExercise) {
    return (
      <div className="space-y-2">
        <button
          onClick={() => onSelect(null)}
          className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors duration-200"
        >
          <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="m15 18-6-6 6-6" />
          </svg>
          All Exercises
        </button>

        <div className="w-full text-left p-3 rounded-xl border border-primary bg-primary/5 ring-1 ring-primary/30 shadow-sm">
          <div className="flex items-center gap-3">
            <span className="text-2xl leading-none" aria-hidden="true">
              {selectedExercise.icon}
            </span>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <span className="text-sm font-semibold text-foreground">
                  {selectedExercise.name}
                </span>
                <span className="text-[9px] font-bold uppercase text-primary bg-primary/10 px-1.5 py-0.5 rounded-full tracking-wider">
                  Active
                </span>
              </div>
              <p className="text-xs text-muted-foreground mt-0.5 line-clamp-2">
                {selectedExercise.description}
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // ── Full list: no exercise selected ──
  return (
    <div className="space-y-3">
      <h3 className="text-xs font-bold uppercase tracking-widest text-muted-foreground">
        Choose Exercise
      </h3>
      <div className="grid grid-cols-1 gap-2">
        {EXERCISES.map((exercise) => (
          <button
            key={exercise.id}
            onClick={() => onSelect(exercise)}
            disabled={disabled}
            className={`
              group w-full text-left p-3.5 rounded-xl border transition-all duration-200
              border-border/40 bg-card hover:border-primary/30 hover:bg-primary/[0.02]
              ${disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer active:scale-[0.98]"}
            `}
          >
            <div className="flex items-center gap-3">
              <span className="text-2xl leading-none" aria-hidden="true">
                {exercise.icon}
              </span>
              <div className="flex-1 min-w-0">
                <span className="text-sm font-semibold text-foreground">
                  {exercise.name}
                </span>
                <p className="text-xs text-muted-foreground mt-0.5 line-clamp-2">
                  {exercise.description}
                </p>
              </div>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
