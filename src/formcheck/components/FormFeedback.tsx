"use client";

import { FormFeedbackData } from "../types";

interface FormFeedbackProps {
  feedback: FormFeedbackData;
}

/**
 * FormFeedback — Real-time form quality banner that changes color and message.
 */
export default function FormFeedback({ feedback }: FormFeedbackProps) {
  const colorMap = {
    good: {
      bg: "bg-emerald-500/10",
      border: "border-emerald-500/30",
      text: "text-emerald-500",
      dot: "bg-emerald-500",
      label: "Good Form",
    },
    warning: {
      bg: "bg-amber-500/10",
      border: "border-amber-500/30",
      text: "text-amber-500",
      dot: "bg-amber-500",
      label: "Watch It",
    },
    bad: {
      bg: "bg-red-500/10",
      border: "border-red-500/30",
      text: "text-red-500",
      dot: "bg-red-500",
      label: "Fix Form",
    },
  };

  const style = colorMap[feedback.quality];

  return (
    <div
      className={`
        flex items-center gap-3 px-4 py-3 rounded-xl border transition-all duration-300
        ${style.bg} ${style.border}
      `}
    >
      {/* Animated dot */}
      <span className="relative flex h-2.5 w-2.5 flex-shrink-0">
        <span
          className={`absolute inline-flex h-full w-full rounded-full opacity-75 animate-ping ${style.dot}`}
        />
        <span
          className={`relative inline-flex h-2.5 w-2.5 rounded-full ${style.dot}`}
        />
      </span>

      <div className="flex-1 min-w-0">
        <span className={`text-[10px] font-bold uppercase tracking-widest ${style.text}`}>
          {style.label}
        </span>
        <p className="text-sm text-foreground mt-0.5 line-clamp-2">
          {feedback.message}
        </p>
      </div>
    </div>
  );
}
