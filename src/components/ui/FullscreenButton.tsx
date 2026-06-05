"use client";

import { Maximize2, Minimize2 } from "lucide-react";

interface FullscreenButtonProps {
  isFullscreen: boolean;
  onClick: () => void;
  className?: string;
}

export default function FullscreenButton({
  isFullscreen,
  onClick,
  className = "",
}: FullscreenButtonProps) {
  return (
    <button
      onClick={onClick}
      className={`w-8 h-8 flex items-center justify-center rounded-full bg-background/25 border border-transparent hover:bg-background/20 hover:border-foreground/10 text-foreground/40 hover:text-foreground transition-all duration-300 pointer-events-auto backdrop-blur-sm ${className}`}
      aria-label={isFullscreen ? "Exit Fullscreen" : "Enter Fullscreen"}
    >
      {isFullscreen ? (
        <Minimize2 className="w-4 h-4" />
      ) : (
        <Maximize2 className="w-4 h-4" />
      )}
    </button>
  );
}
