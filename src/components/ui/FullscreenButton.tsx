"use client";

import { Maximize2, Minimize2 } from "lucide-react";

interface FullscreenButtonProps {
  isFullscreen: boolean;
  onClick: () => void;
  className?: string;
}

/**
 * A reusable, premium floating button that triggers a fullscreen toggle.
 * Styled with blur backdrop, borders, and smooth hover state transitions.
 */
export default function FullscreenButton({
  isFullscreen,
  onClick,
  className = "",
}: FullscreenButtonProps) {
  return (
    <button
      onClick={onClick}
      className={`
        w-8 h-8 flex items-center justify-center rounded-full 
        bg-background/25 border border-transparent 
        hover:bg-background/40 hover:border-border/30 
        text-foreground/60 hover:text-foreground 
        transition-all duration-300 pointer-events-auto 
        backdrop-blur-sm shadow-sm
        ${className}
      `}
      title={isFullscreen ? "Exit Fullscreen" : "Enter Fullscreen"}
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
