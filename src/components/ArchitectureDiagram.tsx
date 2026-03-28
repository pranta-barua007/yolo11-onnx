import React from "react";

/**
 * A vector-based architecture diagram for the YOLO Edge Runner.
 * Visualises the non-blocking worker architecture and the WebGPU / WASM pipeline.
 *
 * Theme-aware: inherits fill and stroke from CSS custom properties via
 * `currentColor` so it works seamlessly in both light and dark mode.
 */
const ArchitectureDiagram = ({ className }: { className?: string }) => {
  return (
    <svg
      viewBox="0 0 800 450"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      role="img"
      aria-label="System architecture diagram showing the Main Thread (UI) communicating with a Web Worker (Non-Blocking) via Transferable Arrays"
    >
      {/* ── Defs ────────────────────────────────────── */}
      <defs>
        <linearGradient id="mainGrad" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="var(--color-primary, #7c3aed)" stopOpacity="0.06" />
          <stop offset="100%" stopColor="var(--color-primary, #7c3aed)" stopOpacity="0" />
        </linearGradient>
        <linearGradient id="workerGrad" x1="100%" y1="0%" x2="0%" y2="0%">
          <stop offset="0%" stopColor="var(--color-primary, #7c3aed)" stopOpacity="0.06" />
          <stop offset="100%" stopColor="var(--color-primary, #7c3aed)" stopOpacity="0" />
        </linearGradient>
        <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L0,6 L9,3 z" className="fill-muted-foreground/40" />
        </marker>
      </defs>

      {/* ── Thread containers ─────────────────────── */}
      <rect x="50" y="50" width="320" height="350" rx="20"
        fill="url(#mainGrad)"
        className="stroke-primary/30" strokeWidth="1" strokeDasharray="4 4" />
      <rect x="430" y="50" width="320" height="350" rx="20"
        fill="url(#workerGrad)"
        className="stroke-primary/30" strokeWidth="1" strokeDasharray="4 4" />

      {/* ── Labels ────────────────────────────────── */}
      <text x="210" y="80" textAnchor="middle" className="fill-primary" fontSize="13" fontWeight="600" letterSpacing="0.05em">
        MAIN THREAD (UI)
      </text>
      <text x="590" y="80" textAnchor="middle" className="fill-primary" fontSize="13" fontWeight="600" letterSpacing="0.05em">
        WEB WORKER (NON-BLOCKING)
      </text>

      {/* ── UI boxes (left) ───────────────────────── */}
      <rect x="90" y="110" width="240" height="50" rx="12"
        className="fill-card stroke-border" strokeWidth="1" />
      <text x="210" y="140" textAnchor="middle" className="fill-foreground" fontSize="12" fontWeight="500">
        Camera / Image Input
      </text>

      <rect x="90" y="180" width="240" height="50" rx="12"
        className="fill-card stroke-border" strokeWidth="1" />
      <text x="210" y="210" textAnchor="middle" className="fill-foreground" fontSize="12" fontWeight="500">
        Canvas Pre-Processing
      </text>

      <rect x="90" y="310" width="240" height="50" rx="12"
        className="fill-primary" />
      <text x="210" y="340" textAnchor="middle" className="fill-primary-foreground" fontSize="12" fontWeight="700">
        Real-Time Visualization
      </text>

      {/* ── Worker boxes (right) ──────────────────── */}
      <rect x="470" y="110" width="240" height="50" rx="12"
        className="fill-card stroke-border" strokeWidth="1" />
      <text x="590" y="140" textAnchor="middle" className="fill-foreground" fontSize="12" fontWeight="500">
        Inference Pipeline
      </text>

      <rect x="470" y="180" width="240" height="80" rx="12"
        className="fill-card stroke-primary" strokeWidth="1.5" />
      <text x="590" y="210" textAnchor="middle" className="fill-primary" fontSize="13" fontWeight="700">
        ONNX Runtime Web
      </text>
      <circle cx="530" cy="240" r="10" className="fill-amber-500/20" />
      <text x="530" y="244" textAnchor="middle" className="fill-amber-600 dark:fill-amber-400" fontSize="9" fontWeight="700">GPU</text>
      <circle cx="650" cy="240" r="10" className="fill-blue-500/20" />
      <text x="650" y="244" textAnchor="middle" className="fill-blue-600 dark:fill-blue-400" fontSize="9" fontWeight="700">CPU</text>

      <rect x="470" y="280" width="240" height="50" rx="12"
        className="fill-card stroke-border" strokeWidth="1" />
      <text x="590" y="310" textAnchor="middle" className="fill-muted-foreground" fontSize="11" fontWeight="500">
        IndexedDB (Model Cache)
      </text>

      {/* ── Arrows ────────────────────────────────── */}
      <path d="M330 205 H470" className="stroke-muted-foreground/40" strokeWidth="1.5" markerEnd="url(#arrow)" />
      <path d="M470 135 H330" className="stroke-muted-foreground/40" strokeWidth="1.5" markerEnd="url(#arrow)" />

      {/* ── Flow labels ───────────────────────────── */}
      <rect x="345" y="185" width="110" height="20" rx="10" className="fill-muted" />
      <text x="400" y="199" textAnchor="middle" className="fill-muted-foreground" fontSize="9" fontWeight="600">
        Transferable Array
      </text>

      <rect x="345" y="115" width="110" height="20" rx="10" className="fill-muted" />
      <text x="400" y="129" textAnchor="middle" className="fill-muted-foreground" fontSize="9" fontWeight="600">
        Detections / Masks
      </text>
    </svg>
  );
};

export default ArchitectureDiagram;
