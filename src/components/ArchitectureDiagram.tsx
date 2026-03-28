import React from "react";

/**
 * A premium, vector-based architecture diagram for the YOLO Edge Runner.
 * Visualizes the non-blocking worker architecture and the WebGPU/WASM pipeline.
 */
const ArchitectureDiagram = ({ className }: { className?: string }) => {
  return (
    <svg
      viewBox="0 0 800 450"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
    >
      {/* Background Gradients */}
      <defs>
        <linearGradient id="mainGrad" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#0d9488" stopOpacity="0.05" />
          <stop offset="100%" stopColor="#0d9488" stopOpacity="0" />
        </linearGradient>
        <linearGradient id="workerGrad" x1="100%" y1="0%" x2="0%" y2="0%">
          <stop offset="0%" stopColor="#7c3aed" stopOpacity="0.05" />
          <stop offset="100%" stopColor="#7c3aed" stopOpacity="0" />
        </linearGradient>
        <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
          <feDropShadow dx="0" dy="4" stdDeviation="15" floodOpacity="0.08" />
        </filter>
      </defs>

      {/* Thread Containers */}
      <rect x="50" y="50" width="320" height="350" rx="20" fill="url(#mainGrad)" stroke="#0d9488" strokeWidth="1" strokeDasharray="4 4" />
      <rect x="430" y="50" width="320" height="350" rx="20" fill="url(#workerGrad)" stroke="#7c3aed" strokeWidth="1" strokeDasharray="4 4" />

      {/* Labels */}
      <text x="210" y="80" textAnchor="middle" fill="#0d9488" fontSize="14" fontWeight="bold" letterSpacing="0.05em">MAIN THREAD (UI)</text>
      <text x="590" y="80" textAnchor="middle" fill="#7c3aed" fontSize="14" fontWeight="bold" letterSpacing="0.05em">WEB WORKER (NON-BLOCKING)</text>

      {/* UI Components */}
      <g filter="url(#shadow)">
        <rect x="90" y="110" width="240" height="50" rx="10" fill="white" stroke="#e2e8f0" />
        <text x="210" y="140" textAnchor="middle" fill="#1e293b" fontSize="12" fontWeight="600">Camera / Image Input</text>
        
        <rect x="90" y="180" width="240" height="50" rx="10" fill="white" stroke="#e2e8f0" />
        <text x="210" y="210" textAnchor="middle" fill="#1e293b" fontSize="12" fontWeight="600">Canvas Pre-processing</text>
        
        <rect x="90" y="310" width="240" height="50" rx="10" fill="#0d9488" />
        <text x="210" y="340" textAnchor="middle" fill="white" fontSize="12" fontWeight="bold">Real-time Visualization</text>
      </g>

      {/* Worker Components */}
      <g filter="url(#shadow)">
        <rect x="470" y="110" width="240" height="50" rx="10" fill="white" stroke="#e2e8f0" />
        <text x="590" y="140" textAnchor="middle" fill="#1e293b" fontSize="12" fontWeight="600">Inference Pipeline</text>
        
        <rect x="470" y="180" width="240" height="80" rx="10" fill="white" stroke="#7c3aed" strokeWidth="2" />
        <text x="590" y="210" textAnchor="middle" fill="#7c3aed" fontSize="13" fontWeight="bold">ONNX Runtime Web</text>
        <circle cx="530" cy="240" r="10" fill="#f59e0b" fillOpacity="0.2" />
        <text x="530" y="244" textAnchor="middle" fill="#b45309" fontSize="9" fontWeight="bold">GPU</text>
        <circle cx="650" cy="240" r="10" fill="#3b82f6" fillOpacity="0.2" />
        <text x="650" y="244" textAnchor="middle" fill="#1d4ed8" fontSize="9" fontWeight="bold">CPU</text>
        
        <rect x="470" y="280" width="240" height="50" rx="10" fill="white" stroke="#e2e8f0" />
        <text x="590" y="310" textAnchor="middle" fill="#1e293b" fontSize="11" fontWeight="500">IndexedDB (Model Cache)</text>
      </g>

      {/* Communication Arrows */}
      <path d="M330 205 H470" stroke="#cbd5e1" strokeWidth="2" markerEnd="url(#arrow)" />
      <path d="M470 135 H330" stroke="#cbd5e1" strokeWidth="2" markerEnd="url(#arrow)" />
      
      {/* Arrow Marker Definitions */}
      <defs>
        <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L0,6 L9,3 z" fill="#cbd5e1" />
        </marker>
      </defs>

      {/* Flow Labels */}
      <rect x="345" y="185" width="110" height="20" rx="10" fill="#f1f5f9" />
      <text x="400" y="199" textAnchor="middle" fill="#64748b" fontSize="9" fontWeight="bold">Transferable Array</text>
      
      <rect x="345" y="115" width="110" height="20" rx="10" fill="#f1f5f9" />
      <text x="400" y="129" textAnchor="middle" fill="#64748b" fontSize="9" fontWeight="bold">Detections / Masks</text>
    </svg>
  );
};

export default ArchitectureDiagram;
