"use client";

import { X } from "lucide-react";
import { useMediaDisplay } from "./MediaDisplayContext";
import Placeholder from "./Placeholder";

export default function MediaArea() {
  const {
    state: { cameraStream, imgSrc, isModelLoaded, modelStatus },
    actions: { onCameraLoad, onImageLoad, onCameraToggle, onImageToggle },
    meta: { inputCanvasRef, cameraRef, imgRef, overlayRef },
  } = useMediaDisplay();

  const showPlaceholder = !imgSrc && !cameraStream;
  const hasMedia = !!(cameraStream || imgSrc);

  return (
    <div
      className={`relative flex-1 min-h-[540px] min-h-0 overflow-hidden bg-transparent flex items-center justify-center transition-opacity duration-300 ${!isModelLoaded ? "pointer-events-none opacity-60" : "opacity-100"
        }`}
    >
      {/* Hidden canvas used for inference */}
      <canvas ref={inputCanvasRef} className="hidden" />

      {/* Loading overlay when model not ready */}
      {!isModelLoaded ? (
        <div className="absolute inset-0 z-30 flex flex-col items-center justify-center bg-background/80 backdrop-blur-md gap-3 rounded-b-[20px]">
          <div className="w-8 h-8 border-2 border-primary border-t-transparent stroke-primary rounded-full animate-spin shadow-[0_0_15px_rgba(168,85,247,0.5)]" />
          <span className="text-sm font-medium text-foreground tracking-wide">{modelStatus}</span>
        </div>
      ) : null}

      {/* Placeholder / Example Grid */}
      {showPlaceholder ? <Placeholder /> : null}

      {/* Media Wrapper for correct overlay alignment */}
      {hasMedia && (
        <div className="absolute inset-4 md:inset-6 flex justify-center items-center pointer-events-none overflow-hidden">
          <div className="relative inline-flex justify-center items-center max-w-full max-h-full pointer-events-auto rounded-[1rem]">
            {/* Video for camera feed */}
            <video
              ref={cameraRef}
              className={`max-w-full max-h-full w-auto h-auto rounded-lg shadow-sm ${!cameraStream ? 'hidden' : 'block'}`}
              onLoadedData={onCameraLoad}
              autoPlay
              playsInline
              muted
            />

            {/* Image for static input */}
            {imgSrc ? (
              /* eslint-disable-next-line @next/next/no-img-element */
              <img
                id="img"
                ref={imgRef}
                src={imgSrc}
                onLoad={onImageLoad}
                className="max-w-full max-h-full w-auto h-auto rounded-lg shadow-sm"
                alt="Input"
                style={{ maxHeight: '100%', maxWidth: '100%' }}
              />
            ) : null}

            {/* Overlay canvas */}
            <canvas ref={overlayRef} className="absolute inset-0 w-full h-full pointer-events-none rounded-lg" style={{ maxHeight: '100%', maxWidth: '100%' }} />
            
            {/* Close button - Anchored to the image wrapper */}
            <button
              className="absolute -top-3 -right-3 z-10 w-8 h-8 flex items-center justify-center rounded-full bg-background border border-foreground/20 hover:bg-destructive/20 hover:border-destructive/40 text-muted-foreground hover:text-destructive transition-all duration-300 shadow-xl"
              onClick={() => {
                if (cameraStream) onCameraToggle();
                if (imgSrc) onImageToggle();
              }}
              aria-label="Close"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
