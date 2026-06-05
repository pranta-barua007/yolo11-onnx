"use client";

import { useRef } from "react";
import { X } from "lucide-react";
import { useMediaDisplay } from "./MediaDisplayContext";
import Placeholder from "./Placeholder";
import { useFullscreen } from "../../hooks/useFullscreen";
import FullscreenButton from "../ui/FullscreenButton";

export default function MediaArea() {
  const {
    state: { cameraStream, imgSrc, isModelLoaded, modelStatus },
    actions: { onCameraLoad, onImageLoad, onCameraToggle, onImageToggle },
    meta: { inputCanvasRef, cameraRef, imgRef, overlayRef },
  } = useMediaDisplay();

  const containerRef = useRef<HTMLDivElement>(null);
  const { isFullscreen, toggleFullscreen } = useFullscreen(containerRef);

  const showPlaceholder = !imgSrc && !cameraStream;
  const hasMedia = !!(cameraStream || imgSrc);

  return (
    <div
      className={`relative flex-1 min-h-135 overflow-hidden bg-transparent flex items-center justify-center transition-opacity duration-300 ${!isModelLoaded ? "pointer-events-none opacity-60" : "opacity-100"
        }`}
    >
      {/* Hidden canvas used for inference */}
      <canvas ref={inputCanvasRef} className="hidden" />

      {/* Loading overlay when model not ready */}
      {!isModelLoaded ? (
        <div className="absolute inset-0 z-30 flex flex-col items-center justify-center bg-background/80 backdrop-blur-md gap-3 rounded-b-4xl">
          <div className="w-8 h-8 border-2 border-primary border-t-transparent stroke-primary rounded-full animate-spin shadow-[0_0_15px_rgba(168,85,247,0.5)]" />
          <span className="text-sm font-medium text-foreground tracking-wide">{modelStatus}</span>
        </div>
      ) : null}

      {/* Placeholder / Example Grid */}
      {showPlaceholder ? <Placeholder /> : null}

      {/* Media Wrapper for correct overlay alignment */}
      {hasMedia && (
        <div className="absolute inset-1 sm:inset-3 md:inset-4 flex justify-center items-center pointer-events-none">
          <div 
            ref={containerRef}
            className="relative inline-block pointer-events-auto leading-none fullscreen:flex fullscreen:items-center fullscreen:justify-center fullscreen:bg-black fullscreen:w-screen fullscreen:h-screen"
          >
            {/* Video for camera feed */}
            <video
              ref={cameraRef}
              className={`rounded-lg shadow-sm ${!cameraStream ? 'hidden' : 'block'} fullscreen:max-h-screen fullscreen:max-w-full`}
              style={isFullscreen ? { maxHeight: '100vh', maxWidth: '100%' } : { maxHeight: 'calc(100vh - 13rem)', maxWidth: '100%' }}
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
                className="rounded-lg shadow-sm fullscreen:max-h-screen fullscreen:max-w-full"
                alt="Input"
                style={isFullscreen ? { maxHeight: '100vh', maxWidth: '100%' } : { maxHeight: 'calc(100vh - 13rem)', maxWidth: '100%' }}
              />
            ) : null}

            {/* Overlay canvas */}
            <canvas ref={overlayRef} className="absolute inset-0 w-full h-full pointer-events-none rounded-lg" />

            {/* Close button - Inside fullscreen container */}
            <button
              className="absolute top-2 right-2 z-10 w-8 h-8 flex items-center justify-center rounded-full bg-destructive/25 border border-transparent hover:bg-destructive/20 hover:border-destructive/30 text-destructive/40 hover:text-destructive transition-all duration-300 pointer-events-auto backdrop-blur-sm"
              onClick={() => {
                if (isFullscreen) toggleFullscreen();
                if (cameraStream) onCameraToggle();
                if (imgSrc) onImageToggle();
              }}
              aria-label="Close"
            >
              <X className="w-4 h-4" />
            </button>

            {/* Fullscreen toggle button */}
            <FullscreenButton
              isFullscreen={isFullscreen}
              onClick={toggleFullscreen}
              className="absolute top-2 right-12 z-10"
            />
          </div>
        </div>
      )}
    </div>
  );
}
