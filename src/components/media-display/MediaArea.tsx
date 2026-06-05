"use client";

import { useRef, useEffect } from "react";
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

  const isFirstRender = useRef(true);

  // Restart camera detection and ensure video plays when toggling fullscreen
  useEffect(() => {
    if (isFirstRender.current) {
      isFirstRender.current = false;
      return;
    }

    if (cameraStream) {
      const timer = setTimeout(() => {
        if (cameraRef.current) {
          cameraRef.current.play()
            .then(() => {
              onCameraLoad();
            })
            .catch(err => {
              console.error("[MediaArea] Error playing video on fullscreen toggle:", err);
              // Still trigger load/processing in case it's just a browser restriction
              onCameraLoad();
            });
        }
      }, 150);
      return () => clearTimeout(timer);
    }
  }, [isFullscreen, cameraStream, onCameraLoad, cameraRef]);

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
        <div 
          ref={containerRef}
          className="absolute inset-1 sm:inset-3 md:inset-4 flex justify-center items-center pointer-events-none fullscreen:bg-black fullscreen:inset-0 fullscreen:w-screen fullscreen:h-screen fullscreen:pointer-events-auto fullscreen:z-50"
        >
          <div 
            className={`relative pointer-events-auto leading-none ${
              isFullscreen && cameraStream ? "w-full h-full" : "inline-block"
            }`}
          >
            {/* Video for camera feed */}
            <video
              ref={cameraRef}
              className={`rounded-lg shadow-sm ${!cameraStream ? 'hidden' : 'block'}`}
              style={isFullscreen ? { height: '100%', width: '100%', objectFit: 'cover' } : { maxHeight: 'calc(100vh - 13rem)', maxWidth: '100%' }}
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
                className="rounded-lg shadow-sm"
                alt="Input"
                style={isFullscreen ? { maxHeight: '100vh', maxWidth: '100vw' } : { maxHeight: 'calc(100vh - 13rem)', maxWidth: '100%' }}
              />
            ) : null}

            {/* Overlay canvas */}
            <canvas 
              ref={overlayRef} 
              className="absolute inset-0 pointer-events-none rounded-lg" 
              style={isFullscreen && cameraStream ? { height: '100%', width: '100%', objectFit: 'cover' } : { width: '100%', height: '100%' }}
            />
          </div>

          {/* Close button - Anchored to the parent container */}
          <button
            className="absolute top-0 right-0 z-10 w-8 h-8 flex items-center justify-center rounded-full bg-destructive/25 border border-transparent hover:bg-destructive/20 hover:border-destructive/30 text-destructive/40 hover:text-destructive transition-all duration-300 pointer-events-auto backdrop-blur-sm md:top-2 md:right-2"
            onClick={() => {
              if (isFullscreen) toggleFullscreen();
              if (cameraStream) onCameraToggle();
              if (imgSrc) onImageToggle();
            }}
            aria-label="Close"
          >
            <X className="w-4 h-4" />
          </button>

          {/* Fullscreen Toggle Button */}
          <FullscreenButton
            isFullscreen={isFullscreen}
            onClick={toggleFullscreen}
            className="absolute top-0 right-10 md:top-2 md:right-12 z-10"
          />
        </div>
      )}
    </div>
  );
}
