"use client";

import { useEffect, useRef, useCallback } from "react";
import { Exercise } from "../types";
import { useYoloModel } from "../../hooks/useYoloModel";
import { useCamera } from "../../hooks/useCamera";
import { useImageProcessing } from "../../hooks/useImageProcessing";
import { useFps } from "../../hooks/useFps";
import { useExerciseTracker } from "../useExerciseTracker";
import AngleOverlay from "./AngleOverlay";
import RepCounter from "./RepCounter";
import FormFeedback from "./FormFeedback";
import SessionSummary from "./SessionSummary";
import ExercisePicker from "./ExercisePicker";
import { useFullscreen } from "../../hooks/useFullscreen";
import FullscreenButton from "../../components/ui/FullscreenButton";

interface FormCheckCameraProps {
  exercise: Exercise | null;
  onSelectExercise: (exercise: Exercise | null) => void;
}

/**
 * FormCheckCamera — Main camera view for the FormCheck app.
 *
 * Composes useYoloModel, useCamera, useImageProcessing, and useFps
 * to create a full pose-tracking fitness view.
 *
 * Forces the model to yolo11n-pose for pose estimation.
 */
export default function FormCheckCamera({
  exercise,
  onSelectExercise,
}: FormCheckCameraProps) {
  const {
    isModelLoaded,
    workerRef,
    workerReadyRef,
    modelStatus,
    config,
  } = useYoloModel("yolo11n-pose");

  const {
    cameras,
    cameraStream,
    selectedDeviceId,
    setSelectedDeviceId,
    toggleCamera,
    refreshCameras,
  } = useCamera();

  const { fps, tick: fpsTick, reset: fpsReset } = useFps();

  const {
    inferenceTime,
    details,
    overlayRef,
    cameraRef,
    inputCanvasRef,
    processCamera,
    stopCameraProcessing,
    clearOverlay,
  } = useImageProcessing();

  const containerRef = useRef<HTMLDivElement>(null);
  const { isFullscreen, toggleFullscreen } = useFullscreen(containerRef);

  const isTracking = !!cameraStream && isModelLoaded;

  const {
    reps,
    repState,
    currentAngle,
    formFeedback,
    sessionStats,
    reset: resetTracker,
  } = useExerciseTracker(exercise, details, isTracking);

  // Start camera processing when camera loads
  const handleCameraLoad = useCallback(() => {
    fpsReset();
    processCamera(config, workerRef, workerReadyRef, fpsTick);
  }, [fpsReset, processCamera, config, workerRef, workerReadyRef, fpsTick]);

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
              handleCameraLoad();
            })
            .catch(err => {
              console.error("[FormCheckCamera] Error playing video on fullscreen toggle:", err);
              handleCameraLoad();
            });
        }
      }, 150);
      return () => clearTimeout(timer);
    }
  }, [isFullscreen, cameraStream, handleCameraLoad, cameraRef]);

  const handleCameraToggle = () => {
    if (cameraStream) {
      clearOverlay();
      fpsReset();
    }
    toggleCamera();
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopCameraProcessing();
    };
  }, [stopCameraProcessing]);

  // Bind stream to video element
  useEffect(() => {
    if (cameraRef.current) {
      cameraRef.current.srcObject = cameraStream;
    }
  }, [cameraStream, cameraRef]);

  // Poke for camera labels on mount
  useEffect(() => {
    refreshCameras(true);
  }, [refreshCameras]);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-3 lg:gap-4 lg:h-[calc(100vh-6.5rem)]">
      {/* ── Left Sidebar: Branding + Exercise picker + Stats ── */}
      <aside className="lg:col-span-2 space-y-3 order-2 lg:order-1 overflow-y-auto scrollbar-none">
        {/* Branding */}
        <div className="flex items-center gap-2.5 pb-2 border-b border-border/30">
          <div className="w-8 h-8 rounded-lg bg-linear-to-br from-primary/20 to-primary/5 flex items-center justify-center shrink-0">
            <span className="text-base" aria-hidden="true">🏋️</span>
          </div>
          <div className="min-w-0">
            <h1 className="text-sm font-bold text-foreground tracking-tight leading-tight">
              FormCheck
              <span className="text-primary ml-1">AI</span>
            </h1>
            <p className="text-[10px] text-muted-foreground truncate">
              Pose tracking · on-device
            </p>
          </div>
        </div>

        <ExercisePicker
          selectedExercise={exercise}
          onSelect={onSelectExercise}
          disabled={false}
        />

        {exercise && (
          <>
            <RepCounter
              reps={reps}
              repState={repState}
              currentAngle={currentAngle}
              exerciseName={exercise.name}
            />

            <FormFeedback feedback={formFeedback} />

            {sessionStats.totalReps > 0 && (
              <SessionSummary
                stats={sessionStats}
                exerciseName={exercise.name}
                onReset={resetTracker}
              />
            )}
          </>
        )}
      </aside>

      {/* ── Main Camera Area ── */}
      <section className="lg:col-span-10 flex flex-col gap-2 order-1 lg:order-2">
        {/* Inline status — no container */}
        <div className="flex items-center justify-between px-1">
          <div className="flex items-center gap-2">
            <span
              className={`w-2 h-2 rounded-full ${isModelLoaded ? "bg-emerald-500" : "bg-amber-500 animate-pulse"
                }`}
            />
            <span className="text-xs font-medium text-muted-foreground">
              {isModelLoaded ? "yolo11n-pose" : modelStatus}
            </span>
            {isModelLoaded && cameraStream && (
              <>
                <span className="text-border">·</span>
                <span className="text-xs font-mono text-muted-foreground tabular-nums">
                  {inferenceTime}ms
                </span>
                <span className="text-xs font-mono text-emerald-500 tabular-nums">
                  {fps} FPS
                </span>
              </>
            )}
          </div>

          <div className="flex items-center gap-2">
            {/* Camera selector with video icon */}
            {cameras.length > 1 && (
              <div className="flex items-center gap-1.5">
                <svg className="w-3.5 h-3.5 text-muted-foreground shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="m16 13 5.223 3.482a.5.5 0 0 0 .777-.416V7.87a.5.5 0 0 0-.752-.432L16 10.5" />
                  <rect x="2" y="6" width="14" height="12" rx="2" />
                </svg>
                <select
                  value={selectedDeviceId}
                  onChange={(e) => setSelectedDeviceId(e.target.value)}
                  className="text-xs bg-transparent border-none text-muted-foreground cursor-pointer pr-4"
                >
                  {cameras.map((cam) => (
                    <option key={cam.deviceId} value={cam.deviceId}>
                      {cam.label || `Camera ${cam.deviceId.slice(0, 8)}`}
                    </option>
                  ))}
                </select>
              </div>
            )}
          </div>
        </div>

        {/* Camera viewport — uses same pattern as home page MediaArea */}
        <div className="relative flex-1 min-h-135 overflow-hidden bg-transparent flex items-center justify-center">
          {/* Hidden inference canvas */}
          <canvas ref={inputCanvasRef} className="hidden" />

          {/* Loading state */}
          {!isModelLoaded && (
            <div className="absolute inset-0 z-30 flex flex-col items-center justify-center bg-background/80 backdrop-blur-md gap-3 rounded-xl">
              <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin" />
              <span className="text-sm font-medium text-foreground">{modelStatus}</span>
            </div>
          )}

          {/* Camera not started */}
          {isModelLoaded && !cameraStream && (
            <div className="flex flex-col items-center gap-4 p-8 text-center rounded-xl border border-border/40 bg-card">
              <div className="w-16 h-16 rounded-2xl bg-primary/10 flex items-center justify-center">
                <svg className="w-8 h-8 text-primary" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z" />
                  <circle cx="12" cy="13" r="3" />
                </svg>
              </div>
              <div className="space-y-1.5">
                <h3 className="text-lg font-semibold text-foreground">Start Camera</h3>
                <p className="text-sm text-muted-foreground max-w-xs">
                  {exercise
                    ? `Position yourself so your full body is visible for ${exercise.name} tracking.`
                    : "Select an exercise and start your camera to begin tracking."}
                </p>
              </div>
              <button
                onClick={handleCameraToggle}
                disabled={!isModelLoaded}
                className="px-6 py-2.5 text-sm font-semibold text-primary-foreground bg-primary rounded-xl hover:bg-primary/90 active:scale-95 transition-all duration-200 shadow-sm"
              >
                Open Camera
              </button>
            </div>
          )}

          {/* Active camera feed — same inline-block pattern as home page */}
          {cameraStream && (
            <div 
              ref={containerRef}
              className="absolute inset-1 sm:inset-3 md:inset-4 flex justify-center items-center pointer-events-none fullscreen:bg-black fullscreen:inset-0 fullscreen:w-screen fullscreen:h-screen fullscreen:pointer-events-auto fullscreen:z-50"
            >
              <div 
                className="relative inline-block pointer-events-auto leading-none"
              >
                <video
                  ref={cameraRef}
                  className="rounded-lg shadow-sm block"
                  style={isFullscreen ? { maxHeight: '100vh', maxWidth: '100vw' } : { maxHeight: 'calc(100vh - 8rem)', maxWidth: '100%' }}
                  onLoadedData={handleCameraLoad}
                  autoPlay
                  playsInline
                  muted
                />

                {/* Overlay canvas — sized to match video exactly */}
                <canvas
                  ref={overlayRef}
                  className="absolute inset-0 w-full h-full pointer-events-none rounded-lg"
                />

                {/* Angle arcs drawn on top */}
                <AngleOverlay
                  overlayRef={overlayRef}
                  details={details}
                  exercise={exercise}
                />

                {/* Stop button — Inside fullscreen container */}
                <button
                  onClick={() => {
                    if (isFullscreen) toggleFullscreen();
                    handleCameraToggle();
                  }}
                  className="absolute top-2 right-2 z-10 w-8 h-8 flex items-center justify-center rounded-full bg-destructive/25 border border-transparent hover:bg-destructive/20 hover:border-destructive/30 text-destructive/40 hover:text-destructive transition-all duration-300 pointer-events-auto backdrop-blur-sm"
                  aria-label="Stop Camera"
                >
                  <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="18" y1="6" x2="6" y2="18" />
                    <line x1="6" y1="6" x2="18" y2="18" />
                  </svg>
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
      </section>
    </div>
  );
}
