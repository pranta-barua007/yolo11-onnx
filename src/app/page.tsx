"use client";

import { useState, useEffect, useCallback } from "react";
import MediaDisplay from "../components/MediaDisplay";
import ModelStatus from "../components/ModelStatus";
import { useYoloModel } from "../hooks/useYoloModel";
import { useCamera } from "../hooks/useCamera";
import { useImageProcessing } from "../hooks/useImageProcessing";
import { useFps } from "../hooks/useFps";
import "../styles/styles.css";

export default function Home() {
  const {
    customModels,
    isModelLoaded,
    warmUpTime,
    workerRef,
    workerReadyRef,
    modelStatus,
    device,
    setDevice,
    modelName,
    setModelName,
    config,
    addCustomModel,
    activeClasses,
    scoreThreshold,
    setScoreThreshold,
    reloadModel,
    removeCustomModel,
  } = useYoloModel();

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
    imgSrc,
    inferenceTime,
    details,
    imgRef,
    overlayRef,
    cameraRef,
    inputCanvasRef,
    openImageRef,
    openImage,
    processImage,
    processCamera,
    stopCameraProcessing,
    redrawOverlay,
    saveResult,
    toggleImage,
    clearOverlay,
    setImgSrc,
  } = useImageProcessing();

  const [selectedDetectionIdx, setSelectedDetectionIdx] = useState<number | null>(null);
  const [selectedKeypointIdx, setSelectedKeypointIdx] = useState<number | null>(null);

  const handleImageLoad = useCallback(() => {
    fpsReset();
    setSelectedDetectionIdx(null);
    processImage(config, workerRef, workerReadyRef);
  }, [fpsReset, processImage, config, workerRef, workerReadyRef]);

  const handleCameraLoad = useCallback(() => {
    fpsReset();
    setSelectedDetectionIdx(null);
    processCamera(config, workerRef, workerReadyRef, fpsTick);
  }, [fpsReset, processCamera, config, workerRef, workerReadyRef, fpsTick]);

  const handleCameraToggle = useCallback(() => {
    if (cameraStream) {
      clearOverlay();
      fpsReset();
    }
    toggleCamera();
  }, [cameraStream, clearOverlay, fpsReset, toggleCamera]);

  const handleSelectDetection = (idx: number | null) => {
    setSelectedDetectionIdx(idx);
    setSelectedKeypointIdx(null); // Reset keypoint when changing detection
    redrawOverlay(details, idx, null, activeClasses);
  };

  const handleSelectKeypoint = (kpIdx: number | null) => {
    setSelectedKeypointIdx(kpIdx);
    redrawOverlay(details, selectedDetectionIdx, kpIdx, activeClasses);
  };

  // Cleanup camera stream and processing when component unmounts
  useEffect(() => {
    return () => {
      stopCameraProcessing();
      if (cameraStream) {
        cameraStream.getTracks().forEach((track) => track.stop());
      }
    };
  }, [cameraStream, stopCameraProcessing]);

  // Set camera stream to video element
  useEffect(() => {
    if (cameraRef.current) {
      cameraRef.current.srcObject = cameraStream;
    }
  }, [cameraStream, cameraRef]);

  return (
    <div className="min-h-screen text-foreground font-sans selection:bg-primary/30 selection:text-primary">

      <main className="mx-auto p-2 sm:p-3 md:p-4 lg:p-6 grid grid-cols-1 lg:grid-cols-12 gap-3 sm:gap-4 lg:gap-6">

        {/* Sidebar Status (Detections) */}
        <aside className="lg:col-span-4 xl:col-span-3 space-y-6 flex flex-col order-2 lg:order-1 lg:h-[calc(100vh-120px)] lg:min-h-170 h-auto sticky top-24">
          <ModelStatus
            details={details}
            selectedDetectionIdx={selectedDetectionIdx}
            onSelectDetection={handleSelectDetection}
            selectedKeypointIdx={selectedKeypointIdx}
            onSelectKeypoint={handleSelectKeypoint}
            onSave={saveResult}
            classes={activeClasses}
          />
        </aside>

        {/* Main Display Area */}
        <section className="lg:col-span-8 xl:col-span-9 flex flex-col gap-6 order-1 lg:order-2 lg:h-[calc(100vh-120px)] lg:min-h-170 lg:sticky top-24">
          <MediaDisplay
            inputCanvasRef={inputCanvasRef}
            cameraRef={cameraRef}
            imgRef={imgRef}
            overlayRef={overlayRef}
            cameraStream={cameraStream}
            imgSrc={imgSrc}
            onCameraLoad={handleCameraLoad}
            onImageLoad={handleImageLoad}
            onImageSelect={setImgSrc}
            onCameraToggle={handleCameraToggle}
            onImageToggle={toggleImage}
            openImageRef={openImageRef}
            onOpenImage={openImage}
            modelName={modelName}
            setModelName={setModelName}
            device={device}
            setDevice={setDevice}
            isModelLoaded={isModelLoaded}
            modelStatus={modelStatus}
            warmUpTime={warmUpTime}
            inferenceTime={inferenceTime}
            fps={fps}
            scoreThreshold={scoreThreshold}
            setScoreThreshold={setScoreThreshold}
            cameras={cameras}
            selectedDeviceId={selectedDeviceId}
            setSelectedDeviceId={setSelectedDeviceId}
            refreshCameras={refreshCameras}
            customModels={customModels}
            addCustomModel={addCustomModel}
            removeCustomModel={removeCustomModel}
            reloadModel={reloadModel}
          />
        </section>

      </main>
    </div>
  );
}