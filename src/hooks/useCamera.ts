"use client";

import { useEffect, useState, useCallback } from "react";

export function useCamera() {
  const [cameras, setCameras] = useState<MediaDeviceInfo[]>([]);
  const [cameraStream, setCameraStream] = useState<MediaStream | null>(null);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>(() => {
    if (typeof window !== "undefined") {
      return localStorage.getItem("selectedDeviceId") || "";
    }
    return "";
  });

  useEffect(() => {
    if (selectedDeviceId) {
      localStorage.setItem("selectedDeviceId", selectedDeviceId);
    }
  }, [selectedDeviceId]);

  const stopCamera = useCallback(() => {
    if (cameraStream) {
      cameraStream.getTracks().forEach((track) => track.stop());
      setCameraStream(null);
    }
  }, [cameraStream]);

  const toggleCamera = async () => {
    if (cameraStream) {
      stopCamera();
    } else {
      try {
        const constraints: MediaStreamConstraints = {
          video: selectedDeviceId ? { deviceId: { exact: selectedDeviceId } } : true,
          audio: false,
        };

        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        setCameraStream(stream);
      } catch (error) {
        console.error("Error toggling camera:", error);
      }
    }
  };

  const refreshCameras = useCallback(async (forcePoke = false) => {
    try {
      let devices = await navigator.mediaDevices.enumerateDevices();
      let videoDevices = devices.filter((device) => device.kind === "videoinput");

      // Only poke if requested AND labels are missing
      const needsPoke = forcePoke && (videoDevices.length === 0 || videoDevices.every(d => !d.label));

      if (needsPoke && typeof navigator !== "undefined" && navigator.mediaDevices.getUserMedia) {
        try {
          const stream = await navigator.mediaDevices.getUserMedia({ video: true });
          stream.getTracks().forEach(track => track.stop());
          devices = await navigator.mediaDevices.enumerateDevices();
          videoDevices = devices.filter((device) => device.kind === "videoinput");
        } catch (err) {
          console.warn("[useCamera] Permission denied or no camera found:", err);
        }
      }

      setCameras(videoDevices);
      if (videoDevices.length > 0 && !selectedDeviceId) {
        setSelectedDeviceId(videoDevices[0].deviceId);
      }
    } catch (error) {
      console.error("Error refreshing cameras:", error);
    }
  }, [selectedDeviceId]);

  useEffect(() => {
    // Initial scan on mount (no poke, just see what's there)
    // Wrapped in Promise to avoid cascading render warning in some lint environments
    Promise.resolve().then(() => refreshCameras(false));

    // Listen for hardware changes
    const handleChange = () => refreshCameras(false);
    navigator.mediaDevices.addEventListener("devicechange", handleChange);
    return () => navigator.mediaDevices.removeEventListener("devicechange", handleChange);
  }, [refreshCameras]);

  return {
    cameras,
    cameraStream,
    selectedDeviceId,
    setSelectedDeviceId,
    toggleCamera,
    stopCamera,
    refreshCameras,
  };
}