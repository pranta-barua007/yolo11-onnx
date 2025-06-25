"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { InferenceSession } from "onnxruntime-web";

import { model_loader } from "../utils/model_loader";
import { inference_pipeline } from "../utils/inference_pipeline";
import { draw_bounding_boxes } from "../utils/draw_bounding_boxes";

import { Box } from "../utils/types";
import classes from "../utils/yolo_classes.json";
import "../styles/styles.css";

interface CustomModel {
  name: string;
  url: string;
}

const input_shape = [1, 3, 640, 640];
const iou_threshold = 0.35;
const score_threshold = 0.45;
const config = { input_shape, iou_threshold, score_threshold };

function Home() {
  // Refs and states
  const deviceRef = useRef<HTMLSelectElement>(null);
  const modelRef = useRef<HTMLSelectElement>(null);
  const [customModels, setCustomModels] = useState<CustomModel[]>([]);
  const [cameras, setCameras] = useState<MediaDeviceInfo[]>([]);
  const cameraSelectorRef = useRef<HTMLSelectElement>(null);
  const [cameraStream, setCameraStream] = useState<MediaStream | null>(null);
  const sessionRef = useRef<InferenceSession>(null); // ONNX session

  // Media refs
  const imgRef = useRef<HTMLImageElement>(null);
  const [imgSrc, setImgSrc] = useState<string | null>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const cameraRef = useRef<HTMLVideoElement>(null);
  const inputCanvasRef = useRef<HTMLCanvasElement>(null);

  // Model status and inference info
  const [warmUpTime, setWarmUpTime] = useState<string>("0");
  const [inferenceTime, setInferenceTime] = useState<string>("0");
  const modelStatusRef = useRef<HTMLParagraphElement>(null);
  const [details, setDetails] = useState<Box[]>([]);
  const openImageRef = useRef<HTMLInputElement>(null);
  const [isModelLoaded, setIsModelLoaded] = useState<boolean>(false);

  useEffect(() => {
    loadModel();
    getCameras();
  }, []);

  const loadModel = useCallback(async () => {
    if (!modelStatusRef.current || !deviceRef.current || !modelRef.current) return;
    const modelStatusEl = modelStatusRef.current;
    modelStatusEl.textContent = "Loading model...";
    modelStatusEl.style.color = "red";
    setIsModelLoaded(false);

    const device = deviceRef.current.value;
    const selectedModel = modelRef.current.value;
    const customModel = customModels.find((model) => model.url === selectedModel);
    const model_path = customModel
      ? customModel.url
      : `/models/${selectedModel}.onnx`;

    try {
      const start = performance.now();
      const yolo_model = await model_loader(device, model_path, config);
      const end = performance.now();
      sessionRef.current = yolo_model;

      modelStatusEl.textContent = "Model loaded";
      modelStatusEl.style.color = "green";
      setWarmUpTime((end - start).toFixed(2));
      setIsModelLoaded(true);
    } catch (error) {
      if (modelStatusEl) {
        modelStatusEl.textContent = "Model loading failed";
        modelStatusEl.style.color = "red";
      }
      console.error(error);
    }
  }, [customModels]);

  const handle_AddModel = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const fileName = file.name.replace(".onnx", "");
      setCustomModels((prevModels) => [
        ...prevModels,
        { name: fileName, url: URL.createObjectURL(file) },
      ]);
    }
  }, []);

  const handle_OpenImage = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setImgSrc(URL.createObjectURL(file));
      if (openImageRef.current) openImageRef.current.disabled = true;
      event.target.value = "";
    }
  }, []);

  const handle_ImageLoad = useCallback(async () => {
    if (!imgRef.current || !overlayRef.current || !sessionRef.current) return;
    overlayRef.current.width = imgRef.current.width;
    overlayRef.current.height = imgRef.current.height;
    const [results, resultsInferenceTime] = await inference_pipeline(
      imgRef.current,
      sessionRef.current,
      config,
      overlayRef.current
    );
    setDetails(results);
    setInferenceTime(resultsInferenceTime);
    await draw_bounding_boxes(results, overlayRef.current);
  }, []);

  const getCameras = useCallback(async () => {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter((device) => device.kind === "videoinput");
      setCameras(videoDevices);
    } catch (error) {
      console.error("Error getting cameras:", error);
    }
  }, []);

  const handle_ToggleCamera = useCallback(async () => {
    if (cameraStream) {
      cameraStream.getTracks().forEach((track) => track.stop());
      if (cameraRef.current) cameraRef.current.srcObject = null;
      setCameraStream(null);
      if (overlayRef.current) {
        overlayRef.current.width = 0;
        overlayRef.current.height = 0;
      }
    } else {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { deviceId: cameraSelectorRef.current?.value },
          audio: false,
        });
        setCameraStream(stream);
        if (cameraRef.current) cameraRef.current.srcObject = stream;
      } catch (error) {
        console.error("Error toggling camera:", error);
      }
    }
  }, [cameraStream]);

  const handle_cameraLoad = useCallback(() => {
    if (!cameraRef.current || !inputCanvasRef.current || !overlayRef.current) return;
    const inputCtx = inputCanvasRef.current.getContext("2d", { willReadFrequently: true });
    if (!inputCtx) return;
    inputCtx.canvas.width = cameraRef.current.videoWidth;
    inputCtx.canvas.height = cameraRef.current.videoHeight;
    overlayRef.current.width = cameraRef.current.videoWidth;
    overlayRef.current.height = cameraRef.current.videoHeight;
    handle_frame_continuous(inputCtx);
  }, [sessionRef.current]);

  const handle_frame_continuous = useCallback(
    async (ctx: CanvasRenderingContext2D) => {
      if (!cameraRef.current || !cameraRef.current.srcObject) return;
      ctx.drawImage(
        cameraRef.current,
        0,
        0,
        cameraRef.current.videoWidth,
        cameraRef.current.videoHeight
      );
      if (inputCanvasRef.current && overlayRef.current && sessionRef.current) {
        const [results, resultsInferenceTime] = await inference_pipeline(
          inputCanvasRef.current,
          sessionRef.current,
          config,
          overlayRef.current
        );
        setDetails(results);
        setInferenceTime(resultsInferenceTime);
        await draw_bounding_boxes(results, overlayRef.current);
      }
      requestAnimationFrame(() => handle_frame_continuous(ctx));
    },
    []
  );

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-4 space-y-8">
      <h1 className="text-4xl font-bold">Yolo Segmentation</h1>

      {/* Settings Container */}
      <div id="setting-container" className="container w-full max-w-3xl flex flex-wrap gap-4 justify-center">
        <div className="flex flex-col">
          <label htmlFor="device-selector">Backend:</label>
          <select name="device-selector" ref={deviceRef} onChange={loadModel}>
            <option value="webgpu">webGPU</option>
            <option value="wasm">Wasm(cpu)</option>
          </select>
        </div>
        <div className="flex flex-col">
          <label htmlFor="model-selector">Model:</label>
          <select name="model-selector" ref={modelRef} onChange={loadModel}>
            <option value="yolo11n-seg">yolo11n-2.6M</option>
            <option value="yolo11s-seg">yolo11s-9.4M</option>
            {customModels.map((model, index) => (
              <option key={index} value={model.url}>
                {model.name}
              </option>
            ))}
          </select>
        </div>
        <div className="flex flex-col">
          <label htmlFor="camera-selector">Select Camera:</label>
          <select ref={cameraSelectorRef}>
            {cameras.map((camera, index) => (
              <option key={index} value={camera.deviceId}>
                {camera.label || `Camera ${index + 1}`}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Media Container */}
      <div className="relative w-full max-w-[720px] border border-slate-500 rounded-lg overflow-hidden">
        {/* Hidden canvas used for inference */}
        <canvas ref={inputCanvasRef} className="hidden" />
        {/* Video for camera feed */}
        <video
          ref={cameraRef}
          className="w-full rounded-lg"
          onLoadedData={handle_cameraLoad}
          autoPlay
          playsInline
          muted
          hidden={!cameraStream}
        />
        {/* Image for static input */}
        {imgSrc && (
          <img
            id="img"
            ref={imgRef}
            src={imgSrc}
            onLoad={handle_ImageLoad}
            className="w-full rounded-lg"
            alt="Input"
          />
        )}
        {/* Overlay canvas */}
        <canvas ref={overlayRef} className="absolute top-0 left-0 w-full h-full pointer-events-none" />
      </div>

      {/* Controls */}
      <div id="btn-container" className="container w-full max-w-3xl flex flex-wrap gap-4 justify-center">
        <button
          className="btn"
          disabled={!!cameraStream || !isModelLoaded}
          onClick={() => {
            if (!imgSrc) {
              openImageRef.current?.click();
            } else {
              setImgSrc(null);
              if (openImageRef.current) openImageRef.current.disabled = false;
              if (overlayRef.current) {
                overlayRef.current.width = 0;
                overlayRef.current.height = 0;
              }
            }
          }}
        >
          {imgSrc ? "Close Image" : "Open Image"}
          <input
            type="file"
            accept="image/*"
            hidden
            ref={openImageRef}
            onChange={handle_OpenImage}
          />
        </button>
        <button
          className="btn"
          onClick={handle_ToggleCamera}
          disabled={cameras.length === 0 || !!imgSrc || !isModelLoaded}
        >
          {cameraStream ? "Close Camera" : "Open Camera"}
        </button>
        <label className="btn cursor-pointer">
          <input type="file" accept=".onnx" onChange={handle_AddModel} hidden />
          <span>Add model</span>
        </label>
      </div>

      {/* Model Status and Details */}
      <div id="model-status-container" className="container w-full max-w-3xl text-center">
        <div id="inferenct-time-container" className="flex flex-col sm:flex-row justify-evenly text-xl my-6">
          <p>
            Warm up time: <span className="text-lime-500">{warmUpTime}ms</span>
          </p>
          <p>
            Inference time: <span className="text-lime-500">{inferenceTime}ms</span>
          </p>
        </div>
        <p ref={modelStatusRef} className={isModelLoaded ? "" : "animate-text-loading"}>
          Model not loaded
        </p>
        <details className="text-gray-200 group mt-4" open>
          <summary className="my-5 hover:text-gray-400 cursor-pointer transition-colors duration-300">
            Detected objects
          </summary>
          <div className="transition-all duration-300 ease-in-out transform origin-top group-open:animate-details-show">
            <table className="w-full text-left border-collapse table-auto text-sm bg-gray-800 rounded-md overflow-hidden">
              <thead className="bg-gray-700">
                <tr>
                  <th className="border-b border-gray-600 p-4 text-gray-100">#</th>
                  <th className="border-b border-gray-600 p-4 text-gray-100">Class</th>
                  <th className="border-b border-gray-600 p-4 text-gray-100">Confidence</th>
                </tr>
              </thead>
              <tbody>
                {details.map((item, index) => (
                  <tr key={index} className="hover:bg-gray-700 transition-colors text-gray-300">
                    <td className="border-b border-gray-600 p-4">{index + 1}</td>
                    <td className="border-b border-gray-600 p-4">{classes[item.class_idx]}</td>
                    <td className="border-b border-gray-600 p-4">{(item.score * 100).toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      </div>
    </div>
  );
}

export default Home;