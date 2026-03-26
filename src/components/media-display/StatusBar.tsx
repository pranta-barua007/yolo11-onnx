"use client";

import { useSyncExternalStore } from "react";
import { useState } from "react";
import { Cpu, Package, Video, ChevronDown, ChevronUp, Sliders, Gauge, RefreshCw } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useMediaDisplay } from "./MediaDisplayContext";
import AddModelDialog from "../AddModelDialog";

/** SSR-safe mount detection without useEffect + setState (fixes react-hooks/set-state-in-effect) */
const emptySubscribe = () => () => { };
function useIsMounted() {
  return useSyncExternalStore(
    emptySubscribe,
    () => true,  // client: always mounted
    () => false  // server: never mounted
  );
}

function DeviceSelect() {
  const { state: { device, isModelLoaded }, actions: { setDevice } } = useMediaDisplay();
  return (
    <div className="flex items-center gap-2">
      <Cpu className="w-3.5 h-3.5 text-slate-400" />
      <Select value={device} onValueChange={setDevice} defaultValue="webgpu">
        <SelectTrigger className="h-8 w-full md:w-[120px] text-[11px] font-bold uppercase tracking-wider bg-slate-50 border-slate-200 rounded-full">
          <div className="flex items-center gap-2">
            <div
              className={`w-2 h-2 rounded-full transition-all duration-500 ${isModelLoaded
                ? "bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.7)]"
                : "bg-amber-400 animate-pulse"
                }`}
            />
            <SelectValue placeholder="Select Device" />
          </div>
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="webgpu" className="text-xs uppercase">WebGPU</SelectItem>
          <SelectItem value="wasm" className="text-xs uppercase">WASM</SelectItem>
        </SelectContent>
      </Select>
    </div>
  );
}

function ModelSelect() {
  const {
    state: { modelName, customModels },
    actions: { setModelName, addCustomModel },
  } = useMediaDisplay();

  const [dialogOpen, setDialogOpen] = useState(false);

  return (
    <div className="flex items-center gap-2">
      <Package className="w-3.5 h-3.5 text-slate-400" />
      <Select value={modelName} onValueChange={setModelName} defaultValue="yolo11n-seg">
        <SelectTrigger className="h-8 w-full md:min-w-[140px] text-[11px] font-bold text-slate-700 bg-slate-50 border-slate-200 font-mono rounded-full">
          <SelectValue placeholder="Select Model" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="yolo11n-seg" className="text-xs font-mono">High Speed</SelectItem>
          <SelectItem value="yolo11s-seg" className="text-xs font-mono">High Accuracy</SelectItem>
          {customModels.filter(model => model.url).map((model) => (
            <SelectItem key={model.url} value={model.url} className="text-xs font-mono">
              {model.name}
            </SelectItem>
          ))}
          <div className="p-2 border-t border-slate-100 mt-1">
            <button
              onClick={(e) => {
                e.preventDefault();
                e.stopPropagation();
                setDialogOpen(true);
              }}
              className="w-full text-[10px] font-bold uppercase tracking-wider py-1.5 px-2 bg-slate-50 hover:bg-slate-100 text-slate-600 rounded transition-colors"
            >
              + Add Custom Model
            </button>
          </div>
        </SelectContent>
      </Select>

      <AddModelDialog
        open={dialogOpen}
        onOpenChange={setDialogOpen}
        onAddModel={addCustomModel}
      />
    </div>
  );
}

function CameraSelect() {
  const {
    state: { cameras, selectedDeviceId },
    actions: { setSelectedDeviceId },
  } = useMediaDisplay();

  return (
    <div className="flex items-center gap-2">
      <Video className="w-3.5 h-3.5 text-slate-400" />
      <Select value={selectedDeviceId} onValueChange={setSelectedDeviceId}>
        <SelectTrigger className="h-8 w-full md:w-[160px] text-[11px] font-medium text-slate-700 bg-slate-50 border-slate-200 rounded-full">
          <SelectValue placeholder="Select Camera" />
        </SelectTrigger>
        <SelectContent>
          {cameras.filter(cam => cam.deviceId).map((cam) => (
            <SelectItem key={cam.deviceId} value={cam.deviceId} className="text-xs">
              {cam.label || `Camera ${cam.deviceId.slice(0, 5)}`}
            </SelectItem>
          ))}
          {cameras.filter(cam => cam.deviceId).length === 0 && (
            <SelectItem value="no-camera" disabled className="text-xs">No cameras found</SelectItem>
          )}
        </SelectContent>
      </Select>
    </div>
  );
}

function ConfidenceSlider() {
  const {
    state: { scoreThreshold, cameraStream, imgSrc },
    actions: { setScoreThreshold },
  } = useMediaDisplay();

  const isInferring = !!cameraStream || !!imgSrc;

  return (
    <div className={`flex items-center gap-2 px-3 h-8 bg-slate-50 border border-slate-200 rounded-full transition-opacity ${isInferring ? "opacity-50 pointer-events-none" : ""}`}>
      <Gauge className="w-3.5 h-3.5 text-slate-400" />
      <span className="text-[11px] font-bold uppercase tracking-wider text-slate-500">Conf.</span>
      <input
        type="range"
        min={0.05}
        max={0.95}
        step={0.05}
        value={scoreThreshold}
        onChange={(e) => setScoreThreshold(parseFloat(e.target.value))}
        disabled={isInferring}
        className="w-20 h-1 accent-teal-500 cursor-pointer disabled:cursor-not-allowed"
      />
      <span className="text-[11px] font-bold text-teal-600 w-7 text-right tabular-nums">
        {(scoreThreshold * 100).toFixed(0)}%
      </span>
    </div>
  );
}

function PerformanceMetrics() {
  const {
    state: { isModelLoaded, modelStatus, inferenceTime, warmUpTime, fps, cameraStream },
    actions: { reloadModel },
  } = useMediaDisplay();

  return (
    <div className="flex items-center justify-between sm:justify-end gap-3 px-1 sm:px-0">
      {!isModelLoaded ? (
        <div className="flex items-center gap-2 text-amber-600">
          <div className="w-3 h-3 border-2 border-current border-t-transparent rounded-full animate-spin" />
          <span className="text-[11px] font-medium">{modelStatus}</span>
        </div>
      ) : (
        <div className="flex items-center gap-2 sm:gap-3 w-full sm:w-auto">
          {fps > 0 && (
            <>
              <div className="flex items-center gap-1 text-[11px] whitespace-nowrap">
                <span className="font-medium text-slate-400">FPS</span>
                <span className="font-bold text-indigo-600">{fps}</span>
              </div>
              <div className="w-px h-3 bg-slate-200" />
            </>
          )}
          <div className="flex items-center gap-1 text-[11px] whitespace-nowrap">
            <span className="font-medium text-slate-400">Inf.</span>
            <span className="font-bold text-teal-600">{inferenceTime}ms</span>
          </div>
          <div className="w-px h-3 bg-slate-200" />
          <div className="flex items-center gap-1 text-[11px] whitespace-nowrap">
            <span className="font-medium text-slate-400 hidden sm:inline">Warmup</span>
            <span className="font-medium text-slate-400 sm:hidden" title="Warmup">W.</span>
            <span className="font-bold text-teal-600">{warmUpTime}ms</span>
          </div>
          
          <div className="ml-auto flex items-center gap-2">
            <span className="text-[11px] font-semibold text-emerald-600 flex items-center gap-1 whitespace-nowrap">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 inline-block" />
              <span className="hidden sm:inline">Ready</span>
            </span>
            <button
              onClick={reloadModel}
              disabled={!!cameraStream}
              title="Re-download model (clear cache)"
              className="p-1 rounded-md hover:bg-slate-100 text-slate-400 hover:text-slate-600 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
            >
              <RefreshCw className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

const StatusBarSkeleton = (
  <div className="w-full flex items-center justify-between animate-pulse">
    <div className="flex gap-4">
      <div className="h-8 w-24 bg-slate-100 rounded-full" />
      <div className="h-8 w-32 bg-slate-100 rounded-md hidden md:block" />
      <div className="h-8 w-40 bg-slate-100 rounded-md hidden md:block" />
    </div>
    <div className="h-8 w-32 bg-slate-100 rounded-md" />
  </div>
);

export default function StatusBar() {
  const mounted = useIsMounted();
  const [showSettings, setShowSettings] = useState(false);

  return (
    <div className="flex-none bg-white border-b border-slate-100 px-4 py-3 lg:py-0 min-h-[64px] lg:h-[64px] flex flex-col lg:flex-row items-stretch lg:items-center justify-between gap-4 overflow-x-auto overflow-y-hidden scrollbar-hide">
      {mounted ? (
        <>
          <div className={`flex flex-col md:flex-row items-stretch md:items-center gap-2 md:gap-4 transition-all duration-300 ${!showSettings ? 'hidden md:flex' : 'flex'}`}>
            <DeviceSelect />
            <ModelSelect />
            <CameraSelect />
            <ConfidenceSlider />
          </div>

          {/* Mobile Settings Toggle */}
          <div className="md:hidden flex items-center justify-between gap-4 mt-1">
            <button
              onClick={() => setShowSettings(!showSettings)}
              className="flex items-center gap-2 px-3 py-1.5 bg-slate-50 rounded-lg border border-slate-200 text-slate-600 font-medium text-[10px] uppercase tracking-wider transition-colors hover:bg-slate-100"
            >
              <Sliders className="w-3 h-3 text-slate-400" />
              {showSettings ? "Hide Options" : "Show Options"}
              {showSettings ? <ChevronUp className="w-3 h-3 text-slate-400" /> : <ChevronDown className="w-3 h-3 text-slate-400" />}
            </button>
          </div>

          <PerformanceMetrics />
        </>
      ) : (
        StatusBarSkeleton
      )}
    </div>
  );
}
