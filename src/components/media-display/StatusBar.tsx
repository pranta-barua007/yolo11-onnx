"use client";

import { useSyncExternalStore } from "react";
import { useState } from "react";
import { Cpu, Package, Video, ChevronDown, ChevronUp, Sliders, Gauge, RefreshCw, Trash2 } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useMediaDisplay } from "./MediaDisplayContext";
import AddModelDialog from "../AddModelDialog";

function CapabilityBadges({ capabilities = [] }: { capabilities?: ("D" | "S" | "P")[] }) {
  if (!capabilities.length) return null;
  
  const colors = {
    D: "text-emerald-500 bg-emerald-500/10 border-emerald-500/20",
    S: "text-blue-500 bg-blue-500/10 border-blue-500/20",
    P: "text-purple-500 bg-purple-500/10 border-purple-500/20"
  };

  return (
    <div className="flex items-center gap-1 ml-1.5 shrink-0">
      {capabilities.map(cap => (
        <span 
          key={cap} 
          className={`px-1 rounded-[4px] border text-[9px] font-bold leading-none py-[2px] ${colors[cap]}`}
          title={cap === 'D' ? 'Detection' : cap === 'S' ? 'Segmentation' : 'Pose'}
        >
          {cap}
        </span>
      ))}
    </div>
  );
}

function DeviceSelect() {
  const { state: { device, isModelLoaded }, actions: { setDevice } } = useMediaDisplay();
  return (
    <div className="flex items-center gap-2">
      <Cpu className="w-3.5 h-3.5 text-muted-foreground" />
      <Select value={device} onValueChange={setDevice} defaultValue="webgpu">
        <SelectTrigger className="h-8 w-full md:w-[120px] text-[11px] font-bold uppercase tracking-wider bg-background border-border/40 rounded-full text-foreground hover:bg-muted/50 transition-colors">
          <div className="flex items-center gap-2">
            <div
              className={`w-2 h-2 rounded-full transition-all duration-500 ${isModelLoaded
                ? "bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.7)]"
                : "bg-amber-400 shadow-[0_0_8px_rgba(251,191,36,0.7)] animate-pulse"
                }`}
            />
            <SelectValue placeholder="Select Device" />
          </div>
        </SelectTrigger>
        <SelectContent className="bg-background/95 backdrop-blur-xl border-border/50">
          <SelectItem value="webgpu" className="text-xs uppercase focus:bg-primary/20">WebGPU</SelectItem>
          <SelectItem value="wasm" className="text-xs uppercase focus:bg-primary/20">WASM</SelectItem>
        </SelectContent>
      </Select>
    </div>
  );
}

function ModelSelect() {
  const {
    state: { modelName, customModels },
    actions: { setModelName, addCustomModel, removeCustomModel },
  } = useMediaDisplay();

  const [dialogOpen, setDialogOpen] = useState(false);

  return (
    <div className="flex items-center gap-2">
      <Package className="w-3.5 h-3.5 text-muted-foreground" />
      <Select value={modelName} onValueChange={setModelName} defaultValue="yolo11n-seg">
        <SelectTrigger className="h-8 w-full md:min-w-[140px] text-[11px] font-bold text-foreground bg-background border-border/40 font-mono rounded-full hover:bg-muted/50 transition-colors">
          <SelectValue placeholder="Select Model" />
        </SelectTrigger>
        <SelectContent className="bg-background/95 backdrop-blur-xl border-border/50">
          <SelectItem value="yolo11n-seg" className="text-xs font-mono focus:bg-primary/20">
            <div className="flex items-center justify-between w-full gap-2">
              <span>High Speed</span>
              <CapabilityBadges capabilities={["D", "S"]} />
            </div>
          </SelectItem>
          <SelectItem value="yolo11s-seg" className="text-xs font-mono focus:bg-primary/20">
            <div className="flex items-center justify-between w-full gap-2">
              <span>High Accuracy</span>
              <CapabilityBadges capabilities={["D", "S"]} />
            </div>
          </SelectItem>
          {customModels.filter(model => model.url).map((model) => (
            <SelectItem key={model.url} value={model.url} className="text-xs font-mono focus:bg-primary/20 group pr-8">
              <div className="flex items-center justify-between w-full min-w-[140px] gap-2">
                <span className="truncate">{model.name}</span>
                <div className="flex items-center gap-1">
                  <CapabilityBadges capabilities={model.capabilities} />
                  <button
                    onPointerDown={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                    }}
                    onPointerUp={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                    }}
                    onClick={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      removeCustomModel(model.url);
                    }}
                    className="opacity-0 group-hover:opacity-100 p-0.5 rounded text-destructive hover:bg-destructive/10 transition-opacity ml-1 flex-shrink-0"
                    title="Delete custom model"
                  >
                    <Trash2 className="w-3 h-3" />
                  </button>
                </div>
              </div>
            </SelectItem>
          ))}
          <div className="p-2 border-t border-foreground/10 mt-1">
            <button
              onClick={(e) => {
                e.preventDefault();
                e.stopPropagation();
                setDialogOpen(true);
              }}
              className="w-full text-[10px] font-bold uppercase tracking-wider py-1.5 px-2 hover:bg-accent text-muted-foreground hover:text-accent-foreground rounded transition-all duration-300"
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
      <Video className="w-3.5 h-3.5 text-muted-foreground" />
      <Select value={selectedDeviceId} onValueChange={setSelectedDeviceId}>
        <SelectTrigger className="h-8 w-full md:w-[160px] text-[11px] font-medium text-foreground bg-background border-border/40 rounded-full hover:bg-muted/50 transition-colors">
          <SelectValue placeholder="Select Camera" />
        </SelectTrigger>
        <SelectContent className="bg-background/95 backdrop-blur-xl border-border/50">
          {cameras.filter(cam => cam.deviceId).map((cam) => (
            <SelectItem key={cam.deviceId} value={cam.deviceId} className="text-xs focus:bg-primary/20">
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
    <div className={`flex items-center gap-2 px-3 h-8 bg-background border border-border/40 rounded-full transition-opacity ${isInferring ? "opacity-50 pointer-events-none" : ""}`}>
      <Gauge className="w-3.5 h-3.5 text-muted-foreground" />
      <span className="text-[11px] font-bold uppercase tracking-wider text-muted-foreground">Conf.</span>
      <input
        type="range"
        min={0.05}
        max={0.95}
        step={0.05}
        value={scoreThreshold}
        onChange={(e) => setScoreThreshold(parseFloat(e.target.value))}
        disabled={isInferring}
        className="w-20 h-1 accent-primary cursor-pointer disabled:cursor-not-allowed"
      />
      <span className="text-[11px] font-bold text-primary w-7 text-right tabular-nums drop-shadow-[0_0_8px_rgba(168,85,247,0.5)]">
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
        <div className="flex items-center gap-2 text-primary">
          <div className="w-3 h-3 border-2 border-current border-t-transparent rounded-full animate-spin shadow-[0_0_10px_rgba(168,85,247,0.5)]" />
          <span className="text-[11px] font-medium animate-pulse">{modelStatus}</span>
        </div>
      ) : (
        <div className="flex items-center gap-2 sm:gap-3 w-full sm:w-auto">
          <div className="flex items-center gap-1 text-[11px]">
            <span className="font-medium text-muted-foreground">FPS</span>
            <span className="font-bold text-fuchsia-400">
              {fps > 0 ? fps : "--"}
            </span>
          </div>
          <div className="w-px h-3 bg-foreground/10" />
          <div className="flex items-center gap-1 text-[11px]">
            <span className="font-medium text-muted-foreground">Inf.</span>
            <span className="font-bold text-primary">{inferenceTime}ms</span>
          </div>
          <div className="w-px h-3 bg-foreground/10" />
          <div className="flex items-center gap-1 text-[11px]">
            <span className="font-medium text-muted-foreground hidden sm:inline">Warmup</span>
            <span className="font-medium text-muted-foreground sm:hidden" title="Warmup">W.</span>
            <span className="font-bold text-primary/70">{warmUpTime}ms</span>
          </div>
          
          <div className="ml-auto flex items-center gap-2">
            <span className="text-[11px] font-semibold text-emerald-600 dark:text-emerald-400 flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 dark:bg-emerald-400 inline-block" />
              <span className="hidden sm:inline">Ready</span>
            </span>
            <button
              onClick={reloadModel}
              disabled={!!cameraStream}
              title="Re-download model (clear cache)"
              className="p-1 rounded-md hover:bg-muted text-muted-foreground hover:text-foreground transition-all duration-300 disabled:opacity-40 disabled:cursor-not-allowed"
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
      <div className="h-8 w-16 bg-foreground/5 rounded-full" />
      <div className="h-8 w-24 bg-foreground/5 rounded-full hidden md:block opacity-60" />
      <div className="h-8 w-28 bg-foreground/5 rounded-full hidden md:block opacity-40" />
    </div>
    <div className="h-8 w-32 bg-foreground/5 rounded-full opacity-60" />
  </div>
);

/** SSR-safe mount detection without useEffect + setState (fixes react-hooks/set-state-in-effect) */
const emptySubscribe = () => () => { };
function useIsMounted() {
  return useSyncExternalStore(
    emptySubscribe,
    () => true,  // client: always mounted
    () => false  // server: never mounted
  );
}

export default function StatusBar() {
  const mounted = useIsMounted();
  const [showSettings, setShowSettings] = useState(false);

  return (
    <div className="flex-none bg-transparent border-b border-border/40 px-4 py-3 flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-4 min-h-[64px]">
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
              className="flex items-center gap-2 px-3 py-1.5 bg-muted/50 rounded-lg border border-border/50 text-muted-foreground font-medium text-[10px] uppercase tracking-wider transition-all hover:bg-muted hover:text-foreground"
            >
              <Sliders className="w-3 h-3 text-muted-foreground" />
              {showSettings ? "Hide Options" : "Show Options"}
              {showSettings ? <ChevronUp className="w-3 h-3 text-muted-foreground" /> : <ChevronDown className="w-3 h-3 text-muted-foreground" />}
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
