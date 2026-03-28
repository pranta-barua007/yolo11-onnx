"use client";

import { useState, useRef, useCallback } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Upload, FileJson, X, Plus, AlertCircle, CheckCircle2 } from "lucide-react";
import { CustomModel } from "@/utils/types";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";

interface AddModelDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onAddModel: (model: CustomModel & { buffer?: ArrayBuffer }) => void;
}

export default function AddModelDialog({ open, onOpenChange, onAddModel }: AddModelDialogProps) {
  const [modelFile, setModelFile] = useState<File | null>(null);
  const [classes, setClasses] = useState<string[]>([]);
  const [manualInput, setManualInput] = useState("");
  const [classError, setClassError] = useState<string | null>(null);
  const [classMode, setClassMode] = useState<"json" | "manual">("json");
  const [manualCaps, setManualCaps] = useState<("Q" | "I8")[]>([]);
  const [isValidating, setIsValidating] = useState(false);

  const modelInputRef = useRef<HTMLInputElement>(null);
  const classInputRef = useRef<HTMLInputElement>(null);

  const resetState = useCallback(() => {
    setModelFile(null);
    setClasses([]);
    setManualInput("");
    setClassError(null);
    setClassMode("json");
    setManualCaps([]);
    setIsValidating(false);
  }, []);

  const handleModelSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (!file.name.endsWith(".onnx")) {
        return;
      }
      setModelFile(file);
    }
    e.target.value = "";
  };

  const parseClassesFromJSON = (text: string): string[] | null => {
    try {
      const parsed = JSON.parse(text);
      // Support both array format and { classes: [...] } format
      if (Array.isArray(parsed)) {
        const valid = parsed.filter((c) => typeof c === "string" && c.trim());
        return valid.length > 0 ? valid.map((c: string) => c.trim()) : null;
      }
      if (parsed.classes && Array.isArray(parsed.classes)) {
        const valid = parsed.classes.filter((c: unknown) => typeof c === "string" && (c as string).trim());
        return valid.length > 0 ? valid.map((c: string) => c.trim()) : null;
      }
      return null;
    } catch {
      return null;
    }
  };

  const handleClassFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = () => {
      const text = reader.result as string;
      const parsed = parseClassesFromJSON(text);
      if (parsed) {
        setClasses(parsed);
        setClassError(null);
      } else {
        setClassError("Invalid JSON. Expected [\"class1\", \"class2\"] or { \"classes\": [...] }");
        setClasses([]);
      }
    };
    reader.readAsText(file);
    e.target.value = "";
  };

  const handleManualParse = () => {
    if (!manualInput.trim()) {
      setClassError("Please enter class names");
      return;
    }

    // Try JSON first
    const jsonParsed = parseClassesFromJSON(manualInput);
    if (jsonParsed) {
      setClasses(jsonParsed);
      setClassError(null);
      return;
    }

    // Otherwise split by comma
    const split = manualInput
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
    if (split.length > 0) {
      setClasses(split);
      setClassError(null);
    } else {
      setClassError("No valid classes found");
    }
  };

  const removeClass = (idx: number) => {
    setClasses((prev) => prev.filter((_, i) => i !== idx));
  };

  const handleSubmit = async () => {
    if (!modelFile || classes.length === 0) return;

    setIsValidating(true);
    setClassError(null);

    try {
      // Read file as ArrayBuffer for caching
      const buffer = await modelFile.arrayBuffer();

      // Perform non-blocking structural validation via Web Worker
      const capabilities = await new Promise<("D" | "S" | "P" | "Q" | "I8")[]>((resolve, reject) => {
        const worker = new Worker(new URL("../workers/validationWorker.ts", import.meta.url), { type: "module" });
        worker.onmessage = (e) => {
          if (e.data.status === "success") {
            resolve(e.data.capabilities);
          } else {
            reject(new Error(e.data.error || "Invalid ONNX Model"));
          }
          worker.terminate();
        };
        worker.onerror = () => {
          reject(new Error("Validation worker crashed"));
          worker.terminate();
        };
        // Safely transfer a copy to the worker so we don't detach the primary buffer
        const bufferCopy = buffer.slice(0);
        worker.postMessage({ buffer: bufferCopy, classesConfig: classes }, [bufferCopy]);
      });

      const cacheKey = `custom:${modelFile.name.replace(".onnx", "")}`;

      const finalCaps = Array.from(new Set([...capabilities, ...manualCaps])) as ("D" | "S" | "P" | "Q" | "I8")[];

      onAddModel({
        name: modelFile.name.replace(".onnx", ""),
        url: cacheKey,
        classes,
        capabilities: finalCaps,
        buffer,
      });
      resetState();
      onOpenChange(false);
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : "Ensure it is a valid YOLO ONNX format.";
      setClassError(`Failed to validate model. ${errorMessage}`);
    } finally {
      setIsValidating(false);
    }
  };

  const isValid = modelFile && classes.length > 0;

  return (
    <Dialog
      open={open}
      onOpenChange={(val) => {
        if (!val) resetState();
        onOpenChange(val);
      }}
    >
      <DialogContent className="sm:max-w-[520px] max-h-[85vh] overflow-y-auto border-border/50 bg-popover rounded-2xl">
        <DialogHeader>
          <DialogTitle className="text-lg font-semibold text-foreground">Add Custom Model</DialogTitle>
          <DialogDescription className="text-muted-foreground text-sm">
            Upload an ONNX model and define its classes.
          </DialogDescription>
        </DialogHeader>

        {/* ── Step 1: Model File ── */}
        <div className="space-y-3">
          <Label className="font-semibold text-sm text-foreground">Model File (.onnx)</Label>
          {modelFile ? (
            <div className="flex items-center gap-3 p-3 bg-primary/10 border border-primary/20 rounded-lg">
              <CheckCircle2 className="w-5 h-5 text-primary shrink-0" />
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-primary truncate">{modelFile.name}</p>
                <p className="text-xs text-primary/70">
                  {(modelFile.size / (1024 * 1024)).toFixed(2)} MB
                </p>
              </div>
              <button
                onClick={() => setModelFile(null)}
                className="p-1 hover:bg-primary/20 rounded transition-colors"
              >
                <X className="w-4 h-4 text-primary" />
              </button>
            </div>
          ) : (
            <button
              onClick={() => modelInputRef.current?.click()}
              className="w-full p-6 border-2 border-dashed border-border rounded-lg hover:border-primary/50 hover:bg-muted/30 transition-all flex flex-col items-center gap-2 group"
            >
              <Upload className="w-8 h-8 text-muted-foreground group-hover:text-foreground transition-colors" />
              <span className="text-sm text-muted-foreground font-medium">Click to upload .onnx model</span>
            </button>
          )}
          <input
            ref={modelInputRef}
            type="file"
            accept=".onnx"
            onChange={handleModelSelect}
            className="hidden"
          />
        </div>

        {/* ── Step 2: Classes ── */}
        <div className="space-y-3">
          <Label className="font-semibold text-sm text-foreground">Model Classes</Label>

          {/* Mode Toggle */}
          <Tabs
            value={classMode}
            onValueChange={(val: string) => {
              if (val) {
                setClassMode(val as "json" | "manual");
                setClassError(null);
              }
            }}
            className="w-full"
          >
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="json" className="flex items-center gap-2">
                <FileJson className="w-3.5 h-3.5" />
                JSON File
              </TabsTrigger>
              <TabsTrigger value="manual" className="flex items-center gap-2">
                <Plus className="w-3.5 h-3.5" />
                Manual Entry
              </TabsTrigger>
            </TabsList>
          </Tabs>

          {classMode === "json" ? (
            <div>
              <button
                onClick={() => classInputRef.current?.click()}
                className="w-full p-4 border-2 border-dashed border-border rounded-lg hover:border-primary/50 hover:bg-muted/30 transition-all flex items-center gap-3"
              >
                <FileJson className="w-5 h-5 text-muted-foreground" />
                <span className="text-sm text-muted-foreground">Upload classes JSON</span>
              </button>
              <input
                ref={classInputRef}
                type="file"
                accept=".json"
                onChange={handleClassFileUpload}
                className="hidden"
              />
              <p className="text-[10px] text-muted-foreground mt-1.5">
                Format: [&quot;class1&quot;, &quot;class2&quot;] or {`{ "classes": [...] }`}
              </p>
            </div>
          ) : (
            <div className="space-y-2">
              <div className="flex gap-2">
                <Input
                  value={manualInput}
                  onChange={(e) => setManualInput(e.target.value)}
                  placeholder='person, bicycle, car...'
                  className="text-sm"
                  onKeyDown={(e) => { if (e.key === "Enter") handleManualParse(); }}
                />
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleManualParse}
                  className="shrink-0"
                >
                  Parse
                </Button>
              </div>
              <p className="text-[10px] text-muted-foreground">
                Comma-separated class names or JSON array
              </p>
            </div>
          )}

          {/* Error */}
          {classError ? (
            <div className="flex items-center gap-2 text-destructive bg-destructive/10 border border-destructive/20 p-2.5 rounded-lg">
              <AlertCircle className="w-4 h-4 shrink-0" />
              <span className="text-xs">{classError}</span>
            </div>
          ) : null}

          {/* Parsed Classes Preview */}
          {classes.length > 0 ? (
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-primary" />
                <span className="text-xs font-semibold text-primary">
                  {classes.length} classes found
                </span>
              </div>
              <div className="flex flex-wrap gap-1.5 p-3 bg-muted/40 dark:bg-black/40 border border-border/40 rounded-lg max-h-32 overflow-y-auto">
                {classes.map((cls, idx) => (
                  <div
                    key={idx}
                    className="flex items-center gap-1.5 text-xs px-2 py-1 bg-background/50 border border-border/40 rounded-md text-foreground"
                  >
                    <span className="text-[10px] text-muted-foreground font-mono">{idx}</span>
                    <span className="font-medium">{cls}</span>
                    <button
                      onClick={() => removeClass(idx)}
                      className="ml-1 hover:text-destructive transition-colors focus:outline-none"
                    >
                      <X className="w-3 h-3" />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          ) : null}
        </div>

        {/* ── Step 3: Experimental/Quantized Flags ── */}
        <div className="space-y-3 pt-2">
          <Label className="font-semibold text-sm text-foreground">Model Precision (Manual)</Label>
          <ToggleGroup 
            type="multiple" 
            variant="outline" 
            className="justify-start gap-2"
            value={manualCaps}
            onValueChange={(val) => setManualCaps(val as ("Q" | "I8")[])}
          >
            <ToggleGroupItem value="Q" className="h-8 text-[11px] gap-2 data-[state=on]:bg-amber-500/10 data-[state=on]:text-amber-500 data-[state=on]:border-amber-500/30">
              <CheckCircle2 className={`w-3 h-3 ${manualCaps.includes("Q") ? "opacity-100" : "opacity-0"}`} />
              Quantized (FP16)
            </ToggleGroupItem>
            <ToggleGroupItem value="I8" className="h-8 text-[11px] gap-2 data-[state=on]:bg-orange-600/10 data-[state=on]:text-orange-600 data-[state=on]:border-orange-600/30">
              <CheckCircle2 className={`w-3 h-3 ${manualCaps.includes("I8") ? "opacity-100" : "opacity-0"}`} />
              Quantized (INT8)
            </ToggleGroupItem>
          </ToggleGroup>
          <p className="text-[10px] text-muted-foreground">
            Auto-detection will also scan for quantization, but you can explicitly tag them here.
          </p>
        </div>

        {/* ── Actions ── */}
        <DialogFooter className="pt-4 border-t border-border/50 mt-6">
          <Button variant="outline" onClick={() => onOpenChange(false)} className="border-border/50 text-muted-foreground hover:bg-foreground/5 hover:text-foreground">
            Cancel
          </Button>
          <Button
            onClick={handleSubmit}
            disabled={!isValid || isValidating}
            className="bg-primary hover:bg-primary/90 text-primary-foreground min-w-[100px] shadow-[0_0_15px_rgba(var(--primary),0.3)] disabled:shadow-none transition-all duration-300"
          >
            {isValidating ? (
              <span className="flex items-center gap-2">
                <span className="w-3.5 h-3.5 border-2 border-primary-foreground border-t-transparent rounded-full animate-spin" />
                Validating...
              </span>
            ) : "Add Model"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
