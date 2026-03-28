import * as ort from "onnxruntime-web";

self.onmessage = async (e: MessageEvent) => {
  const { buffer, classesConfig } = e.data;
  
  // Use a fast WASM-only session with zero graph optimizations 
  // just to quickly extract the architecture metadata.
  let session: ort.InferenceSession | null = null;
  const start = performance.now();

  try {
    session = await ort.InferenceSession.create(new Uint8Array(buffer), {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "disabled",
      logSeverityLevel: 4, 
    });

    const outputNames = session.outputNames;
    const inputName = session.inputNames[0] || "images";
    
    // We need to run a tiny 1x1 dummy to inspect the output dimensions,
    // since ONNXRuntime-Web doesn't publicly expose getOutputMetadata without running it.
    // Assuming standard 640x640 base size to avoid dynamic shape errors
    const dummy = new ort.Tensor("float32", new Float32Array(1 * 3 * 640 * 640), [1, 3, 640, 640]);
    const warmupOutput = await session.run({ [inputName]: dummy });

    const output0 = warmupOutput[outputNames[0]];
    const output1 = outputNames.length > 1 ? warmupOutput[outputNames[1]] : null;
    
    const capabilities: ("D" | "S" | "P")[] = ["D"]; // YOLO base
    
    if (output0) {
      const NUM_CHANNELS = output0.dims[1];
      const NUM_SCORES = classesConfig ? classesConfig.length : 80;
      const NUM_MASK_WEIGHTS = Math.max(0, NUM_CHANNELS - (4 + NUM_SCORES));

      if (output1 && output1.dims.length === 4 && NUM_MASK_WEIGHTS > 0) {
        capabilities.push("S");
      } else if (NUM_MASK_WEIGHTS === 51) {
        capabilities.push("P"); // Keypoints (17 * 3 = 51)
      }
    }
    
    for (const name of outputNames) {
      warmupOutput[name]?.dispose();
    }
    dummy.dispose();

    const timeMs = (performance.now() - start).toFixed(2);

    self.postMessage({
      status: "success",
      capabilities,
      timeMs
    });

  } catch (error: unknown) {
    self.postMessage({
      status: "error",
      error: error instanceof Error ? error.message : "Invalid ONNX model structure."
    });
  }
};
