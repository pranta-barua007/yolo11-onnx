import defaultClasses from "./yolo_classes.json";
import { Colors } from "./img_preprocess";

const SKELETON = [
  [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12],
  [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2],
  [1, 3], [2, 4], [3, 5], [4, 6]
];

/**
 * Draw bounding boxes ON TOP of existing canvas content (e.g., segmentation masks).
 * Does NOT clear the canvas.
 * @param predictions - Detected objects.
 * @param overlay_el - Canvas element to draw on.
 * @param filterIndex - If set, only draw that box fully; dim others.
 * @param classes - Class labels array. Defaults to built-in yolo_classes.json.
 */
export async function draw_bounding_boxes(
  predictions: Array<{ bbox: number[]; class_idx: number; score: number; keypoints?: {x:number, y:number, score:number}[] }>,
  overlay_el: HTMLCanvasElement,
  filterIndex: number | null = null,
  classes: string[] = defaultClasses,
  selectedKeypointIdx: number | null = null
): Promise<void> {
  const ctx = overlay_el.getContext("2d");
  if (!ctx) return;

  // NOTE: We do NOT clearRect here — masks drawn by inference_pipeline must be preserved.

  const diagonalLength = Math.sqrt(Math.pow(overlay_el.width, 2) + Math.pow(overlay_el.height, 2));
  const lineWidth = diagonalLength / 250;
  const fontSize = Math.max(14, Math.round(diagonalLength / 60));
  const pad = Math.round(fontSize * 0.3);

  predictions.forEach((predict, idx) => {
    const isFiltered = filterIndex !== null && idx !== filterIndex;
    const alpha = isFiltered ? 0.15 : 0.8;

    const borderColor = Colors.getColor(predict.class_idx, alpha);
    const rgbaBorderColor = `rgba(${borderColor[0]}, ${borderColor[1]}, ${borderColor[2]}, ${alpha})`;

    const [x1, y1, width, height] = predict.bbox;

    ctx.lineWidth = lineWidth;
    ctx.strokeStyle = rgbaBorderColor;
    ctx.strokeRect(x1, y1, width, height);

    // Only draw label for non-dimmed boxes
    if (!isFiltered) {
      ctx.fillStyle = rgbaBorderColor;
      ctx.font = `bold ${fontSize}px Arial`;
      const text = `${classes[predict.class_idx]} ${predict.score.toFixed(2)}`;
      const textWidth = ctx.measureText(text).width;

      let textY = y1 - pad;
      let rectY = y1 - fontSize - pad * 2;
      if (rectY < 0) {
        textY = y1 + fontSize + pad;
        rectY = y1 + 1;
      }

      ctx.fillRect(x1 - 1, rectY, textWidth + pad * 2, fontSize + pad * 2);
      ctx.fillStyle = "white";
      ctx.fillText(text, x1 + pad - 1, textY);
    }

    // --- Draw Pose Keypoints ---
    if (predict.keypoints && predict.keypoints.length === 17) {
      // Draw skeletal lines
      ctx.lineWidth = Math.max(1, lineWidth * 0.5);
      SKELETON.forEach(([idx1, idx2]) => {
        const p1 = predict.keypoints![idx1];
        const p2 = predict.keypoints![idx2];
        if (p1.score > 0.5 && p2.score > 0.5) {
          ctx.strokeStyle = "rgba(0, 255, 0, 0.8)"; // Green lines
          ctx.beginPath();
          ctx.moveTo(p1.x, p1.y);
          ctx.lineTo(p2.x, p2.y);
          ctx.stroke();
        }
      });

      // Draw keypoint dots
      predict.keypoints.forEach((kp, kpIdx) => {
        if (kp.score > 0.5) {
          const isSelectedKeypoint = !isFiltered && selectedKeypointIdx === kpIdx;
          
          if (isSelectedKeypoint) {
            ctx.fillStyle = "rgba(255, 255, 255, 1.0)"; // White highlight
            ctx.beginPath();
            ctx.arc(kp.x, kp.y, lineWidth * 3, 0, 2 * Math.PI);
            ctx.fill();
            ctx.strokeStyle = "rgba(255, 0, 0, 1.0)";
            ctx.lineWidth = 2;
            ctx.stroke();
          } else {
            ctx.fillStyle = "rgba(255, 0, 0, 1.0)"; // Red dots
            ctx.beginPath();
            ctx.arc(kp.x, kp.y, lineWidth * 1.5, 0, 2 * Math.PI);
            ctx.fill();
          }
        }
      });
    }
  });
}
