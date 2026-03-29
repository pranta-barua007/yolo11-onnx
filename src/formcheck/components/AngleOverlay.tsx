"use client";

import { useEffect, useRef } from "react";
import { Exercise, Keypoint } from "../types";
import { getJointAngle } from "../angle-utils";
import { MIN_KEYPOINT_CONFIDENCE } from "../constants";
import { Box } from "../../utils/types";

interface AngleOverlayProps {
  overlayRef: React.RefObject<HTMLCanvasElement | null>;
  details: Box[];
  exercise: Exercise | null;
}

/**
 * AngleOverlay — Draws angle arcs and degree labels on the canvas overlay.
 *
 * This draws AFTER the pipeline's draw_bounding_boxes has already rendered,
 * adding angle annotations on top of the existing skeleton visualization.
 */
export default function AngleOverlay({
  overlayRef,
  details,
  exercise,
}: AngleOverlayProps) {
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    if (!exercise || details.length === 0) return;

    // Use rAF to draw after the pipeline has rendered
    rafRef.current = requestAnimationFrame(() => {
      const canvas = overlayRef.current;
      if (!canvas) return;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const person = details.find(
        (box) => box.keypoints && box.keypoints.length === 17
      );
      if (!person?.keypoints) return;

      const keypoints = person.keypoints as Keypoint[];
      const allJoints = [exercise.primaryJoint, ...(exercise.secondaryJoints ?? [])];

      for (const joint of allJoints) {
        const [idxA, idxB, idxC] = joint.indices;
        const a = keypoints[idxA];
        const b = keypoints[idxB]; // Vertex
        const c = keypoints[idxC];

        if (
          !a || !b || !c ||
          a.score < MIN_KEYPOINT_CONFIDENCE ||
          b.score < MIN_KEYPOINT_CONFIDENCE ||
          c.score < MIN_KEYPOINT_CONFIDENCE
        ) continue;

        const angle = getJointAngle(keypoints, joint);
        if (angle === null) continue;

        const isPrimary = joint.name === exercise.primaryJoint.name;
        const arcRadius = isPrimary ? 30 : 20;

        // ── Draw angle arc ──
        const startAngle = Math.atan2(a.y - b.y, a.x - b.x);
        const endAngle = Math.atan2(c.y - b.y, c.x - b.x);

        ctx.beginPath();
        ctx.arc(b.x, b.y, arcRadius, startAngle, endAngle, false);
        ctx.strokeStyle = isPrimary
          ? "rgba(168, 85, 247, 0.8)"  // Purple for primary
          : "rgba(99, 102, 241, 0.5)"; // Indigo for secondary
        ctx.lineWidth = isPrimary ? 3 : 2;
        ctx.stroke();

        // ── Draw angle label ──
        const midAngle = (startAngle + endAngle) / 2;
        const labelRadius = arcRadius + 16;
        const labelX = b.x + Math.cos(midAngle) * labelRadius;
        const labelY = b.y + Math.sin(midAngle) * labelRadius;

        const fontSize = isPrimary ? 14 : 11;
        ctx.font = `bold ${fontSize}px system-ui, sans-serif`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";

        // Background pill
        const text = `${angle}°`;
        const textMetrics = ctx.measureText(text);
        const pillW = textMetrics.width + 10;
        const pillH = fontSize + 6;

        ctx.fillStyle = isPrimary
          ? "rgba(168, 85, 247, 0.85)"
          : "rgba(99, 102, 241, 0.7)";
        ctx.beginPath();
        ctx.roundRect(labelX - pillW / 2, labelY - pillH / 2, pillW, pillH, 4);
        ctx.fill();

        // Text
        ctx.fillStyle = "white";
        ctx.fillText(text, labelX, labelY);
      }
    });

    return () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
      }
    };
  }, [details, exercise, overlayRef]);

  // This component renders nothing — it draws directly to the shared canvas
  return null;
}
