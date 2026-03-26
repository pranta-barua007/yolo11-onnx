"use client";

import { Box } from "../utils/types";
import defaultClasses from "../utils/yolo_classes.json";
import { Colors } from "../utils/img_preprocess";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Download } from "lucide-react";

interface ModelStatusProps {
  details: Box[];
  selectedDetectionIdx: number | null;
  onSelectDetection: (idx: number | null) => void;
  onSave: () => void;
  classes?: string[];
}

export default function ModelStatus({
  details,
  selectedDetectionIdx,
  onSelectDetection,
  onSave,
  classes = defaultClasses,
}: ModelStatusProps) {
  const handleRowClick = (idx: number) => {
    onSelectDetection(selectedDetectionIdx === idx ? null : idx);
  };

  return (
    <Card className="glass-panel w-full p-4 h-full flex flex-col border-0 rounded-[20px]">

      {/* Header row: label + count + save */}
      <div className="flex items-center justify-between mb-3 px-1">
        <h2 className="text-xs font-bold uppercase tracking-wider text-muted-foreground">
          Detections{details.length > 0 && ` (${details.length})`}
        </h2>
        <button
          onClick={onSave}
          title="Save result as PNG"
          className="flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[11px] font-semibold text-primary bg-primary/10 hover:bg-primary/20 border border-primary/20 transition-all duration-300 disabled:opacity-40 disabled:cursor-not-allowed hover:shadow-[0_0_15px_rgba(168,85,247,0.2)]"
          disabled={details.length === 0}
        >
          <Download className="w-3.5 h-3.5" />
          Save
        </button>
      </div>

      <div className="flex-1 overflow-auto rounded-xl border border-white/5 bg-black/10 scrollbar-hide">
        <table className="w-full text-left text-sm">
          <thead className="bg-black/20 sticky top-0 backdrop-blur-md z-10">
            <tr>
              <th className="p-3 font-semibold text-muted-foreground w-12 text-left">#</th>
              <th className="p-3 font-semibold text-muted-foreground text-left">Class</th>
              <th className="p-3 font-semibold text-muted-foreground text-right">Conf.</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-white/5">
            {details.map((item, index) => {
              const isSelected = selectedDetectionIdx === index;
              const isDimmed = selectedDetectionIdx !== null && !isSelected;
              const color = Colors.getColor(item.class_idx, 1.0);
              const dotColor = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;

              return (
                <tr
                  key={index}
                  onClick={() => handleRowClick(index)}
                  className={`cursor-pointer transition-colors duration-200 ${isSelected
                      ? "bg-primary/20 ring-1 ring-inset ring-primary/40 shadow-[inset_0_0_15px_rgba(168,85,247,0.1)]"
                      : isDimmed
                        ? "opacity-30 hover:opacity-100"
                        : "hover:bg-white/5"
                    }`}
                >
                  <td className="p-3 text-muted-foreground text-left">{index + 1}</td>
                  <td className="p-3 text-left">
                    <span className="flex items-center gap-2">
                      <span
                        className="inline-block w-2.5 h-2.5 rounded-full flex-shrink-0"
                        style={{ backgroundColor: dotColor, boxShadow: `0 0 10px ${dotColor}` }}
                      />
                      <span className="font-medium text-foreground truncate max-w-[120px]" title={classes[item.class_idx]}>{classes[item.class_idx]}</span>
                    </span>
                  </td>
                  <td className="p-3 text-right">
                    <Badge
                      variant={item.score > 0.8 ? "default" : "secondary"}
                      className={item.score > 0.8 ? "bg-primary hover:bg-primary/80 text-primary-foreground shadow-[0_0_12px_rgba(168,85,247,0.4)]" : "bg-white/10 text-muted-foreground hover:bg-white/20"}
                    >
                      {(item.score * 100).toFixed(1)}%
                    </Badge>
                  </td>
                </tr>
              );
            })}
            {details.length === 0 && (
              <tr>
                <td colSpan={3} className="p-8 text-center text-muted-foreground italic">
                  No detections yet
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </Card>
  );
}