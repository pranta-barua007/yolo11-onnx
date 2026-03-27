"use client";

import { Box } from "../utils/types";
import defaultClasses from "../utils/yolo_classes.json";
import { Colors } from "../utils/img_preprocess";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Download, ChevronLeft } from "lucide-react";

interface ModelStatusProps {
  details: Box[];
  selectedDetectionIdx: number | null;
  onSelectDetection: (idx: number | null) => void;
  selectedKeypointIdx: number | null;
  onSelectKeypoint: (idx: number | null) => void;
  onSave: () => void;
  classes?: string[];
}

const COCO_KEYPOINTS = [
  "Nose", "L-Eye", "R-Eye", "L-Ear", "R-Ear",
  "L-Shoulder", "R-Shoulder", "L-Elbow", "R-Elbow", "L-Wrist", "R-Wrist",
  "L-Hip", "R-Hip", "L-Knee", "R-Knee", "L-Ankle", "R-Ankle"
];

const GROUPS = [
  { name: "Face", indices: [0, 1, 2, 3, 4], color: "text-blue-500" },
  { name: "Arms / Shoulders", indices: [5, 6, 7, 8, 9, 10], color: "text-emerald-500" },
  { name: "Legs / Hips", indices: [11, 12, 13, 14, 15, 16], color: "text-amber-500" }
];

export default function ModelStatus({
  details,
  selectedDetectionIdx,
  onSelectDetection,
  selectedKeypointIdx,
  onSelectKeypoint,
  onSave,
  classes = defaultClasses,
}: ModelStatusProps) {
  const handleRowClick = (idx: number) => {
    onSelectDetection(selectedDetectionIdx === idx ? null : idx);
  };

  return (
    <Card className="w-full h-full flex flex-col border-border/40 shadow-sm bg-card rounded-2xl overflow-hidden p-4">

      {/* Header row: label + count + save */}
      <div className="flex items-center justify-between mb-3 px-1">
        <h2 className="text-xs font-bold uppercase tracking-wider text-muted-foreground">
          Detections{details.length > 0 && ` (${details.length})`}
        </h2>
        <button
          onClick={onSave}
          title="Save result as PNG"
          className="flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[11px] font-semibold text-primary bg-primary/10 hover:bg-primary/20 border border-primary/20 transition-all duration-300 disabled:opacity-40 disabled:cursor-not-allowed"
          disabled={details.length === 0}
        >
          <Download className="w-3.5 h-3.5" />
          Save
        </button>
      </div>

      <div className="flex-1 overflow-auto rounded-xl border border-border/40 bg-transparent scrollbar-hide flex flex-col">
        {selectedDetectionIdx === null ? (
          <table className="w-full text-left text-sm">
            <thead className="bg-muted/50 sticky top-0 backdrop-blur-md z-10">
              <tr>
                <th className="p-3 font-semibold text-muted-foreground w-12 text-left">#</th>
                <th className="p-3 font-semibold text-muted-foreground text-left">Class</th>
                <th className="p-3 font-semibold text-muted-foreground text-right">Conf.</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/40">
              {details.map((item, index) => {
                const color = Colors.getColor(item.class_idx, 1.0);
                const dotColor = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;

                return (
                  <tr
                    key={index}
                    onClick={() => handleRowClick(index)}
                    className="cursor-pointer transition-colors duration-200 hover:bg-muted/50"
                  >
                    <td className="p-3 text-muted-foreground text-left">{index + 1}</td>
                    <td className="p-3 text-left">
                      <span className="flex items-center gap-2">
                        <span
                          className="inline-block w-2.5 h-2.5 rounded-full flex-shrink-0"
                          style={{ backgroundColor: dotColor }}
                        />
                        <span className="font-medium text-foreground truncate max-w-[120px]" title={classes[item.class_idx]}>{classes[item.class_idx]}</span>
                      </span>
                    </td>
                    <td className="p-3 text-right">
                      <Badge
                        variant={item.score > 0.8 ? "default" : "secondary"}
                        className={item.score > 0.8 ? "bg-primary hover:bg-primary/90 text-primary-foreground" : "bg-secondary hover:bg-secondary/80 text-secondary-foreground"}
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
        ) : (
          <div className="flex flex-col h-full animate-in fade-in slide-in-from-left-2 duration-300">
            {/* Active Detection Header */}
            <div className="p-4 border-b border-border/40 flex items-center justify-between sticky top-0 bg-card/80 backdrop-blur-sm z-10">
              <div className="flex items-center gap-3">
                <button 
                  onClick={() => onSelectDetection(null)}
                  className="p-1.5 hover:bg-muted rounded-md transition-colors mr-1"
                >
                  <ChevronLeft className="w-4 h-4" />
                </button>
                <div>
                  <h3 className="text-[13px] font-bold text-foreground">#{selectedDetectionIdx + 1} {classes[details[selectedDetectionIdx].class_idx]}</h3>
                  <p className="text-[10px] text-muted-foreground uppercase tracking-tight">Pose Breakdown</p>
                </div>
              </div>
              <Badge className="bg-primary/20 text-primary border-none text-[10px] font-bold">{(details[selectedDetectionIdx].score * 100).toFixed(1)}%</Badge>
            </div>

            {/* Keypoints list */}
            <div className="p-4 space-y-5 overflow-y-auto h-full scrollbar-hide">
              {GROUPS.map((group) => (
                <div key={group.name}>
                  <h4 className={`text-[10px] font-bold uppercase tracking-wider ${group.color} mb-2 flex items-center gap-2`}>
                    <div className={`w-1 h-3 rounded-full bg-current opacity-40`} />
                    {group.name}
                  </h4>
                  <div className="grid grid-cols-1 gap-1">
                    {group.indices.map((kpIdx) => {
                      const kp = details[selectedDetectionIdx].keypoints?.[kpIdx];
                      const isHighConf = (kp?.score ?? 0) > 0.5;
                      const isSelected = selectedKeypointIdx === kpIdx;

                      return (
                        <div 
                          key={kpIdx}
                          onMouseEnter={() => onSelectKeypoint(kpIdx)}
                          onMouseLeave={() => onSelectKeypoint(null)}
                          onClick={() => onSelectKeypoint(isSelected ? null : kpIdx)}
                          className={`
                            group flex items-center justify-between p-2 rounded-lg 
                            transition-all duration-200 border cursor-pointer
                            ${isSelected 
                              ? "bg-primary/10 border-primary/30 shadow-sm" 
                              : "border-transparent hover:bg-muted font-medium hover:border-border/40"
                            }
                          `}
                        >
                          <span className={`text-xs ${isSelected ? "text-primary font-bold" : "text-foreground"} flex items-center gap-2`}>
                            <div className={`w-1.5 h-1.5 rounded-full ${isHighConf ? "bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.5)]" : "bg-muted-foreground/30"}`} />
                            {COCO_KEYPOINTS[kpIdx]}
                          </span>
                          <span className={`text-[10px] tabular-nums ${isHighConf ? "text-muted-foreground" : "text-muted-foreground/40 font-normal italic"}`}>
                            {kp ? `${(kp.score * 100).toPrecision(2)}%` : "N/A"}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              ))}
              
              {!details[selectedDetectionIdx].keypoints && (
                <div className="flex flex-col items-center justify-center p-8 text-center text-muted-foreground">
                  <p className="text-xs italic">No pose data available for this model.</p>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </Card>
  );
}