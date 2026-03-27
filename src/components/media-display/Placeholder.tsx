"use client";

import { Image as ImageIcon, Camera } from "lucide-react";
import { Card } from "@/components/ui/card";
import { useMediaDisplay } from "./MediaDisplayContext";

const EXAMPLE_IMAGES = [
  '/ex1.jpg',
  '/ex2.jpg',
  '/ex3.jpg',
  '/ex4.jpg',
];

export default function Placeholder() {
  const {
    actions: { onImageSelect, onCameraToggle, onOpenImage },
    meta: { openImageRef },
  } = useMediaDisplay();

  return (
    <div className="text-center p-8 max-w-2xl w-full flex flex-col items-center">
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 w-full max-w-lg mb-8">
        <Card
          className="p-6 cursor-pointer bg-card border-border/40 shadow-sm hover:shadow-md hover:border-primary/30 flex flex-col items-center gap-4 group transition-all duration-300 rounded-2xl"
          onClick={() => openImageRef.current?.click()}
        >
          <div className="p-4 bg-primary/10 rounded-2xl group-hover:bg-primary/15 transition-colors duration-300">
            <ImageIcon className="w-8 h-8 text-primary transition-colors duration-300" />
          </div>
          <div className="text-center">
            <h3 className="font-semibold text-foreground text-lg mb-1">Upload Image</h3>
            <p className="text-sm text-muted-foreground">Analyze a local file</p>
          </div>
          <input
            type="file"
            accept="image/*"
            hidden
            ref={openImageRef}
            onChange={onOpenImage}
          />
        </Card>

        <Card
          className="p-6 cursor-pointer bg-card border-border/40 shadow-sm hover:shadow-md hover:border-primary/30 flex flex-col items-center gap-4 group transition-all duration-300 rounded-2xl"
          onClick={onCameraToggle}
        >
          <div className="p-4 bg-primary/10 rounded-2xl group-hover:bg-primary/15 transition-colors duration-300">
            <Camera className="w-8 h-8 text-primary transition-colors duration-300" />
          </div>
          <div className="text-center">
            <h3 className="font-semibold text-foreground text-lg mb-1">Open Camera</h3>
            <p className="text-sm text-muted-foreground">Real-time detection</p>
          </div>
        </Card>
      </div>

      <div className="relative flex items-center w-full max-w-md mb-8">
        <div className="grow border-t border-foreground/10"></div>
        <span className="shrink-0 mx-4 text-muted-foreground text-xs font-semibold tracking-widest uppercase">Or try an example</span>
        <div className="grow border-t border-foreground/10"></div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 w-full">
        {EXAMPLE_IMAGES.map((src, i) => (
          <button
            key={i}
            onClick={() => onImageSelect(src)}
            className="relative group/img overflow-hidden rounded-2xl aspect-square shadow-sm ring-1 ring-border/50 hover:ring-2 hover:ring-primary/50 hover:shadow-md transition-all duration-300"
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={src} alt={`Example ${i + 1}`} className="w-full h-full object-cover transition-transform duration-500 group-hover/img:scale-105" />
          </button>
        ))}
      </div>
    </div>
  );
}
