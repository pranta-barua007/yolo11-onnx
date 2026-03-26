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
          className="p-6 cursor-pointer glass-panel border-white/5 hover:bg-white/10 hover:border-primary/50 flex flex-col items-center gap-4 group transition-all duration-500 rounded-3xl"
          onClick={() => openImageRef.current?.click()}
        >
          <div className="p-4 bg-primary/10 rounded-[1.25rem] group-hover:bg-primary/20 group-hover:scale-110 group-hover:shadow-[0_0_20px_rgba(168,85,247,0.4)] transition-all duration-500">
            <ImageIcon className="w-8 h-8 text-primary group-hover:text-white transition-colors" />
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
          className="p-6 cursor-pointer glass-panel border-white/5 hover:bg-white/10 hover:border-fuchsia-500/50 flex flex-col items-center gap-4 group transition-all duration-500 rounded-3xl"
          onClick={onCameraToggle}
        >
          <div className="p-4 bg-fuchsia-500/10 rounded-[1.25rem] group-hover:bg-fuchsia-500/20 group-hover:scale-110 group-hover:shadow-[0_0_20px_rgba(217,70,239,0.4)] transition-all duration-500">
            <Camera className="w-8 h-8 text-fuchsia-500 group-hover:text-white transition-colors" />
          </div>
          <div className="text-center">
            <h3 className="font-semibold text-foreground text-lg mb-1">Open Camera</h3>
            <p className="text-sm text-muted-foreground">Real-time detection</p>
          </div>
        </Card>
      </div>

      <div className="relative flex items-center w-full max-w-md mb-8">
        <div className="grow border-t border-white/10"></div>
        <span className="shrink-0 mx-4 text-muted-foreground text-xs font-semibold tracking-widest uppercase">Or try an example</span>
        <div className="grow border-t border-white/10"></div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 w-full">
        {EXAMPLE_IMAGES.map((src, i) => (
          <button
            key={i}
            onClick={() => onImageSelect(src)}
            className="relative group/img overflow-hidden rounded-[1.25rem] aspect-square border border-white/10 hover:border-primary/50 hover:shadow-[0_0_25px_rgba(168,85,247,0.3)] transition-all duration-500"
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={src} alt={`Example ${i + 1}`} className="w-full h-full object-cover opacity-70 group-hover/img:opacity-100 transition-opacity duration-500 group-hover/img:scale-105" />
            <div className="absolute inset-0 bg-gradient-to-t from-background/90 to-transparent opacity-0 group-hover/img:opacity-100 transition-opacity duration-500" />
          </button>
        ))}
      </div>
    </div>
  );
}
