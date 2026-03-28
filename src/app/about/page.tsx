import { Metadata } from "next";
import Link from "next/link";
import Image from "next/image";
import {
    Cpu,
    Layers,
    Zap,
    ScanLine,
    Globe,
    GitBranch,
    ChevronRight,
    BarChart2,
    Sliders,
    Download,
    ExternalLink,
    ShieldCheck,
    Box,
    Terminal,
} from "lucide-react";
import Header from "@/components/Header";
import ArchitectureDiagram from "@/components/ArchitectureDiagram";

export const metadata: Metadata = {
    title: "About | YOLO Edge Runner",
    description:
        "Technical overview of the YOLO Edge Runner — a high-performance, browser-native segmentation tool powered by YOLO11 and WebGPU.",
};

const NAV_SECTIONS = [
    { id: "overview", label: "Overview" },
    { id: "architecture", label: "Architecture" },
    { id: "tech", label: "Technology" },
    { id: "quantization", label: "Quantization" },
    { id: "attribution", label: "Attribution" },
];

const FEATURES = [
    {
        icon: ScanLine,
        title: "Instance Segmentation",
        desc: "High-precision pixel-level segmentation using YOLO11-Seg for real-time object extraction.",
        color: "teal",
    },
    {
        icon: ShieldCheck,
        title: "Privacy-First",
        desc: "Zero server uploads. All inference happens locally in your browser via ONNX Runtime Web.",
        color: "purple",
    },
    {
        icon: Zap,
        title: "WebGPU Acceleration",
        desc: "Leverages the latest WebGPU API for near-native GPU inference speed on modern browsers.",
        color: "amber",
    },
    {
        icon: Box,
        title: "Multi-Model Support",
        desc: "Seamlessly switch between Nano (n) and Small (s) variants or load calibrated custom models.",
        color: "blue",
    },
];

const colorMap: Record<string, string> = {
    teal: "bg-teal-50 text-teal-600 border-teal-100",
    purple: "bg-purple-50 text-purple-600 border-purple-100",
    amber: "bg-amber-50 text-amber-600 border-amber-100",
    blue: "bg-blue-50 text-blue-600 border-blue-100",
};

export default function AboutPage() {
    return (
        <div className="min-h-screen bg-[#fafafa] text-slate-900 font-sans selection:bg-teal-100 selection:text-teal-900">
            <Header />

            {/* Sticky Navigation */}
            <div className="sticky top-[65px] z-20 bg-white/80 backdrop-blur-md border-b border-slate-200/60">
                <div className="max-w-6xl mx-auto px-6 overflow-x-auto overflow-y-hidden">
                    <nav className="flex items-center gap-2 py-3">
                        {NAV_SECTIONS.map((s) => (
                            <a
                                key={s.id}
                                href={`#${s.id}`}
                                className="flex-shrink-0 px-4 py-1.5 text-xs font-bold uppercase tracking-wider text-slate-500 hover:text-teal-700 hover:bg-teal-50/50 rounded-lg transition-all"
                            >
                                {s.label}
                            </a>
                        ))}
                    </nav>
                </div>
            </div>

            <main className="max-w-6xl mx-auto px-6 py-16 space-y-32">

                {/* ── HERO SECTION ────────────────────────────────────────── */}
                <section id="overview" className="scroll-mt-32">
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
                        <div className="space-y-6">
                            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-teal-50 border border-teal-100/50 text-teal-700 text-[10px] font-black uppercase tracking-[0.2em]">
                                Edge Computing
                            </div>
                            <h1 className="text-5xl md:text-6xl font-black text-slate-900 tracking-tight leading-[1.1]">
                                Real-time AI,
                                <br />
                                <span className="text-transparent bg-clip-text bg-gradient-to-r from-teal-600 to-indigo-600">
                                    In Your Browser.
                                </span>
                            </h1>
                            <p className="text-lg text-slate-600 leading-relaxed max-w-xl">
                                YOLO Edge Runner is a state-of-the-art vision application that brings the power of **YOLO11** 
                                to the browser edge. By combining WebGPU acceleration with quantized model inference, 
                                we achieve near-native performance without a backend.
                            </p>
                            <div className="flex items-center gap-4 pt-4">
                                <Link
                                    href="/"
                                    className="h-12 px-6 bg-slate-900 hover:bg-slate-800 text-white text-sm font-bold rounded-xl flex items-center gap-2 transition-all shadow-lg shadow-slate-900/10 active:scale-95"
                                >
                                    Open App <ChevronRight className="w-4 h-4" />
                                </Link>
                                <a
                                    href="https://github.com/pranta-barua007/yolo11-onnx"
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="h-12 px-6 bg-white border border-slate-200 hover:bg-slate-50 text-slate-700 text-sm font-bold rounded-xl flex items-center gap-2 transition-all active:scale-95"
                                >
                                    <GitBranch className="w-4 h-4" /> Source
                                </a>
                            </div>
                        </div>

                        <div className="relative group">
                            <div className="absolute -inset-4 bg-gradient-to-tr from-teal-100/40 to-indigo-100/40 rounded-[2rem] blur-2xl group-hover:blur-3xl transition-all" />
                            <div className="relative bg-white border border-slate-200/60 rounded-[2rem] p-8 shadow-xl">
                                <ArchitectureDiagram className="w-full h-auto drop-shadow-sm" />
                                <div className="mt-8 grid grid-cols-2 gap-4">
                                    <div className="p-4 bg-slate-50 rounded-2xl border border-slate-100">
                                        <div className="text-2xl font-black text-slate-900">~290ms</div>
                                        <div className="text-[10px] uppercase font-bold text-slate-400 tracking-widest mt-1 text-nowrap">Avg. Inference (GPU)</div>
                                    </div>
                                    <div className="p-4 bg-slate-50 rounded-2xl border border-slate-100">
                                        <div className="text-2xl font-black text-slate-900">0ms</div>
                                        <div className="text-[10px] uppercase font-bold text-slate-400 tracking-widest mt-1">Server Latency</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </section>

                {/* ── ARCHITECTURE DEEP DIVE ──────────────────────────────── */}
                <section id="architecture" className="scroll-mt-32">
                    <div className="max-w-3xl space-y-12">
                        <div className="space-y-4">
                            <h2 className="text-3xl font-black text-slate-900">Non-Blocking Worker Pattern</h2>
                            <p className="text-slate-600 leading-relaxed">
                                To maintain a smooth **60FPS UI**, we decouple the heavy inference logic from the main thread. 
                                The application utilizes a sophisticated **Web Worker architecture** that handles the entire 
                                ONNX lifecycle.
                            </p>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                            <div className="space-y-3">
                                <div className="flex items-center gap-2 text-teal-600">
                                    <Terminal className="w-5 h-5" />
                                    <h4 className="font-bold">Execution Provider (EP)</h4>
                                </div>
                                <p className="text-sm text-slate-500 leading-relaxed">
                                    The engine automatically benchmarks your hardware to select the best provider: 
                                    **WebGPU** for modern GPUs, or multi-threaded **WASM** for cross-compatibility.
                                </p>
                            </div>
                            <div className="space-y-3">
                                <div className="flex items-center gap-2 text-indigo-600">
                                    <Layers className="w-5 h-5" />
                                    <h4 className="font-bold">Memory Management</h4>
                                </div>
                                <p className="text-sm text-slate-500 leading-relaxed">
                                    We use **Transferable Objects** to move pixel data between threads with zero-copy overhead, 
                                    ensuring maximum throughput for high-resolution cameras.
                                </p>
                            </div>
                        </div>
                    </div>
                </section>

                {/* ── TECHNOLOGY ──────────────────────────────────────────── */}
                <section id="tech" className="scroll-mt-32">
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                        <div className="md:col-span-1 border-l-4 border-teal-500 pl-6 py-2">
                            <h2 className="text-3xl font-black text-slate-900 mb-4 tracking-tight uppercase tracking-widest">Tech Stack</h2>
                            <p className="text-sm text-slate-500 font-medium">Built on the bleeding edge of the Open Web.</p>
                        </div>
                        <div className="md:col-span-2 grid grid-cols-1 sm:grid-cols-2 gap-4">
                            {[
                                { name: "YOLO11", meta: "Ultralytics Engine", icon: ScanLine },
                                { name: "ONNX Runtime", meta: "Microsoft AI Core", icon: Cpu },
                                { name: "Next.js 15", meta: "App Router / RSC", icon: Box },
                                { name: "WebGPU", meta: "W3C Next-gen Graphics", icon: Zap },
                            ].map((tech) => (
                                <div key={tech.name} className="p-6 bg-white border border-slate-200/60 rounded-2xl hover:border-teal-300 transition-colors group">
                                    <tech.icon className="w-6 h-6 text-slate-400 group-hover:text-teal-600 transition-colors mb-4" />
                                    <h4 className="font-black text-slate-900">{tech.name}</h4>
                                    <p className="text-xs text-slate-400 font-bold uppercase tracking-wider mt-1">{tech.meta}</p>
                                </div>
                            ))}
                        </div>
                    </div>
                </section>

                {/* ── QUANTIZATION ────────────────────────────────────────── */}
                <section id="quantization" className="scroll-mt-32">
                    <div className="bg-slate-900 rounded-[3rem] p-12 text-white overflow-hidden relative group">
                        <div className="absolute top-0 right-0 w-96 h-96 bg-teal-500/20 blur-[100px] rounded-full -translate-y-1/2 translate-x-1/2 group-hover:bg-teal-500/30 transition-all duration-700" />
                        <div className="relative z-10 grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
                            <div className="space-y-6">
                                <h2 className="text-4xl font-black tracking-tight leading-none">FP16 & INT8<br />Quantization</h2>
                                <p className="text-slate-400 text-lg leading-relaxed">
                                    To run efficiently at the edge, we utilize **Half-Precision (FP16)** calibration. 
                                    This reduces model size by **50%** while leveraging hardware-level bit-shifting 
                                    on modern GPUs via the `shader-f16` extension.
                                </p>
                                <div className="space-y-4 pt-4">
                                    <div className="flex items-center gap-3">
                                        <div className="w-5 h-5 rounded bg-teal-500 flex items-center justify-center text-[10px] font-bold">1/2</div>
                                        <span className="text-sm font-bold text-slate-300">50% Memory Reduction</span>
                                    </div>
                                    <div className="flex items-center gap-3">
                                        <div className="w-5 h-5 rounded bg-indigo-500 flex items-center justify-center text-[10px] font-bold">2X</div>
                                        <span className="text-sm font-bold text-slate-300">Throughput Acceleration on WebGPU</span>
                                    </div>
                                </div>
                            </div>
                            <div className="bg-white/5 border border-white/10 backdrop-blur-xl rounded-3xl p-8 space-y-6">
                                <h4 className="text-sm font-bold text-teal-400 uppercase tracking-widest">Precision Utility</h4>
                                <div className="font-mono text-[10px] text-slate-400 p-4 bg-black/40 rounded-xl overflow-x-auto whitespace-pre">
{`/** Bit-depth Transformation Utility **/
function encodeFloat16(val) {
   // IEEE 754 float32 to float16
   exponent = exponent - 127 + 15;
   return sign | (exponent << 10) | (mantissa >> 13);
}`}
                                </div>
                                <p className="text-xs text-slate-500 leading-relaxed italic">
                                    Our custom precision-agnostic engine ensures that your camera input 
                                    always matches the internal bit-depth of the selected model.
                                </p>
                            </div>
                        </div>
                    </div>
                </section>

                {/* ── ATTRIBUTION ─────────────────────────────────────────── */}
                <section id="attribution" className="scroll-mt-32">
                    <div className="max-w-4xl mx-auto text-center space-y-12">
                        <div className="space-y-4">
                            <h2 className="text-3xl font-black text-slate-900 uppercase tracking-[0.3em]">Attribution</h2>
                            <div className="w-20 h-1 bg-teal-600 mx-auto rounded-full" />
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-stretch text-left">
                            <div className="bg-white border border-slate-200 rounded-[2.5rem] p-8 space-y-4 flex flex-col justify-between">
                                <div className="space-y-4">
                                    <h3 className="text-xl font-black text-slate-900 flex items-center gap-2">
                                        Ultralytics
                                        <div className="text-[10px] bg-slate-100 px-2 py-0.5 rounded-md uppercase font-black text-slate-400">Engine API</div>
                                    </h3>
                                    <p className="text-sm text-slate-500 leading-relaxed">
                                        YOLOv11 is the world's most advanced vision AI. Special thanks to the **Ultralytics** team 
                                        for their open-source contribution to the computer vision community.
                                    </p>
                                </div>
                                <a 
                                    href="https://ultralytics.com" 
                                    target="_blank" 
                                    rel="noopener noreferrer"
                                    className="text-teal-600 text-xs font-bold flex items-center gap-1 hover:underline group"
                                >
                                    ultralytics.com <ExternalLink className="w-3 h-3 group-hover:translate-x-0.5 transition-transform" />
                                </a>
                            </div>

                            <div className="bg-white border border-slate-200 rounded-[2.5rem] p-8 space-y-4 flex flex-col justify-between">
                                <div className="space-y-4">
                                    <h3 className="text-xl font-black text-slate-900 flex items-center gap-2">
                                        ONNX Runtime
                                        <div className="text-[10px] bg-slate-100 px-2 py-0.5 rounded-md uppercase font-black text-slate-400">Microsoft</div>
                                    </h3>
                                    <p className="text-sm text-slate-500 leading-relaxed">
                                        Microsoft's ONNX Runtime is the backbone of our inference engine, 
                                        providing the high-performance WebGPU kernels that make this app possible.
                                    </p>
                                </div>
                                <a 
                                    href="https://onnxruntime.ai" 
                                    target="_blank" 
                                    rel="noopener noreferrer"
                                    className="text-teal-600 text-xs font-bold flex items-center gap-1 hover:underline group"
                                >
                                    onnxruntime.ai <ExternalLink className="w-3 h-3 group-hover:translate-x-0.5 transition-transform" />
                                </a>
                            </div>
                        </div>

                        {/* Author */}
                        <div className="pt-12">
                            <div className="inline-flex flex-col items-center gap-4">
                                <Image
                                    src="https://github.com/pranta-barua007.png"
                                    alt="Pranta Barua"
                                    width={80}
                                    height={80}
                                    className="rounded-3xl border-2 border-slate-100 p-1 shadow-lg"
                                />
                                <div className="space-y-1">
                                    <h4 className="font-black text-slate-900">Developed by Pranta Barua</h4>
                                    <p className="text-xs text-slate-400 font-bold uppercase tracking-wider italic">Edge Vision Specialist</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </section>

            </main>

            <footer className="border-t border-slate-200/60 bg-white py-12">
                <div className="max-w-6xl mx-auto px-6 flex flex-col sm:flex-row items-center justify-between gap-6">
                    <div className="flex items-center gap-3">
                        <div className="w-8 h-8 bg-slate-900 rounded-xl flex items-center justify-center text-white font-black text-xs shadow-lg shadow-slate-900/10 italic">Y</div>
                        <span className="text-sm text-slate-400 font-medium tracking-tight">© 2024 YOLO Edge Runner. No data tracking. All rights reserved.</span>
                    </div>
                    <div className="flex items-center gap-6">
                        <a href="https://github.com/pranta-barua007/yolo11-onnx" className="text-slate-400 hover:text-slate-900 transition-colors">
                            <GitBranch className="w-5 h-5" />
                        </a>
                    </div>
                </div>
            </footer>
        </div>
    );
}
