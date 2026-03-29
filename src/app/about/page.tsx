import { Metadata } from "next";
import Link from "next/link";
import Image from "next/image";
import {
    Cpu,
    Zap,
    ScanLine,
    GitBranch,
    ChevronRight,
    Layers,
    ExternalLink,
    Box,
    Terminal,
    Eye,
    PersonStanding,
} from "lucide-react";
import ArchitectureDiagram from "@/components/ArchitectureDiagram";
import { Card } from "@/components/ui/card";

export const metadata: Metadata = {
    title: "About | YOLO Edge Runner",
    description:
        "Technical overview of the YOLO Edge Runner — a high-performance, browser-native AI tool powered by YOLO11, ONNX Runtime Web, and WebGPU.",
};

const NAV_SECTIONS = [
    { id: "overview", label: "Overview" },
    { id: "tasks", label: "Tasks" },
    { id: "architecture", label: "Architecture" },
    { id: "tech", label: "Technology" },
    { id: "quantization", label: "Quantization" },
    { id: "attribution", label: "Attribution" },
];

const STATS = [
    { value: "~290ms", label: "Inference (GPU)", color: "text-primary" },
    { value: "0ms", label: "Server Latency", color: "text-emerald-500 dark:text-emerald-400" },
    { value: "60fps", label: "UI Refresh", color: "text-muted-foreground" },
];

const TASKS = [
    {
        icon: ScanLine,
        name: "Object Detection",
        tag: "D",
        tagColor: "text-emerald-500 bg-emerald-500/10 border-emerald-500/20",
        description:
            "Identify and locate objects in images with bounding boxes and confidence scores. Supports 80 COCO classes out of the box.",
    },
    {
        icon: Eye,
        name: "Instance Segmentation",
        tag: "S",
        tagColor: "text-blue-500 bg-blue-500/10 border-blue-500/20",
        description:
            "Pixel-level masks for each detected object. Separate overlapping instances with distinct color-coded regions.",
    },
    {
        icon: PersonStanding,
        name: "Pose Estimation",
        tag: "P",
        tagColor: "text-purple-500 bg-purple-500/10 border-purple-500/20",
        description:
            "17-keypoint skeleton detection per person. Tracks body joints including eyes, shoulders, elbows, wrists, hips, knees, and ankles.",
    },
];

const TECH_STACK = [
    { icon: ScanLine, name: "YOLO11", meta: "Ultralytics Engine" },
    { icon: Cpu, name: "ONNX Runtime", meta: "Microsoft AI Core" },
    { icon: Box, name: "Next.js 16", meta: "App Router / RSC" },
    { icon: Zap, name: "WebGPU", meta: "W3C Next-Gen Graphics" },
];


export default function AboutPage() {
    return (
        <div className="min-h-screen bg-background text-foreground font-sans selection:bg-primary/30 selection:text-primary">

            {/* Sticky Section Nav */}
            <div className="sticky top-[65px] z-20 bg-background/80 backdrop-blur-md border-b border-border/50">
                <div className="max-w-5xl mx-auto px-6 overflow-x-auto overflow-y-hidden">
                    <nav className="flex items-center gap-1 py-2.5" aria-label="Page sections">
                        {NAV_SECTIONS.map((s) => (
                            <a
                                key={s.id}
                                href={`#${s.id}`}
                                className="flex-shrink-0 px-3 py-1.5 text-xs font-medium rounded-full text-muted-foreground hover:text-foreground hover:bg-foreground/5 transition-colors duration-200"
                            >
                                {s.label}
                            </a>
                        ))}
                    </nav>
                </div>
            </div>

            <main className="max-w-5xl mx-auto px-6 py-16 space-y-24">

                {/* ── HERO ─────────────────────────────────────── */}
                <section id="overview" className="scroll-mt-28 space-y-16">
                    <div className="text-center space-y-6 max-w-2xl mx-auto">
                        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/10 border border-primary/20 text-primary text-[10px] font-bold uppercase tracking-widest">
                            Edge Computing
                        </div>
                        <h1 className="text-4xl sm:text-5xl md:text-6xl font-bold text-foreground tracking-tight leading-[1.1] text-wrap-balance">
                            Real-Time AI,{" "}
                            <span className="text-gradient">In Your Browser</span>
                        </h1>
                        <p className="text-base md:text-lg text-muted-foreground leading-relaxed max-w-xl mx-auto text-pretty">
                            YOLO Edge Runner brings YOLO11 object detection, segmentation, and pose estimation to the browser.
                            WebGPU-accelerated inference with zero server overhead — your data never leaves the device.
                        </p>

                        <div className="flex items-center justify-center gap-3 pt-2">
                            <Link
                                href="/"
                                className="h-10 px-6 bg-primary hover:bg-primary/90 text-primary-foreground text-sm font-medium rounded-full inline-flex items-center gap-2 transition-colors duration-200 shadow-sm ring-1 ring-primary/20 active:scale-[0.98]"
                            >
                                Open App <ChevronRight className="w-4 h-4" aria-hidden="true" />
                            </Link>
                            <a
                                href="https://github.com/pranta-barua007/yolo11-onnx"
                                target="_blank"
                                rel="noopener noreferrer"
                                className="h-10 px-6 bg-secondary hover:bg-secondary/80 text-secondary-foreground text-sm font-medium rounded-full inline-flex items-center gap-2 transition-colors duration-200 active:scale-[0.98]"
                            >
                                <GitBranch className="w-4 h-4" aria-hidden="true" /> Source
                            </a>
                        </div>
                    </div>

                    {/* Stats Row */}
                    <Card className="border-border/40 bg-card/50 backdrop-blur-sm p-0 overflow-hidden">
                        <div className="grid grid-cols-3 divide-x divide-border/40">
                            {STATS.map((stat) => (
                                <div key={stat.label} className="flex flex-col items-center gap-1 py-6 px-4">
                                    <span className={`text-2xl sm:text-3xl md:text-4xl font-bold tracking-tight tabular-nums ${stat.color}`}>
                                        {stat.value}
                                    </span>
                                    <span className="text-[10px] sm:text-xs font-medium text-muted-foreground uppercase tracking-wider">
                                        {stat.label}
                                    </span>
                                </div>
                            ))}
                        </div>
                    </Card>
                </section>

                {/* ── TASKS ────────────────────────────────────── */}
                <section id="tasks" className="scroll-mt-28 space-y-8">
                    <div className="space-y-3">
                        <h2 className="text-2xl font-bold text-foreground tracking-tight text-wrap-balance">
                            Supported Tasks
                        </h2>
                        <p className="text-muted-foreground max-w-lg">
                            Three computer vision tasks, all running entirely in-browser with a single unified pipeline.
                        </p>
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                        {TASKS.map((task) => (
                            <Card
                                key={task.name}
                                className="group border-border/40 bg-card p-6 space-y-4 hover:border-primary/30 transition-colors duration-200"
                            >
                                <div className="flex items-center justify-between">
                                    <div className="w-10 h-10 rounded-lg bg-muted flex items-center justify-center text-muted-foreground group-hover:bg-primary/10 group-hover:text-primary transition-colors duration-200">
                                        <task.icon className="w-5 h-5" aria-hidden="true" />
                                    </div>
                                    <span className={`px-1.5 py-0.5 rounded text-[9px] font-bold leading-none border ${task.tagColor}`}>
                                        {task.tag}
                                    </span>
                                </div>
                                <div className="space-y-1.5">
                                    <h3 className="text-sm font-semibold text-foreground">{task.name}</h3>
                                    <p className="text-xs text-muted-foreground leading-relaxed">
                                        {task.description}
                                    </p>
                                </div>
                            </Card>
                        ))}
                    </div>
                </section>

                {/* ── ARCHITECTURE DIAGRAM (free-standing) ──────── */}
                <section id="architecture" className="scroll-mt-28 space-y-16">
                    <div className="flex flex-col items-center gap-6">
                        <h2 className="text-xs font-bold uppercase tracking-widest text-muted-foreground">
                            System Architecture
                        </h2>
                        <ArchitectureDiagram className="w-full h-auto max-w-4xl mx-auto" />
                    </div>

                    {/* Architecture Deep Dive */}
                    <div className="space-y-8">
                        <div className="space-y-3">
                            <h2 className="text-2xl font-bold text-foreground tracking-tight text-wrap-balance">
                                Non-Blocking Worker Pattern
                            </h2>
                            <p className="text-muted-foreground leading-relaxed max-w-2xl">
                                To maintain a smooth 60 FPS UI, inference runs off the main thread.
                                A dedicated Web Worker handles the entire ONNX lifecycle independently.
                            </p>
                        </div>

                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                            <Card className="border-border/40 bg-card p-6 flex gap-4">
                                <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0">
                                    <Terminal className="w-5 h-5 text-primary" aria-hidden="true" />
                                </div>
                                <div className="space-y-1 min-w-0">
                                    <h3 className="text-sm font-semibold text-foreground">Execution Provider (EP)</h3>
                                    <p className="text-sm text-muted-foreground leading-relaxed">
                                        The engine benchmarks your hardware to select the best provider:
                                        {" "}<code className="text-xs bg-muted px-1.5 py-0.5 rounded font-mono">WebGPU</code> for
                                        modern GPUs, or multi-threaded
                                        {" "}<code className="text-xs bg-muted px-1.5 py-0.5 rounded font-mono">WASM</code> for cross-compatibility.
                                    </p>
                                </div>
                            </Card>

                            <Card className="border-border/40 bg-card p-6 flex gap-4">
                                <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0">
                                    <Layers className="w-5 h-5 text-primary" aria-hidden="true" />
                                </div>
                                <div className="space-y-1 min-w-0">
                                    <h3 className="text-sm font-semibold text-foreground">Memory Management</h3>
                                    <p className="text-sm text-muted-foreground leading-relaxed">
                                        <strong>Transferable Objects</strong> move pixel data between
                                        threads with zero-copy overhead, ensuring maximum throughput
                                        for high-resolution cameras.
                                    </p>
                                </div>
                            </Card>
                        </div>
                    </div>
                </section>

                {/* ── TECH STACK ────────────────────────────────── */}
                <section id="tech" className="scroll-mt-28 space-y-8">
                    <div className="space-y-3">
                        <h2 className="text-2xl font-bold text-foreground tracking-tight text-wrap-balance">
                            Technology
                        </h2>
                        <p className="text-muted-foreground max-w-lg">
                            Built on the bleeding edge of the Open Web platform.
                        </p>
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                        {TECH_STACK.map((tech) => (
                            <Card
                                key={tech.name}
                                className="group border-border/40 bg-card p-5 flex items-center gap-4 hover:border-primary/30 transition-colors duration-200"
                            >
                                <div className="w-10 h-10 rounded-lg bg-muted flex items-center justify-center text-muted-foreground group-hover:bg-primary/10 group-hover:text-primary transition-colors duration-200">
                                    <tech.icon className="w-5 h-5" aria-hidden="true" />
                                </div>
                                <div className="min-w-0">
                                    <div className="text-sm font-semibold text-foreground">{tech.name}</div>
                                    <div className="text-xs text-muted-foreground">{tech.meta}</div>
                                </div>
                            </Card>
                        ))}
                    </div>
                </section>

                {/* ── QUANTIZATION ──────────────────────────────── */}
                <section id="quantization" className="scroll-mt-28">
                    <Card className="border-border/40 bg-card overflow-hidden">
                        <div className="grid grid-cols-1 lg:grid-cols-2">
                            {/* Text Side */}
                            <div className="p-8 md:p-10 space-y-6">
                                <h2 className="text-2xl font-bold text-foreground tracking-tight text-wrap-balance">
                                    FP16 &amp; INT8 Quantization
                                </h2>
                                <p className="text-muted-foreground leading-relaxed">
                                    To run efficiently at the edge, we use <strong>Half-Precision (FP16)</strong> calibration.
                                    This reduces model size by <strong>50%</strong> while leveraging hardware-level
                                    bit-shifting on modern GPUs via the{" "}
                                    <code className="text-xs bg-muted px-1.5 py-0.5 rounded font-mono">shader-f16</code> extension.
                                </p>
                                <div className="flex flex-wrap gap-2">
                                    <span className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-primary/10 text-primary text-xs font-medium rounded-full">
                                        <span className="w-4 h-4 rounded bg-primary/20 flex items-center justify-center text-[9px] font-bold" aria-hidden="true">½</span>
                                        50% Memory Reduction
                                    </span>
                                    <span className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-muted text-muted-foreground text-xs font-medium rounded-full">
                                        <Zap className="w-3.5 h-3.5" aria-hidden="true" />
                                        WebGPU Throughput Acceleration
                                    </span>
                                </div>
                            </div>

                            {/* Code Side */}
                            <div className="bg-muted/50 dark:bg-muted/30 border-t lg:border-t-0 lg:border-l border-border/40 p-8 md:p-10 space-y-4">
                                <div className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
                                    Precision Utility
                                </div>
                                <div className="bg-background rounded-xl p-5 font-mono text-xs leading-relaxed border border-border/40 overflow-x-auto">
                                    <pre className="text-primary">
{`/** Bit-depth Transformation Utility */
function encodeFloat16(val) {
  // IEEE 754 float32 → float16
  exponent = exponent - 127 + 15;
  return sign | (exponent << 10)
    | (mantissa >> 13);
}`}
                                    </pre>
                                </div>
                            </div>
                        </div>
                    </Card>
                </section>

                {/* ── ATTRIBUTION ───────────────────────────────── */}
                <section id="attribution" className="scroll-mt-28 space-y-8">
                    <div className="space-y-3 text-center">
                        <h2 className="text-2xl font-bold text-foreground tracking-tight">
                            Attribution
                        </h2>
                        <p className="text-muted-foreground text-sm">
                            Built with open-source tools from incredible teams.
                        </p>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <Card className="border-border/40 bg-card p-6 space-y-4 hover:border-primary/30 transition-colors duration-200">
                            <div className="flex items-center justify-between">
                                <h3 className="text-base font-semibold text-foreground">Ultralytics</h3>
                                <span className="text-[10px] font-bold uppercase text-primary bg-primary/10 px-2 py-0.5 rounded-full tracking-wider">
                                    Engine
                                </span>
                            </div>
                            <p className="text-sm text-muted-foreground leading-relaxed">
                                YOLO11 is an object detection, segmentation, and pose estimation model.
                                Thanks to the <strong>Ultralytics</strong> team for their open-source contribution to the computer vision community.
                            </p>
                            <a
                                href="https://ultralytics.com"
                                target="_blank"
                                rel="noopener noreferrer"
                                className="inline-flex items-center gap-1.5 text-primary text-xs font-medium hover:underline decoration-primary/50 underline-offset-4"
                            >
                                ultralytics.com <ExternalLink className="w-3 h-3" aria-hidden="true" />
                            </a>
                        </Card>

                        <Card className="border-border/40 bg-card p-6 space-y-4 hover:border-primary/30 transition-colors duration-200">
                            <div className="flex items-center justify-between">
                                <h3 className="text-base font-semibold text-foreground">ONNX Runtime</h3>
                                <span className="text-[10px] font-bold uppercase text-muted-foreground bg-muted px-2 py-0.5 rounded-full tracking-wider">
                                    Microsoft
                                </span>
                            </div>
                            <p className="text-sm text-muted-foreground leading-relaxed">
                                Microsoft&apos;s ONNX Runtime provides the high-performance
                                WebGPU kernels that power inference in this application.
                            </p>
                            <a
                                href="https://onnxruntime.ai"
                                target="_blank"
                                rel="noopener noreferrer"
                                className="inline-flex items-center gap-1.5 text-primary text-xs font-medium hover:underline decoration-primary/50 underline-offset-4"
                            >
                                onnxruntime.ai <ExternalLink className="w-3 h-3" aria-hidden="true" />
                            </a>
                        </Card>

                        <Card className="border-border/40 bg-card p-6 space-y-4 hover:border-primary/30 transition-colors duration-200">
                            <div className="flex items-center justify-between">
                                <h3 className="text-base font-semibold text-foreground text-pretty pr-2">Multi-Task Web</h3>
                                <span className="text-[10px] font-bold uppercase text-emerald-500 bg-emerald-500/10 px-2 py-0.5 rounded-full tracking-wider">
                                    Reference
                                </span>
                            </div>
                            <p className="text-sm text-muted-foreground leading-relaxed">
                                Huge credits to <strong>nomi30701</strong> for their fantastic repository which served as a crucial foundation and reference for the multi-task web implementation.
                            </p>
                            <a
                                href="https://github.com/nomi30701/yolo-multi-task-onnxruntime-web"
                                target="_blank"
                                rel="noopener noreferrer"
                                className="inline-flex items-center gap-1.5 text-primary text-xs font-medium hover:underline decoration-primary/50 underline-offset-4"
                            >
                                GitHub <ExternalLink className="w-3 h-3" aria-hidden="true" />
                            </a>
                        </Card>
                    </div>
                </section>

                {/* ── FOOTER ───────────────────────────────────── */}
                <div className="pt-8 pb-4 flex flex-col items-center gap-3 text-center border-t border-border/30">
                    <a
                        href="https://github.com/pranta-barua007"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex flex-col items-center gap-3 group"
                    >
                        <Image
                            src="https://github.com/pranta-barua007.png"
                            alt="Pranta Barua"
                            width={48}
                            height={48}
                            className="rounded-full border-2 border-border/40 group-hover:border-primary/30 transition-colors duration-200"
                        />
                        <span className="text-xs font-medium text-muted-foreground group-hover:text-foreground transition-colors duration-200">
                            Developed by Pranta Barua
                        </span>
                    </a>
                </div>

            </main>
        </div>
    );
}
