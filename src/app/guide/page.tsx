import { Metadata } from "next";
import Link from "next/link";
import {
    Monitor,
    Code2,
    Camera,
    SlidersHorizontal,
    ImageIcon,
    Download,
    Palette,
    GitBranch,
    Terminal,
    FolderTree,
    Layers,
    Plus,
    RefreshCcw,
    ChevronRight,
    Globe,
    Cpu,
    Eye,
    MousePointer2,
    Save,
} from "lucide-react";
import { Card } from "@/components/ui/card";

export const metadata: Metadata = {
    title: "Guide | YOLO Edge Runner",
    description:
        "Step-by-step guide for using the YOLO Edge Runner — for app users and developers.",
};

const NAV_SECTIONS = [
    { id: "app-user", label: "App User" },
    { id: "developer", label: "Developer" },
];

/* ── Step component ─────────────────────────────── */
function Step({
    step,
    icon: Icon,
    title,
    children,
}: {
    step: number;
    icon: React.ElementType;
    title: string;
    children: React.ReactNode;
}) {
    return (
        <div className="flex gap-5">
            <div className="flex flex-col items-center">
                <div className="w-10 h-10 rounded-full bg-primary/10 text-primary flex items-center justify-center text-sm font-bold ring-1 ring-primary/20 flex-shrink-0">
                    {step}
                </div>
                <div className="w-px flex-1 bg-border/50 mt-2" />
            </div>
            <div className="pb-12 min-w-0">
                <div className="flex items-center gap-2 mb-2">
                    <Icon className="w-4 h-4 text-primary flex-shrink-0" aria-hidden="true" />
                    <h3 className="text-base font-semibold text-foreground">{title}</h3>
                </div>
                <div className="text-sm text-muted-foreground leading-relaxed space-y-3">
                    {children}
                </div>
            </div>
        </div>
    );
}

/* ── KeyValue badge ─────────────────────────────── */
function KV({ label, value }: { label: string; value: string }) {
    return (
        <div className="inline-flex items-center gap-2 px-3 py-1.5 bg-muted rounded-lg text-xs">
            <span className="text-muted-foreground font-medium">{label}</span>
            <span className="text-foreground font-bold font-mono">{value}</span>
        </div>
    );
}


export default function GuidePage() {
    return (
        <div className="min-h-screen bg-background text-foreground font-sans selection:bg-primary/30 selection:text-primary">

            {/* Sticky Section Nav */}
            <div className="sticky top-[65px] z-20 bg-background/80 backdrop-blur-md border-b border-border/50">
                <div className="max-w-4xl mx-auto px-6">
                    <nav className="flex items-center gap-1 py-2.5" aria-label="Guide sections">
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

            <main className="max-w-4xl mx-auto px-6 py-16 space-y-24">

                {/* ── HERO ─────────────────────────────────── */}
                <section className="text-center space-y-4 max-w-2xl mx-auto">
                    <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/10 border border-primary/20 text-primary text-[10px] font-bold uppercase tracking-widest">
                        Getting Started
                    </div>
                    <h1 className="text-4xl sm:text-5xl font-bold text-foreground tracking-tight leading-[1.1] text-wrap-balance">
                        Guide
                    </h1>
                    <p className="text-base text-muted-foreground leading-relaxed max-w-lg mx-auto text-pretty">
                        Whether you want to run inference right now or build on top of the codebase — pick your path below.
                    </p>
                </section>

                {/* ── TAB CARDS (link to sections) ──────────── */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <a href="#app-user">
                        <Card className="group border-border/40 bg-card p-6 space-y-3 hover:border-primary/30 transition-colors duration-200 cursor-pointer h-full">
                            <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center text-primary group-hover:scale-105 transition-transform duration-200">
                                <Monitor className="w-6 h-6" aria-hidden="true" />
                            </div>
                            <h2 className="text-lg font-semibold text-foreground">App User</h2>
                            <p className="text-sm text-muted-foreground leading-relaxed">
                                Just want to use the app? Open it in your browser, pick a model, and start detecting objects in images or camera feeds.
                            </p>
                            <span className="inline-flex items-center gap-1 text-primary text-xs font-medium">
                                Jump to guide <ChevronRight className="w-3.5 h-3.5" aria-hidden="true" />
                            </span>
                        </Card>
                    </a>
                    <a href="#developer">
                        <Card className="group border-border/40 bg-card p-6 space-y-3 hover:border-primary/30 transition-colors duration-200 cursor-pointer h-full">
                            <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center text-primary group-hover:scale-105 transition-transform duration-200">
                                <Code2 className="w-6 h-6" aria-hidden="true" />
                            </div>
                            <h2 className="text-lg font-semibold text-foreground">Developer</h2>
                            <p className="text-sm text-muted-foreground leading-relaxed">
                                Clone the repo, run it locally, add your own ONNX models, or extend the pipeline with custom post-processing.
                            </p>
                            <span className="inline-flex items-center gap-1 text-primary text-xs font-medium">
                                Jump to guide <ChevronRight className="w-3.5 h-3.5" aria-hidden="true" />
                            </span>
                        </Card>
                    </a>
                </div>

                {/* ═══════════════════════════════════════════ */}
                {/* ── APP USER GUIDE ═════════════════════════ */}
                {/* ═══════════════════════════════════════════ */}
                <section id="app-user" className="scroll-mt-28 space-y-10">
                    <div className="space-y-3">
                        <div className="flex items-center gap-3">
                            <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
                                <Monitor className="w-5 h-5 text-primary" aria-hidden="true" />
                            </div>
                            <div>
                                <h2 className="text-2xl font-bold text-foreground tracking-tight">
                                    App User Guide
                                </h2>
                                <p className="text-sm text-muted-foreground">
                                    No installation needed — everything runs in the browser.
                                </p>
                            </div>
                        </div>
                    </div>

                    {/* Prerequisites */}
                    <Card className="border-border/40 bg-card p-6 space-y-3">
                        <h3 className="text-sm font-semibold text-foreground">Requirements</h3>
                        <div className="flex flex-wrap gap-2">
                            <KV label="Browser" value="Chrome 113+ / Edge 113+" />
                            <KV label="Best With" value="WebGPU-capable GPU" />
                            <KV label="Fallback" value="WASM (any modern browser)" />
                        </div>
                        <p className="text-xs text-muted-foreground">
                            WebGPU delivers the best performance. If your browser or hardware doesn&apos;t support it,
                            the app automatically falls back to multi-threaded WASM.
                        </p>
                    </Card>

                    {/* Steps */}
                    <div>
                        <Step step={1} icon={Globe} title="Open the App">
                            <p>
                                Navigate to the <Link href="/" className="text-primary font-medium hover:underline underline-offset-4">Workspace</Link> page.
                                The app will instantly begin loading the default model (<code className="text-xs bg-muted px-1.5 py-0.5 rounded font-mono">yolo11n-seg</code>).
                            </p>
                            <p>
                                First load downloads ~13MB. After that, the model is cached in your browser&apos;s IndexedDB — no re-download needed.
                            </p>
                        </Step>

                        <Step step={2} icon={Cpu} title="Select Model & Device">
                            <p>
                                Use the toolbar at the top of the workspace to pick your model and execution provider:
                            </p>
                            <div className="flex flex-wrap gap-2 mt-2">
                                <KV label="Model" value="YOLO11 Nano (D+S)" />
                                <KV label="Model" value="YOLO11 Nano Pose (D+P)" />
                                <KV label="Model" value="YOLO11 Small (D+Q)" />
                            </div>
                            <div className="flex flex-wrap gap-2 mt-2">
                                <KV label="Device" value="WebGPU" />
                                <KV label="Device" value="WASM" />
                            </div>
                        </Step>

                        <Step step={3} icon={ImageIcon} title="Upload an Image">
                            <p>
                                Click <strong>Upload Image</strong> or drag-and-drop any image onto the workspace.
                                You can also click one of the example thumbnails at the bottom to try a pre-loaded sample.
                            </p>
                            <p>
                                Inference runs automatically once the image is loaded. You&apos;ll see bounding boxes, segmentation masks,
                                or pose skeletons drawn over the image depending on the active model.
                            </p>
                        </Step>

                        <Step step={4} icon={Camera} title="Use Your Camera">
                            <p>
                                Click <strong>Open Camera</strong> to start real-time detection from your webcam.
                                Grant camera permissions when prompted. The app runs inference continuously at up to 60 FPS.
                            </p>
                            <p>
                                If you have multiple cameras (e.g., front and back), use the camera selector dropdown to switch between them.
                            </p>
                        </Step>

                        <Step step={5} icon={SlidersHorizontal} title="Adjust Confidence">
                            <p>
                                The <strong>CONF.</strong> slider in the toolbar controls the minimum confidence threshold.
                                Drag it to filter out low-confidence detections. Default is <strong>55%</strong>.
                            </p>
                        </Step>

                        <Step step={6} icon={Eye} title="Inspect Detections">
                            <p>
                                The sidebar panel lists all detected objects with their class name and confidence score.
                                Click any detection to highlight it on the canvas. For pose models, clicking a person
                                reveals their individual keypoints.
                            </p>
                        </Step>

                        <Step step={7} icon={Save} title="Save Results">
                            <p>
                                Click <strong>Save</strong> in the sidebar to download the annotated image
                                with all bounding boxes, masks, and labels burned in.
                            </p>
                        </Step>

                        <Step step={8} icon={Plus} title="Add Custom Models">
                            <p>
                                Click the <strong>+</strong> button next to the model selector to add your own ONNX model.
                                Provide a URL to any publicly hosted <code className="text-xs bg-muted px-1.5 py-0.5 rounded font-mono">.onnx</code> file,
                                select the task type (Detection, Segmentation, or Pose), and tag its precision.
                            </p>
                        </Step>
                    </div>

                    {/* Tips */}
                    <Card className="border-primary/20 bg-primary/5 p-6 space-y-3">
                        <h3 className="text-sm font-semibold text-primary">Tips for Best Performance</h3>
                        <ul className="text-sm text-muted-foreground space-y-2 list-disc list-inside">
                            <li>Use <strong>Chrome</strong> or <strong>Edge</strong> for WebGPU support — Firefox and Safari use WASM fallback.</li>
                            <li>Close other GPU-intensive tabs (games, video editors) for smoother inference.</li>
                            <li>The first inference after load is a &quot;warm-up&quot; and may be slower. Subsequent frames are faster.</li>
                            <li>If the model feels slow, switch to <code className="text-xs bg-muted px-1.5 py-0.5 rounded font-mono">WASM</code> — some integrated GPUs perform better on CPU.</li>
                        </ul>
                    </Card>
                </section>


                {/* ═══════════════════════════════════════════ */}
                {/* ── DEVELOPER GUIDE ════════════════════════ */}
                {/* ═══════════════════════════════════════════ */}
                <section id="developer" className="scroll-mt-28 space-y-10">
                    <div className="space-y-3">
                        <div className="flex items-center gap-3">
                            <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
                                <Code2 className="w-5 h-5 text-primary" aria-hidden="true" />
                            </div>
                            <div>
                                <h2 className="text-2xl font-bold text-foreground tracking-tight">
                                    Developer Guide
                                </h2>
                                <p className="text-sm text-muted-foreground">
                                    Clone, build, extend, or bring your own model.
                                </p>
                            </div>
                        </div>
                    </div>

                    {/* Prerequisites */}
                    <Card className="border-border/40 bg-card p-6 space-y-3">
                        <h3 className="text-sm font-semibold text-foreground">Prerequisites</h3>
                        <div className="flex flex-wrap gap-2">
                            <KV label="Node.js" value="≥ 18" />
                            <KV label="pnpm" value="≥ 9" />
                            <KV label="Framework" value="Next.js 16" />
                        </div>
                    </Card>

                    {/* Steps */}
                    <div>
                        <Step step={1} icon={GitBranch} title="Clone the Repository">
                            <div className="bg-muted/50 rounded-xl p-4 font-mono text-xs overflow-x-auto border border-border/40">
                                <pre className="text-foreground">{`git clone https://github.com/pranta-barua007/yolo11-onnx.git
cd yolo11-onnx`}</pre>
                            </div>
                        </Step>

                        <Step step={2} icon={Download} title="Install Dependencies">
                            <div className="bg-muted/50 rounded-xl p-4 font-mono text-xs overflow-x-auto border border-border/40">
                                <pre className="text-foreground">pnpm install</pre>
                            </div>
                        </Step>

                        <Step step={3} icon={Terminal} title="Start Development Server">
                            <div className="bg-muted/50 rounded-xl p-4 font-mono text-xs overflow-x-auto border border-border/40">
                                <pre className="text-foreground">pnpm dev</pre>
                            </div>
                            <p>
                                Open <code className="text-xs bg-muted px-1.5 py-0.5 rounded font-mono">http://localhost:3000</code> in
                                a WebGPU-capable browser. The app hot-reloads on file changes.
                            </p>
                        </Step>

                        <Step step={4} icon={FolderTree} title="Project Structure">
                            <div className="bg-muted/50 rounded-xl p-4 font-mono text-[11px] overflow-x-auto border border-border/40 leading-relaxed">
                                <pre className="text-foreground">{`src/
├── app/              # Next.js App Router pages
│   ├── page.tsx      # Workspace (main inference UI)
│   ├── about/        # Technical overview
│   └── guide/        # This guide
├── components/       # UI components
│   ├── Header.tsx    # Global navigation
│   ├── MediaDisplay/ # Image/camera display
│   └── ModelStatus/  # Detection sidebar
├── hooks/            # React hooks
│   ├── useYoloModel  # Model lifecycle
│   ├── useCamera     # Camera stream
│   └── useFps        # FPS counter
├── workers/          # Web Worker pipeline
│   └── workerPipeline.ts
└── utils/            # Inference utilities
    ├── img_preprocess.ts
    ├── mask_processing.ts
    └── draw_bounding_boxes.ts`}</pre>
                            </div>
                        </Step>

                        <Step step={5} icon={Layers} title="Add Your Own ONNX Model">
                            <p>
                                Place your <code className="text-xs bg-muted px-1.5 py-0.5 rounded font-mono">.onnx</code> model
                                in the <code className="text-xs bg-muted px-1.5 py-0.5 rounded font-mono">public/models/</code> directory.
                                The model must follow the standard YOLO output format:
                            </p>
                            <ul className="list-disc list-inside space-y-1 mt-2">
                                <li><strong>Detection:</strong> Output shape <code className="text-xs bg-muted px-1.5 py-0.5 rounded font-mono">[1, 4+C, N]</code> where C = classes, N = proposals</li>
                                <li><strong>Segmentation:</strong> Two outputs — detections + prototype masks <code className="text-xs bg-muted px-1.5 py-0.5 rounded font-mono">[1, 32, H, W]</code></li>
                                <li><strong>Pose:</strong> Output shape includes 17 × 3 keypoint channels</li>
                            </ul>
                            <p className="mt-2">
                                You can add models at runtime via the <strong>+ button</strong> in the UI.
                                For built-in models, add a new <code className="text-xs bg-muted px-1.5 py-0.5 rounded font-mono">SelectItem</code> in{" "}
                                <code className="text-xs bg-muted px-1.5 py-0.5 rounded font-mono">StatusBar.tsx</code>.
                            </p>
                        </Step>

                        <Step step={6} icon={RefreshCcw} title="Build for Production">
                            <div className="bg-muted/50 rounded-xl p-4 font-mono text-xs overflow-x-auto border border-border/40">
                                <pre className="text-foreground">{`pnpm build
# Static output → out/ (configured for GitHub Pages)`}</pre>
                            </div>
                            <p>
                                The project is configured with <code className="text-xs bg-muted px-1.5 py-0.5 rounded font-mono">output: &quot;export&quot;</code> for
                                static hosting. Deploy the <code className="text-xs bg-muted px-1.5 py-0.5 rounded font-mono">out/</code> directory
                                to any static host (GitHub Pages, Vercel, Netlify, S3).
                            </p>
                        </Step>

                        <Step step={7} icon={Palette} title="Customize the UI">
                            <p>
                                The design system uses <strong>Tailwind CSS</strong> with shadcn-style tokens.
                                Theme colors are defined in <code className="text-xs bg-muted px-1.5 py-0.5 rounded font-mono">globals.css</code> as
                                CSS custom properties. Both light and dark modes are supported via <code className="text-xs bg-muted px-1.5 py-0.5 rounded font-mono">next-themes</code>.
                            </p>
                            <p>
                                To change the primary accent, update the <code className="text-xs bg-muted px-1.5 py-0.5 rounded font-mono">--primary</code> HSL
                                values in your CSS.
                            </p>
                        </Step>

                        <Step step={8} icon={MousePointer2} title="Extend the Pipeline">
                            <p>
                                The inference pipeline lives in <code className="text-xs bg-muted px-1.5 py-0.5 rounded font-mono">src/workers/workerPipeline.ts</code>.
                                All model loading, pre-processing, inference, and post-processing happens inside this Web Worker.
                            </p>
                            <p>
                                To add custom post-processing (e.g., counting objects, tracking across frames),
                                modify the worker&apos;s message handler or add new utility functions in <code className="text-xs bg-muted px-1.5 py-0.5 rounded font-mono">src/utils/</code>.
                            </p>
                        </Step>
                    </div>

                    {/* Common Issues */}
                    <Card className="border-amber-500/20 bg-amber-500/5 p-6 space-y-3">
                        <h3 className="text-sm font-semibold text-amber-600 dark:text-amber-400">Common Issues</h3>
                        <ul className="text-sm text-muted-foreground space-y-3">
                            <li>
                                <strong className="text-foreground">Model cache stale?</strong>{" "}
                                Clear the <code className="text-xs bg-muted px-1.5 py-0.5 rounded font-mono">yolo-model-cache</code> in
                                DevTools → Application → Cache Storage. The app caches models in IndexedDB and doesn&apos;t auto-invalidate.
                            </li>
                            <li>
                                <strong className="text-foreground">WebGPU not available?</strong>{" "}
                                Check <code className="text-xs bg-muted px-1.5 py-0.5 rounded font-mono">chrome://flags/#enable-unsafe-webgpu</code>.
                                On Linux, you may also need <code className="text-xs bg-muted px-1.5 py-0.5 rounded font-mono">--enable-features=Vulkan</code>.
                            </li>
                            <li>
                                <strong className="text-foreground">Camera permission denied?</strong>{" "}
                                Ensure you&apos;re on <code className="text-xs bg-muted px-1.5 py-0.5 rounded font-mono">localhost</code> or HTTPS.
                                Browsers block camera access on insecure origins.
                            </li>
                        </ul>
                    </Card>
                </section>

                {/* ── CTA ───────────────────────────────────── */}
                <div className="text-center space-y-4 pb-8">
                    <p className="text-muted-foreground text-sm">Ready to start?</p>
                    <div className="flex items-center justify-center gap-3">
                        <Link
                            href="/"
                            className="h-10 px-6 bg-primary hover:bg-primary/90 text-primary-foreground text-sm font-medium rounded-full inline-flex items-center gap-2 transition-colors duration-200 shadow-sm ring-1 ring-primary/20 active:scale-[0.98]"
                        >
                            Open Workspace <ChevronRight className="w-4 h-4" aria-hidden="true" />
                        </Link>
                        <a
                            href="https://github.com/pranta-barua007/yolo11-onnx"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="h-10 px-6 bg-secondary hover:bg-secondary/80 text-secondary-foreground text-sm font-medium rounded-full inline-flex items-center gap-2 transition-colors duration-200 active:scale-[0.98]"
                        >
                            <GitBranch className="w-4 h-4" aria-hidden="true" /> Source Code
                        </a>
                    </div>
                </div>

            </main>
        </div>
    );
}
