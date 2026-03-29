"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Info, BookOpen, Zap } from "lucide-react";
import { ThemeToggle } from "./ThemeToggle";
import { useState, useRef, useEffect } from "react";

export default function Header() {
    const rawPathname = usePathname();
    const pathname = rawPathname === "/" ? "/" : rawPathname.replace(/\/+$/, "");
    const isAppsRoute = pathname.startsWith("/apps");

    const [appsOpen, setAppsOpen] = useState(false);
    const appsRef = useRef<HTMLDivElement>(null);

    // Close dropdown on outside click
    useEffect(() => {
        const handleClick = (e: MouseEvent) => {
            if (appsRef.current && !appsRef.current.contains(e.target as Node)) {
                setAppsOpen(false);
            }
        };
        document.addEventListener("mousedown", handleClick);
        return () => document.removeEventListener("mousedown", handleClick);
    }, []);

    return (
        <header className="sticky top-0 z-30 w-full px-3 md:px-6 py-2.5 border-b border-border/50 bg-background/80 backdrop-blur-md">
            <div className="flex items-center justify-between mx-auto w-full">
                <div className="flex items-center gap-2 sm:gap-4">
                    {/* Brand */}
                    <Link href="/" className="flex items-center gap-3 active:scale-95 transition-transform duration-200">
                        <div className="w-10 h-10 bg-primary rounded-full flex items-center justify-center text-primary-foreground shadow-sm ring-1 ring-primary/20">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round" className="w-6 h-6">
                                <path d="M5 3l7 9 7-9M12 12v9" />
                            </svg>
                        </div>
                        <span className="hidden sm:flex text-lg font-bold text-foreground tracking-tight items-baseline">
                            YOLO
                            <span className="text-primary ml-1.5 font-bold">Real-time</span>
                        </span>
                    </Link>

                    {/* Nav Links */}
                    <nav className="flex items-center gap-1.5">
                        <Link
                            href="/"
                            className={`hidden sm:inline-flex px-4 py-2 text-sm font-medium rounded-full transition-colors duration-200 ${pathname === "/"
                                ? "bg-primary/10 text-primary ring-1 ring-primary/30"
                                : "text-muted-foreground hover:text-foreground hover:bg-foreground/5"
                                }`}
                        >
                            Workspace
                        </Link>
                        <Link
                            href="/about"
                            className={`px-3 sm:px-4 py-1.5 sm:py-2 text-sm font-medium rounded-full transition-colors duration-200 flex items-center gap-1.5 ${pathname === "/about"
                                ? "bg-primary/10 text-primary ring-1 ring-primary/30"
                                : "text-muted-foreground hover:text-foreground hover:bg-foreground/5"
                                }`}
                        >
                            <Info className="w-4 h-4" aria-hidden="true" />
                            About
                        </Link>
                        <Link
                            href="/guide"
                            className={`px-3 sm:px-4 py-1.5 sm:py-2 text-sm font-medium rounded-full transition-colors duration-200 flex items-center gap-1.5 ${pathname === "/guide"
                                ? "bg-primary/10 text-primary ring-1 ring-primary/30"
                                : "text-muted-foreground hover:text-foreground hover:bg-foreground/5"
                                }`}
                        >
                            <BookOpen className="w-4 h-4" aria-hidden="true" />
                            Guide
                        </Link>

                        {/* Apps dropdown */}
                        <div ref={appsRef} className="relative">
                            <button
                                onClick={() => setAppsOpen((prev) => !prev)}
                                className={`px-3 sm:px-4 py-1.5 sm:py-2 text-sm font-medium rounded-full transition-colors duration-200 flex items-center gap-1.5 ${isAppsRoute
                                    ? "bg-primary/10 text-primary ring-1 ring-primary/30"
                                    : "text-muted-foreground hover:text-foreground hover:bg-foreground/5"
                                    }`}
                            >
                                <Zap className="w-4 h-4" aria-hidden="true" />
                                Apps
                                <svg className={`w-3 h-3 transition-transform duration-200 ${appsOpen ? "rotate-180" : ""}`} viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                    <path d="M3 5l3 3 3-3" />
                                </svg>
                            </button>

                            {appsOpen && (
                                <div className="absolute top-full left-0 mt-1.5 w-52 rounded-xl border border-border/50 bg-popover/95 backdrop-blur-md shadow-lg p-1.5 z-50">
                                    <Link
                                        href="/apps/formcheck"
                                        onClick={() => setAppsOpen(false)}
                                        className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors duration-150 ${pathname === "/apps/formcheck"
                                            ? "bg-primary/10 text-primary"
                                            : "text-foreground hover:bg-muted/50"
                                            }`}
                                    >
                                        <span className="text-lg" aria-hidden="true">🏋️</span>
                                        <div>
                                            <span className="font-medium">FormCheck AI</span>
                                            <p className="text-[10px] text-muted-foreground mt-0.5">Pose tracking & reps</p>
                                        </div>
                                    </Link>
                                </div>
                            )}
                        </div>
                    </nav>
                </div>

                <div className="flex items-center gap-2 sm:gap-3">
                    <a
                        href="https://github.com/pranta-barua007/yolo11-onnx"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex items-center gap-1.5 px-2.5 sm:px-3 py-1.5 text-xs font-bold text-muted-foreground hover:text-foreground transition-colors duration-200 active:scale-95"
                    >
                        <svg viewBox="0 0 24 24" fill="currentColor" className="w-4 h-4 text-amber-400">
                            <path d="M12 2l3.09 6.26L22 9.27l-5 4.87L18.18 22 12 18.27 5.82 22 7 14.14l-5-4.87 6.91-1.01L12 2z" />
                        </svg>
                        <span className="hidden sm:inline">GitHub</span>
                    </a>
                    <ThemeToggle />
                </div>
            </div>
        </header>
    );
}
