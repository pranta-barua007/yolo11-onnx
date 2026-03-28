"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Info, BookOpen } from "lucide-react";
import { ThemeToggle } from "./ThemeToggle";

export default function Header() {
    const rawPathname = usePathname();
    const pathname = rawPathname === "/" ? "/" : rawPathname.replace(/\/+$/, "");

    return (
        <header className="sticky top-0 z-30 w-full px-3 md:px-6 py-2.5 border-b border-border/50 bg-background/80 backdrop-blur-md">
            <div className="flex items-center justify-between mx-auto w-full">
                <div className="flex items-center gap-4">
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
                    <nav className="flex items-center gap-1.5 sm:gap-2 ml-2 sm:ml-4">
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
                        GitHub
                    </a>
                    <ThemeToggle />
                </div>
            </div>
        </header>
    );
}
