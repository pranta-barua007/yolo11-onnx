"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Info } from "lucide-react";
import { ThemeToggle } from "./ThemeToggle";

interface HeaderProps {
    rightSlot?: React.ReactNode;
}

export default function Header({ rightSlot }: HeaderProps) {
    const pathname = usePathname();

    return (
        <header className="sticky top-0 z-30 w-full px-4 md:px-6 py-3 border-b border-border/50 bg-background/80 backdrop-blur-md transition-all duration-500">
            <div className="flex items-center justify-between max-w-[1600px] mx-auto w-full">
                <div className="flex items-center gap-4">
                    {/* Brand */}
                    <Link href="/" className="flex items-center gap-3 active:scale-95 transition-transform duration-200">
                        <div className="w-10 h-10 bg-primary rounded-xl flex items-center justify-center text-primary-foreground font-black text-2xl shadow-sm ring-1 ring-primary/20">
                            D
                        </div>
                        <span className="hidden sm:flex text-lg font-bold text-foreground tracking-tight items-baseline">
                            YOLO
                            <span className="text-primary ml-1.5 font-bold">Real-time</span>
                            <span className="text-muted-foreground font-medium text-xs ml-3 hidden lg:inline border-l border-border/50 pl-3">
                                Segmentation App
                            </span>
                        </span>
                    </Link>

                    {/* Nav Links */}
                    <nav className="flex items-center gap-2 ml-4">
                        <Link
                            href="/"
                            className={`px-4 py-2 text-sm font-medium rounded-full transition-all duration-300 ${pathname === "/"
                                ? "bg-primary/10 text-primary shadow-[inset_0_1px_1px_rgba(255,255,255,0.05)] ring-1 ring-primary/30"
                                : "text-muted-foreground hover:text-foreground hover:bg-foreground/5"
                                }`}
                        >
                            Workspace
                        </Link>
                        <Link
                            href="/about"
                            className={`px-4 py-2 text-sm font-medium rounded-full transition-all duration-300 flex items-center gap-2 ${pathname === "/about"
                                ? "bg-primary/10 text-primary shadow-[inset_0_1px_1px_rgba(255,255,255,0.05)] ring-1 ring-primary/30"
                                : "text-muted-foreground hover:text-foreground hover:bg-foreground/5"
                                }`}
                        >
                            <Info className="w-4 h-4" />
                            About
                        </Link>
                    </nav>
                </div>

                <div className="flex items-center gap-2">
                    {rightSlot}
                    <ThemeToggle />
                </div>
            </div>
        </header>
    );
}
