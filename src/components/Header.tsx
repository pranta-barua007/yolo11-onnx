"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Info } from "lucide-react";

interface HeaderProps {
    rightSlot?: React.ReactNode;
}

export default function Header({ rightSlot }: HeaderProps) {
    const pathname = usePathname();

    return (
        <div className="px-4 md:px-6 pt-4">
            <header className="glass-panel sticky top-4 z-30 px-4 md:px-6 py-3 flex items-center justify-between rounded-full max-w-[1600px] mx-auto transition-all duration-500">
                <div className="flex items-center gap-4">
                    {/* Brand */}
                    <Link href="/" className="flex items-center gap-3 group">
                        <div className="w-9 h-9 bg-gradient-to-br from-indigo-500 via-purple-500 to-fuchsia-500 rounded-[10px] flex items-center justify-center text-white font-bold text-xl shadow-lg ring-1 ring-white/20 transition-transform duration-300 group-hover:scale-105 group-hover:shadow-[0_0_20px_rgba(168,85,247,0.4)]">D</div>
                        <span className="hidden sm:flex text-lg md:text-xl font-bold text-foreground tracking-tight items-baseline">
                            YOLO <span className="text-gradient ml-1.5 font-extrabold animate-gradient">Real-time</span>
                            <span className="text-muted-foreground font-medium text-xs md:text-sm ml-3 hidden lg:inline border-l border-border/50 pl-3">Segmentation App</span>
                        </span>
                    </Link>

                    {/* Nav Links */}
                    <nav className="flex items-center gap-2 ml-4">
                        <Link
                            href="/"
                            className={`px-4 py-2 text-sm font-medium rounded-full transition-all duration-300 ${pathname === "/"
                                ? "bg-primary/10 text-primary shadow-[inset_0_1px_1px_rgba(255,255,255,0.05)] ring-1 ring-primary/30"
                                : "text-muted-foreground hover:text-foreground hover:bg-white/5"
                                }`}
                        >
                            Workspace
                        </Link>
                        <Link
                            href="/about"
                            className={`px-4 py-2 text-sm font-medium rounded-full transition-all duration-300 flex items-center gap-2 ${pathname === "/about"
                                ? "bg-primary/10 text-primary shadow-[inset_0_1px_1px_rgba(255,255,255,0.05)] ring-1 ring-primary/30"
                                : "text-muted-foreground hover:text-foreground hover:bg-white/5"
                                }`}
                        >
                            <Info className="w-4 h-4" />
                            About
                        </Link>
                    </nav>
                </div>

                {rightSlot && <div className="flex items-center gap-2">{rightSlot}</div>}
            </header>
        </div>
    );
}
