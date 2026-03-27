"use client"

import * as React from "react"
import { Moon, Sun, Check } from "lucide-react"
import { useTheme } from "next-themes"

import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"

export function ThemeToggle() {
  const { theme, setTheme } = useTheme()

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="icon" className="w-9 h-9 rounded-full bg-foreground/5 hover:bg-foreground/10 border border-foreground/10 text-muted-foreground hover:text-foreground transition-all">
          <Sun className="h-[1.2rem] w-[1.2rem] rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
          <Moon className="absolute h-[1.2rem] w-[1.2rem] rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
          <span className="sr-only">Toggle theme</span>
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="bg-background/95 backdrop-blur-xl border-border/50 min-w-[120px] p-1.5 animate-in fade-in zoom-in-95 duration-200">
        <DropdownMenuItem 
          onClick={() => setTheme("light")} 
          className={`flex items-center justify-between gap-2 px-3 py-2 text-xs font-medium cursor-pointer rounded-lg transition-colors ${theme === "light" ? "bg-primary/15 text-primary" : "focus:bg-muted"}`}
        >
          Light
          {theme === "light" && <Check className="w-3.5 h-3.5" />}
        </DropdownMenuItem>
        <DropdownMenuItem 
          onClick={() => setTheme("dark")} 
          className={`flex items-center justify-between gap-2 px-3 py-2 text-xs font-medium cursor-pointer rounded-lg transition-colors ${theme === "dark" ? "bg-primary/15 text-primary" : "focus:bg-muted"}`}
        >
          Dark
          {theme === "dark" && <Check className="w-3.5 h-3.5" />}
        </DropdownMenuItem>
        <DropdownMenuItem 
          onClick={() => setTheme("system")} 
          className={`flex items-center justify-between gap-2 px-3 py-2 text-xs font-medium cursor-pointer rounded-lg transition-colors ${theme === "system" ? "bg-primary/15 text-primary" : "focus:bg-muted"}`}
        >
          System
          {theme === "system" && <Check className="w-3.5 h-3.5" />}
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
