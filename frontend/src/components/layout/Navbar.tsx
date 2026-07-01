import Link from "next/link";
import { Activity, History, Home, LayoutDashboard } from "lucide-react";
import { Button } from "@/components/ui/button";

export const Navbar = () => {
  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container mx-auto flex h-14 items-center px-4">
        <div className="mr-4 flex">
          <Link href="/" className="mr-6 flex items-center space-x-2">
            <Activity className="h-6 w-6 text-primary" />
            <span className="hidden font-bold sm:inline-block">
              MedVQA Platform
            </span>
          </Link>
          <nav className="flex items-center space-x-6 text-sm font-medium">
            <Link
              href="/analyze"
              className="transition-colors hover:text-foreground/80 text-foreground/60 flex items-center gap-2"
            >
              <LayoutDashboard className="w-4 h-4" /> Analyze
            </Link>
            <Link
              href="/history"
              className="transition-colors hover:text-foreground/80 text-foreground/60 flex items-center gap-2"
            >
              <History className="w-4 h-4" /> History
            </Link>
          </nav>
        </div>
        <div className="flex flex-1 items-center justify-between space-x-2 md:justify-end">
          <div className="w-full flex-1 md:w-auto md:flex-none">
            {/* Could add a global search here */}
          </div>
          <nav className="flex items-center">
            <Link href="/status">
              <Button variant="ghost" size="sm" className="gap-2">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
                </span>
                System Status
              </Button>
            </Link>
          </nav>
        </div>
      </div>
    </header>
  );
};
