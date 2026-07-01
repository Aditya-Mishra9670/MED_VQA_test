"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";
import { Navbar } from "./layout/Navbar";

export function Providers({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(() => new QueryClient());

  return (
    <QueryClientProvider client={queryClient}>
      <Navbar />
      <main className="flex-1 container mx-auto px-4 py-8">
        {children}
      </main>
    </QueryClientProvider>
  );
}
