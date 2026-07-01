"use client";

import { useHealth } from "@/lib/hooks/queries";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Activity, Cpu, ServerCrash, CheckCircle2 } from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";

export default function StatusPage() {
  const { data: health, isLoading, isError, error } = useHealth();

  if (isError) {
    return (
      <div className="max-w-3xl mx-auto py-12 space-y-6">
        <div className="flex items-center gap-3 text-destructive">
          <ServerCrash className="w-8 h-8" />
          <h1 className="text-3xl font-bold">System Status</h1>
        </div>
        <Card className="border-destructive bg-destructive/5">
          <CardHeader>
            <CardTitle className="text-destructive">Backend Offline</CardTitle>
            <CardDescription>
              Could not connect to the API server. Please ensure the FastAPI backend is running.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm font-mono text-muted-foreground">
              Error details: {error instanceof Error ? error.message : "Unknown error"}
            </p>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="max-w-3xl mx-auto py-12 space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Activity className="w-8 h-8 text-primary" />
          <h1 className="text-3xl font-bold">System Status</h1>
        </div>
        {!isLoading && health && (
          <Badge variant="outline" className="bg-green-500/10 text-green-600 border-green-500/20 gap-1.5 py-1">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
            </span>
            Operational
          </Badge>
        )}
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        {/* Core API Status */}
        <Card>
          <CardHeader className="pb-4">
            <CardTitle className="text-lg flex items-center gap-2">
              <Activity className="w-5 h-5 text-muted-foreground" /> API Service
            </CardTitle>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <div className="space-y-2">
                <Skeleton className="h-4 w-[100px]" />
                <Skeleton className="h-4 w-[150px]" />
              </div>
            ) : (
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-muted-foreground text-sm">Status</span>
                  <span className="font-medium capitalize">{health?.status}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-muted-foreground text-sm">Active Device</span>
                  <Badge variant="secondary" className="font-mono">
                    <Cpu className="w-3 h-3 mr-1" /> {health?.device}
                  </Badge>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Models Status */}
        <Card>
          <CardHeader className="pb-4">
            <CardTitle className="text-lg flex items-center gap-2">
              <CheckCircle2 className="w-5 h-5 text-muted-foreground" /> Loaded Models
            </CardTitle>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <div className="space-y-2">
                <Skeleton className="h-4 w-full" />
                <Skeleton className="h-4 w-[80%]" />
                <Skeleton className="h-4 w-[90%]" />
              </div>
            ) : (
              <div className="space-y-3">
                {health?.models_loaded && Object.entries(health.models_loaded).map(([model, loaded]) => (
                  <div key={model} className="flex justify-between items-center">
                    <span className="text-sm font-medium">{model}</span>
                    <Badge variant={loaded ? "default" : "secondary"}>
                      {loaded ? "Loaded" : "Unloaded"}
                    </Badge>
                  </div>
                ))}
                {(!health?.models_loaded || Object.keys(health.models_loaded).length === 0) && (
                  <p className="text-sm text-muted-foreground italic">Models are loaded lazily on first request.</p>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
      <p className="text-xs text-center text-muted-foreground pt-4">
        This page automatically refreshes every 30 seconds.
      </p>
    </div>
  );
}
