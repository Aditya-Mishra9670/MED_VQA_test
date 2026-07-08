"use client";

import { useHistory } from "@/lib/store/history";
import { useParams, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";
import { ArrowLeft, Download, Maximize2, Layers, Copy } from "lucide-react";
import { getOutputUrl } from "@/lib/api/client";
import { Badge } from "@/components/ui/badge";
import { toast } from "sonner";

export default function HeatmapWorkspace() {
  const params = useParams();
  const id = params.id as string;
  const { history, isLoaded } = useHistory();
  const router = useRouter();
  
  const [opacity, setOpacity] = useState(0.7);
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [showBoxes, setShowBoxes] = useState(true);
  const [showMask, setShowMask] = useState(true);
  const [isFullscreen, setIsFullscreen] = useState(false);

  const item = history.find(h => h.id === id);

  useEffect(() => {
    if (isLoaded && !item) {
      router.push('/history');
    }
  }, [isLoaded, item, router]);

  if (!isLoaded || !item) {
    return <div className="p-8 text-center">Loading...</div>;
  }

  const { originalImageBase64, request, response } = item;
  const hasHeatmap = !!response.heatmap_url;
  const hasAttention = !!response.attention_overlay_url;
  const hasBoxes = !!response.boxes_image_url;
  const hasMask = !!response.mask_overlay_url;
  const hasAnyOverlay = hasHeatmap || hasAttention || hasBoxes || hasMask;

  const downloadHeatmap = () => {
    if (!hasHeatmap) return;
    const a = document.createElement('a');
    a.href = getOutputUrl(response.heatmap_url);
    a.download = `heatmap-${id}.png`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <div className={`flex flex-col ${isFullscreen ? 'fixed inset-0 z-50 bg-background p-4' : 'h-[calc(100vh-6rem)]'}`}>
      
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-4">
          {!isFullscreen && (
            <Button variant="ghost" size="icon" onClick={() => router.back()}>
              <ArrowLeft className="w-5 h-5" />
            </Button>
          )}
          <div>
            <h1 className="text-2xl font-bold">Analysis Results</h1>
            <p className="text-muted-foreground text-sm">Case ID: {id.split('-')[0]}</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {hasHeatmap && (
            <Button variant="outline" size="sm" onClick={downloadHeatmap} className="gap-2">
              <Download className="w-4 h-4" /> Download Heatmap
            </Button>
          )}
          <Button variant="outline" size="sm" onClick={() => setIsFullscreen(!isFullscreen)} className="gap-2">
            <Maximize2 className="w-4 h-4" /> {isFullscreen ? "Exit Fullscreen" : "Fullscreen"}
          </Button>
        </div>
      </div>

      <div className={`grid grid-cols-1 ${isFullscreen ? 'lg:grid-cols-4' : 'lg:grid-cols-12'} gap-6 flex-1 min-h-0`}>
        
        {/* Results Panel */}
        <div className={`${isFullscreen ? 'lg:col-span-1' : 'lg:col-span-4'} flex flex-col gap-6 overflow-auto`}>
          <Card>
            <CardHeader className="pb-3 border-b bg-muted/20">
              <CardTitle className="text-sm text-muted-foreground uppercase tracking-wider">Clinical Query</CardTitle>
              <p className="text-lg font-medium">{request.question}</p>
            </CardHeader>
            <CardContent className="pt-4">
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between items-center mb-1">
                    <h3 className="text-sm font-semibold text-primary">STLLaVA-Med Diagnosis</h3>
                    <Button variant="ghost" size="sm" className="h-6 px-2 text-xs" onClick={() => { navigator.clipboard.writeText(response.answer); toast.success("Copied to clipboard!"); }}>
                      <Copy className="w-3 h-3 mr-1" /> Copy
                    </Button>
                  </div>
                  <p className="text-base leading-relaxed bg-primary/5 p-4 rounded-lg border-l-4 border-primary">
                    {response.answer}
                  </p>
                </div>
                
                <div className="grid grid-cols-2 gap-4 text-sm border-t pt-4">
                  <div>
                    <p className="text-muted-foreground">Inference Time</p>
                    <p className="font-medium">{response.inference_time_seconds.toFixed(2)}s</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Model</p>
                    <Badge variant="secondary">{response.model_name}</Badge>
                  </div>
                </div>

                {response.metadata?.localization_enabled && response.bounding_boxes && (
                  <div className="border-t pt-4">
                    <p className="text-muted-foreground mb-2 text-sm">Detected Regions</p>
                    <div className="flex flex-wrap gap-2">
                      {response.bounding_boxes.map((box, i) => (
                        <Badge key={i} variant="outline" className="border-blue-500/30 bg-blue-500/10 text-blue-700">
                          {box.label} ({(box.score * 100).toFixed(0)}%)
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Controls */}
          {hasAnyOverlay && (
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-semibold flex items-center gap-2">
                  <Layers className="w-4 h-4" /> Overlay Controls
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex flex-wrap gap-2 mb-4">
                    {hasHeatmap && (
                      <Button variant={showHeatmap ? "default" : "outline"} size="sm" onClick={() => setShowHeatmap(!showHeatmap)}>
                        Grad-CAM Heatmap
                      </Button>
                    )}
                    {hasAttention && (
                      <Button variant={showHeatmap ? "default" : "outline"} size="sm" onClick={() => setShowHeatmap(!showHeatmap)}>
                        Attention Rollout
                      </Button>
                    )}
                    {hasBoxes && (
                      <Button variant={showBoxes ? "default" : "outline"} size="sm" onClick={() => setShowBoxes(!showBoxes)}>
                        Bounding Boxes
                      </Button>
                    )}
                    {hasMask && (
                      <Button variant={showMask ? "default" : "outline"} size="sm" onClick={() => setShowMask(!showMask)}>
                        Segmentation
                      </Button>
                    )}
                  </div>
                  <div>
                    <div className="flex justify-between mb-2 text-sm">
                      <label>Overlay Opacity</label>
                      <span className="font-mono">{Math.round(opacity * 100)}%</span>
                    </div>
                    <Slider 
                      value={[opacity]} 
                      onValueChange={(v: any) => setOpacity(v[0])} 
                      max={1} 
                      step={0.01} 
                    />
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </div>

        {/* Viewer Panel */}
        <div className={`${isFullscreen ? 'lg:col-span-3' : 'lg:col-span-8'} bg-black rounded-xl overflow-hidden relative flex items-center justify-center border shadow-inner`}>
          <div className="absolute inset-4 flex items-center justify-center">
            <div className="relative max-w-full max-h-full">
              {/* Original Image */}
               {/* eslint-disable-next-line @next/next/no-img-element */}
              <img 
                src={originalImageBase64} 
                alt="Original Scan" 
                className="max-w-full max-h-full object-contain"
              />
              
              {/* Heatmap Overlay */}
              {hasHeatmap && showHeatmap && (
                <>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img 
                    src={getOutputUrl(response.heatmap_url)} 
                    alt="Grad-CAM Heatmap" 
                    className="absolute inset-0 w-full h-full object-contain transition-opacity duration-200"
                    style={{ opacity, mixBlendMode: 'screen' }}
                  />
                </>
              )}

              {/* Attention Rollout Overlay */}
              {hasAttention && showHeatmap && !hasHeatmap && (
                <>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img 
                    src={getOutputUrl(response.attention_overlay_url)} 
                    alt="Attention Rollout" 
                    className="absolute inset-0 w-full h-full object-contain transition-opacity duration-200 pointer-events-none"
                    style={{ opacity: opacity }}
                  />
                </>
              )}

              {/* Segmentation Mask Overlay */}
              {hasMask && showMask && (
                <>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img 
                    src={getOutputUrl(response.mask_overlay_url)} 
                    alt="Segmentation Masks" 
                    className="absolute inset-0 w-full h-full object-contain transition-opacity duration-200 pointer-events-none"
                    style={{ opacity: opacity * 0.8 }}
                  />
                </>
              )}

              {/* Bounding Boxes Overlay */}
              {hasBoxes && showBoxes && (
                <>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img 
                    src={getOutputUrl(response.boxes_image_url)} 
                    alt="Bounding Boxes" 
                    className="absolute inset-0 w-full h-full object-contain transition-opacity duration-200 pointer-events-none"
                    style={{ opacity: opacity }}
                  />
                </>
              )}
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}
