"use client";

import { useState, useRef } from "react";
import { ImageUploader } from "@/components/vqa/ImageUploader";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { usePredict } from "@/lib/hooks/queries";
import { useHistory, fileToBase64 } from "@/lib/store/history";
import { toast } from "sonner";
import { Loader2, ArrowRight, Activity, BrainCircuit } from "lucide-react";
import { useRouter } from "next/navigation";
import { getOutputUrl } from "@/lib/api/client";
import { Copy, XCircle } from "lucide-react";

const SUGGESTED_QUESTIONS = [
  "What abnormalities are visible?",
  "Is there evidence of pneumonia?",
  "Describe the findings.",
  "What pathology is present?",
  "Summarize the scan.",
];

export default function AnalyzePage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [question, setQuestion] = useState("");
  
  const predictMutation = usePredict();
  const { addHistoryItem } = useHistory();
  const router = useRouter();
  
  const [uploadProgress, setUploadProgress] = useState(0);
  const abortControllerRef = useRef<AbortController | null>(null);

  const handleAsk = async () => {
    if (!selectedFile) {
      toast.error("Please upload an image first.");
      return;
    }
    if (!question.trim()) {
      toast.error("Please enter a question.");
      return;
    }

    abortControllerRef.current = new AbortController();
    setUploadProgress(0);

    try {
      const response = await predictMutation.mutateAsync({
        image: selectedFile,
        question: question,
        enable_gradcam: true,
        enable_localization: true,
        signal: abortControllerRef.current.signal,
        onUploadProgress: (progressEvent) => {
          if (progressEvent.total) {
            const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
            setUploadProgress(percentCompleted);
          }
        }
      });

      const base64Image = await fileToBase64(selectedFile);
      
      const historyId = crypto.randomUUID();
      addHistoryItem({
        id: historyId,
        timestamp: new Date().toISOString(),
        originalImageBase64: base64Image,
        request: { question },
        response,
      });

      toast.success("Analysis complete!");
      // Navigate to heatmap workspace which also serves as the detailed result view
      router.push(`/heatmap/${historyId}`);

    } catch (error: any) {
      if (error?.name === 'CanceledError' || error?.message?.includes('canceled')) {
        toast.info("Request cancelled.");
        return;
      }

      let errorMsg = error?.response?.data?.detail || error.message || "An unknown error occurred.";
      if (error?.response?.status === 413) {
        errorMsg = "Image is too large. Maximum size is 50MB.";
      } else if (error?.response?.status === 503) {
        errorMsg = "Server is busy or GPU is out of memory. Please try again later.";
      } else if (error.code === 'ECONNABORTED' || error.message.includes('timeout')) {
        errorMsg = "Request timed out. The server might be processing a heavy load.";
      }

      toast.error("Analysis failed", {
        description: errorMsg,
      });
    } finally {
      abortControllerRef.current = null;
      setUploadProgress(0);
    }
  };

  const handleCancel = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
  };

  return (
    <div className="h-[calc(100vh-6rem)]">
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 h-full">
        
        {/* LEFT PANEL: Uploader and Details */}
        <div className="lg:col-span-3 space-y-6 flex flex-col h-full">
          <Card className="flex-1">
            <CardHeader>
              <CardTitle>Image Input</CardTitle>
              <CardDescription>Upload a medical scan for analysis.</CardDescription>
            </CardHeader>
            <CardContent>
              <ImageUploader 
                onImageSelected={setSelectedFile} 
                onClear={() => setSelectedFile(null)} 
                selectedFile={selectedFile}
              />
            </CardContent>
          </Card>
        </div>

        {/* CENTER PANEL: Large Viewer */}
        <div className="lg:col-span-6 bg-black/5 rounded-xl border flex flex-col overflow-hidden relative">
          <div className="absolute top-4 left-4 z-10 bg-background/80 backdrop-blur px-3 py-1.5 rounded-md text-sm font-medium border shadow-sm">
            Primary Viewer
          </div>
          {selectedFile ? (
            <div className="flex-1 overflow-auto flex items-center justify-center p-4">
               {/* eslint-disable-next-line @next/next/no-img-element */}
              <img 
                src={URL.createObjectURL(selectedFile)} 
                alt="Scan viewer" 
                className="max-w-full max-h-full object-contain shadow-md border bg-white"
              />
            </div>
          ) : (
            <div className="flex-1 flex flex-col items-center justify-center text-muted-foreground p-8 text-center space-y-4">
              <Activity className="w-16 h-16 opacity-20" />
              <p>Upload a medical image to view it here.</p>
            </div>
          )}
        </div>

        {/* RIGHT PANEL: Question and Analysis */}
        <div className="lg:col-span-3 flex flex-col gap-6">
          <Card className="flex-1 flex flex-col">
            <CardHeader>
              <CardTitle>Clinical Query</CardTitle>
              <CardDescription>Ask STLLaVA-Med about the scan.</CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col flex-1 gap-6">
              <div className="space-y-3">
                <Label htmlFor="question">Question</Label>
                <Input 
                  id="question" 
                  placeholder="e.g. What abnormalities are visible?" 
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  disabled={predictMutation.isPending}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") handleAsk();
                  }}
                />
              </div>

              <div className="space-y-3">
                <Label className="text-muted-foreground text-xs uppercase font-semibold">Suggested Prompts</Label>
                <div className="flex flex-wrap gap-2">
                  {SUGGESTED_QUESTIONS.map(q => (
                    <Badge 
                      key={q} 
                      variant="secondary" 
                      className="cursor-pointer font-normal hover:bg-secondary/80"
                      onClick={() => setQuestion(q)}
                    >
                      {q}
                    </Badge>
                  ))}
                </div>
              </div>

              <div className="mt-auto pt-6 border-t flex flex-col gap-3">
                {predictMutation.isPending ? (
                  <div className="flex gap-2 w-full">
                    <Button 
                      className="flex-1 h-12 text-lg gap-2" 
                      disabled
                    >
                      <Loader2 className="w-5 h-5 animate-spin" /> 
                      {uploadProgress > 0 && uploadProgress < 100 
                        ? `Uploading (${uploadProgress}%)` 
                        : "Processing..."}
                    </Button>
                    <Button 
                      variant="destructive" 
                      className="h-12 px-4" 
                      onClick={handleCancel}
                      title="Cancel Request"
                    >
                      <XCircle className="w-5 h-5" />
                    </Button>
                  </div>
                ) : (
                  <Button 
                    className="w-full h-12 text-lg gap-2" 
                    onClick={handleAsk}
                    disabled={!selectedFile || !question.trim()}
                  >
                    <BrainCircuit className="w-5 h-5" /> Analyze Scan
                  </Button>
                )}
                {predictMutation.isPending && uploadProgress === 100 && (
                  <p className="text-xs text-center text-muted-foreground animate-pulse">
                    This may take 10-30 seconds depending on GPU availability...
                  </p>
                )}
                {predictMutation.isError && !predictMutation.isPending && (
                  <Button variant="outline" className="w-full text-destructive border-destructive/20 hover:bg-destructive/10" onClick={handleAsk}>
                    Retry Failed Request
                  </Button>
                )}
              </div>
            </CardContent>
          </Card>
        </div>
        
      </div>
    </div>
  );
}

function Badge({ children, className, variant = "default", onClick }: { children: React.ReactNode, className?: string, variant?: "default" | "secondary" | "outline", onClick?: () => void }) {
  const base = "inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2";
  const variants = {
    default: "border-transparent bg-primary text-primary-foreground hover:bg-primary/80",
    secondary: "border-transparent bg-secondary text-secondary-foreground hover:bg-secondary/80",
    outline: "text-foreground",
  };
  return (
    <div className={`${base} ${variants[variant]} ${className || ""}`} onClick={onClick}>
      {children}
    </div>
  );
}
