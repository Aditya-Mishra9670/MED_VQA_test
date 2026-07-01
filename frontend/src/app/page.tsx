import Link from "next/link";
import { ArrowRight, Brain, FileSearch, ShieldCheck } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

export default function Home() {
  return (
    <div className="flex flex-col items-center max-w-5xl mx-auto space-y-12 py-12">
      {/* Hero Section */}
      <section className="text-center space-y-6">
        <h1 className="text-5xl font-extrabold tracking-tight lg:text-6xl text-primary">
          Medical VQA Platform
        </h1>
        <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
          Advanced clinical image analysis using STLLaVA-Med and Grad-CAM explainability. 
          Upload scans, ask clinical questions, and receive localized diagnostic insights.
        </p>
        <div className="flex justify-center gap-4 pt-4">
          <Link href="/analyze">
            <Button size="lg" className="gap-2">
              Start Analysis <ArrowRight className="w-5 h-5" />
            </Button>
          </Link>
          <Link href="/history">
            <Button size="lg" variant="outline">
              View Recent Cases
            </Button>
          </Link>
        </div>
      </section>

      {/* Features Grid */}
      <section className="grid md:grid-cols-3 gap-6 w-full">
        <Card>
          <CardHeader>
            <Brain className="w-10 h-10 text-primary mb-2" />
            <CardTitle>Specialized AI Model</CardTitle>
            <CardDescription>
              Powered by STLLaVA-Med, specifically fine-tuned for medical reasoning.
            </CardDescription>
          </CardHeader>
          <CardContent>
            Achieve highly accurate open-ended medical answers across various modalities including X-Ray, CT, and MRI.
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader>
            <FileSearch className="w-10 h-10 text-primary mb-2" />
            <CardTitle>Visual Evidence</CardTitle>
            <CardDescription>
              Explainable AI via Grad-CAM and Attention Rollout.
            </CardDescription>
          </CardHeader>
          <CardContent>
            Understand exactly which regions of the medical scan influenced the AI's diagnostic response.
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <ShieldCheck className="w-10 h-10 text-primary mb-2" />
            <CardTitle>Precise Localization</CardTitle>
            <CardDescription>
              Grounding DINO and SAM2 integration.
            </CardDescription>
          </CardHeader>
          <CardContent>
            Automatically generate bounding boxes and pixel-perfect segmentation masks for identified pathologies.
          </CardContent>
        </Card>
      </section>

      {/* Accepted Formats */}
      <section className="w-full text-center space-y-4 pt-8 border-t">
        <h3 className="text-lg font-medium">Supported Medical Image Formats</h3>
        <div className="flex justify-center gap-4 text-sm font-semibold text-muted-foreground">
          <span className="bg-secondary px-3 py-1 rounded-md">DICOM (.dcm)</span>
          <span className="bg-secondary px-3 py-1 rounded-md">PNG</span>
          <span className="bg-secondary px-3 py-1 rounded-md">JPEG / JPG</span>
        </div>
      </section>
    </div>
  );
}
