"use client";

import React, { useCallback, useState, useEffect } from "react";
import { UploadCloud, X, FileImage } from "lucide-react";
import { Button } from "@/components/ui/button";

interface ImageUploaderProps {
  onImageSelected: (file: File) => void;
  onClear: () => void;
  selectedFile?: File | null;
}

export const ImageUploader: React.FC<ImageUploaderProps> = ({
  onImageSelected,
  onClear,
  selectedFile,
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const handleFile = useCallback((file: File) => {
    if (!file.type.startsWith("image/") && !file.name.endsWith(".dicom")) {
      alert("Please upload a valid image file (PNG, JPG, DICOM)."); // simple validation
      return;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    onImageSelected(file);
  }, [onImageSelected]);

  useEffect(() => {
    const handlePaste = (e: ClipboardEvent) => {
      if (e.clipboardData?.items) {
        for (const item of Array.from(e.clipboardData.items)) {
          if (item.type.startsWith("image/")) {
            const file = item.getAsFile();
            if (file) {
              handleFile(file);
              break;
            }
          }
        }
      }
    };
    
    if (!selectedFile) {
      document.addEventListener("paste", handlePaste);
    }
    return () => {
      document.removeEventListener("paste", handlePaste);
    };
  }, [handleFile, selectedFile]);

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const onDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0]);
    }
  }, [handleFile]);

  const onChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFile(e.target.files[0]);
    }
  };

  const handleClear = () => {
    setPreviewUrl(null);
    onClear();
  };

  if (selectedFile && previewUrl) {
    return (
      <div className="relative border rounded-xl overflow-hidden bg-muted/50 p-2 group">
        <div className="absolute top-4 right-4 z-10 opacity-0 group-hover:opacity-100 transition-opacity">
          <Button variant="destructive" size="icon" onClick={handleClear}>
            <X className="w-4 h-4" />
          </Button>
        </div>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={previewUrl}
          alt="Selected Medical Scan"
          className="w-full h-auto object-contain max-h-[500px] rounded-lg border bg-white shadow-sm"
        />
        <div className="mt-2 flex items-center justify-between text-sm text-muted-foreground px-2">
          <span className="flex items-center gap-2">
            <FileImage className="w-4 h-4" /> {selectedFile.name}
          </span>
          <span>{(selectedFile.size / 1024 / 1024).toFixed(2)} MB</span>
        </div>
      </div>
    );
  }

  return (
    <div
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
      className={`border-2 border-dashed rounded-xl p-10 flex flex-col items-center justify-center text-center cursor-pointer transition-colors ${
        isDragging
          ? "border-primary bg-primary/5"
          : "border-muted-foreground/25 hover:border-primary/50 hover:bg-muted/50"
      }`}
    >
      <input
        type="file"
        id="file-upload"
        className="hidden"
        accept="image/*,.dicom"
        onChange={onChange}
      />
      <label htmlFor="file-upload" className="cursor-pointer flex flex-col items-center">
        <div className="w-16 h-16 bg-primary/10 text-primary rounded-full flex items-center justify-center mb-4">
          <UploadCloud className="w-8 h-8" />
        </div>
        <h3 className="text-lg font-semibold mb-1">Drag & Drop Image or Paste</h3>
        <p className="text-sm text-muted-foreground mb-4 max-w-[250px]">
          Support for PNG, JPG, JPEG, and DICOM medical scans. You can also paste directly (Ctrl+V).
        </p>
        <Button variant="secondary" onClick={() => document.getElementById("file-upload")?.click()}>
          Browse Files
        </Button>
      </label>
    </div>
  );
};
