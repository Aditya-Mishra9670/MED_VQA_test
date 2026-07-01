"use client";

import { useState, useEffect } from "react";
import { PredictResponse } from "@/types/api";

export interface HistoryItem {
  id: string;
  timestamp: string;
  originalImageBase64: string; // Stored as base64 for local viewing
  request: {
    question: string;
  };
  response: PredictResponse;
}

const HISTORY_KEY = "medvqa_history";

export const useHistory = () => {
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    const stored = localStorage.getItem(HISTORY_KEY);
    if (stored) {
      try {
        setHistory(JSON.parse(stored));
      } catch (e) {
        console.error("Failed to parse history", e);
      }
    }
    setIsLoaded(true);
  }, []);

  const addHistoryItem = (item: HistoryItem) => {
    const newHistory = [item, ...history];
    setHistory(newHistory);
    try {
      localStorage.setItem(HISTORY_KEY, JSON.stringify(newHistory));
    } catch (e) {
      console.warn("Could not save to local storage (might be full)", e);
    }
  };

  const clearHistory = () => {
    setHistory([]);
    localStorage.removeItem(HISTORY_KEY);
  };

  const deleteItem = (id: string) => {
    const newHistory = history.filter(item => item.id !== id);
    setHistory(newHistory);
    localStorage.setItem(HISTORY_KEY, JSON.stringify(newHistory));
  };

  return { history, isLoaded, addHistoryItem, clearHistory, deleteItem };
};

export const fileToBase64 = (file: File): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = error => reject(error);
  });
};
