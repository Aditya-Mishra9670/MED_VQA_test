export interface BoundingBox {
  x: number;
  y: number;
  w: number;
  h: number;
  score: number;
  label: string;
}

export interface PredictResponse {
  answer: string;
  question: string;
  inference_time_seconds: number;
  model_name: string;
  confidence?: number;
  heatmap_url?: string;
  overlay_url?: string;
  attention_url?: string;
  attention_overlay_url?: string;
  bounding_boxes?: BoundingBox[];
  boxes_image_url?: string;
  mask_overlay_url?: string;
  mask_urls?: string[];
  metadata: Record<string, any>;
}

export interface HealthResponse {
  status: string;
  models_loaded: Record<string, any>;
  device: string;
}

export interface ErrorResponse {
  error: string;
  detail?: string;
}
