import { useQuery, useMutation } from '@tanstack/react-query';
import { apiClient } from '../api/client';
import { HealthResponse, PredictResponse } from '../../types/api';

// Fetch Health Status
export const useHealth = () => {
  return useQuery({
    queryKey: ['health'],
    queryFn: async (): Promise<HealthResponse> => {
      const response = await apiClient.get('/health');
      return response.data;
    },
    refetchInterval: 30000, // Auto-refresh every 30 seconds
  });
};

export interface PredictMutationParams {
  image: File;
  question: string;
  enable_gradcam?: boolean;
  enable_attention?: boolean;
  enable_localization?: boolean;
  localization_prompt?: string;
  max_new_tokens?: number;
  temperature?: number;
  onUploadProgress?: (progressEvent: any) => void;
  signal?: AbortSignal;
}

// Predict VQA
export const usePredict = () => {
  return useMutation({
    mutationFn: async (params: PredictMutationParams): Promise<PredictResponse> => {
      const formData = new FormData();
      formData.append('image', params.image);
      formData.append('question', params.question);
      
      // Defaults aligned with backend schemas
      formData.append('enable_gradcam', String(params.enable_gradcam ?? true));
      formData.append('enable_attention', String(params.enable_attention ?? false));
      formData.append('enable_localization', String(params.enable_localization ?? false));
      
      if (params.localization_prompt) {
        formData.append('localization_prompt', params.localization_prompt);
      }
      if (params.max_new_tokens !== undefined) {
        formData.append('max_new_tokens', String(params.max_new_tokens));
      }
      if (params.temperature !== undefined) {
        formData.append('temperature', String(params.temperature));
      }

      const response = await apiClient.post('/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: params.onUploadProgress,
        signal: params.signal,
      });

      return response.data;
    },
  });
};
