import axios from 'axios';

// Ensure the backend URL is correctly formatted
const BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export const apiClient = axios.create({
  baseURL: BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// For outputs that might need absolute URLs if served directly by FastAPI
export const getOutputUrl = (path?: string) => {
  if (!path) return '';
  if (path.startsWith('http')) return path;
  return `${BASE_URL}${path.startsWith('/') ? path : '/' + path}`;
};
