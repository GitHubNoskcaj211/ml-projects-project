export function makeBackendURL(path: string) {
  const url = (import.meta.env.DEV) ? `/api/${path}` : new URL(path, import.meta.env.VITE_BACKEND_URL);
  return url.toString();
}

export function makeMLBackendURL(path: string) {
  const url = (import.meta.env.DEV) ? `/ml-api/${path}` : new URL(path, import.meta.env.VITE_ML_BACKEND_URL);
  return url.toString();
}
