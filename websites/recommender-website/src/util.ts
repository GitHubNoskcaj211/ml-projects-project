export function makeBackendURL(path: string) {
  const url = (import.meta.env.DEV) ? `/api/${path}` : new URL(path, import.meta.env.VITE_BACKEND_URL);
  return url.toString();
}

