export function makeBackendURL(path: string) {
  const url = new URL(path, import.meta.env.VITE_BACKEND_URL);
  return url.toString();
}

