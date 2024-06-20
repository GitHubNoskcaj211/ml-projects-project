import { auth } from "./firebase";

export async function backendAuthFetch(path: string, init?: RequestInit | undefined) {
  const options = init || {};
  options.mode = "cors";
  options.headers = options.headers || {};
  
  const token = await auth.currentUser!.getIdToken();
  const headers = new Headers(options.headers);
  headers.set("Authorization", `Bearer ${token}`);
  options.headers = headers;

  return await fetch(makeBackendURL(path), options);
}

export function makeBackendURL(path: string) {
  const url = (import.meta.env.DEV) ? `/api/${path}` : new URL(path, import.meta.env.VITE_BACKEND_URL);
  return url.toString();
}

export function makeMLBackendURL(path: string) {
  const url = (import.meta.env.DEV) ? `/ml-api/${path}` : new URL(path, import.meta.env.VITE_ML_BACKEND_URL);
  return url.toString();
}
