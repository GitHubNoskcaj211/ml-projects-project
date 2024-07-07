import { auth } from "./firebase";

async function authFetch(url: string, init?: RequestInit | undefined) {
  const options = init || {};
  options.mode = "cors";
  options.headers = options.headers || {};
  
  const token = await auth.currentUser!.getIdToken();
  const headers = new Headers(options.headers);
  headers.set("Authorization", `Bearer ${token}`);
  options.headers = headers;

  return await fetch(url, options);
}

export async function backendAuthFetch(url: string, init?: RequestInit | undefined) {
  return await authFetch(makeBackendURL(url), init);
}

export async function mlBackendAuthFetch(url: string, init?: RequestInit | undefined) {
  return await authFetch(makeMLBackendURL(url), init);
}

export function makeBackendURL(path: string) {
  const url = (import.meta.env.DEV) ? `/api/${path}` : new URL(path, import.meta.env.VITE_BACKEND_URL);
  return url.toString();
}

export function makeMLBackendURL(path: string) {
  const url = (import.meta.env.DEV) ? `/ml-api/${path}` : new URL(path, import.meta.env.VITE_ML_BACKEND_URL);
  return url.toString();
}

export function delay(ms: number) {
  return new Promise(resolve => setTimeout(resolve, ms));
}
