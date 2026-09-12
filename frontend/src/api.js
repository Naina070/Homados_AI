const explicit = import.meta.env.VITE_API_URL;

export const API_BASE = (explicit || "").replace(/\/$/, "");

export function apiUrl(path) {
  if (!API_BASE) return path;
  return `${API_BASE}${path}`;
}

export async function getJson(path) {
  const res = await fetch(apiUrl(path));
  if (!res.ok) throw new Error(`${path} ${res.status}`);
  return res.json();
}

export async function postJson(path, body) {
  const res = await fetch(apiUrl(path), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${path} ${res.status}`);
  return res.json();
}

export async function postFile(path, file, extra = {}) {
  const form = new FormData();
  form.append("file", file);
  Object.entries(extra).forEach(([k, v]) => {
    if (v != null) form.append(k, v);
  });
  const res = await fetch(apiUrl(path), { method: "POST", body: form });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `${path} ${res.status}`);
  }
  return res.json();
}
