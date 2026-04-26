const API_BASE = import.meta.env.VITE_API_URL ?? "";

export async function predictImage(file) {
  const body = new FormData();
  body.append("image", file);

  const response = await fetch(`${API_BASE}/predict`, { method: "POST", body });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail ?? "Request failed");
  }

  return response.json();
}
