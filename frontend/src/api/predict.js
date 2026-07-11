export async function predictImage(file) {
  const body = new FormData();
  body.append("image", file);

  const response = await fetch("/api/predict", { method: "POST", body });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail ?? "Request failed");
  }

  return response.json();
}
