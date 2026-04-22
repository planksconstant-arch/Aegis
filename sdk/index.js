const defaultBaseUrl = process.env.LOCAL_IDE_AGENT_URL || "http://127.0.0.1:8765";

async function request(path, options = {}) {
  const response = await fetch(`${defaultBaseUrl}${path}`, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Bridge request failed: ${response.status} ${text}`);
  }

  return response.json();
}

export async function registerClient(payload) {
  return request("/clients/register", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export async function observe(payload) {
  return request("/observe", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export async function act(payload) {
  return request("/act", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export async function sendFeedback(payload) {
  return request("/feedback", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export async function getMemory(userId = "default") {
  const encoded = encodeURIComponent(userId);
  return request(`/memory?user_id=${encoded}`);
}
