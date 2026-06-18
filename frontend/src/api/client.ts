import type {
  HealthResponse,
  SessionDetail,
  SessionSummary,
} from "./types";

const API_BASE = "/api";

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text().catch(() => `HTTP ${res.status}`);
    throw new Error(text || `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`);
  return handleResponse<HealthResponse>(res);
}

export async function createSession(
  input: string
): Promise<{ session_id: string; status: string }> {
  const res = await fetch(`${API_BASE}/sessions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ input }),
  });
  return handleResponse(res);
}

export async function getSession(sessionId: string): Promise<SessionDetail> {
  const res = await fetch(`${API_BASE}/sessions/${sessionId}`);
  return handleResponse<SessionDetail>(res);
}

export async function listSessions(): Promise<SessionSummary[]> {
  const res = await fetch(`${API_BASE}/sessions`);
  return handleResponse<SessionSummary[]>(res);
}

export async function postMessage(
  sessionId: string,
  message: string
): Promise<{ message_id: string }> {
  const res = await fetch(`${API_BASE}/sessions/${sessionId}/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });
  if (res.status === 409) {
    throw new Error("Session pipeline is not finished yet");
  }
  return handleResponse(res);
}

export async function deleteSession(
  sessionId: string
): Promise<{ deleted: boolean }> {
  const res = await fetch(`${API_BASE}/sessions/${sessionId}`, {
    method: "DELETE",
  });
  return handleResponse(res);
}

export function getStreamUrl(sessionId: string): string {
  return `${API_BASE}/sessions/${sessionId}/stream`;
}
