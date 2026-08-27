import { afterEach, describe, expect, it, vi } from "vitest";
import { createSession, getHealth } from "./client";

describe("API client", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("surfaces the backend error body when a request fails", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response("Backend unavailable", {
          status: 503,
          headers: { "Content-Type": "text/plain" },
        })
      )
    );

    await expect(getHealth()).rejects.toThrow("Backend unavailable");
  });

  it("posts the browser-created session ID", async () => {
    const sessionId = "11111111-1111-4111-8111-111111111111";
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ session_id: sessionId, status: "running" }), {
          status: 202,
          headers: { "Content-Type": "application/json" },
        }),
      ),
    );

    await createSession("A claim", sessionId);

    expect(fetch).toHaveBeenCalledWith(
      "/api/sessions",
      expect.objectContaining({
        body: JSON.stringify({ input: "A claim", session_id: sessionId }),
      }),
    );
  });
});
