import { afterEach, describe, expect, it, vi } from "vitest";
import { getHealth } from "./client";

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
});
