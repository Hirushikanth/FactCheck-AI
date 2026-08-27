import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { createElement } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { AppContext } from "../app-context";
import { createInitialActivityState } from "../activity/reducer";
import { createSession, listSessions } from "../api/client";
import { appendAssistantMessage } from "./chatMessages";
import { SessionScreen } from "./SessionScreen";

vi.mock("../api/client", () => ({
  createSession: vi.fn(),
  getSession: vi.fn(),
  listSessions: vi.fn(),
  postMessage: vi.fn(),
}));

vi.mock("../hooks/useSessionStream", () => ({
  useSessionStream: vi.fn(),
}));

import { useSessionStream } from "../hooks/useSessionStream";

describe("appendAssistantMessage", () => {
  it("does not append a dialogue reply already restored from session history", () => {
    const existing = [
      { role: "user" as const, content: "Where is Earth?" },
      { role: "assistant" as const, content: "Earth is the third planet from the Sun." },
    ];

    expect(appendAssistantMessage(existing, "Earth is the third planet from the Sun.")).toEqual(existing);
  });
});

describe("SessionScreen session identity", () => {
  beforeEach(() => {
    Object.defineProperty(HTMLElement.prototype, "scrollIntoView", {
      configurable: true,
      value: vi.fn(),
    });
    vi.mocked(listSessions).mockResolvedValue([]);
    vi.mocked(useSessionStream).mockReturnValue({
      state: {
        activity: createInitialActivityState(),
        thinkingEnabled: false,
        pipelineDone: false,
        sessionStatus: null,
        pipelineError: null,
      },
      setThinkingEnabled: vi.fn(),
      connectStream: vi.fn(),
      startNewActivity: vi.fn(),
    } as unknown as ReturnType<typeof useSessionStream>);
  });

  it("creates and posts one UUID, then clears it when creation fails", async () => {
    const sessionId = "11111111-1111-4111-8111-111111111111";
    const createSessionMock = vi.mocked(createSession);
    createSessionMock.mockRejectedValueOnce(new Error("Backend unavailable"));
    const randomUuid = vi
      .spyOn(globalThis.crypto, "randomUUID")
      .mockReturnValueOnce("22222222-2222-4222-8222-222222222222")
      .mockReturnValueOnce(sessionId);
    const setActiveSessionId = vi.fn();
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });

    render(
      createElement(
        QueryClientProvider,
        { client: queryClient },
        createElement(
          AppContext.Provider,
          {
            value: {
              activeTab: "session",
              setActiveTab: vi.fn(),
              activeSessionId: null,
              setActiveSessionId,
              activeSession: null,
              setActiveSession: vi.fn(),
            },
          },
          createElement(SessionScreen),
        ),
      ),
    );

    await userEvent.type(screen.getByPlaceholderText("Enter a claim to fact-check…"), "A claim");
    await userEvent.click(screen.getByRole("button", { name: "Send" }));

    await waitFor(() => expect(createSessionMock).toHaveBeenCalledWith("A claim", sessionId));
    expect(randomUuid).toHaveBeenCalledTimes(2);
    expect(setActiveSessionId).toHaveBeenNthCalledWith(1, sessionId);
    expect(setActiveSessionId).toHaveBeenNthCalledWith(2, null);
    expect(setActiveSessionId.mock.invocationCallOrder[0]).toBeLessThan(
      createSessionMock.mock.invocationCallOrder[0],
    );
  });
});
