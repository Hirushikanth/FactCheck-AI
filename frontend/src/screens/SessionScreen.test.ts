import { describe, expect, it } from "vitest";
import { appendAssistantMessage } from "./chatMessages";

describe("appendAssistantMessage", () => {
  it("does not append a dialogue reply already restored from session history", () => {
    const existing = [
      { role: "user" as const, content: "Where is Earth?" },
      { role: "assistant" as const, content: "Earth is the third planet from the Sun." },
    ];

    expect(appendAssistantMessage(existing, "Earth is the third planet from the Sun.")).toEqual(existing);
  });
});
