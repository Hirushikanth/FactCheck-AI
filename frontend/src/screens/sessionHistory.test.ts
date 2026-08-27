import { describe, expect, it } from "vitest";
import { buildHistoricChatMessages } from "./sessionHistory";

describe("buildHistoricChatMessages", () => {
  it("keeps the initial user message when returning to an active session", () => {
    const messages = buildHistoricChatMessages({
      raw_input: "The Moon is made of cheese.",
      final_report: "# Fact-Check Report",
      runs: [{ run_id: "run-1", raw_input: "The Moon is made of cheese." }],
      messages: [],
    });

    expect(messages).toEqual(expect.arrayContaining([
      expect.objectContaining({ role: "user", content: "The Moon is made of cheese." }),
    ]));
  });
});
