import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { ActivityTimeline } from "./ActivityTimeline";
import { createInitialActivityState, reduceActivityEvent } from "../activity/reducer";

function activeTimeline() {
  let state = createInitialActivityState();
  state = reduceActivityEvent(state, {
    type: "agent_progress",
    data: {
      agent: "verifier",
      stage: "claim_verification",
      status: "started",
      message: "Checking claim 1 of 1.",
    },
  }, 100);
  state = reduceActivityEvent(state, {
    type: "search_progress",
    data: { claim_index: 0, query_index: 0, total_queries: 1, status: "started" },
  }, 200);
  state = reduceActivityEvent(state, {
    type: "agent_progress",
    data: {
      agent: "verifier",
      stage: "claim_verification",
      status: "retrying",
      attempt: 2,
      max_attempts: 3,
      message: "Model unavailable — retrying (2/3)",
    },
  }, 300);
  state = reduceActivityEvent(state, {
    type: "thinking_chunk",
    data: {
      agent: "verifier",
      stage: "claim_verification",
      claim_index: 0,
      text: "Comparing the evidence.",
    },
  }, 400);
  return state;
}

describe("ActivityTimeline", () => {
  it("renders an accessible agent hierarchy with nested verifier activity", () => {
    render(<ActivityTimeline timeline={activeTimeline()} thinkingEnabled />);

    expect(screen.getByRole("heading", { name: "Activity" })).toBeInTheDocument();
    expect(screen.getByText("Verifier")).toBeInTheDocument();
    expect(screen.getByText("Claim 1 — searching evidence")).toBeInTheDocument();
    expect(screen.getByText("Model unavailable — retrying (2/3)")).toBeInTheDocument();
    expect(screen.getByText("Model-emitted thinking — experimental, not evidence")).toBeInTheDocument();
  });

  it("collapses and expands the activity and inline thinking with keyboard controls", async () => {
    const user = userEvent.setup();
    render(<ActivityTimeline timeline={activeTimeline()} thinkingEnabled />);

    const activityToggle = screen.getByRole("button", { name: /activity/i });
    await user.click(activityToggle);
    expect(activityToggle).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByText("Verifier")).not.toBeInTheDocument();
    activityToggle.focus();
    await user.keyboard("{Enter}");
    expect(activityToggle).toHaveAttribute("aria-expanded", "true");

    const thinkingToggle = screen.getByRole("button", { name: /model-emitted thinking/i });
    await user.click(thinkingToggle);
    expect(thinkingToggle).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByText("Comparing the evidence.")).not.toBeInTheDocument();
  });

  it("does not render an empty thinking region when thinking is unsupported", () => {
    const timeline = createInitialActivityState();
    render(<ActivityTimeline timeline={timeline} thinkingEnabled />);
    expect(screen.queryByText(/model-emitted thinking/i)).not.toBeInTheDocument();
  });

  it("uses dialogue-specific copy and only shows the dialogue agent", () => {
    let timeline = createInitialActivityState();
    timeline = reduceActivityEvent(timeline, {
      type: "agent_progress",
      data: {
        agent: "dialogue",
        stage: "response",
        status: "completed",
        message: "Response ready.",
      },
    }, 100);

    render(<ActivityTimeline timeline={timeline} mode="dialogue" />);

    expect(screen.getByText("Follow-up complete — the source-grounded response is ready.")).toBeInTheDocument();
    expect(screen.getByText("Dialogue")).toBeInTheDocument();
    expect(screen.queryByText("Extractor")).not.toBeInTheDocument();
  });
});
