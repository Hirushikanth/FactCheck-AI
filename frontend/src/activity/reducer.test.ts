import { describe, expect, it } from "vitest";
import {
  createInitialActivityState,
  reduceActivityEvent,
  type ActivityEvent,
} from "./reducer";

const event = (type: string, data: Record<string, unknown>): ActivityEvent => ({
  type,
  data,
});

describe("activity reducer", () => {
  it("records agent duration and completion summary", () => {
    let state = createInitialActivityState();
    state = reduceActivityEvent(
      state,
      event("agent_progress", {
        agent: "extractor",
        stage: "claim_extraction",
        status: "started",
        message: "Extractor started.",
      }),
      100,
    );
    state = reduceActivityEvent(
      state,
      event("agent_progress", {
        agent: "extractor",
        stage: "claim_extraction",
        status: "completed",
        message: "3 claims extracted.",
      }),
      2_200,
    );

    expect(state.agents.extractor.status).toBe("completed");
    expect(state.agents.extractor.durationMs).toBe(2_100);
    expect(state.agents.extractor.summary).toBe("3 claims extracted.");
  });

  it("nests search progress and verdict results below verifier claims", () => {
    let state = createInitialActivityState();
    state = reduceActivityEvent(
      state,
      event("agent_progress", {
        agent: "verifier",
        stage: "claim_verification",
        status: "started",
        message: "Checking claim 1 of 2.",
      }),
      100,
    );
    state = reduceActivityEvent(
      state,
      event("search_progress", {
        claim_index: 0,
        query_index: 0,
        total_queries: 2,
        provider: "web",
        status: "started",
      }),
      200,
    );
    state = reduceActivityEvent(
      state,
      event("search_progress", {
        claim_index: 0,
        query_index: 0,
        total_queries: 2,
        provider: "web",
        status: "completed",
        result_count: 8,
      }),
      300,
    );
    state = reduceActivityEvent(
      state,
      event("verdict_ready", {
        claim: "The claim is true.",
        verdict: "SUPPORTED",
        confidence: 0.9,
        index: 0,
        total: 2,
        processing_status: "ok",
      }),
      400,
    );

    const claim = state.agents.verifier.claims[0];
    expect(claim.status).toBe("completed");
    expect(claim.verdictLabel).toBe("verified");
    expect(claim.searches[0].resultCount).toBe(8);
    expect(claim.searches[0].provider).toBe("web");
  });

  it("keeps retry copy human-friendly and records degraded verdicts", () => {
    let state = createInitialActivityState();
    state = reduceActivityEvent(
      state,
      event("agent_progress", {
        agent: "verifier",
        stage: "claim_verification",
        status: "retrying",
        attempt: 2,
        max_attempts: 3,
        message: "Model unavailable — retrying (2/3)",
      }),
      100,
    );
    state = reduceActivityEvent(
      state,
      event("verdict_ready", {
        claim: "A claim",
        verdict: "INSUFFICIENT_EVIDENCE",
        confidence: 0,
        index: 0,
        total: 1,
        processing_status: "degraded",
        degraded_reason: "Verification completed in degraded mode.",
      }),
      200,
    );

    expect(state.agents.verifier.status).toBe("degraded");
    expect(state.agents.verifier.actions.some((action) => action.message.includes("retrying (2/3)"))).toBe(true);
    expect(state.agents.verifier.claims[0].status).toBe("degraded");
    expect(state.agents.verifier.claims[0].summary).toBe("Verification completed in degraded mode.");
  });

  it("attaches thinking to the current claim action and keeps newest text", () => {
    let state = createInitialActivityState();
    state = reduceActivityEvent(
      state,
      event("search_progress", {
        claim_index: 1,
        query_index: 0,
        total_queries: 1,
        status: "started",
      }),
      100,
    );
    state = reduceActivityEvent(
      state,
      event("thinking_chunk", {
        agent: "verifier",
        stage: "claim_verification",
        claim_index: 1,
        text: "First thought. ",
      }),
      200,
    );
    state = reduceActivityEvent(
      state,
      event("thinking_chunk", {
        agent: "verifier",
        stage: "claim_verification",
        claim_index: 1,
        text: "Second thought.",
        truncated: true,
      }),
      300,
    );

    const claim = state.agents.verifier.claims[1];
    expect(claim.thinking).toBe("First thought. Second thought.");
    expect(claim.thinkingTruncated).toBe(true);
    expect(state.thinkingSupported).toBe(true);
  });

  it("ignores unknown events", () => {
    const state = createInitialActivityState();
    expect(reduceActivityEvent(state, event("something_new", { value: 1 }))).toEqual(state);
  });
});
