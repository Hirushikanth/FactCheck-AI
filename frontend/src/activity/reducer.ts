import type {
  ActivityAction,
  ActivityAgent,
  ActivityClaim,
  ActivitySearch,
  ActivityTimelineState,
  PipelineAgent,
  TimelineAgentStatus,
} from "./types";
import { AGENT_LABELS, AGENT_ORDER } from "./types";

export interface ActivityEvent {
  type: string;
  data: Record<string, unknown>;
}

const THINKING_MAX_CHARS = 12_000;

function createAgent(agent: PipelineAgent): ActivityAgent {
  return {
    agent,
    label: AGENT_LABELS[agent],
    status: "pending",
    actions: [],
    claims: [],
  };
}

export function createInitialActivityState(): ActivityTimelineState {
  return {
    agents: {
      extractor: createAgent("extractor"),
      verifier: createAgent("verifier"),
      reporter: createAgent("reporter"),
      dialogue: createAgent("dialogue"),
    },
    expanded: true,
    thinkingSupported: false,
    thinkingTruncated: false,
  };
}

function asAgent(value: unknown): PipelineAgent | null {
  return typeof value === "string" && AGENT_ORDER.includes(value as PipelineAgent)
    ? (value as PipelineAgent)
    : null;
}

function text(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}

function number(value: unknown, fallback = 0): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function status(value: unknown): ActivityAgent["status"] {
  if (value === "degraded") return "degraded";
  if (value === "completed") return "completed";
  if (value === "started" || value === "retrying") return "active";
  return "pending";
}

function actionStatus(value: unknown): ActivityAction["status"] {
  if (value === "degraded") return "degraded";
  if (value === "completed") return "completed";
  if (value === "failed") return "failed";
  if (value === "started" || value === "retrying") return "active";
  return "pending";
}

function claimLabel(verdict: string | undefined): string | undefined {
  if (!verdict) return undefined;
  const labels: Record<string, string> = {
    SUPPORTED: "verified",
    REFUTED: "refuted",
    INSUFFICIENT_EVIDENCE: "insufficient evidence",
    CONFLICTING_EVIDENCE: "conflicting evidence",
  };
  return labels[verdict] ?? verdict.toLowerCase().replaceAll("_", " ");
}

function cloneAgents(state: ActivityTimelineState): Record<PipelineAgent, ActivityAgent> {
  return Object.fromEntries(
    AGENT_ORDER.map((agent) => [
      agent,
      {
        ...state.agents[agent],
        actions: state.agents[agent].actions.map((action) => ({
          ...action,
          searches: action.searches.map((search) => ({ ...search })),
        })),
        claims: state.agents[agent].claims.map((claim) => ({
          ...claim,
          searches: claim.searches.map((search) => ({ ...search })),
        })),
      },
    ]),
  ) as Record<PipelineAgent, ActivityAgent>;
}

function ensureAction(
  agent: ActivityAgent,
  stage: string,
  message: string,
  now: number,
  claimIndex?: number,
): ActivityAction {
  const actionId = `${stage}:${claimIndex ?? "agent"}`;
  const existing = agent.actions.find((action) => action.id === actionId);
  if (existing) {
    existing.message = message || existing.message;
    return existing;
  }
  const action: ActivityAction = {
    id: actionId,
    stage,
    message,
    status: "active",
    claimIndex,
    thinking: "",
    thinkingTruncated: false,
    startedAt: now,
    searches: [],
  };
  agent.actions.push(action);
  return action;
}

function ensureClaim(agent: ActivityAgent, index: number, claimText = ""): ActivityClaim {
  const existing = agent.claims.find((claim) => claim.index === index);
  if (existing) {
    if (claimText) existing.claim = claimText;
    return existing;
  }
  // Keep the indexed list stable when events for a later claim arrive first.
  for (let placeholderIndex = 0; placeholderIndex < index; placeholderIndex += 1) {
    if (!agent.claims.some((claim) => claim.index === placeholderIndex)) {
      agent.claims.push({
        index: placeholderIndex,
        claim: `Claim ${placeholderIndex + 1}`,
        status: "pending",
        thinking: "",
        thinkingTruncated: false,
        searches: [],
      });
    }
  }
  const claim: ActivityClaim = {
    index,
    claim: claimText || `Claim ${index + 1}`,
    status: "pending",
    thinking: "",
    thinkingTruncated: false,
    searches: [],
  };
  agent.claims.push(claim);
  agent.claims.sort((a, b) => a.index - b.index);
  return claim;
}

function applyThinking(target: { thinking: string; thinkingTruncated: boolean }, chunk: string, truncated: boolean): boolean {
  if (!chunk && !truncated) return false;
  const combined = `${target.thinking}${chunk}`;
  if (combined.length > THINKING_MAX_CHARS) {
    target.thinking = combined.slice(-THINKING_MAX_CHARS);
    target.thinkingTruncated = true;
  } else {
    target.thinking = combined;
  }
  target.thinkingTruncated ||= truncated;
  return true;
}

function updateAgentStatus(
  agent: ActivityAgent,
  nextStatus: TimelineAgentStatus,
  now: number,
  message: string,
): void {
  if (!agent.startedAt) agent.startedAt = now;
  agent.status = nextStatus;
  if (message) agent.summary = message;
  if (nextStatus === "completed" || nextStatus === "degraded") {
    agent.completedAt = now;
    agent.durationMs = Math.max(0, now - (agent.startedAt ?? now));
  }
}

export function reduceActivityEvent(
  state: ActivityTimelineState,
  event: ActivityEvent,
  now = typeof performance === "undefined" ? Date.now() : performance.now(),
): ActivityTimelineState {
  const data = event.data ?? {};
  const nextAgents = cloneAgents(state);

  if (event.type === "agent_start") {
    const agent = asAgent(data.agent);
    if (!agent) return state;
    updateAgentStatus(nextAgents[agent], "active", now, "");
    return { ...state, agents: nextAgents };
  }

  if (event.type === "agent_progress") {
    const agentName = asAgent(data.agent);
    if (!agentName) return state;
    const agent = nextAgents[agentName];
    const stage = text(data.stage, `${agentName}_stage`);
    const message = text(data.message);
    const eventStatus = text(data.status);
    const claimMatch = message.match(/claim\s+(\d+)/i);
    const claimIndex = claimMatch
      ? Math.max(0, Number(claimMatch[1]) - 1)
      : agentName === "verifier" && stage === "claim_verification" && agent.claims.length > 0
        ? agent.claims[agent.claims.length - 1].index
        : undefined;
    const claimAction = agentName === "verifier" && stage === "claim_verification";
    const action = ensureAction(agent, stage, message, now, claimAction ? claimIndex : undefined);
    action.status = actionStatus(eventStatus);
    if (eventStatus === "completed" || eventStatus === "degraded") {
      action.completedAt = now;
      action.startedAt ??= now;
    }
    if (claimAction && claimIndex !== undefined) {
      const claim = ensureClaim(agent, claimIndex);
      claim.status = action.status;
      claim.actionId = action.id;
      if (eventStatus === "degraded") claim.summary = message;
      if (eventStatus === "retrying") claim.summary = message;
    }

    // Claim-level verifier progress describes a child action, not the whole agent.
    if (!(claimAction && claimIndex !== undefined)) {
      updateAgentStatus(agent, status(data.status), now, message);
    } else if (agent.status === "pending") {
      updateAgentStatus(agent, "active", now, "");
    }
    return { ...state, agents: nextAgents };
  }

  if (event.type === "claim_found") {
    const agent = nextAgents.verifier;
    ensureClaim(agent, Math.max(0, Math.floor(number(data.index))), text(data.claim));
    return { ...state, agents: nextAgents };
  }

  if (event.type === "search_progress") {
    const agent = nextAgents.verifier;
    const claimIndex = Math.max(0, Math.floor(number(data.claim_index)));
    const queryIndex = Math.max(0, Math.floor(number(data.query_index)));
    const claim = ensureClaim(agent, claimIndex);
    const id = `search:${claimIndex}:${queryIndex}`;
    const search: ActivitySearch = claim.searches.find((item) => item.id === id) ?? {
      id,
      queryIndex,
      totalQueries: Math.max(0, Math.floor(number(data.total_queries))),
      status: "pending",
    };
    search.status = data.status === "completed" ? "completed" : data.status === "failed" ? "failed" : "active";
    if (typeof data.provider === "string") search.provider = data.provider;
    if (typeof data.result_count === "number") search.resultCount = Math.max(0, data.result_count);
    if (!claim.searches.some((item) => item.id === id)) claim.searches.push(search);
    claim.status = search.status === "failed" ? "failed" : "active";
    const action = ensureAction(agent, "claim_verification", `Claim ${claimIndex + 1} — searching evidence`, now, claimIndex);
    action.status = search.status === "failed" ? "failed" : "active";
    action.searches = claim.searches.map((item) => ({ ...item }));
    claim.actionId = action.id;
    if (agent.status === "pending") updateAgentStatus(agent, "active", now, "");
    return { ...state, agents: nextAgents };
  }

  if (event.type === "verdict_ready") {
    const agent = nextAgents.verifier;
    const index = Math.max(0, Math.floor(number(data.index)));
    const verdict = text(data.verdict);
    const claim = ensureClaim(agent, index, text(data.claim));
    claim.verdict = verdict;
    claim.verdictLabel = claimLabel(verdict);
    claim.confidence = number(data.confidence);
    claim.status = data.processing_status === "degraded" || data.processing_status === "error" ? "degraded" : "completed";
    claim.summary = text(data.degraded_reason, claim.verdictLabel ? `Claim ${index + 1} — ${claim.verdictLabel}.` : "Claim checked.");
    const action = ensureAction(agent, "claim_verification", `Claim ${index + 1} — ${claim.verdictLabel ?? "checked"}`, now, index);
    action.status = claim.status;
    action.message = claim.summary;
    claim.actionId = action.id;
    if (claim.status === "degraded") agent.status = "degraded";
    return { ...state, agents: nextAgents };
  }

  if (event.type === "thinking_chunk") {
    const agentName = asAgent(data.agent);
    if (!agentName) return state;
    const agent = nextAgents[agentName];
    const claimIndex = typeof data.claim_index === "number" ? Math.max(0, Math.floor(data.claim_index)) : undefined;
    const action = ensureAction(agent, text(data.stage, "activity"), "", now, claimIndex);
    const changed = applyThinking(action, text(data.text), Boolean(data.truncated));
    if (!changed) return state;
    if (claimIndex !== undefined) {
      const claim = ensureClaim(agent, claimIndex);
      applyThinking(claim, text(data.text), Boolean(data.truncated));
      claim.actionId = action.id;
    }
    return {
      ...state,
      agents: nextAgents,
      thinkingSupported: true,
      thinkingTruncated: state.thinkingTruncated || Boolean(data.truncated) || action.thinkingTruncated,
    };
  }

  if (event.type === "pipeline_done") {
    for (const agent of AGENT_ORDER) {
      const item = nextAgents[agent];
      if (item.status === "active") updateAgentStatus(item, "completed", now, item.summary ?? "Complete.");
    }
    return { ...state, agents: nextAgents };
  }

  if (event.type === "pipeline_error") {
    const agentName = asAgent(data.agent);
    if (!agentName) return state;
    updateAgentStatus(nextAgents[agentName], "degraded", now, "Pipeline could not complete.");
    return { ...state, agents: nextAgents };
  }

  return state;
}
