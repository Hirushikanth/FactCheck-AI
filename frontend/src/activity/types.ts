export type PipelineAgent = "extractor" | "verifier" | "reporter" | "dialogue";
export type TimelineAgentStatus = "pending" | "active" | "completed" | "degraded";
export type ActivityActionStatus = "pending" | "active" | "completed" | "degraded" | "failed";

export interface ActivitySearch {
  id: string;
  queryIndex: number;
  totalQueries: number;
  provider?: string;
  status: "pending" | "active" | "completed" | "failed";
  resultCount?: number;
}

export interface ActivityAction {
  id: string;
  stage: string;
  message: string;
  status: ActivityActionStatus;
  claimIndex?: number;
  thinking: string;
  thinkingTruncated: boolean;
  startedAt?: number;
  completedAt?: number;
  searches: ActivitySearch[];
}

export interface ActivityClaim {
  index: number;
  claim: string;
  status: ActivityActionStatus;
  verdict?: string;
  verdictLabel?: string;
  confidence?: number;
  summary?: string;
  thinking: string;
  thinkingTruncated: boolean;
  searches: ActivitySearch[];
  actionId?: string;
}

export interface ActivityAgent {
  agent: PipelineAgent;
  label: string;
  status: TimelineAgentStatus;
  startedAt?: number;
  completedAt?: number;
  durationMs?: number;
  summary?: string;
  actions: ActivityAction[];
  claims: ActivityClaim[];
}

export interface ActivityTimelineState {
  agents: Record<PipelineAgent, ActivityAgent>;
  expanded: boolean;
  thinkingSupported: boolean;
  thinkingTruncated: boolean;
}

export const AGENT_ORDER: PipelineAgent[] = [
  "extractor",
  "verifier",
  "reporter",
  "dialogue",
];

export const AGENT_LABELS: Record<PipelineAgent, string> = {
  extractor: "Extractor",
  verifier: "Verifier",
  reporter: "Reporter",
  dialogue: "Dialogue",
};
