// Mirrors backend/app/schemas/sessions.py and backend/factcheck/state.py

export type Verdict =
  | "SUPPORTED"
  | "REFUTED"
  | "INSUFFICIENT_EVIDENCE"
  | "CONFLICTING_EVIDENCE";

export type SessionStatus = "running" | "done" | "error";
export type RunTrigger = "initial" | "dialogue";

export interface ClaimResult {
  claim: string;
  verdict: Verdict;
  confidence: number;
  evidence: string[];
  sources: string[];
  reasoning: string;
  search_queries: string[];
  source_sentence?: string | null;
  fidelity_status?: string | null;
  processing_status?: "ok" | "error" | "degraded";
  processing_error?: string | null;
}

export interface FactCheckRunSummary {
  run_id: string;
  sequence: number;
  raw_input: string;
  status: string;
  triggered_by: RunTrigger;
  created_at: number;
}

export interface DialogueMessage {
  role: "user" | "assistant";
  content: string;
  created_at: number;
}

export interface SessionSummary {
  session_id: string;
  raw_input: string;
  status: SessionStatus;
  created_at: number;
  updated_at: number;
}

export interface SessionDetail {
  session_id: string;
  active_run_id: string | null;
  raw_input: string;
  status: SessionStatus;
  final_report: string | null;
  error: string | null;
  claim_results: ClaimResult[];
  messages: DialogueMessage[];
  runs: FactCheckRunSummary[];
  created_at: number;
  updated_at: number;
}

export interface HealthResponse {
  status: "ok" | "error";
  ollama_reachable: boolean;
  model_loaded: boolean;
  ollama_base_url: string;
  ollama_model: string;
}

// SSE event payloads
export interface SseStreamOpen {
  session_id: string;
  run_id: string | null;
  replay_count: number;
  hub_state: "open" | "closed";
  server_time: string;
}

export interface SseAgentStart {
  agent: "extractor" | "verifier" | "reporter" | "dialogue";
  timestamp: string;
}

export interface SseClaimFound {
  claim: string;
  index: number;
  total: number;
}

export interface SseVerdictReady {
  claim: string;
  verdict: Verdict;
  confidence: number;
  index: number;
  total: number;
}

export interface SseReportReady {
  final_report: string;
}

export interface SsePipelineDone {
  session_id: string;
  duration_seconds: number;
}

export interface SsePipelineError {
  error: string;
  agent: string;
}

export interface SseDialogueReply {
  message: string;
}
