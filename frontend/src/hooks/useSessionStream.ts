import { useCallback, useEffect, useRef, useState } from "react";
import { getStreamUrl, getSession } from "../api/client";
import type {
  SseAgentStart,
  SseClaimFound,
  SseDialogueReply,
  SsePipelineDone,
  SsePipelineError,
  SseReportReady,
  SseStreamOpen,
  SseVerdictReady,
} from "../api/types";

export type PipelineAgent = "extractor" | "verifier" | "reporter" | "dialogue";
export type PipelineStep = {
  agent: PipelineAgent;
  status: "done" | "active" | "pending";
};

export type StreamStatus =
  | "idle"
  | "connecting"
  | "connected"
  | "closed"
  | "error";

export interface StreamState {
  streamStatus: StreamStatus;
  activeAgents: Set<PipelineAgent>;
  completedAgents: Set<PipelineAgent>;
  claimsFound: SseClaimFound[];
  verdicts: SseVerdictReady[];
  finalReport: string | null;
  dialogueReplies: string[];
  pipelineError: string | null;
  pipelineDone: boolean;
  // Refreshed session after pipeline_done
  sessionStatus: "running" | "done" | "error" | null;
}

interface UseSessionStreamOptions {
  onSessionRefreshed?: (sessionId: string) => void;
  onDialogueReply?: (reply: string) => void;
  onReportReady?: (report: string) => void;
}

const AGENT_ORDER: PipelineAgent[] = [
  "extractor",
  "verifier",
  "reporter",
  "dialogue",
];

const INITIAL_STATE: StreamState = {
  streamStatus: "idle",
  activeAgents: new Set(),
  completedAgents: new Set(),
  claimsFound: [],
  verdicts: [],
  finalReport: null,
  dialogueReplies: [],
  pipelineError: null,
  pipelineDone: false,
  sessionStatus: null,
};

export function useSessionStream(
  sessionId: string | null,
  options: UseSessionStreamOptions = {}
) {
  const [state, setState] = useState<StreamState>(INITIAL_STATE);

  // Refs to avoid stale closures in async callbacks
  const abortRef = useRef<AbortController | null>(null);
  const reconnectCountRef = useRef(0);
  const orphanedRetryUsedRef = useRef(false);
  const sessionStatusRef = useRef<StreamState["sessionStatus"]>(null);
  const optionsRef = useRef(options);
  optionsRef.current = options;

  const reset = useCallback(() => {
    setState(INITIAL_STATE);
    reconnectCountRef.current = 0;
    orphanedRetryUsedRef.current = false;
    sessionStatusRef.current = null;
  }, []);

  const abort = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
  }, []);

  // Forward-declared so connectStream can call itself recursively
  const connectStreamRef = useRef<((id: string) => void) | null>(null);

  const handleEvent = useCallback(
    async (eventName: string, data: Record<string, unknown>, sessionId: string) => {
      if (eventName === "stream_open") {
        const payload = data as unknown as SseStreamOpen;
        setState((s) => ({
          ...s,
          streamStatus: "connected",
          pipelineDone: false,
          pipelineError: null,
        }));
        void payload;
        return;
      }

      if (eventName === "agent_start") {
        const payload = data as unknown as SseAgentStart;
        const agent = payload.agent as PipelineAgent;
        setState((s) => ({
          ...s,
          activeAgents: new Set([...s.activeAgents, agent]),
        }));
        return;
      }

      if (eventName === "claim_found") {
        const payload = data as unknown as SseClaimFound;
        setState((s) => ({
          ...s,
          claimsFound: [...s.claimsFound, payload],
        }));
        return;
      }

      if (eventName === "verdict_ready") {
        const payload = data as unknown as SseVerdictReady;
        setState((s) => ({
          ...s,
          verdicts: [...s.verdicts, payload],
          // Mark verifier as still active (moves to done on pipeline_done)
        }));
        return;
      }

      if (eventName === "report_ready") {
        const payload = data as unknown as SseReportReady;
        setState((s) => ({
          ...s,
          finalReport: payload.final_report,
          completedAgents: new Set([...s.completedAgents, "reporter"]),
          activeAgents: new Set(
            [...s.activeAgents].filter((a) => a !== "reporter")
          ),
        }));
        optionsRef.current.onReportReady?.(payload.final_report);
        return;
      }

      if (eventName === "dialogue_reply") {
        const payload = data as unknown as SseDialogueReply;
        setState((s) => ({
          ...s,
          dialogueReplies: [...s.dialogueReplies, payload.message],
        }));
        optionsRef.current.onDialogueReply?.(payload.message);
        return;
      }

      if (eventName === "pipeline_error") {
        const payload = data as unknown as SsePipelineError;
        setState((s) => ({
          ...s,
          streamStatus: "error",
          pipelineError: payload.error,
          sessionStatus: "error",
        }));
        sessionStatusRef.current = "error";

        // Still fetch session for final state
        try {
          const session = await getSession(sessionId);
          sessionStatusRef.current = session.status;
          optionsRef.current.onSessionRefreshed?.(sessionId);
        } catch {
          // ignore
        }
        return;
      }

      if (eventName === "pipeline_done") {
        void (data as unknown as SsePipelineDone);

        // Mark all in-flight agents as completed
        setState((s) => {
          const newCompleted = new Set([...s.completedAgents, ...s.activeAgents]);
          return {
            ...s,
            completedAgents: newCompleted,
            activeAgents: new Set<PipelineAgent>(),
            pipelineDone: true,
          };
        });

        // Fetch authoritative session state
        try {
          const session = await getSession(sessionId);
          sessionStatusRef.current = session.status;
          setState((s) => ({
            ...s,
            sessionStatus: session.status,
            finalReport: session.final_report ?? s.finalReport,
          }));
          optionsRef.current.onSessionRefreshed?.(sessionId);

          // If dialogue triggered a new factcheck, reconnect
          if (session.status === "running") {
            reconnectCountRef.current = 0;
            orphanedRetryUsedRef.current = false;
            connectStreamRef.current?.(sessionId);
          }
        } catch {
          // ignore secondary fetch errors
        }
        return;
      }
    },
    []
  );

  const readSSE = useCallback(
    async (
      response: Response,
      signal: AbortSignal,
      sessionId: string
    ): Promise<void> => {
      const reader = response.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop() ?? "";

        for (const block of parts) {
          if (!block.trim()) continue;

          let eventName = "message";
          const dataLines: string[] = [];

          for (const line of block.split("\n")) {
            if (line.startsWith(":")) continue;
            if (line.startsWith("event:")) {
              eventName = line.slice(6).trim();
            } else if (line.startsWith("data:")) {
              dataLines.push(line.slice(5).trimStart());
            }
          }

          if (dataLines.length === 0) continue;
          if (signal.aborted) return;

          let data: Record<string, unknown> = {};
          try {
            data = JSON.parse(dataLines.join("\n"));
          } catch {
            data = { raw: dataLines.join("\n") };
          }

          await handleEvent(eventName, data, sessionId);
        }
      }
    },
    [handleEvent]
  );

  const connectStream = useCallback(
    async (id: string) => {
      abort();

      const controller = new AbortController();
      abortRef.current = controller;
      const { signal } = controller;

      setState((s) => ({
        ...s,
        streamStatus: "connecting",
      }));

      const url = getStreamUrl(id);

      try {
        const response = await fetch(url, { signal });

        if (!response.ok) {
          // Handle 409 error codes
          if (response.status === 409) {
            let detail: { code?: string; session_status?: string; hint?: string } = {};
            try {
              const body = await response.json();
              detail = body.detail ?? {};
            } catch {
              // ignore
            }

            const code = detail.code ?? "unknown";

            if (code === "stream_missed") {
              setState((s) => ({
                ...s,
                streamStatus: "closed",
                sessionStatus: (detail.session_status as StreamState["sessionStatus"]) ?? null,
              }));
              try {
                const session = await getSession(id);
                sessionStatusRef.current = session.status;
                optionsRef.current.onSessionRefreshed?.(id);
              } catch {
                // ignore
              }
              return;
            }

            if (code === "pipeline_orphaned") {
              if (!orphanedRetryUsedRef.current) {
                orphanedRetryUsedRef.current = true;
                setState((s) => ({ ...s, streamStatus: "connecting" }));
                await new Promise((r) => setTimeout(r, 1000));
                if (!signal.aborted) connectStream(id);
                return;
              }
              setState((s) => ({ ...s, streamStatus: "error" }));
              try {
                await getSession(id);
                optionsRef.current.onSessionRefreshed?.(id);
              } catch {
                // ignore
              }
              return;
            }

            setState((s) => ({ ...s, streamStatus: "error" }));
            return;
          }

          setState((s) => ({
            ...s,
            streamStatus: "error",
          }));
          return;
        }

        await readSSE(response, signal, id);

        // Reconnect if still running and not explicitly aborted
        if (
          !signal.aborted &&
          sessionStatusRef.current === "running" &&
          reconnectCountRef.current < 1
        ) {
          reconnectCountRef.current += 1;
          setState((s) => ({ ...s, streamStatus: "connecting" }));
          connectStream(id);
        }
      } catch (err) {
        if (signal.aborted) return;

        if (
          sessionStatusRef.current === "running" &&
          reconnectCountRef.current < 1
        ) {
          reconnectCountRef.current += 1;
          setState((s) => ({ ...s, streamStatus: "connecting" }));
          connectStream(id);
          return;
        }

        setState((s) => ({
          ...s,
          streamStatus: "error",
          pipelineError: err instanceof Error ? err.message : "Stream error",
        }));
        if (id) {
          try {
            await getSession(id);
            optionsRef.current.onSessionRefreshed?.(id);
          } catch {
            // ignore
          }
        }
      } finally {
        if (abortRef.current === controller) {
          abortRef.current = null;
        }
        setState((s) =>
          s.streamStatus === "connected"
            ? { ...s, streamStatus: "closed" }
            : s
        );
      }
    },
    [abort, readSSE]
  );

  // Keep ref in sync for recursive calls
  connectStreamRef.current = connectStream;

  useEffect(() => {
    if (!sessionId) {
      abort();
      reset();
      return;
    }

    reset();
    connectStream(sessionId);

    return () => {
      abort();
    };
  }, [sessionId]); // eslint-disable-line react-hooks/exhaustive-deps

  return { state, connectStream, abort, reset };
}

export function buildPipelineSteps(
  completedAgents: Set<PipelineAgent>,
  activeAgents: Set<PipelineAgent>
): PipelineStep[] {
  return AGENT_ORDER.map((agent) => {
    if (completedAgents.has(agent)) return { agent, status: "done" };
    if (activeAgents.has(agent)) return { agent, status: "active" };
    return { agent, status: "pending" };
  });
}
