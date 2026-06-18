import {
  useState,
  useRef,
  useEffect,
  useCallback,
  useReducer,
} from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  IconMessageCircle,
  IconPlus,
  IconArrowUp,
  IconDownload,
  IconAlertCircle,
  IconLayoutSidebar,
} from "@tabler/icons-react";
import { createSession, getSession, listSessions, postMessage } from "../api/client";
import type { SessionSummary } from "../api/types";
import { useApp } from "../App";
import { useSessionStream, buildPipelineSteps } from "../hooks/useSessionStream";
import { PipelineStepper } from "../components/PipelineStepper";
import { MessageBubble } from "../components/MessageBubble";
import type { ChatMessage } from "../components/MessageBubble";
import { truncate } from "../lib/format";

const INTERIM_VERIFYING_PREFIX = "Verifying that claim now";

const INITIAL_MESSAGES: ChatMessage[] = [
  {
    role: "system",
    content:
      "Hello! Submit a claim and I'll verify it using multiple sources. I'll extract the core assertion, search for evidence, and give you a structured verdict.",
  },
];

// ── Session screen ────────────────────────────────────────────────────────────
export function SessionScreen() {
  const { activeSessionId, setActiveSessionId, setActiveSession, setActiveTab } =
    useApp();
  const queryClient = useQueryClient();
  const [sidebarOpen, toggleSidebar] = useReducer((s: boolean) => !s, true);

  const [chatMessages, setChatMessages] = useState<ChatMessage[]>(INITIAL_MESSAGES);
  const [inputValue, setInputValue] = useState("");
  const [isBusy, setIsBusy] = useState(false);
  const [statusError, setStatusError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Helper: append (or replace) the final report as a markdown bot bubble.
  // Removes the interim "Verifying…" bubble and deduplicates.
  const appendReportMessage = useCallback((report: string) => {
    setChatMessages((prev) => {
      // Skip if this exact report is already the last assistant message
      const lastAssistant = [...prev].reverse().find((m) => m.role === "assistant");
      if (lastAssistant?.content === report) return prev;

      // Strip any interim "Verifying…" assistant bubble
      const withoutInterim = prev.filter(
        (m) => !(m.role === "assistant" && m.content.startsWith(INTERIM_VERIFYING_PREFIX))
      );

      return [...withoutInterim, { role: "assistant", content: report, markdown: true }];
    });
  }, []);

  // Fetch sessions list for sidebar
  const { data: sessions = [] } = useQuery<SessionSummary[]>({
    queryKey: ["sessions"],
    queryFn: listSessions,
    refetchInterval: 5_000,
  });

  // SSE for active session
  const { state: streamState } = useSessionStream(activeSessionId, {
    onReportReady: (report) => {
      appendReportMessage(report);
    },
    onSessionRefreshed: async (id) => {
      const session = await getSession(id).catch(() => null);
      if (!session) return;
      queryClient.setQueryData(["session", id], session);
      queryClient.invalidateQueries({ queryKey: ["sessions"] });
      setActiveSession(session);

      if (session.status === "done") {
        setIsBusy(false);
        // Surface final_report in chat (dedup with onReportReady)
        if (session.final_report) {
          appendReportMessage(session.final_report);
        }
        // Append final dialogue reply if present and not already shown
        if (session.messages.length > 0) {
          const last = session.messages[session.messages.length - 1];
          if (last.role === "assistant") {
            setChatMessages((prev) => {
              const alreadyAdded = prev.some(
                (m) => m.role === "assistant" && m.content === last.content
              );
              if (alreadyAdded) return prev;
              return [...prev, { role: "assistant", content: last.content }];
            });
          }
        }
      }
      if (session.status === "error") {
        setIsBusy(false);
        setStatusError(session.error ?? "Pipeline failed.");
      }
    },
    onDialogueReply: (reply) => {
      setChatMessages((prev) => [
        ...prev,
        { role: "assistant", content: reply },
      ]);
    },
  });

  const pipelineSteps = buildPipelineSteps(
    streamState.completedAgents,
    streamState.activeAgents
  );

  const showStepper =
    activeSessionId !== null &&
    pipelineSteps.some((s) => s.status !== "pending");

  // Scroll chat to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatMessages, showStepper]);

  const handleSelectSession = useCallback(
    async (session: SessionSummary) => {
      if (session.session_id === activeSessionId) return;
      setActiveSessionId(session.session_id);
      setIsBusy(session.status === "running");
      setStatusError(null);

      try {
        const detail = await getSession(session.session_id);
        setActiveSession(detail);

        // Reconstruct chat from stored history
        const msgs: ChatMessage[] = [
          {
            role: "system",
            content:
              "Hello! Submit a claim and I'll verify it using multiple sources.",
          },
          { role: "user", content: detail.raw_input },
        ];

        // Show the report if the session completed
        if (detail.final_report) {
          msgs.push({ role: "assistant", content: detail.final_report, markdown: true });
        }

        // Append any dialogue turns that came after
        for (const m of detail.messages) {
          msgs.push({ role: m.role, content: m.content });
        }

        setChatMessages(msgs);
      } catch {
        // ignore — SSE will catch up on reconnect
      }
    },
    [activeSessionId, setActiveSessionId, setActiveSession]
  );

  const handleNewSession = useCallback(() => {
    setActiveSessionId(null);
    setActiveSession(null);
    setIsBusy(false);
    setStatusError(null);
    setInputValue("");
    setChatMessages(INITIAL_MESSAGES);
  }, [setActiveSessionId, setActiveSession]);

  const handleSend = useCallback(async () => {
    const text = inputValue.trim();
    if (!text || isBusy) return;
    setInputValue("");
    setStatusError(null);

    setChatMessages((prev) => [...prev, { role: "user", content: text }]);

    if (!activeSessionId) {
      setIsBusy(true);
      try {
        const result = await createSession(text);
        setActiveSessionId(result.session_id);
        queryClient.invalidateQueries({ queryKey: ["sessions"] });

        // Interim message — will be replaced by the actual report via onReportReady
        setChatMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content:
              `${INTERIM_VERIFYING_PREFIX} — running the pipeline across multiple sources. Results in a moment.`,
          },
        ]);
      } catch (err) {
        setIsBusy(false);
        setStatusError(err instanceof Error ? err.message : "Failed to create session");
      }
      return;
    }

    // Existing session → follow-up dialogue message
    setIsBusy(true);
    try {
      await postMessage(activeSessionId, text);
      queryClient.invalidateQueries({ queryKey: ["sessions"] });
    } catch (err) {
      setIsBusy(false);
      setStatusError(err instanceof Error ? err.message : "Failed to send message");
    }
  }, [inputValue, isBusy, activeSessionId, setActiveSessionId, queryClient]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend]
  );

  const handleViewResults = useCallback(() => {
    setActiveTab("results");
  }, [setActiveTab]);

  return (
    <div className="session-layout">
      {/* ── Sidebar ── */}
      <aside className={`chat-sidebar${sidebarOpen ? "" : " sidebar-collapsed"}`}>
        <span className="sidebar-label">Current session</span>
        {activeSessionId ? (
          <div className="sidebar-item active">
            <IconMessageCircle size={15} />
            <span className="item-text">
              {truncate(
                sessions.find((s) => s.session_id === activeSessionId)
                  ?.raw_input ?? "Active session",
                30
              )}
            </span>
          </div>
        ) : (
          <div className="sidebar-item" style={{ color: "var(--color-text-tertiary)" }}>
            <IconMessageCircle size={15} />
            <span className="item-text">No active session</span>
          </div>
        )}

        <span className="sidebar-label">Recent</span>
        {sessions
          .filter((s) => s.session_id !== activeSessionId)
          .slice(0, 8)
          .map((session) => (
            <div
              key={session.session_id}
              className={`sidebar-item${session.session_id === activeSessionId ? " active" : ""}`}
              onClick={() => handleSelectSession(session)}
            >
              <IconMessageCircle size={15} />
              <span className="item-text">{truncate(session.raw_input, 28)}</span>
              <StatusBadge status={session.status} />
            </div>
          ))}

        <div style={{ flex: 1 }} />
        <div
          className="sidebar-item"
          onClick={handleNewSession}
          style={{ marginTop: "auto", cursor: "pointer" }}
        >
          <IconPlus size={15} />
          <span className="item-text">New session</span>
        </div>
      </aside>

      {/* ── Chat main ── */}
      <div className="chat-main">
        {/* Header */}
        <div className="chat-header">
          <div className="chat-header-left">
            <span className="chat-title">
              {activeSessionId
                ? truncate(
                    sessions.find((s) => s.session_id === activeSessionId)
                      ?.raw_input ?? "Session",
                    40
                  )
                : "New fact-check"}
            </span>
            <span className="chat-sub">
              {activeSessionId
                ? `Session · ${sessions.find((s) => s.session_id === activeSessionId)?.status ?? "—"}`
                : "Start by submitting a claim below"}
            </span>
          </div>
          <div className="chat-actions">
            {streamState.pipelineDone && streamState.sessionStatus === "done" && (
              <button
                className="icon-btn"
                onClick={handleViewResults}
                title="View structured results"
              >
                <IconDownload size={15} />
              </button>
            )}
            <button className="icon-btn" onClick={toggleSidebar} title="Toggle sidebar">
              <IconLayoutSidebar size={15} />
            </button>
          </div>
        </div>

        {/* Error banner */}
        {(statusError || streamState.pipelineError) && (
          <div className="error-banner">
            <IconAlertCircle size={14} />
            <span>{statusError ?? streamState.pipelineError}</span>
          </div>
        )}

        {/* Messages */}
        <div className="messages">
          {chatMessages.map((msg, i) => (
            <MessageBubble key={i} message={msg} />
          ))}
          <div ref={messagesEndRef} />
        </div>

        {/* Pipeline stepper */}
        {showStepper && <PipelineStepper steps={pipelineSteps} />}

        {/* Input */}
        <div className="chat-input-area">
          <div className="input-box">
            <input
              type="text"
              placeholder={
                isBusy
                  ? "Verifying…"
                  : activeSessionId && streamState.sessionStatus === "done"
                    ? "Ask a follow-up question…"
                    : "Enter a claim to fact-check…"
              }
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={
                isBusy ||
                (!!activeSessionId && streamState.sessionStatus === "running")
              }
            />
            <button
              className="send-btn"
              onClick={handleSend}
              disabled={
                !inputValue.trim() ||
                isBusy ||
                (!!activeSessionId && streamState.sessionStatus === "running")
              }
              aria-label="Send"
            >
              <IconArrowUp size={13} stroke={2.5} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// Small status badge for sidebar
function StatusBadge({ status }: { status: string }) {
  if (status === "running") {
    return <span className="badge-sm badge-running">Running</span>;
  }
  if (status === "error") {
    return <span className="badge-sm badge-false">Error</span>;
  }
  return null;
}
