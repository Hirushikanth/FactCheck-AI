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
  IconTrash,
  IconAlertTriangle,
} from "@tabler/icons-react";
import { createSession, getSession, listSessions, postMessage, deleteSession } from "../api/client";
import type { SessionDetail, SessionSummary } from "../api/types";
import { useApp } from "../app-context";
import { useSessionStream } from "../hooks/useSessionStream";
import { ActivityTimeline } from "../components/ActivityTimeline";
import { MessageBubble } from "../components/MessageBubble";
import type { ChatMessage } from "../components/MessageBubble";
import { createInitialActivityState, reduceActivityEvent } from "../activity/reducer";
import type { ActivityTimelineState } from "../activity/types";
import { appendAssistantMessage } from "./chatMessages";
import { buildHistoricChatMessages } from "./sessionHistory";
import { truncate } from "../lib/format";

const INITIAL_MESSAGES: ChatMessage[] = [
  {
    role: "system",
    content:
      "Hello! Submit a claim and I'll verify it using multiple sources. I'll extract the core assertion, search for evidence, and give you a structured verdict.",
  },
];

function restoreActivity(events: Array<{ type: string; data: Record<string, unknown> }>) {
  return events.reduce(
    (timeline, event) => reduceActivityEvent(timeline, event),
    createInitialActivityState(),
  );
}

function restoreActivitySnapshots(session: SessionDetail | null) {
  if (!session) return {};
  const initialRun = session.runs[0];
  return {
    ...(initialRun ? { [initialRun.run_id]: restoreActivity(initialRun.activity_events) } : {}),
    ...Object.fromEntries(
      session.messages
        .filter((message) => message.role === "user")
        .map((message) => [
          `dialogue:${message.id}`,
          restoreActivity(message.activity_events),
        ]),
    ),
  };
}

// ── Session screen ────────────────────────────────────────────────────────────
export function SessionScreen() {
  const { activeSessionId, activeSession, setActiveSessionId, setActiveSession, setActiveTab } =
    useApp();
  const queryClient = useQueryClient();
  const [sidebarOpen, toggleSidebar] = useReducer((s: boolean) => !s, true);

  const [chatMessages, setChatMessages] = useState<ChatMessage[]>(() =>
    activeSession ? buildHistoricChatMessages(activeSession) : INITIAL_MESSAGES,
  );
  const [inputValue, setInputValue] = useState("");
  const [isBusy, setIsBusy] = useState(false);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [activitySnapshots, setActivitySnapshots] = useState<Record<string, ActivityTimelineState>>(() =>
    restoreActivitySnapshots(activeSession),
  );
  const [activeActivityId, setActiveActivityId] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Add the final report once; follow-up dialogue may appear after it.
  const appendReportMessage = useCallback((report: string) => {
    setChatMessages((prev) => {
      // A dialogue response may now follow the report, so deduplicate across the
      // whole conversation rather than only looking at the final assistant turn.
      if (prev.some((message) => message.markdown && message.content === report)) {
        return prev;
      }

      return [...prev, { role: "assistant", content: report, markdown: true }];
    });
  }, []);

  // Fetch sessions list for sidebar
  const { data: sessions = [] } = useQuery<SessionSummary[]>({
    queryKey: ["sessions"],
    queryFn: listSessions,
    refetchInterval: 5_000,
  });

  const [pendingDelete, setPendingDelete] = useState<SessionSummary | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);

  async function confirmDelete() {
    if (!pendingDelete) return;
    setIsDeleting(true);
    try {
      await deleteSession(pendingDelete.session_id);
      queryClient.invalidateQueries({ queryKey: ["sessions"] });
      if (pendingDelete.session_id === activeSessionId) {
        setActiveSessionId(null);
        setActiveSession(null);
      }
      setPendingDelete(null);
    } catch {
      // keep dialog open so the user can retry
    } finally {
      setIsDeleting(false);
    }
  }

  // SSE for active session
  const { state: streamState, setThinkingEnabled, connectStream, startNewActivity } = useSessionStream(activeSessionId, {
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
            setChatMessages((previous) => appendAssistantMessage(previous, last.content));
          }
        }
      }
      if (session.status === "error") {
        setIsBusy(false);
        setStatusError(session.error ?? "Pipeline failed.");
      }
    },
    onDialogueReply: (reply) => {
      setChatMessages((previous) => appendAssistantMessage(previous, reply));
    },
  });

  // Scroll chat to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatMessages, activeActivityId, streamState.activity]);

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
        const initialRun = detail.runs[0];
        setChatMessages(buildHistoricChatMessages(detail));
        setActivitySnapshots({
          ...(initialRun
            ? { [initialRun.run_id]: restoreActivity(initialRun.activity_events) }
            : {}),
          ...Object.fromEntries(
            detail.messages
              .filter((message) => message.role === "user")
              .map((message) => [
                `dialogue:${message.id}`,
                restoreActivity(message.activity_events),
              ]),
          ),
        });
        setActiveActivityId(null);
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
    setActivitySnapshots({});
    setActiveActivityId(null);
  }, [setActiveSessionId, setActiveSession]);

  const handleSend = useCallback(async () => {
    const text = inputValue.trim();
    if (!text || isBusy) return;
    setInputValue("");
    setStatusError(null);

    const activityId = crypto.randomUUID();
    const activityKind = activeSessionId ? "dialogue" : "pipeline";
    if (activeActivityId) {
      setActivitySnapshots((previous) => ({
        ...previous,
        [activeActivityId]: streamState.activity,
      }));
    }
    setActiveActivityId(activityId);
    startNewActivity();
    setChatMessages((prev) => [
      ...prev,
      { role: "user", content: text, activityId, activityKind },
    ]);

    if (!activeSessionId) {
      setIsBusy(true);
      const sessionId = crypto.randomUUID();
      setActiveSessionId(sessionId);
      try {
        const result = await createSession(text, sessionId);
        if (result.session_id !== sessionId) {
          throw new Error("Session ID mismatch");
        }
        queryClient.invalidateQueries({ queryKey: ["sessions"] });

      } catch (err) {
        setIsBusy(false);
        setActiveSessionId(null);
        setStatusError(err instanceof Error ? err.message : "Failed to create session");
      }
      return;
    }

    // Existing session → follow-up dialogue message
    setIsBusy(true);
    try {
      await postMessage(activeSessionId, text);
      connectStream(activeSessionId);
      queryClient.invalidateQueries({ queryKey: ["sessions"] });
    } catch (err) {
      setIsBusy(false);
      setStatusError(err instanceof Error ? err.message : "Failed to send message");
    }
  }, [inputValue, isBusy, activeSessionId, activeActivityId, streamState.activity, setActiveSessionId, queryClient, connectStream, startNewActivity]);

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
            <button
              type="button"
              className="sidebar-delete-btn"
              aria-label="Delete chat"
              title="Delete chat"
              onClick={(e) => {
                e.stopPropagation();
                const current = sessions.find((s) => s.session_id === activeSessionId);
                if (current) setPendingDelete(current);
              }}
            >
              <IconTrash size={14} />
            </button>
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
            <button
              type="button"
              key={session.session_id}
              className={`sidebar-item${session.session_id === activeSessionId ? " active" : ""}`}
              onClick={() => handleSelectSession(session)}
            >
              <IconMessageCircle size={15} />
              <span className="item-text">{truncate(session.raw_input, 28)}</span>
              <StatusBadge status={session.status} />
              <button
                type="button"
                className="sidebar-delete-btn"
                aria-label="Delete chat"
                title="Delete chat"
                onClick={(e) => {
                  e.stopPropagation();
                  setPendingDelete(session);
                }}
              >
                <IconTrash size={14} />
              </button>
            </button>
          ))}

        <div style={{ flex: 1 }} />
        <button
          type="button"
          className="sidebar-item"
          onClick={handleNewSession}
          style={{ marginTop: "auto", cursor: "pointer" }}
        >
          <IconPlus size={15} />
          <span className="item-text">New session</span>
        </button>
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
            <div key={i} className={`chat-turn chat-turn-${msg.role}`}>
              <MessageBubble message={msg} />
              {msg.role === "user" && msg.activityId && (
                <ActivityBubble
                  mode={msg.activityKind}
                  timeline={activitySnapshots[msg.activityId] ?? (msg.activityId === activeActivityId ? streamState.activity : createInitialActivityState())}
                  thinkingEnabled={msg.activityId === activeActivityId && streamState.thinkingEnabled}
                  onThinkingEnabledChange={msg.activityId === activeActivityId ? setThinkingEnabled : undefined}
                />
              )}
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

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

      {pendingDelete && (
        <ConfirmDeleteDialog
          claim={pendingDelete.raw_input}
          isDeleting={isDeleting}
          onCancel={() => !isDeleting && setPendingDelete(null)}
          onConfirm={confirmDelete}
        />
      )}
    </div>
  );
}

// ── Confirm delete dialog ─────────────────────────────────────────────────────
function ConfirmDeleteDialog({
  claim,
  isDeleting,
  onCancel,
  onConfirm,
}: {
  claim: string;
  isDeleting: boolean;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  return (
    <div className="modal-overlay" onClick={onCancel}>
      <div className="confirm-dialog" onClick={(e) => e.stopPropagation()}>
        <div className="confirm-dialog-icon">
          <IconAlertTriangle size={22} stroke={1.6} />
        </div>
        <h3 className="confirm-dialog-title">Delete this chat?</h3>
        <p className="confirm-dialog-text">
          This will permanently remove the fact-check and its conversation from
          your history. This action cannot be undone.
        </p>
        <div className="confirm-dialog-claim">{claim}</div>
        <div className="confirm-dialog-actions">
          <button className="btn-ghost" onClick={onCancel} disabled={isDeleting}>
            Cancel
          </button>
          <button
            className="btn-danger"
            onClick={onConfirm}
            disabled={isDeleting}
          >
            {isDeleting ? "Deleting…" : "Delete chat"}
          </button>
        </div>
      </div>
    </div>
  );
}

function ActivityBubble({
  mode,
  timeline,
  thinkingEnabled,
  onThinkingEnabledChange,
}: React.ComponentProps<typeof ActivityTimeline>) {
  return <div className="activity-message-bubble"><ActivityTimeline mode={mode} timeline={timeline} thinkingEnabled={thinkingEnabled} onThinkingEnabledChange={onThinkingEnabledChange} /></div>;
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
