import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  IconSearch,
  IconFilter,
  IconCalendar,
  IconChevronRight,
  IconShieldQuestion,
} from "@tabler/icons-react";
import { listSessions, getSession } from "../api/client";
import type { SessionSummary } from "../api/types";
import { useApp } from "../App";
import { VerdictBadge } from "../components/VerdictBadge";
import { formatTimestamp, truncate } from "../lib/format";

type FilterVerdict = "all" | "running" | "done" | "error";

export function HistoryScreen() {
  const { setActiveSessionId, setActiveSession, setActiveTab } = useApp();
  const [search, setSearch] = useState("");
  const [filterStatus, setFilterStatus] = useState<FilterVerdict>("all");

  const { data: sessions = [], isLoading } = useQuery<SessionSummary[]>({
    queryKey: ["sessions"],
    queryFn: listSessions,
    refetchInterval: 10_000,
  });

  const filtered = sessions.filter((s) => {
    const matchesSearch =
      search === "" ||
      s.raw_input.toLowerCase().includes(search.toLowerCase());
    const matchesFilter =
      filterStatus === "all" || s.status === filterStatus;
    return matchesSearch && matchesFilter;
  });

  // Aggregate stats from the list
  const total = sessions.length;
  const done = sessions.filter((s) => s.status === "done").length;
  const running = sessions.filter((s) => s.status === "running").length;
  const errored = sessions.filter((s) => s.status === "error").length;

  async function handleRowClick(session: SessionSummary) {
    setActiveSessionId(session.session_id);
    try {
      const detail = await getSession(session.session_id);
      setActiveSession(detail);
    } catch {
      // ignore — Results screen will fetch on its own
    }
    setActiveTab("results");
  }

  return (
    <div className="history-layout">
      {/* ── Toolbar ── */}
      <div className="history-toolbar">
        <div className="search-box">
          <IconSearch size={15} style={{ color: "var(--color-text-tertiary)", flexShrink: 0 }} />
          <input
            type="text"
            placeholder="Search claims…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
        <FilterButton
          icon={<IconFilter size={14} />}
          label={filterStatus === "all" ? "All" : filterStatus}
          onClick={() => {
            const next: FilterVerdict[] = ["all", "done", "running", "error"];
            const cur = next.indexOf(filterStatus);
            setFilterStatus(next[(cur + 1) % next.length]);
          }}
        />
        <FilterButton
          icon={<IconCalendar size={14} />}
          label={new Date().toLocaleDateString([], { month: "short", year: "numeric" })}
        />
      </div>

      {/* ── Stats ── */}
      <div className="history-stats">
        <StatCard label="Total checked" value={total} />
        <StatCard label="Completed" value={done} color="green" />
        <StatCard label="Errors" value={errored} color="red" />
        <StatCard label="Running" value={running} color="amber" />
      </div>

      {/* ── Session list ── */}
      <div className="history-list">
        {isLoading ? (
          [1, 2, 3].map((i) => (
            <div
              key={i}
              className="skeleton"
              style={{ height: 60, borderRadius: 8 }}
            />
          ))
        ) : filtered.length === 0 ? (
          <div className="empty-state">
            <div className="empty-state-icon">
              <IconShieldQuestion size={22} stroke={1.5} />
            </div>
            <p>
              {sessions.length === 0
                ? "No sessions yet. Submit a claim to get started."
                : "No sessions match your search."}
            </p>
          </div>
        ) : (
          filtered.map((session) => (
            <HistoryRow
              key={session.session_id}
              session={session}
              onClick={() => handleRowClick(session)}
            />
          ))
        )}
      </div>
    </div>
  );
}

// ── History row ────────────────────────────────────────────────────────────────
function HistoryRow({
  session,
  onClick,
}: {
  session: SessionSummary;
  onClick: () => void;
}) {
  const verdictBadge =
    session.status === "running"
      ? "running"
      : session.status === "error"
        ? "error"
        : null;

  return (
    <div className="history-row" onClick={onClick} role="button" tabIndex={0}>
      <VerdictBadge verdict={verdictBadge} size="md" />
      <div className="history-claim">
        <div className="history-claim-text">{session.raw_input}</div>
        <div className="history-meta">
          <span>{truncate(session.raw_input.split(" ").slice(0, 2).join(" "), 20)}</span>
          <span>{formatTimestamp(session.created_at)}</span>
          <StatusChip status={session.status} />
        </div>
      </div>
      <div className="history-right">
        <IconChevronRight
          size={14}
          style={{ color: "var(--color-text-tertiary)" }}
        />
      </div>
    </div>
  );
}

function StatusChip({ status }: { status: string }) {
  if (status === "running")
    return <span style={{ color: "var(--color-text-tertiary)", fontSize: 11 }}>Running…</span>;
  if (status === "error")
    return <span style={{ color: "var(--color-false-text)", fontSize: 11 }}>Error</span>;
  if (status === "done")
    return <span style={{ color: "var(--color-true-text)", fontSize: 11 }}>Done</span>;
  return null;
}

function StatCard({
  label,
  value,
  color,
}: {
  label: string;
  value: number;
  color?: "green" | "red" | "amber";
}) {
  const textColor =
    color === "green"
      ? "var(--color-true-text)"
      : color === "red"
        ? "var(--color-false-text)"
        : color === "amber"
          ? "var(--color-mixed-text)"
          : "var(--color-text-primary)";
  return (
    <div className="stat-card">
      <div className="stat-label">{label}</div>
      <div className="stat-val" style={{ color: textColor }}>
        {value}
      </div>
    </div>
  );
}

function FilterButton({
  icon,
  label,
  onClick,
}: {
  icon: React.ReactNode;
  label: string;
  onClick?: () => void;
}) {
  return (
    <button className="filter-btn" onClick={onClick}>
      {icon}
      {label}
    </button>
  );
}
