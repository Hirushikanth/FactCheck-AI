import { useQuery } from "@tanstack/react-query";
import {
  IconCircleCheck,
  IconCircleX,
  IconCircleMinus,
  IconShieldQuestion,
  IconBrain,
  IconSearch,
  IconFileText,
  IconMessage,
} from "@tabler/icons-react";
import { getSession } from "../api/client";
import type { ClaimResult } from "../api/types";
import { useApp } from "../App";
import {
  toUiVerdict,
  dominantVerdict,
  uiVerdictLabel,
  backendVerdictLabel,
  verdictColors,
} from "../lib/verdict";
import { SourceCard } from "../components/SourceCard";
import { truncate } from "../lib/format";

export function ResultsScreen() {
  const { activeSessionId } = useApp();

  const { data: session, isLoading, isError } = useQuery({
    queryKey: ["session", activeSessionId],
    queryFn: () => getSession(activeSessionId!),
    enabled: !!activeSessionId,
  });

  if (!activeSessionId) {
    return (
      <div className="empty-state" style={{ flex: 1 }}>
        <div className="empty-state-icon">
          <IconShieldQuestion size={22} stroke={1.5} />
        </div>
        <p>No session selected. Submit a claim in the Session tab to get started.</p>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="results-loading">
        <div className="skeleton" style={{ height: 120, margin: "28px 28px 0" }} />
        <div className="results-body" style={{ marginTop: 20 }}>
          <div className="results-col">
            {[1, 2, 3].map((i) => (
              <div key={i} className="skeleton" style={{ height: 72, borderRadius: 8 }} />
            ))}
          </div>
          <div className="results-col">
            {[1, 2, 3].map((i) => (
              <div key={i} className="skeleton" style={{ height: 56, borderRadius: 8 }} />
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (isError || !session) {
    return (
      <div className="empty-state" style={{ flex: 1 }}>
        <div className="empty-state-icon">
          <IconCircleX size={22} stroke={1.5} />
        </div>
        <p>Failed to load session results. Please try again.</p>
      </div>
    );
  }

  if (session.status === "running") {
    return (
      <div className="empty-state" style={{ flex: 1 }}>
        <div className="empty-state-icon" style={{ animation: "pulse 1.5s ease-in-out infinite" }}>
          <IconBrain size={22} stroke={1.5} />
        </div>
        <p>Pipeline is still running. Results will appear here when complete.</p>
      </div>
    );
  }

  if (session.status === "error") {
    return (
      <div className="empty-state" style={{ flex: 1 }}>
        <div className="empty-state-icon">
          <IconCircleX size={22} stroke={1.5} />
        </div>
        <p>{session.error ?? "Pipeline failed."}</p>
      </div>
    );
  }

  const claims = session.claim_results;
  const uiVerdicts = claims.map((c) => toUiVerdict(c.verdict));
  const dominant = dominantVerdict(claims.map((c) => c.verdict));
  const avgConfidence =
    claims.length > 0
      ? claims.reduce((sum, c) => sum + c.confidence, 0) / claims.length
      : 0;

  // Collect all unique sources across all claims
  const allSources: { url: string; evidence: string }[] = [];
  for (const claim of claims) {
    claim.sources.forEach((url, i) => {
      if (!allSources.some((s) => s.url === url)) {
        allSources.push({ url, evidence: claim.evidence[i] ?? "" });
      }
    });
  }

  return (
    <div className="results-layout">
      {/* ── Verdict hero ── */}
      <div className="verdict-hero">
        <VerdictIcon uiVerdict={dominant} />
        <div className="verdict-details">
          <div className="verdict-label">Verdict</div>
          <div className="verdict-claim">
            "{truncate(session.raw_input, 120)}"
          </div>
          <div className="verdict-pills">
            <VerdictPill dominant={dominant} />
            {claims.length > 1 && (
              <span className="pill">
                {claims.length} claims
              </span>
            )}
          </div>
          <div className="confidence-bar-wrap">
            <div className="confidence-meta">
              <span>Confidence</span>
              <span className="confidence-val">
                {Math.round(avgConfidence * 100)}%
              </span>
            </div>
            <div className="confidence-bar">
              <div
                className="confidence-fill"
                style={{ width: `${Math.round(avgConfidence * 100)}%`, background: verdictColors(dominant).text }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* ── Per-claim breakdown (multi-claim) ── */}
      {claims.length > 1 && (
        <div className="claims-strip">
          {claims.map((claim, i) => (
            <ClaimChip key={i} claim={claim} uiVerdict={uiVerdicts[i]} />
          ))}
        </div>
      )}

      {/* ── Body: sources + agent reasoning ── */}
      <div className="results-body">
        {/* Left: sources */}
        <div className="results-col">
          <div className="section-heading">Sources</div>
          {allSources.length === 0 ? (
            <p style={{ fontSize: 13, color: "var(--color-text-tertiary)" }}>
              No sources available.
            </p>
          ) : (
            allSources.slice(0, 6).map((s, i) => (
              <SourceCard key={i} url={s.url} excerpt={s.evidence} index={i} />
            ))
          )}
        </div>

        {/* Right: agent reasoning */}
        <div className="results-col">
          <div className="section-heading">Agent reasoning</div>
          <AgentStep
            icon={<IconBrain size={12} />}
            name="Extractor"
            content={
              claims.length === 0
                ? "No claims were extracted."
                : `${claims.length} claim${claims.length === 1 ? "" : "s"} extracted: ${claims.map((c) => `"${truncate(c.claim, 60)}"`).join("; ")}`
            }
          />
          {claims.map((claim, i) => (
            <AgentStep
              key={i}
              icon={<IconSearch size={12} />}
              name={`Verifier${claims.length > 1 ? ` (${i + 1})` : ""}`}
              content={`${backendVerdictLabel(claim.verdict)} · ${Math.round(claim.confidence * 100)}% confidence. ${claim.reasoning}`}
            />
          ))}
          {session.final_report && (
            <AgentStep
              icon={<IconFileText size={12} />}
              name="Reporter"
              content={extractSummary(session.final_report)}
            />
          )}
          {session.messages.length > 0 && (
            <AgentStep
              icon={<IconMessage size={12} />}
              name="Dialogue"
              content={`${session.messages.length} follow-up message${session.messages.length === 1 ? "" : "s"} in this session.`}
            />
          )}
        </div>
      </div>
    </div>
  );
}

// ── Helpers ────────────────────────────────────────────────────────────────────

function VerdictIcon({ uiVerdict }: { uiVerdict: ReturnType<typeof toUiVerdict> }) {
  const colors = verdictColors(uiVerdict);
  const size = 28;
  return (
    <div
      className="verdict-icon-wrap"
      style={{ background: colors.bg, color: colors.text }}
    >
      {uiVerdict === "true" && <IconCircleCheck size={size} stroke={1.5} />}
      {uiVerdict === "false" && <IconCircleX size={size} stroke={1.5} />}
      {uiVerdict === "mixed" && <IconCircleMinus size={size} stroke={1.5} />}
    </div>
  );
}

function VerdictPill({ dominant }: { dominant: ReturnType<typeof dominantVerdict> }) {
  const colors = verdictColors(dominant);
  return (
    <span
      className="pill"
      style={{
        background: colors.bg,
        borderColor: colors.border,
        color: colors.text,
        fontWeight: 500,
      }}
    >
      {uiVerdictLabel(dominant)}
    </span>
  );
}

function ClaimChip({
  claim,
  uiVerdict,
}: {
  claim: ClaimResult;
  uiVerdict: ReturnType<typeof toUiVerdict>;
}) {
  const colors = verdictColors(uiVerdict);
  return (
    <div className="claim-chip" style={{ borderColor: colors.border }}>
      <span
        className="claim-chip-badge"
        style={{ background: colors.bg, color: colors.text }}
      >
        {uiVerdictLabel(uiVerdict)}
      </span>
      <span className="claim-chip-text">{truncate(claim.claim, 80)}</span>
      <span className="claim-chip-conf">
        {Math.round(claim.confidence * 100)}%
      </span>
    </div>
  );
}

function AgentStep({ icon, name, content }: { icon: React.ReactNode; name: string; content: string }) {
  return (
    <div className="agent-step">
      <div className="agent-dot" />
      <div className="agent-content">
        <div className="agent-name">
          <span className="agent-name-icon">{icon}</span>
          {name}
        </div>
        <p>{content}</p>
      </div>
    </div>
  );
}

function extractSummary(report: string): string {
  // Extract first paragraph or sentence from markdown report
  const clean = report.replace(/^#+.*$/gm, "").replace(/\*\*/g, "").trim();
  const firstPara = clean.split(/\n\n+/)[0].trim();
  return truncate(firstPara, 300);
}
