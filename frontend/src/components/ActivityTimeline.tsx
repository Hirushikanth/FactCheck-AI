import { useState } from "react";
import {
  IconAlertTriangle,
  IconBrain,
  IconChevronDown,
  IconChevronRight,
  IconCheck,
  IconCircleDot,
  IconFileText,
  IconMessageCircle,
  IconSearch,
} from "@tabler/icons-react";
import type {
  ActivityAction,
  ActivityAgent,
  ActivityClaim,
  ActivitySearch,
  ActivityTimelineState,
  PipelineAgent,
} from "../activity/types";
import { AGENT_ORDER } from "../activity/types";

interface ActivityTimelineProps {
  timeline: ActivityTimelineState;
  thinkingEnabled?: boolean;
  onThinkingEnabledChange?: (enabled: boolean) => void;
}

const AGENT_ICONS: Record<PipelineAgent, typeof IconBrain> = {
  extractor: IconBrain,
  verifier: IconSearch,
  reporter: IconFileText,
  dialogue: IconMessageCircle,
};

function durationLabel(agent: ActivityAgent): string | null {
  if (agent.durationMs === undefined) return null;
  return `${(agent.durationMs / 1000).toFixed(1)}s`;
}

function statusLabel(status: ActivityAgent["status"]): string {
  if (status === "active") return "Running";
  if (status === "completed") return "Complete";
  if (status === "degraded") return "Needs attention";
  return "Pending";
}

function claimTitle(claim: ActivityClaim): string {
  if (claim.verdictLabel) return `Claim ${claim.index + 1} — ${claim.verdictLabel}`;
  const activeSearch = claim.searches.some((search) => search.status === "active");
  return `Claim ${claim.index + 1} — ${activeSearch ? "searching evidence" : "verifying"}`;
}

function renderSearch(search: ActivitySearch) {
  const state = search.status === "completed" ? "✓" : search.status === "failed" ? "!" : search.status === "active" ? "⌕" : "·";
  const suffix = search.status === "completed" && search.resultCount !== undefined
    ? ` — ${search.resultCount} result${search.resultCount === 1 ? "" : "s"}`
    : search.status === "failed" ? " — search unavailable" : search.status === "active" ? " — searching…" : "";
  return (
    <li key={search.id} className={`activity-search activity-search-${search.status}`}>
      <span className="activity-search-mark" aria-hidden="true">{state}</span>
      <span>
        Query {search.queryIndex + 1} of {search.totalQueries}
        {search.provider ? ` · ${search.provider}` : ""}{suffix}
      </span>
    </li>
  );
}

function ThinkingBlock({
  action,
  expanded,
  onToggle,
}: {
  action: ActivityAction;
  expanded: boolean;
  onToggle: () => void;
}) {
  if (!action.thinking) return null;
  const labelId = `thinking-label-${action.id.replaceAll(/[^a-zA-Z0-9_-]/g, "-")}`;
  return (
    <div className="activity-thinking">
      <button
        type="button"
        className="activity-thinking-toggle"
        aria-expanded={expanded}
        aria-controls={labelId}
        onClick={onToggle}
      >
        <IconBrain size={13} aria-hidden="true" />
        <span>Model-emitted thinking — experimental, not evidence</span>
        {expanded ? <IconChevronDown size={13} aria-hidden="true" /> : <IconChevronRight size={13} aria-hidden="true" />}
      </button>
      {expanded && (
        <div id={labelId} className="activity-thinking-copy" aria-live={action.status === "active" ? "polite" : undefined}>
          {action.thinking}
          {action.status === "active" && <span className="activity-thinking-cursor" aria-hidden="true">▌</span>}
          {action.thinkingTruncated && <span className="activity-thinking-note">Older thinking hidden.</span>}
        </div>
      )}
    </div>
  );
}

function ClaimRow({
  claim,
  thinkingEnabled,
  thinkingOpen,
  onThinkingToggle,
}: {
  claim: ActivityClaim;
  thinkingEnabled: boolean;
  thinkingOpen: boolean;
  onThinkingToggle: () => void;
}) {
  const action: ActivityAction = {
    id: claim.actionId ?? `claim:${claim.index}`,
    stage: "claim_verification",
    message: claimTitle(claim),
    status: claim.status,
    claimIndex: claim.index,
    thinking: claim.thinking,
    thinkingTruncated: claim.thinkingTruncated,
    searches: claim.searches,
  };
  return (
    <li className={`activity-claim activity-claim-${claim.status}`}>
      <div className="activity-claim-heading">
        <span className="activity-claim-mark" aria-hidden="true">
          {claim.status === "completed" ? "✓" : claim.status === "degraded" || claim.status === "failed" ? "⚠" : claim.status === "active" ? "⌕" : "○"}
        </span>
        <span>{claimTitle(claim)}</span>
      </div>
      {claim.summary && claim.summary !== claimTitle(claim) && (
        <p className="activity-claim-summary">{claim.summary}</p>
      )}
      {claim.searches.length > 0 && (
        <ul className="activity-search-list" aria-label={`Evidence searches for claim ${claim.index + 1}`}>
          {claim.searches.map(renderSearch)}
        </ul>
      )}
      {thinkingEnabled && <ThinkingBlock action={action} expanded={thinkingOpen} onToggle={onThinkingToggle} />}
    </li>
  );
}

function AgentRow({
  agent,
  thinkingEnabled,
  openThinking,
  onThinkingToggle,
}: {
  agent: ActivityAgent;
  thinkingEnabled: boolean;
  openThinking: Set<string>;
  onThinkingToggle: (id: string) => void;
}) {
  const AgentIcon = AGENT_ICONS[agent.agent];
  const childClaims = agent.agent === "verifier" ? agent.claims : [];
  const standaloneActions = agent.agent === "verifier"
    ? agent.actions.filter((action) => action.claimIndex === undefined)
    : agent.actions;
  return (
    <li className={`activity-agent activity-agent-${agent.status}`}>
      <div className="activity-agent-heading">
        <span className="activity-agent-mark" aria-hidden="true">
          {agent.status === "completed" ? <IconCheck size={14} stroke={2.5} /> : agent.status === "degraded" ? <IconAlertTriangle size={14} /> : agent.status === "active" ? <IconCircleDot size={14} /> : <AgentIcon size={14} />}
        </span>
        <span className="activity-agent-name">{agent.label}</span>
        <span className="activity-agent-status">{statusLabel(agent.status)}</span>
        {durationLabel(agent) && <span className="activity-agent-duration">{durationLabel(agent)}</span>}
      </div>
      {agent.summary && <p className="activity-agent-summary">{agent.summary}</p>}
      {(childClaims.length > 0 || standaloneActions.length > 0) && (
        <ul className="activity-children" aria-label={`${agent.label} activity details`}>
          {childClaims.map((claim) => (
            <ClaimRow
              key={claim.index}
              claim={claim}
              thinkingEnabled={thinkingEnabled}
              thinkingOpen={openThinking.has(claim.actionId ?? `claim:${claim.index}`) || (claim.status === "active" && !openThinking.has(`closed:${claim.actionId ?? `claim:${claim.index}`}`))}
              onThinkingToggle={() => onThinkingToggle(claim.actionId ?? `claim:${claim.index}`)}
            />
          ))}
          {standaloneActions.map((action) => (
            <li key={action.id} className={`activity-action activity-action-${action.status}`}>
              <span className="activity-action-mark" aria-hidden="true">{action.status === "completed" ? "✓" : action.status === "degraded" ? "⚠" : action.status === "active" ? "●" : "○"}</span>
              <span>{action.message}</span>
              {thinkingEnabled && (
                <ThinkingBlock
                  action={action}
                  expanded={openThinking.has(action.id) || (action.status === "active" && !openThinking.has(`closed:${action.id}`))}
                  onToggle={() => onThinkingToggle(action.id)}
                />
              )}
            </li>
          ))}
        </ul>
      )}
    </li>
  );
}

export function ActivityTimeline({
  timeline,
  thinkingEnabled = false,
  onThinkingEnabledChange,
}: ActivityTimelineProps) {
  const [expanded, setExpanded] = useState(timeline.expanded);
  const [openThinking, setOpenThinking] = useState<Set<string>>(new Set());
  const hasThinking = timeline.thinkingSupported && AGENT_ORDER.some((agent) =>
    timeline.agents[agent].actions.some((action) => action.thinking) || timeline.agents[agent].claims.some((claim) => claim.thinking),
  );

  const toggleThinking = (id: string) => {
    setOpenThinking((previous) => {
      const next = new Set(previous);
      const isOpen = next.has(id) || !next.has(`closed:${id}`);
      if (isOpen) {
        next.delete(id);
        next.add(`closed:${id}`);
      } else {
        next.delete(`closed:${id}`);
        next.add(id);
      }
      return next;
    });
    onThinkingEnabledChange?.(true);
  };

  return (
    <section className="activity-timeline" aria-labelledby="activity-heading">
      <div className="activity-header">
        <h2 id="activity-heading">Activity</h2>
        <button
          type="button"
          className="activity-collapse-button"
          aria-expanded={expanded}
          aria-controls="activity-timeline-content"
          onClick={() => setExpanded((value) => !value)}
        >
          {expanded ? "Hide activity" : "Show activity"}
          {expanded ? <IconChevronDown size={15} aria-hidden="true" /> : <IconChevronRight size={15} aria-hidden="true" />}
        </button>
      </div>
      {expanded && (
        <div id="activity-timeline-content" className="activity-content">
          <ol className="activity-agent-list" aria-label="Fact-check pipeline">
            {AGENT_ORDER.map((agent) => (
              <AgentRow
                key={agent}
                agent={timeline.agents[agent]}
                thinkingEnabled={thinkingEnabled && hasThinking}
                openThinking={openThinking}
                onThinkingToggle={toggleThinking}
              />
            ))}
          </ol>
          {hasThinking && !thinkingEnabled && (
            <button type="button" className="activity-thinking-enable" onClick={() => onThinkingEnabledChange?.(true)}>
              <IconBrain size={13} aria-hidden="true" />
              Show model-emitted thinking
            </button>
          )}
        </div>
      )}
    </section>
  );
}
