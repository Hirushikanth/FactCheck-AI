import { IconCheck, IconLoader, IconMessage, IconSearch, IconFileText, IconBrain } from "@tabler/icons-react";
import type { PipelineAgent, PipelineStep } from "../hooks/useSessionStream";

const LABELS: Record<PipelineAgent, string> = {
  extractor: "Extractor",
  verifier: "Verifier",
  reporter: "Reporter",
  dialogue: "Dialogue",
};

const ICONS: Record<PipelineAgent, React.ReactNode> = {
  extractor: <IconBrain size={11} />,
  verifier: <IconSearch size={11} />,
  reporter: <IconFileText size={11} />,
  dialogue: <IconMessage size={11} />,
};

interface Props {
  steps: PipelineStep[];
}

export function PipelineStepper({ steps }: Props) {
  if (steps.every((s) => s.status === "pending")) return null;

  return (
    <div className="pipeline-status">
      <div className="pipeline-label">Pipeline</div>
      <div className="pipeline-steps">
        {steps.map((step, i) => (
          <span key={step.agent} style={{ display: "flex", alignItems: "center" }}>
            {i > 0 && <div className="p-line" />}
            <span className="p-step">
              <span className={`p-dot ${step.status}`}>
                {step.status === "done" ? (
                  <IconCheck size={11} stroke={2.5} />
                ) : step.status === "active" ? (
                  <span className="spin-icon">{ICONS[step.agent]}</span>
                ) : (
                  <IconLoader size={11} stroke={2} />
                )}
              </span>
              <span className={`p-name ${step.status}`}>{LABELS[step.agent]}</span>
            </span>
          </span>
        ))}
      </div>
    </div>
  );
}
