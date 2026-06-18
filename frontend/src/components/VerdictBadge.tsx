import {
  IconCheck,
  IconX,
  IconMinus,
  IconClock,
  IconAlertCircle,
} from "@tabler/icons-react";
import type { UiVerdict } from "../lib/verdict";

interface Props {
  verdict: UiVerdict | "running" | "error" | null;
  size?: "sm" | "md";
}

export function VerdictBadge({ verdict, size = "md" }: Props) {
  const dim = size === "sm" ? 14 : 16;
  const wh = size === "sm" ? 24 : 28;

  const style: React.CSSProperties = {
    width: wh,
    height: wh,
    borderRadius: 7,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    flexShrink: 0,
  };

  if (verdict === "true") {
    return (
      <div style={{ ...style, background: "var(--color-true-bg)", color: "var(--color-true-text)" }}>
        <IconCheck size={dim} stroke={2.5} />
      </div>
    );
  }
  if (verdict === "false") {
    return (
      <div style={{ ...style, background: "var(--color-false-bg)", color: "var(--color-false-text)" }}>
        <IconX size={dim} stroke={2.5} />
      </div>
    );
  }
  if (verdict === "mixed") {
    return (
      <div style={{ ...style, background: "var(--color-mixed-bg)", color: "var(--color-mixed-text)" }}>
        <IconMinus size={dim} stroke={2.5} />
      </div>
    );
  }
  if (verdict === "running") {
    return (
      <div style={{ ...style, background: "var(--color-bg-secondary)", color: "var(--color-text-tertiary)" }}>
        <IconClock size={dim} stroke={2} />
      </div>
    );
  }
  if (verdict === "error") {
    return (
      <div style={{ ...style, background: "var(--color-false-bg)", color: "var(--color-false-text)" }}>
        <IconAlertCircle size={dim} stroke={2} />
      </div>
    );
  }
  return (
    <div style={{ ...style, background: "var(--color-bg-secondary)", border: "0.5px solid var(--color-border-tertiary)", color: "var(--color-text-tertiary)" }}>
      <IconMinus size={dim} stroke={2} />
    </div>
  );
}
