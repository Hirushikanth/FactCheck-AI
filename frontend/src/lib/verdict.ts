import type { Verdict } from "../api/types";

export type UiVerdict = "true" | "false" | "mixed";

export function toUiVerdict(verdict: Verdict): UiVerdict {
  switch (verdict) {
    case "SUPPORTED":
      return "true";
    case "REFUTED":
      return "false";
    case "CONFLICTING_EVIDENCE":
    case "INSUFFICIENT_EVIDENCE":
      return "mixed";
  }
}

export function uiVerdictLabel(v: UiVerdict): string {
  switch (v) {
    case "true":
      return "True";
    case "false":
      return "False";
    case "mixed":
      return "Mixed";
  }
}

export function backendVerdictLabel(v: Verdict): string {
  switch (v) {
    case "SUPPORTED":
      return "Supported";
    case "REFUTED":
      return "Refuted";
    case "CONFLICTING_EVIDENCE":
      return "Conflicting Evidence";
    case "INSUFFICIENT_EVIDENCE":
      return "Insufficient Evidence";
  }
}

// Returns the worst-case verdict from a list (for hero display on multi-claim results)
export function dominantVerdict(verdicts: Verdict[]): UiVerdict {
  if (verdicts.length === 0) return "mixed";
  if (verdicts.some((v) => v === "REFUTED")) return "false";
  if (
    verdicts.some(
      (v) => v === "CONFLICTING_EVIDENCE" || v === "INSUFFICIENT_EVIDENCE"
    )
  )
    return "mixed";
  return "true";
}

// Color classes derived from design tokens
export function verdictColors(v: UiVerdict) {
  switch (v) {
    case "true":
      return {
        bg: "var(--color-true-bg)",
        text: "var(--color-true-text)",
        border: "var(--color-true-border)",
      };
    case "false":
      return {
        bg: "var(--color-false-bg)",
        text: "var(--color-false-text)",
        border: "var(--color-false-border)",
      };
    case "mixed":
      return {
        bg: "var(--color-mixed-bg)",
        text: "var(--color-mixed-text)",
        border: "var(--color-mixed-border)",
      };
  }
}
