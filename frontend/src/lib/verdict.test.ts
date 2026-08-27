import { describe, expect, it } from "vitest";
import { toUiVerdict, uiVerdictLabel } from "./verdict";

describe("verdict labels", () => {
  it("maps backend verdicts to the user-facing label", () => {
    expect(uiVerdictLabel(toUiVerdict("SUPPORTED"))).toBe("True");
    expect(uiVerdictLabel(toUiVerdict("REFUTED"))).toBe("False");
    expect(uiVerdictLabel(toUiVerdict("INSUFFICIENT_EVIDENCE"))).toBe("Mixed");
  });
});
