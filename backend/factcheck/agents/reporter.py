"""Phase 2 reporter placeholder."""

from __future__ import annotations

from factcheck.state import FactCheckState


def reporter_node(state: FactCheckState) -> dict[str, str]:
    """Finalize the stub report until reporting logic is implemented."""

    return {
        "current_agent": "reporter",
        "final_report": "Phase 1 pipeline scaffold completed.",
        "status": "done",
    }
