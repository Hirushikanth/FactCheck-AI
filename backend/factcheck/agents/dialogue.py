"""Phase 2 dialogue placeholder."""

from __future__ import annotations

from factcheck.state import FactCheckState


def dialogue_node(state: FactCheckState) -> dict[str, str]:
    """Keep dialogue behavior as a no-op until the dialogue phase."""

    return {"current_agent": "dialogue"}
