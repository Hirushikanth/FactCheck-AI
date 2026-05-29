"""Orchestrator node for the fact-checking pipeline."""

from __future__ import annotations

from factcheck.state import FactCheckState


def orchestrator_node(state: FactCheckState) -> dict[str, str]:
    """Mark the pipeline as running before agent execution begins."""

    return {"current_agent": "orchestrator", "status": "running"}
