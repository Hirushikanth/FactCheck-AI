"""Reporter node for the main fact-checking pipeline."""

from __future__ import annotations

from factcheck.reporter import run_reporter
from factcheck.graph.event_bus import event_scope
from factcheck.state import FactCheckState


async def reporter_node(state: FactCheckState) -> dict[str, str]:
    """Generate the final markdown report after claim verification completes."""

    with event_scope(state["session_id"], agent="reporter", stage="report_generation"):
        final_report = await run_reporter(state)
    return {
        "current_agent": "reporter",
        "final_report": final_report,
        "status": "done",
    }
