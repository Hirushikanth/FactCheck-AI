"""Reporter node for the main fact-checking pipeline."""

from __future__ import annotations

from factcheck.reporter import run_reporter
from factcheck.state import FactCheckState


async def reporter_node(state: FactCheckState) -> dict[str, str]:
    """Generate the final markdown report after claim verification completes."""

    final_report = await run_reporter(state)
    return {
        "current_agent": "reporter",
        "final_report": final_report,
        "status": "done",
    }
