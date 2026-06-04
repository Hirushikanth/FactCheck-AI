"""Stream extractor LangGraph updates for local development."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from time import perf_counter
from typing import Any

from factcheck.extractor.graph import build_extractor_graph
from factcheck.extractor.schemas import ExtractorState
from factcheck.streaming.sse import to_jsonable


StreamPayload = dict[str, Any]


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat()


async def stream_extractor_updates(
    input_text: str,
    metadata: str | None = None,
) -> AsyncIterator[StreamPayload]:
    """Yield dev SSE payloads from the extractor subgraph update stream."""

    started_at = perf_counter()
    state_snapshot = ExtractorState(raw_input=input_text, metadata=metadata).model_dump()

    try:
        graph = build_extractor_graph()
        async for chunk in graph.astream(
            ExtractorState(raw_input=input_text, metadata=metadata),
            stream_mode="updates",
        ):
            for node_name, update in chunk.items():
                jsonable_update = to_jsonable(update)
                if isinstance(jsonable_update, dict):
                    state_snapshot.update(jsonable_update)

                yield {
                    "event": "node_update",
                    "data": {
                        "node": node_name,
                        "update": jsonable_update,
                        "timestamp": _utc_timestamp(),
                    },
                }

        yield {
            "event": "graph_done",
            "data": {
                "validated_claims": state_snapshot.get("validated_claims", []),
                "elapsed_ms": int((perf_counter() - started_at) * 1000),
            },
        }
    except Exception as exc:
        yield {
            "event": "pipeline_error",
            "data": {
                "error": str(exc),
                "agent": "extractor",
            },
        }
