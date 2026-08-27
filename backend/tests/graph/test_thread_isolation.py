"""Deterministic tests for session-to-LangGraph thread mapping."""

from __future__ import annotations

from typing import Any, TypedDict

import pytest

from factcheck.graph import runner
from factcheck.graph.checkpoint import thread_config


class _SpyGraph:
    def __init__(self) -> None:
        self.configs: list[dict[str, Any]] = []

    async def astream(self, state, *, config, stream_mode):
        assert stream_mode == "updates"
        self.configs.append(config)
        yield {"reporter": {"final_report": "ok", "status": "done"}}


@pytest.mark.asyncio
async def test_factcheck_runner_passes_session_id_as_thread_id(monkeypatch) -> None:
    graph = _SpyGraph()
    monkeypatch.setattr(runner, "build_graph", lambda: graph)

    await runner.run_factcheck_with_events(session_id="session-a", text="claim A")

    assert graph.configs == [{"configurable": {"thread_id": "session-a"}}]


@pytest.mark.asyncio
async def test_dialogue_runner_passes_session_id_as_thread_id(monkeypatch) -> None:
    from factcheck.dialogue import graph as dialogue_graph_module
    from factcheck.dialogue import run_dialogue

    observed: list[dict[str, Any]] = []

    class _DialogueGraph:
        async def ainvoke(self, state, *, config):
            observed.append(config)
            return {
                **state,
                "dialogue_response": "deterministic response",
                "classified_intent": "clarification",
                "dialogue_history": [],
            }

    monkeypatch.setattr(dialogue_graph_module, "dialogue_graph", _DialogueGraph())

    result = await run_dialogue(
        session_id="session-b",
        user_message="Why?",
        raw_input="claim B",
        claim_results=[],
    )

    assert result["response"] == "deterministic response"
    assert observed == [{"configurable": {"thread_id": "session-b"}}]


@pytest.mark.asyncio
async def test_sqlite_checkpoints_are_isolated_by_thread_id(tmp_path) -> None:
    aiosqlite = pytest.importorskip("aiosqlite")
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from langgraph.graph import END, START, StateGraph

    class _State(TypedDict):
        marker: str

    async with aiosqlite.connect(str(tmp_path / "checkpoints.db")) as connection:
        saver = AsyncSqliteSaver(connection)
        await saver.setup()

        graph = StateGraph(_State)
        graph.add_node("record", lambda state: {"marker": state["marker"]})
        graph.add_edge(START, "record")
        graph.add_edge("record", END)
        compiled = graph.compile(checkpointer=saver)

        await compiled.ainvoke({"marker": "A"}, config=thread_config("session-a"))
        await compiled.ainvoke({"marker": "B"}, config=thread_config("session-b"))

        checkpoint_a = await saver.aget_tuple(thread_config("session-a"))
        checkpoint_b = await saver.aget_tuple(thread_config("session-b"))

    assert checkpoint_a is not None and checkpoint_b is not None
    assert checkpoint_a.checkpoint["channel_values"]["marker"] == "A"
    assert checkpoint_b.checkpoint["channel_values"]["marker"] == "B"
