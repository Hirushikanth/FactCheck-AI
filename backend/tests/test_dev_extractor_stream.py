from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app import main
from factcheck.config import AppSettings
from factcheck.streaming import extractor_runner


def _parse_sse_events(body: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for block in body.strip().split("\n\n"):
        event_name = ""
        data = ""
        for line in block.splitlines():
            if line.startswith("event: "):
                event_name = line.removeprefix("event: ")
            elif line.startswith("data: "):
                data = line.removeprefix("data: ")
        events.append((event_name, json.loads(data)))
    return events


def test_dev_extractor_stream_route_disabled_by_default() -> None:
    app = main.create_app(AppSettings(_env_file=None, dev_stream_enabled=False))
    client = TestClient(app)

    response = client.post(
        "/api/dev/extractor/stream",
        json={"input": "The Earth is round."},
    )

    assert response.status_code == 404


def test_dev_extractor_stream_emits_sse_events(monkeypatch) -> None:
    class FakeExtractorGraph:
        async def astream(self, state, stream_mode: str):
            assert state.raw_input == "The Earth is round."
            assert state.metadata == "smoke"
            assert stream_mode == "updates"
            yield {
                "sentence_splitter": {
                    "contextual_sentences": [{"original_sentence": "The Earth is round."}]
                }
            }
            yield {
                "validation": {
                    "validated_claims": [{"claim_text": "The Earth is round."}],
                }
            }

    monkeypatch.setattr(extractor_runner, "build_extractor_graph", lambda: FakeExtractorGraph())

    app = main.create_app(AppSettings(_env_file=None, dev_stream_enabled=True))
    client = TestClient(app)

    response = client.post(
        "/api/dev/extractor/stream",
        json={"input": "The Earth is round.", "metadata": "smoke"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse_events(response.text)
    assert [event_name for event_name, _ in events] == [
        "node_update",
        "node_update",
        "graph_done",
    ]
    assert events[0][1]["node"] == "sentence_splitter"
    assert events[0][1]["update"] == {
        "contextual_sentences": [{"original_sentence": "The Earth is round."}]
    }
    assert events[1][1]["node"] == "validation"
    assert events[2][1]["validated_claims"] == [{"claim_text": "The Earth is round."}]
    assert isinstance(events[2][1]["elapsed_ms"], int)


def test_dev_extractor_stream_emits_pipeline_error(monkeypatch) -> None:
    class FailingExtractorGraph:
        async def astream(self, state, stream_mode: str):
            raise RuntimeError("extractor exploded")
            yield {}

    monkeypatch.setattr(extractor_runner, "build_extractor_graph", lambda: FailingExtractorGraph())

    app = main.create_app(AppSettings(_env_file=None, dev_stream_enabled=True))
    client = TestClient(app)

    response = client.post(
        "/api/dev/extractor/stream",
        json={"input": "The Earth is round."},
    )

    assert response.status_code == 200
    assert _parse_sse_events(response.text) == [
        ("pipeline_error", {"agent": "extractor", "error": "extractor exploded"})
    ]
