from __future__ import annotations

from factcheck.llm.structured import _parse_json_object


def test_parse_json_object_strips_markdown_json_fences() -> None:
    raw = '```json\n{"verdict": "SUPPORTED", "confidence": 0.8}\n```'

    assert _parse_json_object(raw) == {"verdict": "SUPPORTED", "confidence": 0.8}


def test_parse_json_object_extracts_fenced_json_after_preamble() -> None:
    raw = 'Here is the answer:\n```json\n{"verdict": "REFUTED"}\n```'

    assert _parse_json_object(raw) == {"verdict": "REFUTED"}
