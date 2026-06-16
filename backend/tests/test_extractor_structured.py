from __future__ import annotations

import json

from pydantic import BaseModel

from factcheck.llm.extractor_structured import (
    _repair_json_string_values,
    call_extractor_structured_output,
)


class DemoOutput(BaseModel):
    is_complete_declarative: bool
    reasoning: str = ""


class FakePlainInvoker:
    def __init__(self, responses: list[object]):
        self.responses = list(responses)
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakePlainLlm:
    def __init__(self, responses: list[object]):
        self.plain = FakePlainInvoker(responses)

    def with_structured_output(self, output_class, **kwargs):
        raise AssertionError("json_mode should not be called when plain JSON succeeds")

    async def ainvoke(self, messages):
        return await self.plain.ainvoke(messages)


class FakeJsonModeInvoker:
    async def ainvoke(self, messages):
        return DemoOutput(is_complete_declarative=True, reasoning="from json mode")


class FakeFallbackLlm:
    def __init__(self):
        self.plain = FakePlainInvoker(["not json", "still not json"])

    def with_structured_output(self, output_class, **kwargs):
        return FakeJsonModeInvoker()

    async def ainvoke(self, messages):
        return await self.plain.ainvoke(messages)


def test_repair_json_string_values_fixes_unescaped_reasoning_quotes() -> None:
    broken = (
        '{"is_complete_declarative": true, "reasoning": "The sentence "The Earth is round" is factual"}'
    )
    repaired = _repair_json_string_values(broken)
    parsed = json.loads(repaired)
    assert parsed["is_complete_declarative"] is True
    assert "The Earth is round" in parsed["reasoning"]


async def test_extractor_structured_plain_json_primary() -> None:
    llm = FakePlainLlm(
        ['{"is_complete_declarative": true, "reasoning": "complete sentence"}']
    )

    result = await call_extractor_structured_output(
        llm=llm,
        output_class=DemoOutput,
        messages=[("human", "validate")],
    )

    assert result == DemoOutput(is_complete_declarative=True, reasoning="complete sentence")
    assert llm.plain.calls == 1


async def test_extractor_structured_uses_json_mode_fallback() -> None:
    llm = FakeFallbackLlm()

    result = await call_extractor_structured_output(
        llm=llm,
        output_class=DemoOutput,
        messages=[("human", "validate")],
    )

    assert result == DemoOutput(is_complete_declarative=True, reasoning="from json mode")
    assert llm.plain.calls == 2
