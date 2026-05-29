from __future__ import annotations

from pydantic import BaseModel

from factcheck.extractor.utils.text import remove_following_sentences
from factcheck.extractor.utils.voting import process_with_voting
from factcheck.llm.structured import call_llm_with_structured_output


class DemoOutput(BaseModel):
    value: str


class FakeStructuredInvoker:
    def __init__(self, response):
        self.response = response

    async def ainvoke(self, messages):
        return self.response


class FakeStructuredLlm:
    def __init__(self, response):
        self.response = response

    def with_structured_output(self, output_class):
        return FakeStructuredInvoker(self.response)


class FlakyStructuredInvoker:
    def __init__(self):
        self.calls = 0
        self.messages = []

    async def ainvoke(self, messages):
        self.calls += 1
        self.messages.append(messages)
        if self.calls == 1:
            raise ValueError("invalid structured output")
        return DemoOutput(value="repaired")


class FlakyStructuredLlm:
    def __init__(self):
        self.invoker = FlakyStructuredInvoker()

    def with_structured_output(self, output_class):
        return self.invoker


class NoneThenValidStructuredInvoker:
    def __init__(self):
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        if self.calls == 1:
            return None
        return DemoOutput(value="repaired-from-none")


class NoneThenValidStructuredLlm:
    def __init__(self):
        self.invoker = NoneThenValidStructuredInvoker()

    def with_structured_output(self, output_class):
        return self.invoker


class StructuredNonePlainJsonInvoker:
    async def ainvoke(self, messages):
        return None


class StructuredNonePlainJsonLlm:
    def __init__(self):
        self.structured_invoker = StructuredNonePlainJsonInvoker()
        self.plain_messages = []

    def with_structured_output(self, output_class):
        return self.structured_invoker

    async def ainvoke(self, messages):
        self.plain_messages.append(messages)
        return '{"value": "plain-json"}'


def test_remove_following_sentences_strips_following_context() -> None:
    context = "\n".join(
        [
            "[Preceding Sentences:]",
            "Earlier context.",
            "[Sentence of Interest for current task:]",
            "The claim sentence.",
            "[Following Sentences:]",
            "Later context.",
        ]
    )

    assert remove_following_sentences(context) == "\n".join(
        [
            "[Preceding Sentences:]",
            "Earlier context.",
            "[Sentence of Interest for current task:]",
            "The claim sentence.",
        ]
    )


async def test_process_with_voting_requires_minimum_successes() -> None:
    attempts = iter([(True, "first"), (False, None), (True, "second")])

    async def processor(item, llm):
        return next(attempts)

    results = await process_with_voting(
        items=["sentence"],
        processor=processor,
        llm=object(),
        completions=3,
        min_successes=2,
        result_factory=lambda value, item: f"{item}:{value}",
    )

    assert results == ["sentence:first"]


async def test_structured_llm_helper_returns_parsed_model() -> None:
    response = DemoOutput(value="parsed")

    result = await call_llm_with_structured_output(
        llm=FakeStructuredLlm(response),
        output_class=DemoOutput,
        messages=[("human", "Return structured output.")],
    )

    assert result == response


async def test_structured_llm_helper_retries_once_with_schema_hint() -> None:
    llm = FlakyStructuredLlm()

    result = await call_llm_with_structured_output(
        llm=llm,
        output_class=DemoOutput,
        messages=[("human", "Return structured output.")],
    )

    assert result == DemoOutput(value="repaired")
    assert llm.invoker.calls == 2
    assert "JSON schema" in llm.invoker.messages[1][-1][1]


async def test_structured_llm_helper_retries_when_first_attempt_returns_none() -> None:
    llm = NoneThenValidStructuredLlm()

    result = await call_llm_with_structured_output(
        llm=llm,
        output_class=DemoOutput,
        messages=[("human", "Return structured output.")],
    )

    assert result == DemoOutput(value="repaired-from-none")
    assert llm.invoker.calls == 2


async def test_structured_llm_helper_uses_plain_json_fallback_after_structured_none() -> None:
    llm = StructuredNonePlainJsonLlm()

    result = await call_llm_with_structured_output(
        llm=llm,
        output_class=DemoOutput,
        messages=[("human", "Return structured output.")],
    )

    assert result == DemoOutput(value="plain-json")
    assert "Return only one JSON object" in llm.plain_messages[0][-1][1]
