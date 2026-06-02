from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from factcheck.config import get_settings
from factcheck.llm import concurrency
from factcheck.llm.structured import call_llm_with_structured_output


class DemoOutput(BaseModel):
    value: str


class ConcurrencyTracker:
    def __init__(self) -> None:
        self.in_flight = 0
        self.max_in_flight = 0
        self.lock = asyncio.Lock()

    async def enter(self) -> None:
        async with self.lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)

    async def exit(self) -> None:
        async with self.lock:
            self.in_flight -= 1


class CountingStructuredInvoker:
    def __init__(self, tracker: ConcurrencyTracker) -> None:
        self.tracker = tracker

    async def ainvoke(self, messages):
        await self.tracker.enter()
        try:
            await asyncio.sleep(0.05)
            return DemoOutput(value="ok")
        finally:
            await self.tracker.exit()


class CountingStructuredLlm:
    def __init__(self, tracker: ConcurrencyTracker) -> None:
        self.invoker = CountingStructuredInvoker(tracker)

    def with_structured_output(self, output_class):
        return self.invoker


@pytest.fixture(autouse=True)
def reset_ollama_concurrency(monkeypatch):
    monkeypatch.delenv("OLLAMA_CONCURRENCY", raising=False)
    get_settings.cache_clear()

    reset = getattr(concurrency, "reset_ollama_semaphore_for_tests", None)
    assert reset is not None
    reset()

    yield

    get_settings.cache_clear()
    reset()


@pytest.mark.parametrize("limit", [1, 2])
async def test_structured_llm_calls_respect_ollama_concurrency_limit(
    monkeypatch,
    limit: int,
) -> None:
    monkeypatch.setenv("OLLAMA_CONCURRENCY", str(limit))
    get_settings.cache_clear()
    concurrency.reset_ollama_semaphore_for_tests()

    tracker = ConcurrencyTracker()
    llm = CountingStructuredLlm(tracker)

    await asyncio.gather(
        *(
            call_llm_with_structured_output(
                llm=llm,
                output_class=DemoOutput,
                messages=[("human", f"Return output {index}.")],
            )
            for index in range(5)
        )
    )

    assert tracker.max_in_flight <= limit
