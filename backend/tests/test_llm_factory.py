from __future__ import annotations

from factcheck.config import AppSettings
from factcheck.llm import factory


def test_get_verifier_llm_passes_explicit_num_ctx(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeChatOllama:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(factory, "ChatOllama", FakeChatOllama)

    llm = factory.get_verifier_llm(
        temperature=0.0,
        num_ctx=4096,
        settings=AppSettings(_env_file=None),
    )

    assert isinstance(llm, FakeChatOllama)
    assert captured["num_ctx"] == 4096


def test_get_verifier_llm_uses_settings_num_ctx_when_explicit_value_missing(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeChatOllama:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(factory, "ChatOllama", FakeChatOllama)

    settings = AppSettings(ollama_num_ctx=8192, _env_file=None)
    factory.get_verifier_llm(temperature=0.0, settings=settings)

    assert captured["num_ctx"] == 8192


def test_get_verifier_llm_caps_explicit_num_ctx_to_settings_ceiling(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeChatOllama:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(factory, "ChatOllama", FakeChatOllama)

    settings = AppSettings(ollama_num_ctx=2048, _env_file=None)
    factory.get_verifier_llm(temperature=0.0, num_ctx=4096, settings=settings)

    assert captured["num_ctx"] == 2048


def test_get_reporter_llm_applies_reporter_runtime_options(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeChatOllama:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(factory, "ChatOllama", FakeChatOllama)

    llm = factory.get_reporter_llm(
        temperature=0.1,
        num_ctx=2048,
        num_predict=512,
        settings=AppSettings(ollama_num_ctx=8192, _env_file=None),
    )

    assert isinstance(llm, FakeChatOllama)
    assert captured["temperature"] == 0.1
    assert captured["num_ctx"] == 2048
    assert captured["num_predict"] == 512


def test_get_extractor_llm_applies_num_predict_and_num_ctx(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeChatOllama:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(factory, "ChatOllama", FakeChatOllama)

    llm = factory.get_extractor_llm(
        temperature=0.2,
        num_ctx=8192,
        num_predict=512,
        settings=AppSettings(ollama_num_ctx=8192, _env_file=None),
    )

    assert isinstance(llm, FakeChatOllama)
    assert captured["temperature"] == 0.2
    assert captured["num_ctx"] == 8192
    assert captured["num_predict"] == 512


def test_get_dialogue_llm_uses_dialogue_num_predict(monkeypatch) -> None:
    from factcheck.dialogue.config import DIALOGUE_NUM_PREDICT, MAX_RESPONSE_TOKENS

    captured: dict[str, object] = {}

    class FakeChatOllama:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(factory, "ChatOllama", FakeChatOllama)

    llm = factory.get_dialogue_llm(settings=AppSettings(_env_file=None))

    assert isinstance(llm, FakeChatOllama)
    assert captured["num_predict"] == DIALOGUE_NUM_PREDICT
    assert DIALOGUE_NUM_PREDICT != MAX_RESPONSE_TOKENS
