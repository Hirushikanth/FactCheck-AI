from __future__ import annotations

from factcheck.extractor import prompts


def test_extractor_llm_prompts_route_reasoning_into_json() -> None:
    prompt_values = [
        prompts.SELECTION_SYSTEM_PROMPT,
        prompts.DISAMBIGUATION_SYSTEM_PROMPT,
        prompts.DECOMPOSITION_SYSTEM_PROMPT,
        prompts.VALIDATION_SYSTEM_PROMPT,
    ]

    for prompt in prompt_values:
        first_line = prompt.strip().splitlines()[0]

        assert "Return ONLY one JSON object" in first_line
        assert "reasoning" in prompt
        assert "No markdown. No preamble." in prompt


def test_extractor_llm_prompts_do_not_request_printed_cot() -> None:
    combined_prompts = "\n".join(
        [
            prompts.SELECTION_SYSTEM_PROMPT,
            prompts.DISAMBIGUATION_SYSTEM_PROMPT,
            prompts.DECOMPOSITION_SYSTEM_PROMPT,
            prompts.VALIDATION_SYSTEM_PROMPT,
        ]
    )

    assert "I will now provide step-by-step reasoning" not in combined_prompts
    assert "I will perform a detailed analysis" not in combined_prompts
    assert "I will systematically analyze" not in combined_prompts
