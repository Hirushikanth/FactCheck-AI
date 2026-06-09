from __future__ import annotations

from factcheck.verifier import prompts


def test_evaluator_prompt_starts_with_json_only_instruction() -> None:
    first_line = prompts.EVIDENCE_EVALUATOR_SYSTEM_PROMPT.strip().splitlines()[0]

    assert "ONLY one JSON object" in first_line
    assert "CONFLICTING_EVIDENCE" in prompts.EVIDENCE_EVALUATOR_SYSTEM_PROMPT
    assert "needs_more_evidence=true" in prompts.EVIDENCE_EVALUATOR_SYSTEM_PROMPT


def test_query_prompts_have_initial_and_iterative_variants() -> None:
    assert "Previous queries" in prompts.QUERY_GENERATOR_ITERATIVE_HUMAN_PROMPT
    assert "Missing evidence aspects" in prompts.QUERY_GENERATOR_ITERATIVE_HUMAN_PROMPT
    assert "supporting and contradictory evidence" in prompts.QUERY_GENERATOR_INITIAL_SYSTEM_PROMPT


def test_obsolete_ranker_and_verdict_prompts_are_removed() -> None:
    assert not hasattr(prompts, "EVIDENCE_RANKER_SYSTEM_PROMPT")
    assert not hasattr(prompts, "EVIDENCE_RANKER_HUMAN_PROMPT")
    assert not hasattr(prompts, "VERDICT_SYSTEM_PROMPT")
    assert not hasattr(prompts, "VERDICT_HUMAN_PROMPT")
