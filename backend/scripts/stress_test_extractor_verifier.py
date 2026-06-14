#!/usr/bin/env python3
"""Manual extractor → verifier integration test."""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

env_path = BACKEND_ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())

from factcheck.extractor import run_extractor
from factcheck.verifier import run_verifier

CASES = [
    {
        "id": "pyramids_aliens",
        "input": "The pyramids were built by aliens.",
        "note": "False claim; fidelity should stay faithful, verifier should return a verdict.",
    },
    {
        "id": "compound_berries",
        "input": (
            "Bananas are berries, but strawberries are not, "
            "according to the botanical definitions of fruits."
        ),
        "note": "Two faithful conjuncts; verifier runs once per claim.",
    },
    {
        "id": "morph_tense",
        "input": "Jane was running the firm.",
        "note": "Morph/fidelity path; verifier should accept faithful extractor output.",
    },
    {
        "id": "dangerous_false",
        "input": "Drinking bleach cures COVID-19.",
        "note": "Preserves dangerous false claim through extractor and verifier.",
    },
    {
        "id": "french_revolution_1815",
        "input": "The French Revolution began in 1815 after Napoleon's defeat.",
        "note": "Incomplete decomposition must not yield only SUPPORTED Napoleon defeat.",
        "expect_refuted": True,
    },
]

VALID_VERDICTS = frozenset(
    {"SUPPORTED", "REFUTED", "INSUFFICIENT_EVIDENCE", "CONFLICTING_EVIDENCE"}
)


def _fmt_verdict(result: dict) -> str:
    return (
        f"verdict={result.get('verdict')!r} "
        f"confidence={result.get('confidence')} "
        f"fidelity_status={result.get('fidelity_status')!r} "
        f"queries={len(result.get('search_queries') or [])} "
        f"sources={len(result.get('sources') or [])}"
    )


async def main() -> int:
    print("=" * 72)
    print("EXTRACTOR → VERIFIER INTEGRATION TEST")
    print(f"Ollama: {os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')}")
    print(f"Model:  {os.environ.get('OLLAMA_MODEL', 'mistral:7b')}")
    print("=" * 72)

    total_claims = 0
    verified_ok = 0
    errors = 0

    for case in CASES:
        case_id = case["id"]
        raw_input = case["input"]
        note = case["note"]

        print(f"\n{'=' * 72}")
        print(f"CASE: {case_id}")
        print(f"INPUT: {raw_input}")
        print(f"NOTE: {note}")

        extract_start = time.perf_counter()
        try:
            claims = await run_extractor(raw_input)
        except Exception as exc:
            errors += 1
            print(f"EXTRACTOR ERROR ({time.perf_counter() - extract_start:.1f}s): {exc}")
            continue

        extract_elapsed = time.perf_counter() - extract_start
        print(f"EXTRACTOR: {len(claims)} claim(s) in {extract_elapsed:.1f}s")
        if not claims:
            errors += 1
            print("  (no claims produced)")
            continue

        case_verdicts: list[str] = []
        case_claim_texts: list[str] = []

        for index, claim in enumerate(claims, 1):
            total_claims += 1
            case_claim_texts.append(claim.claim_text)
            print(f"\n  --- claim {index}/{len(claims)} ---")
            print(f"  extractor fidelity: {claim.fidelity_status!r}")
            print(f"  claim_text: {claim.claim_text}")
            print(f"  source_sentence: {claim.original_sentence}")

            verify_start = time.perf_counter()
            try:
                result = await run_verifier(claim)
                verify_elapsed = time.perf_counter() - verify_start

                verdict = result.get("verdict")
                case_verdicts.append(str(verdict))
                fidelity_out = result.get("fidelity_status")
                ok = (
                    verdict in VALID_VERDICTS
                    and fidelity_out == claim.fidelity_status
                    and result.get("claim") == claim.claim_text
                    and result.get("source_sentence") == claim.original_sentence
                )
                if ok:
                    verified_ok += 1
                    status = "OK"
                else:
                    errors += 1
                    status = "MISMATCH"

                print(f"  VERIFIER ({verify_elapsed:.1f}s): {status}")
                print(f"    {_fmt_verdict(result)}")
                if result.get("reasoning"):
                    reasoning = str(result["reasoning"]).replace("\n", " ")
                    print(f"    reasoning: {reasoning[:200]}{'...' if len(reasoning) > 200 else ''}")
            except Exception as exc:
                errors += 1
                verify_elapsed = time.perf_counter() - verify_start
                print(f"  VERIFIER ERROR ({verify_elapsed:.1f}s): {exc}")

        if case.get("expect_refuted"):
            core_claims = [
                (text, verdict)
                for text, verdict in zip(case_claim_texts, case_verdicts, strict=False)
                if "revolution" in text.casefold() or "1815" in text
            ]
            if not core_claims:
                core_claims = list(zip(case_claim_texts, case_verdicts, strict=False))
            if not any(verdict == "REFUTED" for _, verdict in core_claims):
                errors += 1
                print("  CASE EXPECTATION FAILED: expected REFUTED for revolution/1815 claim")
            elif (
                len(claims) == 1
                and "napoleon" in claims[0].claim_text.casefold()
                and "revolution" not in claims[0].claim_text.casefold()
                and case_verdicts == ["SUPPORTED"]
            ):
                errors += 1
                print("  CASE EXPECTATION FAILED: only Napoleon's defeat was checked as SUPPORTED")

    print(f"\n{'=' * 72}")
    print(
        f"SUMMARY: {verified_ok}/{total_claims} claims verified cleanly, "
        f"{errors} error(s)/mismatch(es)"
    )
    print("=" * 72)
    return 0 if errors == 0 and total_claims > 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
