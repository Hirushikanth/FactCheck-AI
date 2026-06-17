#!/usr/bin/env python3
"""Manual extractor stress test — runs varied inputs through the full graph."""

from __future__ import annotations

import asyncio
import os
import sys
import textwrap
import time
from pathlib import Path

# Ensure backend package is importable when run as script
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

CASES: list[dict[str, object]] = [
    {
        "id": "great_wall_myth",
        "input": "The Great Wall of China is visible from space with the naked eye.",
        "expect": "Direct-claim route; myth sentence must extract without selection drop.",
        "checks": lambda claims: bool(claims) and any(
            "great wall" in c.claim_text.casefold() for c in claims
        ),
    },
    {
        "id": "lightning_myth",
        "input": "Lightning never strikes the same place twice.",
        "expect": "Direct-claim route; proverb-like myth must still extract.",
        "checks": lambda claims: bool(claims) and any(
            "lightning" in c.claim_text.casefold() for c in claims
        ),
    },
    {
        "id": "morph_tense",
        "input": "Jane was running the firm.",
        "expect": "Faithful decomposition; tense cleanup OK (runs/running). No fallback.",
        "checks": lambda claims: (
            claims
            and all(c.fidelity_status == "faithful" for c in claims)
            and not any("egypt" in c.claim_text.casefold() for c in claims)
        ),
    },
    {
        "id": "plural_singular",
        "input": "The pyramids were built by aliens.",
        "expect": "Preserves 'aliens'; faithful (pyramid/pyramids morph OK).",
        "checks": lambda claims: (
            claims
            and all(c.fidelity_status == "faithful" for c in claims)
            and any("alien" in c.claim_text.casefold() for c in claims)
            and not any("egypt" in c.claim_text.casefold() or "human" in c.claim_text.casefold() for c in claims)
        ),
    },
    {
        "id": "dangerous_false",
        "input": "Drinking bleach cures COVID-19.",
        "expect": "Preserves false claim verbatim; faithful.",
        "checks": lambda claims: (
            claims
            and all(c.fidelity_status == "faithful" for c in claims)
            and any("bleach" in c.claim_text.casefold() for c in claims)
        ),
    },
    {
        "id": "compound_berries",
        "input": (
            "Bananas are berries, but strawberries are not, "
            "according to the botanical definitions of fruits."
        ),
        "expect": "Two faithful conjuncts; negation preserved on strawberries.",
        "checks": lambda claims: (
            len(claims) >= 2
            and all(c.fidelity_status == "faithful" for c in claims)
            and any("banana" in c.claim_text.casefold() and "berr" in c.claim_text.casefold() for c in claims)
            and any(
                "strawberr" in c.claim_text.casefold()
                and "not" in c.claim_text.casefold()
                and "berr" in c.claim_text.casefold()
                for c in claims
            )
        ),
    },
    {
        "id": "atomic_split",
        "input": (
            "Ada Lovelace wrote notes about the Analytical Engine "
            "and Charles Babbage designed it."
        ),
        "expect": "Two faithful atomic claims from compound sentence.",
        "checks": lambda claims: (
            len(claims) >= 2
            and all(c.fidelity_status == "faithful" for c in claims)
            and any("ada" in c.claim_text.casefold() for c in claims)
            and any("babbage" in c.claim_text.casefold() for c in claims)
        ),
    },
    {
        "id": "pronoun_context",
        "input": (
            "Ada Lovelace was a mathematician. "
            "She wrote notes about Charles Babbage's Analytical Engine."
        ),
        "expect": "Disambiguation may resolve She→Ada; claims faithful.",
        "checks": lambda claims: (
            bool(claims)
            and any("note" in c.claim_text.casefold() or "analytical" in c.claim_text.casefold() for c in claims)
        ),
    },
    {
        "id": "negation_drop_trap",
        "input": "Strawberries are not berries according to botanical definitions.",
        "expect": "Must keep negation; not flip to 'strawberries are berries'.",
        "checks": lambda claims: (
            claims
            and all(
                not (
                    "strawberr" in c.claim_text.casefold()
                    and "berr" in c.claim_text.casefold()
                    and "not" not in c.claim_text.casefold()
                )
                for c in claims
            )
        ),
    },
    {
        "id": "multi_sentence_stress",
        "input": (
            "The Great Wall of China is visible from space. "
            "Jane was running the firm while the pyramids were studied by archaeologists. "
            "Bananas are berries, but strawberries are not."
        ),
        "expect": "Multiple claims; no mass fallback; morph/tense variants tolerated.",
        "checks": lambda claims: len(claims) >= 2,
    },
    {
        "id": "temporal_subordinate_trap",
        "input": "The French Revolution began in 1815 after Napoleon's defeat.",
        "expect": "Must cover revolution/1815; not only Napoleon's defeat.",
        "checks": lambda claims: (
            bool(claims)
            and any(
                "revolution" in c.claim_text.casefold() and "1815" in c.claim_text
                for c in claims
            )
            and not (
                len(claims) == 1
                and "napoleon" in claims[0].claim_text.casefold()
                and "revolution" not in claims[0].claim_text.casefold()
            )
        ),
    },
]


def _format_claims(claims) -> str:
    if not claims:
        return "  (no claims)"
    lines = []
    for i, c in enumerate(claims, 1):
        lines.append(
            f"  [{i}] fidelity={c.fidelity_status!r} complete={c.is_complete_declarative}\n"
            f"      claim: {c.claim_text}\n"
            f"      source: {c.disambiguated_sentence}"
        )
    return "\n".join(lines)


async def main() -> int:
    print("=" * 72)
    print("EXTRACTOR STRESS TEST")
    print(f"Ollama: {os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')}")
    print(f"Model:  {os.environ.get('OLLAMA_MODEL', 'gemma4')}")
    print("=" * 72)

    passed = 0
    failed = 0

    for case in CASES:
        case_id = case["id"]
        raw_input = str(case["input"])
        expect = str(case["expect"])
        checks = case["checks"]

        print(f"\n--- {case_id} ---")
        print(f"INPUT: {raw_input}")
        print(f"EXPECT: {expect}")

        start = time.perf_counter()
        try:
            result = await run_extractor(raw_input)
            claims = result.claims
            elapsed = time.perf_counter() - start
            ok = bool(checks(claims))
            status = "PASS" if ok else "FAIL"
            if ok:
                passed += 1
            else:
                failed += 1
            print(f"RESULT: {status} ({elapsed:.1f}s, {len(claims)} claim(s))")
            print(_format_claims(claims))
        except Exception as exc:
            failed += 1
            elapsed = time.perf_counter() - start
            print(f"RESULT: ERROR ({elapsed:.1f}s) — {exc}")

    print("\n" + "=" * 72)
    print(f"SUMMARY: {passed} passed, {failed} failed, {len(CASES)} total")
    print("=" * 72)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
