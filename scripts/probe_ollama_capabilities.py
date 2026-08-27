#!/usr/bin/env python3
"""Read-only diagnostic for the configured Ollama model's streaming capabilities."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx

# Allow running this script from the repository root or from backend/ without
# requiring an editable package installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from factcheck.config import AppSettings  # noqa: E402
from factcheck.llm.ollama import probe_ollama_capabilities  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check local Ollama reachability, model availability, and thinking streaming."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="print the complete result as machine-readable JSON",
    )
    parser.add_argument(
        "--show-raw-thinking",
        action="store_true",
        help="include model-emitted thinking text in the diagnostic output",
    )
    return parser


async def _run(args: argparse.Namespace) -> dict[str, object]:
    settings = AppSettings()
    async with httpx.AsyncClient(timeout=settings.ollama_timeout) as client:
        return await probe_ollama_capabilities(
            settings,
            client,
            show_raw_thinking=args.show_raw_thinking,
        )


def main() -> int:
    args = _parser().parse_args()
    try:
        result = asyncio.run(_run(args))
    except KeyboardInterrupt:
        return 130

    if args.as_json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"Ollama capability status: {result['status']}")
        print(f"Reachable: {result['reachable']}")
        print(f"Configured model: {result['model']}")
        print(f"Model installed: {result['model_installed']}")
        print(f"Thinking chunks: {result['thinking_chunks']}")
        print(f"Content chunks: {result['content_chunks']}")
        print(f"Elapsed: {result['elapsed_ms']} ms")
        if "error" in result:
            print(f"Diagnostic error: {result['error']}")
        if args.show_raw_thinking and result.get("thinking"):
            print("\nRaw model-emitted thinking:\n" + str(result["thinking"]))

    return 0 if result["status"] != "probe_failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
