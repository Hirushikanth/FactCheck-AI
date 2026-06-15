from __future__ import annotations

import pytest

from factcheck.extractor.utils.fidelity import _ensure_wordnet


@pytest.fixture(scope="session", autouse=True)
def _bootstrap_wordnet() -> None:
    _ensure_wordnet()
