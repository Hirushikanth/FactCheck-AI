#!/usr/bin/env bash
set -euo pipefail

failures=0

version_at_least() {
  local actual="$1"
  local minimum="$2"

  python3 - "$actual" "$minimum" <<'PY'
from __future__ import annotations

import re
import sys


def parts(value: str) -> tuple[int, ...]:
    match = re.search(r"\d+(?:\.\d+)*", value)
    if not match:
        raise SystemExit(1)
    return tuple(int(part) for part in match.group(0).split("."))


actual = parts(sys.argv[1])
minimum = parts(sys.argv[2])
width = max(len(actual), len(minimum))

if actual + (0,) * (width - len(actual)) >= minimum + (0,) * (width - len(minimum)):
    raise SystemExit(0)

raise SystemExit(1)
PY
}

check_command() {
  local name="$1"
  local command="$2"
  local version_command="$3"
  local minimum="${4:-}"

  if command -v "$command" >/dev/null 2>&1; then
    local version
    version="$($version_command 2>&1 | head -n 1)"
    if [ -n "$minimum" ] && ! version_at_least "$version" "$minimum"; then
      printf '%s: %s (requires >= %s)\n' "$name" "$version" "$minimum"
      failures=$((failures + 1))
    else
      printf '%s: %s\n' "$name" "$version"
    fi
  else
    printf '%s: missing\n' "$name"
    failures=$((failures + 1))
  fi
}

check_command "Python" "python3" "python3 --version" "3.11"
check_command "Poetry" "poetry" "poetry --version" "1.8"
check_command "Node.js" "node" "node --version" "20"
check_command "Git" "git" "git --version"
check_command "Ollama" "ollama" "ollama --version"

if [ "$failures" -gt 0 ]; then
  printf '\n%s required tool(s) missing.\n' "$failures"
  exit 1
fi

printf '\nToolchain check passed.\n'
