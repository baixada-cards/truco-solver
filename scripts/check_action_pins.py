"""Reject mutable third-party references in GitHub Actions workflows."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

WORKFLOW_DIRECTORY = Path(".github/workflows")
WORKFLOW_SUFFIXES = {".yaml", ".yml"}
USES_LINE = re.compile(r"^\s*(?:-\s*)?uses\s*:\s*(?P<reference>.*?)\s*$")
FULL_COMMIT = re.compile(
    r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+"
    r"(?:/[A-Za-z0-9_.-]+)*@[0-9a-fA-F]{40}$"
)


@dataclass(frozen=True)
class UnpinnedAction:
    path: Path
    line_number: int
    reference: str


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _action_reference(line: str) -> str | None:
    match = USES_LINE.match(line)
    if match is None:
        return None
    reference = match.group("reference").split("#", 1)[0].strip()
    return _unquote(reference)


def find_unpinned_actions(
    workflow_directory: Path = WORKFLOW_DIRECTORY,
) -> list[UnpinnedAction]:
    failures: list[UnpinnedAction] = []
    workflow_paths = sorted(
        path
        for path in workflow_directory.rglob("*")
        if path.is_file() and path.suffix in WORKFLOW_SUFFIXES
    )
    for path in workflow_paths:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            reference = _action_reference(line)
            if reference is None or reference.startswith("./"):
                continue
            if not FULL_COMMIT.fullmatch(reference):
                failures.append(UnpinnedAction(path, line_number, reference))
    return failures


def main() -> int:
    failures = find_unpinned_actions()
    if not failures:
        print("All non-local GitHub Actions use full commit SHAs.")
        return 0
    print("Mutable or unsupported GitHub Actions references found:")
    for failure in failures:
        print(f"  {failure.path}:{failure.line_number}: {failure.reference}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
