"""Create a clean public snapshot without private git history or case data."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".idea",
    ".vscode",
    "cases",
    "uploads",
    "public_cases",
}

EXCLUDE_FILES = {
    ".DS_Store",
}

EXCLUDE_SUFFIXES = {
    ".pyc",
    ".pyo",
}

PRIVATE_RELATIVE_PREFIXES = (
    Path("evals/private"),
    Path("reports/private"),
    Path(".claude"),
    # Contains real case identifiers used by the private safety scan.
    Path("docs/public-export-checklist.md"),
)


def should_skip(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if any(rel == prefix or prefix in rel.parents for prefix in PRIVATE_RELATIVE_PREFIXES):
        return True
    if any(part in EXCLUDE_DIRS for part in rel.parts):
        return True
    if path.name in EXCLUDE_FILES:
        return True
    if path.suffix in EXCLUDE_SUFFIXES:
        return True
    return False


def copy_public_snapshot(destination: Path) -> None:
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")

    for source in ROOT.rglob("*"):
        if should_skip(source):
            continue
        rel = source.relative_to(ROOT)
        target = destination / rel
        if source.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, help="New directory to create")
    args = parser.parse_args()

    copy_public_snapshot(args.destination)
    print(f"Created public snapshot: {args.destination.resolve()}")
    print("Next: inspect it, run the public safety scan, then initialise a fresh git repo there.")


if __name__ == "__main__":
    main()
