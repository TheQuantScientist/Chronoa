"""Check for large nonignored files before publishing the repository."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


DEFAULT_LIMIT_MB = 25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit-mb",
        type=float,
        default=DEFAULT_LIMIT_MB,
        help="Fail if a tracked or unignored file is larger than this many MB.",
    )
    return parser.parse_args()


def git_files(*args: str) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z", *args],
        check=True,
        stdout=subprocess.PIPE,
    )
    return [Path(item.decode()) for item in result.stdout.split(b"\\0") if item]


def oversized(paths: list[Path], limit_bytes: int) -> list[tuple[Path, int]]:
    rows: list[tuple[Path, int]] = []
    for path in paths:
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size > limit_bytes:
            rows.append((path, size))
    return sorted(rows, key=lambda item: item[1], reverse=True)


def main() -> None:
    args = parse_args()
    limit_bytes = int(args.limit_mb * 1024 * 1024)
    checked = git_files() + git_files("--others", "--exclude-standard")
    offenders = oversized(checked, limit_bytes)
    if not offenders:
        print(f"OK: no tracked or unignored files exceed {args.limit_mb:g} MB.")
        return

    print(f"Files larger than {args.limit_mb:g} MB are visible to git:")
    for path, size in offenders:
        print(f"{size / (1024 * 1024):8.1f} MB  {path}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
