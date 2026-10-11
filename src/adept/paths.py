"""Repository path helpers used by ADEPT scripts and packages."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
BENCHMARK_ROOT = PROJECT_ROOT / "vendor" / "upstream_benchmark"


def add_to_sys_path(path: Path, *, prepend: bool = True) -> None:
    """Add a path to ``sys.path`` once."""
    value = str(path)
    if value in sys.path:
        return
    if prepend:
        sys.path.insert(0, value)
    else:
        sys.path.append(value)


def add_src_to_path() -> None:
    """Make the local ``src`` package importable from source checkouts."""
    add_to_sys_path(SRC_ROOT)


def add_benchmark_to_path() -> None:
    """Make vendored benchmark imports available without changing directories."""
    add_to_sys_path(BENCHMARK_ROOT)
