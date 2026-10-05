"""Small CLI parsing helpers shared by Chronoa runner scripts."""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def split_csv_values(values: Iterable[str] | None, *, default: Sequence[str] = ()) -> list[str]:
    """Expand repeatable comma-separated CLI values while preserving order."""
    raw_values = list(values or default)
    items: list[str] = []
    for raw_value in raw_values:
        for part in str(raw_value).split(','):
            item = part.strip()
            if item:
                items.append(item)
    return items


def unique_preserve_order(values: Iterable[str]) -> list[str]:
    """Return unique values in their first-seen order."""
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


def slugify(value: str) -> str:
    """Build a lowercase filename-safe slug from user-provided text."""
    chars: list[str] = []
    for char in value.strip().lower():
        if char.isalnum():
            chars.append(char)
        elif chars and chars[-1] != '_':
            chars.append('_')
    return ''.join(chars).strip('_')


def expand_selection(
    raw_values: Iterable[str] | None,
    defaults: Sequence[str],
    aliases: dict[str, str] | None = None,
    *,
    all_tokens: set[str] | None = None,
) -> list[str]:
    """Expand repeatable CLI selections with aliases and ``all`` support."""
    aliases = aliases or {}
    all_tokens = all_tokens or {'all'}
    selected: list[str] = []
    for item in split_csv_values(raw_values, default=defaults):
        normalized = item.lower()
        if normalized in all_tokens:
            selected.extend(defaults)
            continue
        selected.append(aliases.get(normalized, item))
    return unique_preserve_order(selected)
