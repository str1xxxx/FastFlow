from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath


def _normalize_pattern(pattern: str) -> str:
    normalized = pattern.strip().replace("\\", "/")
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _match_pattern(path: str, pattern: str) -> bool:
    path_obj = PurePosixPath(path)
    norm = _normalize_pattern(pattern)
    if not norm:
        return False
    # Support both repo-root anchored and recursive matching styles.
    return (
        path_obj.match(norm)
        or path_obj.match(f"**/{norm}")
        or (norm.endswith("/") and path.startswith(norm))
    )


@dataclass(slots=True)
class PathFilter:
    include_patterns: tuple[str, ...] = ()
    exclude_patterns: tuple[str, ...] = ()

    def matches(self, path: str) -> bool:
        if self.include_patterns and not any(
            _match_pattern(path, pattern) for pattern in self.include_patterns
        ):
            return False
        if any(_match_pattern(path, pattern) for pattern in self.exclude_patterns):
            return False
        return True


def build_path_filter(
    include_patterns: list[str] | tuple[str, ...] | None = None,
    exclude_patterns: list[str] | tuple[str, ...] | None = None,
) -> PathFilter:
    include = tuple(_normalize_pattern(pattern) for pattern in (include_patterns or []) if pattern)
    exclude = tuple(_normalize_pattern(pattern) for pattern in (exclude_patterns or []) if pattern)
    return PathFilter(include_patterns=include, exclude_patterns=exclude)

