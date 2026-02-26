from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class FileRecord:
    path: str
    sha256: str
    size: int
    mtime_ns: int

