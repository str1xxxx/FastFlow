from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class FileRecord:
    path: str
    sha256: str
    size: int
    mtime_ns: int


@dataclass(slots=True)
class RemoteSnapshotRecord:
    path: str
    sha256: str | None
    size: int | None
    oid: str | None
