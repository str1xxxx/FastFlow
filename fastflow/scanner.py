from __future__ import annotations

import hashlib
from pathlib import Path

from fastflow.config import CONFIG_FILENAME, STATE_DB_FILENAME
from fastflow.filters import PathFilter
from fastflow.models import FileRecord


EXCLUDED_FILENAMES = {CONFIG_FILENAME, STATE_DB_FILENAME}
HF_LOCAL_CACHE_PREFIX = (".cache", "huggingface")


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def scan_local_files(
    root: Path,
    *,
    previous_records: dict[str, FileRecord] | None = None,
    path_filter: PathFilter | None = None,
) -> list[FileRecord]:
    root = root.resolve()
    records: list[FileRecord] = []
    previous_records = previous_records or {}
    path_filter = path_filter or PathFilter()

    for file_path in sorted(root.rglob("*")):
        if not file_path.is_file():
            continue
        rel_parts = file_path.relative_to(root).parts
        if file_path.name in EXCLUDED_FILENAMES:
            continue
        if rel_parts[:2] == HF_LOCAL_CACHE_PREFIX:
            continue

        stat = file_path.stat()
        relative_path = Path(*rel_parts).as_posix()
        if not path_filter.matches(relative_path):
            continue

        previous = previous_records.get(relative_path)
        if (
            previous is not None
            and previous.size == stat.st_size
            and previous.mtime_ns == stat.st_mtime_ns
        ):
            sha256 = previous.sha256
        else:
            sha256 = _sha256_file(file_path)
        records.append(
            FileRecord(
                path=relative_path,
                sha256=sha256,
                size=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
            )
        )

    return records
