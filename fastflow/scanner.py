from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

from fastflow.config import CONFIG_FILENAME, STATE_DB_FILENAME
from fastflow.filters import PathFilter
from fastflow.models import FileRecord

if TYPE_CHECKING:
    from rich.console import Console


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


def _discover_candidates(root: Path, path_filter: PathFilter) -> tuple[list[tuple[Path, str, int, int]], int]:
    candidates: list[tuple[Path, str, int, int]] = []
    total_bytes = 0

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

        candidates.append((file_path, relative_path, stat.st_size, stat.st_mtime_ns))
        total_bytes += stat.st_size

    return candidates, total_bytes


def _record_from_candidate(
    candidate: tuple[Path, str, int, int],
    previous_records: dict[str, FileRecord],
) -> FileRecord:
    file_path, relative_path, size, mtime_ns = candidate
    previous = previous_records.get(relative_path)
    if previous is not None and previous.size == size and previous.mtime_ns == mtime_ns:
        sha256 = previous.sha256
    else:
        sha256 = _sha256_file(file_path)

    return FileRecord(
        path=relative_path,
        sha256=sha256,
        size=size,
        mtime_ns=mtime_ns,
    )


def scan_local_files(
    root: Path,
    *,
    previous_records: dict[str, FileRecord] | None = None,
    path_filter: PathFilter | None = None,
) -> list[FileRecord]:
    root = root.resolve()
    previous_records = previous_records or {}
    path_filter = path_filter or PathFilter()
    candidates, _ = _discover_candidates(root, path_filter)
    return [_record_from_candidate(candidate, previous_records) for candidate in candidates]


def scan_local_files_with_progress(
    root: Path,
    *,
    previous_records: dict[str, FileRecord] | None = None,
    path_filter: PathFilter | None = None,
    console: "Console | None" = None,
) -> list[FileRecord]:
    from rich.progress import (
        BarColumn,
        DownloadColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeRemainingColumn,
        TransferSpeedColumn,
    )

    root = root.resolve()
    previous_records = previous_records or {}
    path_filter = path_filter or PathFilter()

    if console is not None:
        with console.status("Discovering files to scan..."):
            candidates, total_bytes = _discover_candidates(root, path_filter)
    else:
        candidates, total_bytes = _discover_candidates(root, path_filter)

    total_files = len(candidates)
    if total_files == 0:
        return []

    progress_total = total_bytes if total_bytes > 0 else total_files
    processed_bytes = 0
    records: list[FileRecord] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]Scanning"),
        BarColumn(),
        TaskProgressColumn(),
        DownloadColumn(binary_units=True),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        TextColumn("{task.fields[file_progress]}"),
        TextColumn("{task.fields[path]}"),
        console=console,
        transient=False,
        expand=True,
    ) as progress:
        task_id = progress.add_task(
            "scan",
            total=progress_total,
            file_progress=f"0/{total_files} files",
            path="",
        )

        for index, candidate in enumerate(candidates, start=1):
            file_path, relative_path, size, _ = candidate
            progress.update(task_id, file_progress=f"{index}/{total_files} files", path=relative_path)
            record = _record_from_candidate(candidate, previous_records)
            records.append(record)

            if total_bytes > 0:
                processed_bytes += size
                progress.update(task_id, completed=processed_bytes)
            else:
                progress.advance(task_id, 1)

    return records
