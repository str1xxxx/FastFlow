from __future__ import annotations

from dataclasses import dataclass

from fastflow.models import FileRecord
from fastflow.state_db import load_records, replace_snapshot


@dataclass(slots=True)
class StatusResult:
    new_files: list[FileRecord]
    modified_files: list[FileRecord]
    deleted_files: list[FileRecord]
    snapshot_count: int

    @property
    def has_changes(self) -> bool:
        return bool(self.new_files or self.modified_files or self.deleted_files)


def diff_records(
    previous: dict[str, FileRecord], current_records: list[FileRecord]
) -> StatusResult:
    current_map = {record.path: record for record in current_records}

    new_files: list[FileRecord] = []
    modified_files: list[FileRecord] = []
    deleted_files: list[FileRecord] = []

    for path, record in current_map.items():
        old = previous.get(path)
        if old is None:
            new_files.append(record)
            continue
        if (
            old.sha256 != record.sha256
            or old.size != record.size
            or old.mtime_ns != record.mtime_ns
        ):
            modified_files.append(record)

    for path, record in previous.items():
        if path not in current_map:
            deleted_files.append(record)

    return StatusResult(
        new_files=sorted(new_files, key=lambda r: r.path),
        modified_files=sorted(modified_files, key=lambda r: r.path),
        deleted_files=sorted(deleted_files, key=lambda r: r.path),
        snapshot_count=len(current_records),
    )


async def compute_status(
    db_path,
    current_records: list[FileRecord],
    *,
    previous_records: dict[str, FileRecord] | None = None,
) -> StatusResult:
    previous = previous_records if previous_records is not None else await load_records(db_path)
    return diff_records(previous, current_records)


async def refresh_snapshot(db_path, current_records: list[FileRecord]) -> None:
    await replace_snapshot(db_path, current_records)


async def compute_and_refresh_status(
    db_path, current_records: list[FileRecord]
) -> StatusResult:
    result = await compute_status(db_path, current_records)
    await refresh_snapshot(db_path, current_records)
    return result
