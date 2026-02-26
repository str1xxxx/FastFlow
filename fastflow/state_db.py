from __future__ import annotations

from pathlib import Path

import aiosqlite

from fastflow.models import FileRecord


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS file_state (
    path TEXT PRIMARY KEY,
    sha256 TEXT NOT NULL,
    size INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL
);
"""


async def ensure_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(db_path) as db:
        await db.execute(SCHEMA_SQL)
        await db.commit()


async def load_records(db_path: Path) -> dict[str, FileRecord]:
    await ensure_db(db_path)
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT path, sha256, size, mtime_ns FROM file_state ORDER BY path"
        )
        rows = await cursor.fetchall()
        await cursor.close()

    return {
        str(row["path"]): FileRecord(
            path=str(row["path"]),
            sha256=str(row["sha256"]),
            size=int(row["size"]),
            mtime_ns=int(row["mtime_ns"]),
        )
        for row in rows
    }


async def replace_snapshot(db_path: Path, records: list[FileRecord]) -> None:
    await ensure_db(db_path)
    async with aiosqlite.connect(db_path) as db:
        await db.execute("DELETE FROM file_state")
        if records:
            await db.executemany(
                """
                INSERT INTO file_state (path, sha256, size, mtime_ns)
                VALUES (?, ?, ?, ?)
                """,
                [(r.path, r.sha256, r.size, r.mtime_ns) for r in records],
            )
        await db.commit()

