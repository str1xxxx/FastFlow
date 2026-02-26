from __future__ import annotations

from pathlib import Path

import aiosqlite

from fastflow.models import FileRecord, RemoteSnapshotRecord


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS file_state (
    path TEXT PRIMARY KEY,
    sha256 TEXT NOT NULL,
    size INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL
);
"""

REMOTE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS remote_file_state (
    path TEXT PRIMARY KEY,
    sha256 TEXT,
    size INTEGER,
    oid TEXT,
    updated_at INTEGER NOT NULL
);
"""

META_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sync_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


async def ensure_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(db_path) as db:
        await db.execute(SCHEMA_SQL)
        await db.execute(REMOTE_SCHEMA_SQL)
        await db.execute(META_SCHEMA_SQL)
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


async def load_remote_records(db_path: Path) -> dict[str, RemoteSnapshotRecord]:
    await ensure_db(db_path)
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT path, sha256, size, oid FROM remote_file_state ORDER BY path"
        )
        rows = await cursor.fetchall()
        await cursor.close()

    return {
        str(row["path"]): RemoteSnapshotRecord(
            path=str(row["path"]),
            sha256=None if row["sha256"] is None else str(row["sha256"]),
            size=None if row["size"] is None else int(row["size"]),
            oid=None if row["oid"] is None else str(row["oid"]),
        )
        for row in rows
    }


async def replace_remote_snapshot(db_path: Path, records: list[RemoteSnapshotRecord]) -> None:
    await ensure_db(db_path)
    async with aiosqlite.connect(db_path) as db:
        await db.execute("DELETE FROM remote_file_state")
        if records:
            await db.executemany(
                """
                INSERT INTO remote_file_state (path, sha256, size, oid, updated_at)
                VALUES (?, ?, ?, ?, strftime('%s','now'))
                """,
                [(r.path, r.sha256, r.size, r.oid) for r in records],
            )
        await db.commit()


async def get_meta(db_path: Path, key: str) -> str | None:
    await ensure_db(db_path)
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            "SELECT value FROM sync_meta WHERE key = ?",
            (key,),
        )
        row = await cursor.fetchone()
        await cursor.close()
    if row is None:
        return None
    return str(row[0])


async def set_meta(db_path: Path, key: str, value: str) -> None:
    await ensure_db(db_path)
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            INSERT OR REPLACE INTO sync_meta (key, value) VALUES (?, ?)
            """,
            (key, value),
        )
        await db.commit()
