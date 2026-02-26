from __future__ import annotations

import io
import threading
from dataclasses import dataclass
from typing import BinaryIO

from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)


@dataclass(slots=True)
class TransferTaskHandle:
    task_id: TaskID
    total: int | None
    path: str


class TransferProgressUI:
    def __init__(self, console: Console | None = None) -> None:
        self._console = console
        self._lock = threading.Lock()
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.fields[action]}"),
            TextColumn("{task.fields[path]}"),
            BarColumn(),
            DownloadColumn(binary_units=True),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            TextColumn("{task.fields[state]}"),
            console=console,
            transient=False,
            expand=True,
        )

    def __enter__(self) -> "TransferProgressUI":
        self._progress.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._progress.__exit__(exc_type, exc, tb)

    def add_transfer(self, *, action: str, path: str, total_bytes: int | None) -> TransferTaskHandle:
        with self._lock:
            task_id = self._progress.add_task(
                description=path,
                total=total_bytes,
                completed=0,
                start=False,
                action=action,
                path=path,
                state="queued",
            )
        return TransferTaskHandle(task_id=task_id, total=total_bytes, path=path)

    def start(self, handle: TransferTaskHandle, state: str = "running") -> None:
        with self._lock:
            self._progress.start_task(handle.task_id)
            self._progress.update(handle.task_id, state=state)

    def set_completed(self, handle: TransferTaskHandle, completed: int) -> None:
        with self._lock:
            if handle.total is not None:
                completed = min(completed, handle.total)
            self._progress.update(handle.task_id, completed=completed)

    def advance(self, handle: TransferTaskHandle, delta: int) -> None:
        with self._lock:
            self._progress.update(handle.task_id, advance=max(0, delta))

    def set_total(self, handle: TransferTaskHandle, total_bytes: int) -> None:
        with self._lock:
            self._progress.update(handle.task_id, total=total_bytes)

    def complete(self, handle: TransferTaskHandle, total_bytes: int | None = None) -> None:
        with self._lock:
            kwargs = {"state": "done"}
            if total_bytes is not None:
                kwargs["total"] = total_bytes
                kwargs["completed"] = total_bytes
            elif handle.total is not None:
                kwargs["completed"] = handle.total
            self._progress.update(handle.task_id, **kwargs)

    def fail(self, handle: TransferTaskHandle, message: str = "failed") -> None:
        with self._lock:
            self._progress.update(handle.task_id, state=message)


class ProgressFileReader(io.BufferedIOBase):
    """Binary reader wrapper that reports read progress to Rich."""

    def __init__(
        self,
        file_obj: BinaryIO,
        *,
        progress: TransferProgressUI,
        handle: TransferTaskHandle,
    ) -> None:
        self._file = file_obj
        self._progress = progress
        self._handle = handle
        self._reported = 0

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def seekable(self) -> bool:
        seekable = getattr(self._file, "seekable", None)
        return bool(seekable()) if callable(seekable) else True

    def read(self, size: int = -1):
        data = self._file.read(size)
        if data:
            new_reported = self._reported + len(data)
            if self._handle.total is not None:
                new_reported = min(new_reported, self._handle.total)
            delta = new_reported - self._reported
            self._reported = new_reported
            if delta:
                self._progress.advance(self._handle, delta)
        return data

    def readinto(self, b) -> int:
        data = self.read(len(b))
        if not data:
            return 0
        n = len(data)
        b[:n] = data
        return n

    def tell(self) -> int:
        return self._file.tell()

    def seek(self, offset: int, whence: int = 0) -> int:
        return self._file.seek(offset, whence)

    def fileno(self) -> int:
        return self._file.fileno()

    def close(self) -> None:
        self._file.close()
        super().close()

    @property
    def closed(self) -> bool:  # type: ignore[override]
        return self._file.closed

    def __getattr__(self, name: str):
        return getattr(self._file, name)

    def __enter__(self):
        self._file.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        return self._file.__exit__(exc_type, exc, tb)
