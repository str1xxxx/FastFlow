from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, TypeVar

from rich.console import Console

from fastflow.auth import detect_ssh_public_keys, resolve_hf_token, ssh_only_auth_hint
from fastflow.config import FastFlowConfig, CONFIG_FILENAME, STATE_DB_FILENAME
from fastflow.filters import PathFilter, build_path_filter
from fastflow.models import FileRecord, RemoteSnapshotRecord
from fastflow.scanner import scan_local_files, scan_local_files_with_progress, scan_local_paths
from fastflow.state_db import (
    get_meta,
    load_records,
    load_remote_records,
    replace_remote_snapshot,
    replace_snapshot,
    set_meta,
)
from fastflow.status_service import compute_status, diff_records
from fastflow.transfer_ui import ProgressFileReader, TransferProgressUI


HF_REPO_TYPE = "model"
REMOTE_EXCLUDED_PATHS = {CONFIG_FILENAME, STATE_DB_FILENAME}
LARGE_FILE_THRESHOLD_BYTES = 128 * 1024 * 1024
LARGE_EXTENSIONS = {".rpf"}
DEFAULT_SMALL_FILE_WORKERS = 6
FASTFLOW_TRASH_DIRNAME = ".fastflow_trash"
REMOTE_CACHE_INITIALIZED_META_KEY = "remote_cache_initialized"
REMOTE_CACHE_REPO_ID_META_KEY = "remote_cache_repo_id"
LOCAL_SNAPSHOT_REPO_ID_META_KEY = "local_snapshot_repo_id"
MAX_COMMIT_OPERATIONS = 500
BULK_DELETE_MIN_COUNT = 20
BULK_DELETE_MIN_RATIO = 0.25
BULK_DELETE_ALLOW_ENV = "FASTFLOW_ALLOW_BULK_DELETE"
T = TypeVar("T")


@dataclass(slots=True)
class SyncOptions:
    include_patterns: tuple[str, ...] = ()
    exclude_patterns: tuple[str, ...] = ()
    prefer_conflicts: Literal["remote", "local"] = "remote"
    small_file_workers: int = DEFAULT_SMALL_FILE_WORKERS

    @property
    def path_filter(self) -> PathFilter:
        return build_path_filter(self.include_patterns, self.exclude_patterns)


@dataclass(slots=True)
class RemoteFileRecord:
    path: str
    size: int | None
    sha256: str | None
    oid: str | None


@dataclass(slots=True)
class PullResult:
    remote_file_count: int
    downloaded_paths: list[str]
    skipped_paths: list[str]
    deleted_local_paths: list[str]
    snapshot_count: int
    remote_cache_bootstrapped: bool = False


@dataclass(slots=True)
class PushResult:
    uploaded_paths: list[str]
    deleted_remote_paths: list[str]
    skipped_paths: list[str]
    snapshot_count: int
    remote_cache_bootstrapped: bool = False


@dataclass(slots=True)
class SyncResult:
    downloaded_paths: list[str]
    uploaded_paths: list[str]
    deleted_local_paths: list[str]
    deleted_remote_paths: list[str]
    conflict_paths: list[str]
    skipped_paths: list[str]
    snapshot_count: int
    remote_cache_bootstrapped: bool = False


@dataclass(slots=True)
class _UploadJob:
    path: str
    size: int


@dataclass(slots=True)
class _DownloadJob:
    remote: RemoteFileRecord


def _load_hf_symbols():
    try:
        from huggingface_hub import HfApi, hf_hub_download  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for `ff pull`/`ff push`/`ff sync`. Install dependencies first."
        ) from exc
    return HfApi, hf_hub_download


def _load_hf_commit_symbols():
    try:
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for commit operations. Install dependencies first."
        ) from exc
    return HfApi, CommitOperationAdd, CommitOperationDelete


@contextmanager
def _suppress_hf_progress_bars():
    """Disable huggingface_hub/Xet progress bars while FastFlow renders its own Rich UI."""
    try:
        from huggingface_hub.utils import (  # type: ignore
            are_progress_bars_disabled,
            disable_progress_bars,
            enable_progress_bars,
        )
    except Exception:
        yield
        return

    was_disabled = bool(are_progress_bars_disabled())
    if not was_disabled:
        try:
            disable_progress_bars()
        except Exception:
            pass

    try:
        yield
    finally:
        if not was_disabled:
            try:
                enable_progress_bars()
            except Exception:
                pass


def _token_for_hf(config: FastFlowConfig) -> str | None:
    # `resolve_hf_token` also checks `hf auth login` cache and common token file locations.
    return resolve_hf_token(config.token)


def _iter_exception_chain(exc: BaseException):
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _is_timeout_error(exc: BaseException) -> bool:
    timeout_names = {
        "TimeoutError",
        "TimeoutException",
        "ReadTimeout",
        "ConnectTimeout",
        "ReadTimeoutError",
    }
    for current in _iter_exception_chain(exc):
        if current.__class__.__name__ in timeout_names:
            return True
        message = str(current).lower()
        if "timed out" in message or "timeout" in message:
            return True
    return False


def _is_not_found_error(exc: BaseException) -> bool:
    for current in _iter_exception_chain(exc):
        message = str(current).lower()
        if "404" in message or "not found" in message:
            return True
    return False


def _retry_on_timeout(
    func: Callable[[], T],
    *,
    operation: str,
    max_attempts: int = 3,
    base_delay_seconds: float = 1.0,
) -> T:
    attempt = 1
    while True:
        try:
            return func()
        except Exception as exc:
            if attempt >= max_attempts or not _is_timeout_error(exc):
                raise
            sleep_seconds = base_delay_seconds * (2 ** (attempt - 1))
            time.sleep(sleep_seconds)
            attempt += 1


def _normalize_remote_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("/")


def _extract_sha256_from_lfs(lfs: Any) -> str | None:
    if lfs is None:
        return None
    if isinstance(lfs, dict):
        value = lfs.get("sha256")
        return str(value) if value else None
    value = getattr(lfs, "sha256", None)
    return str(value) if value else None


def _extract_remote_path(entry: Any) -> str | None:
    value = (
        getattr(entry, "rfilename", None)
        or getattr(entry, "path", None)
        or getattr(entry, "name", None)
    )
    if not value:
        return None
    path = _normalize_remote_path(str(value))
    if not path or path.endswith("/"):
        return None
    if path in REMOTE_EXCLUDED_PATHS:
        return None
    if path.startswith(".cache/huggingface/"):
        return None
    return path


def _remote_manifest(config: FastFlowConfig, *, path_filter: PathFilter | None = None) -> list[RemoteFileRecord]:
    HfApi, _ = _load_hf_symbols()
    token = _token_for_hf(config)
    api = HfApi(token=token)
    path_filter = path_filter or PathFilter()

    def _repo_info_call():
        return api.repo_info(
            repo_id=config.repo_id,
            repo_type=HF_REPO_TYPE,
            files_metadata=True,
            token=token,
        )

    try:
        info = _retry_on_timeout(_repo_info_call, operation="repo_info")
        siblings = getattr(info, "siblings", None) or []
    except TypeError:
        siblings = []

    if not siblings:
        def _list_files_call():
            return api.list_repo_files(
                repo_id=config.repo_id,
                repo_type=HF_REPO_TYPE,
                token=token,
            )

        paths = _retry_on_timeout(_list_files_call, operation="list_repo_files")
        result = [
            RemoteFileRecord(
                path=normalized,
                size=None,
                sha256=None,
                oid=None,
            )
            for path in paths
            for normalized in [_normalize_remote_path(path)]
            if normalized not in REMOTE_EXCLUDED_PATHS and path_filter.matches(normalized)
        ]
        result.sort(key=lambda item: item.path)
        return result

    manifest: list[RemoteFileRecord] = []
    for sibling in siblings:
        path = _extract_remote_path(sibling)
        if not path or not path_filter.matches(path):
            continue
        size_value = getattr(sibling, "size", None)
        oid_value = getattr(sibling, "blob_id", None) or getattr(sibling, "oid", None)
        manifest.append(
            RemoteFileRecord(
                path=path,
                size=int(size_value) if size_value is not None else None,
                sha256=_extract_sha256_from_lfs(getattr(sibling, "lfs", None)),
                oid=str(oid_value) if oid_value else None,
            )
        )

    manifest.sort(key=lambda item: item.path)
    return manifest


def _to_remote_snapshot_record(remote: RemoteFileRecord) -> RemoteSnapshotRecord:
    return RemoteSnapshotRecord(
        path=remote.path,
        sha256=remote.sha256,
        size=remote.size,
        oid=remote.oid,
    )


def _merge_remote_snapshot_scope(
    previous_remote: dict[str, RemoteSnapshotRecord],
    *,
    path_filter: PathFilter,
    scope_remote_files: list[RemoteFileRecord],
) -> dict[str, RemoteSnapshotRecord]:
    merged = {path: record for path, record in previous_remote.items() if not path_filter.matches(path)}
    for remote in scope_remote_files:
        merged[remote.path] = _to_remote_snapshot_record(remote)
    return merged


async def _is_remote_cache_initialized(config: FastFlowConfig) -> bool:
    initialized = (await get_meta(config.state_db_path, REMOTE_CACHE_INITIALIZED_META_KEY)) == "1"
    if not initialized:
        return False
    cached_repo_id = (await get_meta(config.state_db_path, REMOTE_CACHE_REPO_ID_META_KEY) or "").strip()
    return cached_repo_id == config.repo_id


async def _set_remote_cache_initialized(config: FastFlowConfig, initialized: bool = True) -> None:
    await set_meta(config.state_db_path, REMOTE_CACHE_INITIALIZED_META_KEY, "1" if initialized else "0")
    await set_meta(
        config.state_db_path,
        REMOTE_CACHE_REPO_ID_META_KEY,
        config.repo_id if initialized else "",
    )


async def _is_local_snapshot_bound_to_repo(config: FastFlowConfig) -> bool:
    snapshot_repo_id = (await get_meta(config.state_db_path, LOCAL_SNAPSHOT_REPO_ID_META_KEY) or "").strip()
    return snapshot_repo_id == config.repo_id


async def _set_local_snapshot_repo_id(config: FastFlowConfig) -> None:
    await set_meta(config.state_db_path, LOCAL_SNAPSHOT_REPO_ID_META_KEY, config.repo_id)


async def _write_remote_snapshot_map(
    config: FastFlowConfig,
    *,
    snapshot_map: dict[str, RemoteSnapshotRecord],
    console: Console | None = None,
) -> None:
    with _phase_timer(console, "remote cache update"):
        await replace_remote_snapshot(
            config.state_db_path,
            sorted(snapshot_map.values(), key=lambda item: item.path),
        )
        await _set_remote_cache_initialized(config, True)


def _local_file_path(local_root: Path, relative_path: str) -> Path:
    return (local_root / Path(relative_path)).resolve()


def _is_large_transfer(path: str, size: int | None) -> bool:
    suffix = Path(path).suffix.lower()
    if suffix in LARGE_EXTENSIONS:
        return True
    return bool(size is not None and size >= LARGE_FILE_THRESHOLD_BYTES)


def _should_download(
    remote: RemoteFileRecord,
    previous_records: dict[str, FileRecord],
    local_root: Path,
) -> bool:
    local_path = _local_file_path(local_root, remote.path)
    old = previous_records.get(remote.path)

    if old is None:
        return True
    if not local_path.exists() or not local_path.is_file():
        return True

    stat = local_path.stat()
    if stat.st_size != old.size or stat.st_mtime_ns != old.mtime_ns:
        return True

    if remote.sha256:
        if old.sha256 != remote.sha256:
            return True
        if remote.size is not None and old.size != remote.size:
            return True
        return False

    if remote.size is not None and old.size != remote.size:
        return True
    return False


def _monitor_file_progress(
    *,
    path: Path,
    ui: TransferProgressUI,
    handle,
    stop_event: threading.Event,
) -> None:
    last_size = 0
    while not stop_event.wait(0.2):
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            continue
        if size > last_size:
            ui.set_completed(handle, size)
            last_size = size
    try:
        final_size = path.stat().st_size
    except FileNotFoundError:
        return
    ui.set_completed(handle, final_size)


def _download_one(
    config: FastFlowConfig,
    remote: RemoteFileRecord,
    *,
    ui: TransferProgressUI | None = None,
    handle=None,
) -> None:
    _, hf_hub_download = _load_hf_symbols()
    token = _token_for_hf(config)
    local_root = config.local_root_path
    target_path = _local_file_path(local_root, remote.path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    stop_event = threading.Event()
    monitor_thread: threading.Thread | None = None
    if ui is not None and handle is not None:
        download_state = "downloading"
        if _is_large_transfer(remote.path, remote.size):
            download_state = "downloading (hf-managed)"
        ui.start(handle, state=download_state)
        monitor_thread = threading.Thread(
            target=_monitor_file_progress,
            kwargs={"path": target_path, "ui": ui, "handle": handle, "stop_event": stop_event},
            daemon=True,
        )
        monitor_thread.start()

    def _call():
        return hf_hub_download(
            repo_id=config.repo_id,
            filename=remote.path,
            repo_type=HF_REPO_TYPE,
            token=token,
            local_dir=str(local_root),
        )

    try:
        _retry_on_timeout(_call, operation=f"download:{remote.path}")
        if ui is not None and handle is not None:
            final_size = target_path.stat().st_size if target_path.exists() else remote.size
            ui.complete(handle, total_bytes=final_size)
    except Exception:
        if ui is not None and handle is not None:
            ui.fail(handle)
        raise
    finally:
        stop_event.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=1.0)


def _cleanup_empty_parents(local_root: Path, path: Path) -> None:
    current = path.parent
    while current != local_root:
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


def _delete_local_file(local_root: Path, relative_path: str) -> bool:
    path = _local_file_path(local_root, relative_path)
    if not path.exists() or not path.is_file():
        return False
    path.unlink()
    _cleanup_empty_parents(local_root, path)
    return True


def _new_trash_session_dir(local_root: Path) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = local_root / FASTFLOW_TRASH_DIRNAME / ts
    candidate = base
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = local_root / FASTFLOW_TRASH_DIRNAME / f"{ts}-{suffix}"
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def _move_local_file_to_trash(
    local_root: Path,
    relative_path: str,
    *,
    trash_session_dir: Path,
) -> bool:
    path = _local_file_path(local_root, relative_path)
    if not path.exists() or not path.is_file():
        return False

    destination = (trash_session_dir / Path(relative_path)).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.replace(destination)
    except OSError:
        # Cross-device or platform-specific move edge cases: fallback to copy+delete.
        import shutil

        shutil.copy2(path, destination)
        path.unlink()

    _cleanup_empty_parents(local_root, path)
    return True


def _upload_one(
    config: FastFlowConfig,
    relative_path: str,
    *,
    ui: TransferProgressUI | None = None,
    handle=None,
) -> None:
    HfApi, _ = _load_hf_symbols()
    token = _token_for_hf(config)
    api = HfApi(token=token)
    local_path = _local_file_path(config.local_root_path, relative_path)
    local_size = local_path.stat().st_size
    use_buffer_wrapper = not _is_large_transfer(relative_path, local_size)

    def _call() -> Any:
        if ui is None or handle is None or not use_buffer_wrapper:
            if ui is not None and handle is not None:
                upload_state = "uploading"
                if not use_buffer_wrapper:
                    upload_state = "uploading (hf-managed)"
                ui.start(handle, state=upload_state)
            return api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=relative_path,
                repo_id=config.repo_id,
                repo_type=HF_REPO_TYPE,
                token=token,
                commit_message=f"FastFlow push: {relative_path}",
            )

        ui.start(handle, state="uploading")
        with local_path.open("rb") as raw:
            wrapped = ProgressFileReader(raw, progress=ui, handle=handle)
            result = api.upload_file(
                path_or_fileobj=wrapped,
                path_in_repo=relative_path,
                repo_id=config.repo_id,
                repo_type=HF_REPO_TYPE,
                token=token,
                commit_message=f"FastFlow push: {relative_path}",
            )
        ui.complete(handle, total_bytes=local_size)
        return result

    try:
        result = _retry_on_timeout(_call, operation=f"upload:{relative_path}")
        if ui is not None and handle is not None and not use_buffer_wrapper:
            ui.complete(handle, total_bytes=local_size)
        return result
    except Exception:
        if ui is not None and handle is not None:
            ui.fail(handle)
        raise


def _delete_remote_one(config: FastFlowConfig, relative_path: str) -> bool:
    HfApi, _ = _load_hf_symbols()
    token = _token_for_hf(config)
    api = HfApi(token=token)

    def _call():
        return api.delete_file(
            path_in_repo=relative_path,
            repo_id=config.repo_id,
            repo_type=HF_REPO_TYPE,
            token=token,
            commit_message=f"FastFlow delete: {relative_path}",
        )

    try:
        _retry_on_timeout(_call, operation=f"delete:{relative_path}")
    except Exception as exc:
        if _is_not_found_error(exc):
            return False
        raise
    return True


def _safe_to_delete_remote(config: FastFlowConfig, relative_path: str) -> bool:
    """Guard against accidental remote deletes if the local file actually still exists."""
    local_path = _local_file_path(config.local_root_path, relative_path)
    return not local_path.exists()


def _run_jobs(
    jobs: list[tuple[str, Callable[[], None]]],
    *,
    max_workers: int,
) -> None:
    if not jobs:
        return
    if max_workers <= 1 or len(jobs) == 1:
        for _, job in jobs:
            job()
        return

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="fastflow-xfer") as executor:
        futures: dict[Future[None], str] = {
            executor.submit(job): label for label, job in jobs
        }
        try:
            for future in as_completed(futures):
                future.result()
        except Exception:
            for future in futures:
                future.cancel()
            raise


def _run_download_jobs(
    config: FastFlowConfig,
    jobs: list[_DownloadJob],
    *,
    console: Console | None,
    small_file_workers: int,
    progress_transient: bool = False,
) -> list[str]:
    if not jobs:
        return []
    downloaded: list[str] = []

    with _suppress_hf_progress_bars():
        with TransferProgressUI(console=console, transient=progress_transient) as ui:
            handles = {
                job.remote.path: ui.add_transfer(
                    action="GET",
                    path=job.remote.path,
                    total_bytes=job.remote.size,
                )
                for job in jobs
            }

            large_jobs = [job for job in jobs if _is_large_transfer(job.remote.path, job.remote.size)]
            small_jobs = [job for job in jobs if job not in large_jobs]

            def make_job(job: _DownloadJob):
                return (
                    job.remote.path,
                    lambda j=job: _download_one(
                        config,
                        j.remote,
                        ui=ui,
                        handle=handles[j.remote.path],
                    ),
                )

            _run_jobs([make_job(job) for job in small_jobs], max_workers=max(1, small_file_workers))
            _run_jobs([make_job(job) for job in large_jobs], max_workers=1)

            downloaded = sorted(job.remote.path for job in jobs)

    return downloaded


def _run_upload_jobs(
    config: FastFlowConfig,
    jobs: list[_UploadJob],
    *,
    console: Console | None,
    small_file_workers: int,
    progress_transient: bool = False,
) -> list[str]:
    if not jobs:
        return []
    uploaded: list[str] = []

    with _suppress_hf_progress_bars():
        with TransferProgressUI(console=console, transient=progress_transient) as ui:
            handles = {
                job.path: ui.add_transfer(
                    action="PUT",
                    path=job.path,
                    total_bytes=job.size,
                )
                for job in jobs
            }

            large_jobs = [job for job in jobs if _is_large_transfer(job.path, job.size)]
            small_jobs = [job for job in jobs if job not in large_jobs]

            def make_job(job: _UploadJob):
                return (
                    job.path,
                    lambda j=job: _upload_one(
                        config,
                        j.path,
                        ui=ui,
                        handle=handles[j.path],
                    ),
                )

            _run_jobs([make_job(job) for job in small_jobs], max_workers=max(1, small_file_workers))
            _run_jobs([make_job(job) for job in large_jobs], max_workers=1)

            uploaded = sorted(job.path for job in jobs)

    return uploaded


def _chunk_list(items: list[T], chunk_size: int) -> list[list[T]]:
    if chunk_size <= 0:
        return [items]
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def _run_remote_commit_jobs(
    config: FastFlowConfig,
    *,
    add_jobs: list[_UploadJob],
    delete_paths: list[str],
    console: Console | None,
    max_operations_per_commit: int = MAX_COMMIT_OPERATIONS,
) -> tuple[list[str], list[str], list[str]]:
    if not add_jobs and not delete_paths:
        return [], [], []

    _require_push_token(config)
    HfApi, CommitOperationAdd, CommitOperationDelete = _load_hf_commit_symbols()
    token = _token_for_hf(config)
    api = HfApi(token=token)

    missing_local_paths: list[str] = []
    staged_operations: list[tuple[str, str, Any]] = []

    seen_add_paths: set[str] = set()
    for job in sorted(add_jobs, key=lambda item: item.path):
        if job.path in seen_add_paths:
            continue
        seen_add_paths.add(job.path)
        local_path = _local_file_path(config.local_root_path, job.path)
        if not local_path.exists() or not local_path.is_file():
            missing_local_paths.append(job.path)
            continue
        staged_operations.append(
            (
                "add",
                job.path,
                CommitOperationAdd(path_in_repo=job.path, path_or_fileobj=str(local_path)),
            )
        )

    for path in sorted(set(delete_paths)):
        if path in seen_add_paths:
            continue
        staged_operations.append(("delete", path, CommitOperationDelete(path_in_repo=path)))

    if not staged_operations:
        return [], [], missing_local_paths

    uploaded_paths: list[str] = []
    deleted_paths: list[str] = []
    operation_batches = _chunk_list(staged_operations, max_operations_per_commit)

    with _suppress_hf_progress_bars():
        for index, batch in enumerate(operation_batches, start=1):
            operations = [item[2] for item in batch]
            label = f"remote commit {index}/{len(operation_batches)}"
            if console is not None:
                console.print(f"[cyan]Phase:[/cyan] {label} ...")

            def _call():
                return api.create_commit(
                    repo_id=config.repo_id,
                    repo_type=HF_REPO_TYPE,
                    token=token,
                    operations=operations,
                    commit_message=f"FastFlow sync batch {index}/{len(operation_batches)}",
                )

            start = time.perf_counter()
            _retry_on_timeout(_call, operation=f"create_commit:{index}")
            elapsed = time.perf_counter() - start
            if console is not None:
                console.print(f"[dim]Phase complete:[/dim] {label} ({elapsed:.1f}s)")

            for op_type, path, _ in batch:
                if op_type == "add":
                    uploaded_paths.append(path)
                else:
                    deleted_paths.append(path)

    return sorted(uploaded_paths), sorted(deleted_paths), sorted(missing_local_paths)


def _remote_changed_since_snapshot(
    remote: RemoteFileRecord,
    snapshot: RemoteSnapshotRecord | None,
) -> bool:
    if snapshot is None:
        return True

    if remote.oid and snapshot.oid:
        if remote.oid != snapshot.oid:
            return True

    if remote.sha256 and snapshot.sha256:
        if remote.sha256 != snapshot.sha256:
            return True

    if remote.size is not None and snapshot.size is not None:
        if remote.size != snapshot.size:
            return True

    # If we don't have comparable metadata, stay conservative and treat as unchanged
    # to avoid destructive false positives on deletes/conflicts.
    return False


def _remote_unchanged_with_confidence(
    remote: RemoteFileRecord,
    snapshot: RemoteSnapshotRecord | None,
) -> bool:
    if snapshot is None:
        return False

    compared_any_field = False

    if remote.oid and snapshot.oid:
        compared_any_field = True
        if remote.oid != snapshot.oid:
            return False

    if remote.sha256 and snapshot.sha256:
        compared_any_field = True
        if remote.sha256 != snapshot.sha256:
            return False

    if remote.size is not None and snapshot.size is not None:
        compared_any_field = True
        if remote.size != snapshot.size:
            return False

    return compared_any_field


def _remote_matches_local(remote: RemoteFileRecord, local: FileRecord | None) -> bool:
    if local is None:
        return False
    if remote.sha256 and remote.sha256 != local.sha256:
        return False
    if remote.size is not None and remote.size != local.size:
        return False
    return True


def _require_push_token(config: FastFlowConfig) -> None:
    if not _token_for_hf(config):
        if detect_ssh_public_keys():
            raise RuntimeError(ssh_only_auth_hint())
        raise RuntimeError(
            "This command requires a Hugging Face token. Set `HF_TOKEN`, run `hf auth login`, "
            "or update `.fastflow.json`."
        )


def _paths_with_filter(paths: list[str] | set[str], path_filter: PathFilter) -> list[str]:
    return sorted(path for path in paths if path_filter.matches(path))


def _allow_bulk_delete() -> bool:
    value = os.getenv(BULK_DELETE_ALLOW_ENV, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _guard_bulk_delete_plan(
    *,
    action: str,
    delete_paths: list[str] | set[str],
    scope_count: int,
) -> None:
    delete_count = len(delete_paths)
    if delete_count == 0:
        return
    if _allow_bulk_delete():
        return

    ratio = (delete_count / scope_count) if scope_count > 0 else 1.0
    if delete_count >= BULK_DELETE_MIN_COUNT and ratio >= BULK_DELETE_MIN_RATIO:
        raise RuntimeError(
            f"Safety stop: refusing bulk {action} ({delete_count} path(s), {ratio:.0%} of scope). "
            f"If this is intentional, set {BULK_DELETE_ALLOW_ENV}=1 and run again."
        )


@contextmanager
def _phase_timer(console: Console | None, label: str):
    start = time.perf_counter()
    if console is not None:
        console.print(f"[cyan]Phase:[/cyan] {label} ...")
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if console is not None:
            console.print(f"[dim]Phase complete:[/dim] {label} ({elapsed:.1f}s)")


def _scan_local_records_for_operation(
    config: FastFlowConfig,
    *,
    previous_records: dict[str, FileRecord],
    path_filter: PathFilter,
    console: Console | None,
) -> list[FileRecord]:
    with _phase_timer(console, "local scan"):
        if console is not None:
            return scan_local_files_with_progress(
                config.local_root_path,
                previous_records=previous_records,
                path_filter=path_filter,
                console=console,
            )
        return scan_local_files(
            config.local_root_path,
            previous_records=previous_records,
            path_filter=path_filter,
        )


def _merge_snapshot_scope(
    previous_records: dict[str, FileRecord],
    *,
    path_filter: PathFilter,
    scope_records: list[FileRecord],
) -> dict[str, FileRecord]:
    merged = {path: record for path, record in previous_records.items() if not path_filter.matches(path)}
    for record in scope_records:
        merged[record.path] = record
    return merged


def _apply_snapshot_removals(snapshot_map: dict[str, FileRecord], paths: list[str] | set[str]) -> None:
    for path in paths:
        snapshot_map.pop(path, None)


def _refresh_snapshot_paths(
    config: FastFlowConfig,
    *,
    previous_records: dict[str, FileRecord],
    paths: list[str] | set[str],
    console: Console | None = None,
) -> dict[str, FileRecord]:
    target_paths = sorted(set(paths))
    if not target_paths:
        return {}

    with _phase_timer(console, f"snapshot rescan ({len(target_paths)} path(s))"):
        refreshed = scan_local_paths(
            config.local_root_path,
            target_paths,
            previous_records=previous_records,
        )
    return {record.path: record for record in refreshed}


async def _write_snapshot_map(
    config: FastFlowConfig,
    *,
    snapshot_map: dict[str, FileRecord],
    console: Console | None = None,
) -> int:
    with _phase_timer(console, "snapshot update"):
        records = sorted(snapshot_map.values(), key=lambda item: item.path)
        await replace_snapshot(config.state_db_path, records)
        await _set_local_snapshot_repo_id(config)
    return len(snapshot_map)


async def pull_from_hf(
    config: FastFlowConfig,
    *,
    include_patterns: tuple[str, ...] = (),
    exclude_patterns: tuple[str, ...] = (),
    console: Console | None = None,
    small_file_workers: int = DEFAULT_SMALL_FILE_WORKERS,
    progress_transient: bool = False,
) -> PullResult:
    local_root = config.local_root_path
    local_root.mkdir(parents=True, exist_ok=True)
    path_filter = build_path_filter(include_patterns, exclude_patterns)

    previous_records = await load_records(config.state_db_path)
    previous_remote_records = await load_remote_records(config.state_db_path)
    remote_cache_initialized = await _is_remote_cache_initialized(config)
    if not remote_cache_initialized:
        previous_remote_records = {}
    remote_cache_bootstrapped = not remote_cache_initialized
    with _phase_timer(console, "remote manifest"):
        remote_files = _remote_manifest(config, path_filter=path_filter)
    remote_map = {item.path: item for item in remote_files}

    with _phase_timer(console, "planning"):
        download_jobs = [
            _DownloadJob(remote=item)
            for item in remote_files
            if _should_download(item, previous_records, local_root)
        ]
        download_job_paths = {job.remote.path for job in download_jobs}
        skipped_paths = sorted(
            item.path for item in remote_files if item.path not in download_job_paths
        )
        if remote_cache_initialized:
            removed_scope_paths = _paths_with_filter(
                set(previous_remote_records) - set(remote_map),
                path_filter,
            )
        else:
            removed_scope_paths = []
        _guard_bulk_delete_plan(
            action="local delete from remote state",
            delete_paths=removed_scope_paths,
            scope_count=max(1, len(remote_map) + len(removed_scope_paths)),
        )
    deleted_local_paths: list[str] = []
    trash_session_dir = _new_trash_session_dir(local_root) if removed_scope_paths else None

    with _phase_timer(console, "downloads"):
        downloaded_paths = _run_download_jobs(
            config,
            download_jobs,
            console=console,
            small_file_workers=small_file_workers,
            progress_transient=progress_transient,
        )

    # Delete only within the filtered scope and only for known remote-deletes.
    with _phase_timer(console, "apply local deletes"):
        for path in removed_scope_paths:
            if trash_session_dir is not None and _move_local_file_to_trash(
                local_root,
                path,
                trash_session_dir=trash_session_dir,
            ):
                deleted_local_paths.append(path)

    snapshot_map = dict(previous_records)
    _apply_snapshot_removals(snapshot_map, removed_scope_paths)
    snapshot_map.update(
        _refresh_snapshot_paths(
            config,
            previous_records=previous_records,
            paths=downloaded_paths,
            console=console,
        )
    )
    snapshot_count = await _write_snapshot_map(config, snapshot_map=snapshot_map, console=console)
    remote_snapshot_map = _merge_remote_snapshot_scope(
        previous_remote_records,
        path_filter=path_filter,
        scope_remote_files=remote_files,
    )
    _apply_snapshot_removals(remote_snapshot_map, removed_scope_paths)
    await _write_remote_snapshot_map(config, snapshot_map=remote_snapshot_map, console=console)

    return PullResult(
        remote_file_count=len(remote_files),
        downloaded_paths=downloaded_paths,
        skipped_paths=skipped_paths,
        deleted_local_paths=sorted(deleted_local_paths),
        snapshot_count=snapshot_count,
        remote_cache_bootstrapped=remote_cache_bootstrapped,
    )


async def push_to_hf(
    config: FastFlowConfig,
    *,
    include_patterns: tuple[str, ...] = (),
    exclude_patterns: tuple[str, ...] = (),
    console: Console | None = None,
    small_file_workers: int = DEFAULT_SMALL_FILE_WORKERS,
    progress_transient: bool = False,
) -> PushResult:
    _require_push_token(config)
    local_root = config.local_root_path
    if not local_root.exists():
        raise FileNotFoundError(f"Configured local_root does not exist: {local_root}")

    path_filter = build_path_filter(include_patterns, exclude_patterns)
    previous_records = await load_records(config.state_db_path)
    local_snapshot_bound = await _is_local_snapshot_bound_to_repo(config)
    previous_remote_records = await load_remote_records(config.state_db_path)
    remote_cache_initialized = await _is_remote_cache_initialized(config)
    if not remote_cache_initialized:
        previous_remote_records = {}
    remote_cache_bootstrapped = not remote_cache_initialized
    with _phase_timer(console, "remote manifest"):
        remote_files_before_push = _remote_manifest(config, path_filter=path_filter)
    remote_paths_before_push = {item.path for item in remote_files_before_push}
    previous_filtered = {
        path: record for path, record in previous_records.items() if path_filter.matches(path)
    }
    current_records = _scan_local_records_for_operation(
        config,
        previous_records=previous_records,
        path_filter=path_filter,
        console=console,
    )
    with _phase_timer(console, "planning"):
        status = await compute_status(
            config.state_db_path,
            current_records,
            previous_records=previous_filtered,
        )

        upload_jobs = [
            _UploadJob(path=record.path, size=record.size)
            for record in sorted([*status.new_files, *status.modified_files], key=lambda item: item.path)
        ]
        upload_paths = {job.path for job in upload_jobs}
        skipped_paths = sorted(
            record.path for record in current_records if record.path not in upload_paths
        )
    skipped_paths_set = set(skipped_paths)
    delete_remote_candidates: list[str] = []
    with _phase_timer(console, "planning remote deletes"):
        deleted_candidates = status.deleted_files if local_snapshot_bound else []
        if not local_snapshot_bound and status.deleted_files and console is not None:
            console.print(
                "[yellow]Local snapshot repo binding is missing/mismatched; "
                "remote delete propagation is disabled for this run.[/yellow]"
            )
        for record in deleted_candidates:
            if not _safe_to_delete_remote(config, record.path):
                continue
            if record.path in remote_paths_before_push:
                delete_remote_candidates.append(record.path)
            else:
                skipped_paths_set.add(record.path)
    _guard_bulk_delete_plan(
        action="remote delete",
        delete_paths=delete_remote_candidates,
        scope_count=max(1, len(remote_paths_before_push)),
    )

    uploaded_paths: list[str] = []
    deleted_remote_paths: list[str] = []
    missing_upload_paths: list[str] = []
    if upload_jobs or delete_remote_candidates:
        uploaded_paths, deleted_remote_paths, missing_upload_paths = _run_remote_commit_jobs(
            config,
            add_jobs=upload_jobs,
            delete_paths=delete_remote_candidates,
            console=console,
        )
    skipped_paths_set.update(missing_upload_paths)

    snapshot_map = _merge_snapshot_scope(
        previous_records,
        path_filter=path_filter,
        scope_records=current_records,
    )
    snapshot_count = await _write_snapshot_map(config, snapshot_map=snapshot_map, console=console)
    with _phase_timer(console, "remote manifest (post-push)"):
        remote_files_after_push = _remote_manifest(config, path_filter=path_filter)
    remote_snapshot_map = _merge_remote_snapshot_scope(
        previous_remote_records,
        path_filter=path_filter,
        scope_remote_files=remote_files_after_push,
    )
    remote_paths_after_push = {item.path for item in remote_files_after_push}
    remote_removed_scope_paths = _paths_with_filter(
        set(previous_remote_records) - remote_paths_after_push,
        path_filter,
    )
    _apply_snapshot_removals(remote_snapshot_map, remote_removed_scope_paths)
    await _write_remote_snapshot_map(config, snapshot_map=remote_snapshot_map, console=console)

    return PushResult(
        uploaded_paths=uploaded_paths,
        deleted_remote_paths=sorted(deleted_remote_paths),
        skipped_paths=sorted(skipped_paths_set),
        snapshot_count=snapshot_count,
        remote_cache_bootstrapped=remote_cache_bootstrapped,
    )


async def sync_with_hf(
    config: FastFlowConfig,
    *,
    include_patterns: tuple[str, ...] = (),
    exclude_patterns: tuple[str, ...] = (),
    prefer_conflicts: Literal["remote", "local"] = "remote",
    console: Console | None = None,
    small_file_workers: int = DEFAULT_SMALL_FILE_WORKERS,
    progress_transient: bool = False,
) -> SyncResult:
    if prefer_conflicts not in {"remote", "local"}:
        raise ValueError("prefer_conflicts must be 'remote' or 'local'")

    if prefer_conflicts == "local":
        _require_push_token(config)

    local_root = config.local_root_path
    local_root.mkdir(parents=True, exist_ok=True)
    can_push = bool(_token_for_hf(config))
    path_filter = build_path_filter(include_patterns, exclude_patterns)

    previous_records = await load_records(config.state_db_path)
    local_snapshot_bound = await _is_local_snapshot_bound_to_repo(config)
    previous_remote_records = await load_remote_records(config.state_db_path)
    remote_cache_initialized = await _is_remote_cache_initialized(config)
    if not remote_cache_initialized:
        previous_remote_records = {}
    remote_cache_bootstrapped = not remote_cache_initialized
    previous_filtered = {
        path: record for path, record in previous_records.items() if path_filter.matches(path)
    }
    current_records = _scan_local_records_for_operation(
        config,
        previous_records=previous_records,
        path_filter=path_filter,
        console=console,
    )
    local_status = diff_records(previous_filtered, current_records)
    local_map = {record.path: record for record in current_records}

    with _phase_timer(console, "remote manifest"):
        remote_files = _remote_manifest(config, path_filter=path_filter)
    remote_map = {record.path: record for record in remote_files}

    with _phase_timer(console, "planning"):
        local_upserts = {record.path: record for record in [*local_status.new_files, *local_status.modified_files]}
        local_deletes = (
            {record.path for record in local_status.deleted_files}
            if local_snapshot_bound
            else set()
        )
        if not local_snapshot_bound and local_status.deleted_files and console is not None:
            console.print(
                "[yellow]Local snapshot repo binding is missing/mismatched; "
                "local-delete conflict propagation is disabled for this run.[/yellow]"
            )

        remote_upserts: dict[str, RemoteFileRecord] = {}
        for remote in remote_files:
            if _remote_changed_since_snapshot(remote, previous_remote_records.get(remote.path)):
                remote_upserts[remote.path] = remote
        remote_deletes = (
            {
                path
                for path in previous_remote_records
                if path_filter.matches(path) and path not in remote_map
            }
            if remote_cache_initialized
            else set()
        )

        all_changed_paths = sorted(set(local_upserts) | local_deletes | set(remote_upserts) | remote_deletes)

        download_jobs_map: dict[str, _DownloadJob] = {}
        upload_jobs_map: dict[str, _UploadJob] = {}
        delete_local_paths: set[str] = set()
        delete_remote_paths: set[str] = set()
        conflict_paths: list[str] = []
        skipped_paths: set[str] = set()

        for path in all_changed_paths:
            local_action = "none"
            remote_action = "none"
            if path in local_upserts:
                local_action = "upsert"
            elif path in local_deletes:
                local_action = "delete"
            if path in remote_upserts:
                remote_action = "upsert"
            elif path in remote_deletes:
                remote_action = "delete"

            if local_action == "none" and remote_action == "none":
                continue

            if local_action == "none":
                if remote_action == "upsert":
                    remote_record = remote_upserts[path]
                    if _remote_matches_local(remote_record, local_map.get(path)):
                        skipped_paths.add(path)
                    else:
                        download_jobs_map[path] = _DownloadJob(remote=remote_record)
                elif remote_action == "delete":
                    delete_local_paths.add(path)
                continue

            if remote_action == "none":
                if local_action == "upsert":
                    record = local_upserts[path]
                    upload_jobs_map[path] = _UploadJob(path=record.path, size=record.size)
                elif local_action == "delete":
                    remote_existing = remote_map.get(path)
                    remote_snapshot = previous_remote_records.get(path)
                    can_propagate_delete = (
                        can_push
                        and remote_existing is not None
                        and _remote_unchanged_with_confidence(remote_existing, remote_snapshot)
                    )
                    if can_propagate_delete:
                        # Non-conflict case: local delete + unchanged remote => propagate delete.
                        delete_remote_paths.add(path)
                    else:
                        # If we cannot prove remote is unchanged (or cannot push),
                        # keep remote as source of truth.
                        if remote_existing is not None:
                            download_jobs_map[path] = _DownloadJob(remote=remote_existing)
                        else:
                            skipped_paths.add(path)
                continue

            # Both sides changed.
            if local_action == "delete" and remote_action == "delete":
                skipped_paths.add(path)
                continue

            if (
                local_action == "upsert"
                and remote_action == "upsert"
                and _remote_matches_local(remote_upserts[path], local_map.get(path))
            ):
                skipped_paths.add(path)
                continue

            conflict_paths.append(path)
            if prefer_conflicts == "remote":
                if remote_action == "upsert":
                    remote_record = remote_upserts[path]
                    if _remote_matches_local(remote_record, local_map.get(path)):
                        skipped_paths.add(path)
                        download_jobs_map.pop(path, None)
                    else:
                        download_jobs_map[path] = _DownloadJob(remote=remote_record)
                    upload_jobs_map.pop(path, None)
                    delete_remote_paths.discard(path)
                elif remote_action == "delete":
                    delete_local_paths.add(path)
                    upload_jobs_map.pop(path, None)
            else:
                _require_push_token(config)
                if local_action == "upsert":
                    record = local_upserts[path]
                    upload_jobs_map[path] = _UploadJob(path=record.path, size=record.size)
                    download_jobs_map.pop(path, None)
                    delete_local_paths.discard(path)
                elif local_action == "delete":
                    delete_remote_paths.add(path)
                    download_jobs_map.pop(path, None)
        _guard_bulk_delete_plan(
            action="local delete from remote state",
            delete_paths=delete_local_paths,
            scope_count=max(1, len(remote_map) + len(remote_deletes)),
        )

    with _phase_timer(console, "downloads"):
        downloaded_paths = _run_download_jobs(
            config,
            sorted(download_jobs_map.values(), key=lambda item: item.remote.path),
            console=console,
            small_file_workers=small_file_workers,
            progress_transient=progress_transient,
        )

    delete_remote_commit_paths: list[str] = []
    for path in sorted(delete_remote_paths):
        if not _safe_to_delete_remote(config, path):
            skipped_paths.add(path)
            continue
        if path not in remote_map:
            skipped_paths.add(path)
            continue
        delete_remote_commit_paths.append(path)
    _guard_bulk_delete_plan(
        action="remote delete",
        delete_paths=delete_remote_commit_paths,
        scope_count=max(1, len(remote_map)),
    )

    uploaded_paths: list[str] = []
    deleted_remote_done: list[str] = []
    missing_upload_paths: list[str] = []
    if upload_jobs_map or delete_remote_commit_paths:
        uploaded_paths, deleted_remote_done, missing_upload_paths = _run_remote_commit_jobs(
            config,
            add_jobs=sorted(upload_jobs_map.values(), key=lambda item: item.path),
            delete_paths=delete_remote_commit_paths,
            console=console,
        )
    skipped_paths.update(missing_upload_paths)

    deleted_local_done: list[str] = []
    trash_session_dir = _new_trash_session_dir(local_root) if delete_local_paths else None
    with _phase_timer(console, "local deletes"):
        for path in sorted(delete_local_paths):
            if trash_session_dir is not None and _move_local_file_to_trash(
                local_root,
                path,
                trash_session_dir=trash_session_dir,
            ):
                deleted_local_done.append(path)

    snapshot_map = _merge_snapshot_scope(
        previous_records,
        path_filter=path_filter,
        scope_records=current_records,
    )
    _apply_snapshot_removals(snapshot_map, deleted_local_done)
    snapshot_map.update(
        _refresh_snapshot_paths(
            config,
            previous_records=previous_records,
            paths=downloaded_paths,
            console=console,
        )
    )
    snapshot_count = await _write_snapshot_map(config, snapshot_map=snapshot_map, console=console)
    remote_mutated = bool(uploaded_paths or deleted_remote_done)
    if remote_mutated:
        with _phase_timer(console, "remote manifest (post-sync)"):
            final_remote_files = _remote_manifest(config, path_filter=path_filter)
    else:
        final_remote_files = remote_files
    final_remote_map = {item.path: item for item in final_remote_files}
    remote_snapshot_map = _merge_remote_snapshot_scope(
        previous_remote_records,
        path_filter=path_filter,
        scope_remote_files=final_remote_files,
    )
    remote_removed_scope_paths = _paths_with_filter(
        set(previous_remote_records) - set(final_remote_map),
        path_filter,
    )
    _apply_snapshot_removals(remote_snapshot_map, remote_removed_scope_paths)
    await _write_remote_snapshot_map(config, snapshot_map=remote_snapshot_map, console=console)
    downloaded_set = set(downloaded_paths)
    uploaded_set = set(uploaded_paths)
    deleted_local_done_set = set(deleted_local_done)
    deleted_remote_done_set = set(deleted_remote_done)
    skipped_paths.update(
        path
        for path in remote_map
        if path_filter.matches(path)
        and path not in downloaded_set
        and path not in uploaded_set
        and path not in deleted_local_done_set
        and path not in deleted_remote_done_set
        and path not in conflict_paths
    )

    return SyncResult(
        downloaded_paths=sorted(downloaded_paths),
        uploaded_paths=sorted(uploaded_paths),
        deleted_local_paths=sorted(deleted_local_done),
        deleted_remote_paths=sorted(deleted_remote_done),
        conflict_paths=sorted(conflict_paths),
        skipped_paths=sorted(skipped_paths),
        snapshot_count=snapshot_count,
        remote_cache_bootstrapped=remote_cache_bootstrapped,
    )
