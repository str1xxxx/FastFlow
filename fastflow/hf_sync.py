from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, TypeVar

from rich.console import Console

from fastflow.auth import detect_ssh_public_keys, resolve_hf_token, ssh_only_auth_hint
from fastflow.config import FastFlowConfig, CONFIG_FILENAME, STATE_DB_FILENAME
from fastflow.filters import PathFilter, build_path_filter
from fastflow.models import FileRecord
from fastflow.scanner import scan_local_files
from fastflow.state_db import load_records, replace_snapshot
from fastflow.status_service import compute_status, diff_records
from fastflow.transfer_ui import ProgressFileReader, TransferProgressUI


HF_REPO_TYPE = "model"
REMOTE_EXCLUDED_PATHS = {CONFIG_FILENAME, STATE_DB_FILENAME}
LARGE_FILE_THRESHOLD_BYTES = 128 * 1024 * 1024
LARGE_EXTENSIONS = {".rpf"}
DEFAULT_SMALL_FILE_WORKERS = 6
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


@dataclass(slots=True)
class PushResult:
    uploaded_paths: list[str]
    deleted_remote_paths: list[str]
    skipped_paths: list[str]
    snapshot_count: int


@dataclass(slots=True)
class SyncResult:
    downloaded_paths: list[str]
    uploaded_paths: list[str]
    deleted_local_paths: list[str]
    deleted_remote_paths: list[str]
    conflict_paths: list[str]
    skipped_paths: list[str]
    snapshot_count: int


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
        ui.start(handle, state="downloading")
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


def _delete_local_file(local_root: Path, relative_path: str) -> bool:
    path = _local_file_path(local_root, relative_path)
    if not path.exists() or not path.is_file():
        return False
    path.unlink()
    current = path.parent
    while current != local_root:
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent
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
                ui.start(handle, state="uploading")
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
) -> list[str]:
    if not jobs:
        return []
    downloaded: list[str] = []

    with _suppress_hf_progress_bars():
        with TransferProgressUI(console=console) as ui:
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
) -> list[str]:
    if not jobs:
        return []
    uploaded: list[str] = []

    with _suppress_hf_progress_bars():
        with TransferProgressUI(console=console) as ui:
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


def _remote_changed_since_snapshot(remote: RemoteFileRecord, snapshot: FileRecord | None) -> bool:
    if snapshot is None:
        return True
    if remote.sha256:
        if remote.sha256 != snapshot.sha256:
            return True
        if remote.size is not None and remote.size != snapshot.size:
            return True
        return False
    if remote.size is not None and remote.size != snapshot.size:
        return True
    return False


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


async def _refresh_full_snapshot(
    config: FastFlowConfig,
    *,
    previous_records: dict[str, FileRecord],
) -> int:
    current_records = scan_local_files(config.local_root_path, previous_records=previous_records)
    await replace_snapshot(config.state_db_path, current_records)
    return len(current_records)


async def pull_from_hf(
    config: FastFlowConfig,
    *,
    include_patterns: tuple[str, ...] = (),
    exclude_patterns: tuple[str, ...] = (),
    console: Console | None = None,
    small_file_workers: int = DEFAULT_SMALL_FILE_WORKERS,
) -> PullResult:
    local_root = config.local_root_path
    local_root.mkdir(parents=True, exist_ok=True)
    path_filter = build_path_filter(include_patterns, exclude_patterns)

    previous_records = await load_records(config.state_db_path)
    remote_files = _remote_manifest(config, path_filter=path_filter)
    remote_map = {item.path: item for item in remote_files}

    download_jobs = [
        _DownloadJob(remote=item)
        for item in remote_files
        if _should_download(item, previous_records, local_root)
    ]
    skipped_paths = sorted(
        item.path for item in remote_files if item.path not in {job.remote.path for job in download_jobs}
    )
    deleted_local_paths: list[str] = []

    downloaded_paths = _run_download_jobs(
        config,
        download_jobs,
        console=console,
        small_file_workers=small_file_workers,
    )

    # Delete only within the filtered scope.
    for path in _paths_with_filter(set(previous_records) - set(remote_map), path_filter):
        if _delete_local_file(local_root, path):
            deleted_local_paths.append(path)

    snapshot_count = await _refresh_full_snapshot(config, previous_records=previous_records)

    return PullResult(
        remote_file_count=len(remote_files),
        downloaded_paths=downloaded_paths,
        skipped_paths=skipped_paths,
        deleted_local_paths=sorted(deleted_local_paths),
        snapshot_count=snapshot_count,
    )


async def push_to_hf(
    config: FastFlowConfig,
    *,
    include_patterns: tuple[str, ...] = (),
    exclude_patterns: tuple[str, ...] = (),
    console: Console | None = None,
    small_file_workers: int = DEFAULT_SMALL_FILE_WORKERS,
) -> PushResult:
    _require_push_token(config)
    local_root = config.local_root_path
    if not local_root.exists():
        raise FileNotFoundError(f"Configured local_root does not exist: {local_root}")

    path_filter = build_path_filter(include_patterns, exclude_patterns)
    previous_records = await load_records(config.state_db_path)
    previous_filtered = {
        path: record for path, record in previous_records.items() if path_filter.matches(path)
    }
    current_records = scan_local_files(
        local_root,
        previous_records=previous_records,
        path_filter=path_filter,
    )
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
    uploaded_paths = _run_upload_jobs(
        config,
        upload_jobs,
        console=console,
        small_file_workers=small_file_workers,
    )

    deleted_remote_paths: list[str] = []
    for record in status.deleted_files:
        if not _safe_to_delete_remote(config, record.path):
            continue
        if _delete_remote_one(config, record.path):
            deleted_remote_paths.append(record.path)

    snapshot_count = await _refresh_full_snapshot(config, previous_records=previous_records)

    return PushResult(
        uploaded_paths=uploaded_paths,
        deleted_remote_paths=sorted(deleted_remote_paths),
        skipped_paths=skipped_paths,
        snapshot_count=snapshot_count,
    )


async def sync_with_hf(
    config: FastFlowConfig,
    *,
    include_patterns: tuple[str, ...] = (),
    exclude_patterns: tuple[str, ...] = (),
    prefer_conflicts: Literal["remote", "local"] = "remote",
    console: Console | None = None,
    small_file_workers: int = DEFAULT_SMALL_FILE_WORKERS,
) -> SyncResult:
    if prefer_conflicts not in {"remote", "local"}:
        raise ValueError("prefer_conflicts must be 'remote' or 'local'")

    if prefer_conflicts == "local":
        _require_push_token(config)

    local_root = config.local_root_path
    local_root.mkdir(parents=True, exist_ok=True)
    path_filter = build_path_filter(include_patterns, exclude_patterns)

    previous_records = await load_records(config.state_db_path)
    previous_filtered = {
        path: record for path, record in previous_records.items() if path_filter.matches(path)
    }
    current_records = scan_local_files(
        local_root,
        previous_records=previous_records,
        path_filter=path_filter,
    )
    local_status = diff_records(previous_filtered, current_records)
    local_map = {record.path: record for record in current_records}

    remote_files = _remote_manifest(config, path_filter=path_filter)
    remote_map = {record.path: record for record in remote_files}

    local_upserts = {record.path: record for record in [*local_status.new_files, *local_status.modified_files]}
    local_deletes = {record.path for record in local_status.deleted_files}

    remote_upserts: dict[str, RemoteFileRecord] = {}
    for remote in remote_files:
        if _remote_changed_since_snapshot(remote, previous_records.get(remote.path)):
            remote_upserts[remote.path] = remote
    remote_deletes = {
        path for path in previous_records if path_filter.matches(path) and path not in remote_map
    }

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
                download_jobs_map[path] = _DownloadJob(remote=remote_upserts[path])
            elif remote_action == "delete":
                delete_local_paths.add(path)
            continue

        if remote_action == "none":
            if local_action == "upsert":
                record = local_upserts[path]
                upload_jobs_map[path] = _UploadJob(path=record.path, size=record.size)
            elif local_action == "delete":
                delete_remote_paths.add(path)
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
                download_jobs_map[path] = _DownloadJob(remote=remote_upserts[path])
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

    downloaded_paths = _run_download_jobs(
        config,
        sorted(download_jobs_map.values(), key=lambda item: item.remote.path),
        console=console,
        small_file_workers=small_file_workers,
    )

    uploaded_paths: list[str] = []
    if upload_jobs_map:
        _require_push_token(config)
        uploaded_paths = _run_upload_jobs(
            config,
            sorted(upload_jobs_map.values(), key=lambda item: item.path),
            console=console,
            small_file_workers=small_file_workers,
        )

    deleted_local_done: list[str] = []
    for path in sorted(delete_local_paths):
        if _delete_local_file(local_root, path):
            deleted_local_done.append(path)

    deleted_remote_done: list[str] = []
    if delete_remote_paths:
        _require_push_token(config)
    for path in sorted(delete_remote_paths):
        if not _safe_to_delete_remote(config, path):
            skipped_paths.add(path)
            continue
        if _delete_remote_one(config, path):
            deleted_remote_done.append(path)

    snapshot_count = await _refresh_full_snapshot(config, previous_records=previous_records)
    skipped_paths.update(
        path
        for path in remote_map
        if path_filter.matches(path)
        and path not in set(downloaded_paths)
        and path not in set(uploaded_paths)
        and path not in delete_local_paths
        and path not in delete_remote_paths
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
    )
