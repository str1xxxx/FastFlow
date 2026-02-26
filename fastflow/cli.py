from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from fastflow.config import (
    FastFlowConfig,
    default_token,
    load_config,
    normalize_repo_id,
    save_config,
)
from fastflow.filters import build_path_filter
from fastflow.hf_sync import pull_from_hf, push_to_hf, sync_with_hf
from fastflow.scanner import scan_local_files
from fastflow.state_db import ensure_db, load_records
from fastflow.status_service import diff_records


app = typer.Typer(help="FastFlow CLI")
console = Console()


def _render_changes(title: str, records) -> None:
    if not records:
        return

    table = Table(title=title)
    table.add_column("Path")
    table.add_column("Size", justify="right")
    table.add_column("Modified (ns)", justify="right")
    table.add_column("SHA-256")

    for record in records:
        table.add_row(record.path, str(record.size), str(record.mtime_ns), record.sha256)

    console.print(table)


def _render_path_summary(title: str, paths: list[str], style: str) -> None:
    if not paths:
        return
    console.print(Text(f"{title} ({len(paths)}):", style=style))
    for path in paths:
        console.print(f"  {path}")


@app.command()
def init(repo_id: str) -> None:
    """Initialize FastFlow config in the current directory."""
    root = Path.cwd().resolve()
    normalized_repo_id = normalize_repo_id(repo_id)
    config = FastFlowConfig(
        repo_id=normalized_repo_id,
        token=default_token(),
        local_root=str(root),
    )
    config_path = save_config(config, root)
    asyncio.run(ensure_db(config.state_db_path))
    console.print(f"[green]Initialized FastFlow[/green] at {root}")
    console.print(f"Config: {config_path}")
    console.print(f"State DB: {config.state_db_path}")
    if normalized_repo_id != repo_id.strip():
        console.print(f"Repo ID normalized: {repo_id} -> {normalized_repo_id}")
    if not config.token:
        console.print(
            "[yellow]HF_TOKEN not found in environment. `token` was initialized as empty.[/yellow]"
        )


async def _status_async(include: tuple[str, ...], exclude: tuple[str, ...]) -> int:
    try:
        config = load_config()
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1

    local_root = config.local_root_path
    if not local_root.exists():
        console.print(f"[red]Configured local_root does not exist: {local_root}[/red]")
        return 1

    path_filter = build_path_filter(include, exclude)
    previous_records = await load_records(config.state_db_path)
    previous_filtered = {
        path: record for path, record in previous_records.items() if path_filter.matches(path)
    }

    console.print(f"Scanning [bold]{local_root}[/bold] ...")
    current_records = scan_local_files(
        local_root,
        previous_records=previous_records,
        path_filter=path_filter,
    )
    result = diff_records(previous_filtered, current_records)

    _render_changes("New", result.new_files)
    _render_changes("Modified", result.modified_files)
    _render_changes("Deleted", result.deleted_files)

    if not result.has_changes:
        console.print("[green]No changes detected.[/green]")

    console.print(
        f"Current scan: {result.snapshot_count} file(s). Snapshot baseline remains in {config.state_db_path}"
    )
    return 0


@app.command()
def status(
    include: list[str] | None = typer.Option(
        None,
        "--include",
        help="Include glob pattern(s) for paths to consider (repeatable).",
    ),
    exclude: list[str] | None = typer.Option(
        None,
        "--exclude",
        help="Exclude glob pattern(s) for paths to ignore (repeatable).",
    ),
) -> None:
    """Show local file changes compared to the last SQLite snapshot."""
    raise typer.Exit(code=asyncio.run(_status_async(tuple(include or ()), tuple(exclude or ()))))


async def _pull_async(include: tuple[str, ...], exclude: tuple[str, ...]) -> int:
    try:
        config = load_config()
        result = await pull_from_hf(
            config,
            include_patterns=include,
            exclude_patterns=exclude,
            console=console,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        console.print(f"[red]{exc}[/red]")
        return 1
    except Exception as exc:
        console.print(f"[red]Pull failed:[/red] {exc}")
        return 1

    _render_path_summary("Downloaded", result.downloaded_paths, "green")
    _render_path_summary("Deleted local (removed upstream)", result.deleted_local_paths, "yellow")
    if not result.downloaded_paths and not result.deleted_local_paths:
        console.print("[green]Local workspace already matches remote snapshot.[/green]")

    console.print(
        f"Remote files: {result.remote_file_count} | Skipped unchanged: {len(result.skipped_paths)}"
    )
    console.print(
        f"Snapshot updated: {result.snapshot_count} tracked file(s) in {config.state_db_path}"
    )
    return 0


@app.command()
def pull(
    include: list[str] | None = typer.Option(
        None,
        "--include",
        help="Include glob pattern(s) for paths to pull (repeatable).",
    ),
    exclude: list[str] | None = typer.Option(
        None,
        "--exclude",
        help="Exclude glob pattern(s) for paths to skip (repeatable).",
    ),
) -> None:
    """Download missing/changed files from Hugging Face and refresh the local snapshot."""
    raise typer.Exit(code=asyncio.run(_pull_async(tuple(include or ()), tuple(exclude or ()))))


async def _push_async(include: tuple[str, ...], exclude: tuple[str, ...]) -> int:
    try:
        config = load_config()
        result = await push_to_hf(
            config,
            include_patterns=include,
            exclude_patterns=exclude,
            console=console,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        console.print(f"[red]{exc}[/red]")
        return 1
    except Exception as exc:
        console.print(f"[red]Push failed:[/red] {exc}")
        return 1

    _render_path_summary("Uploaded", result.uploaded_paths, "green")
    _render_path_summary("Deleted remote", result.deleted_remote_paths, "yellow")

    if not result.uploaded_paths and not result.deleted_remote_paths:
        console.print("[green]No local changes to push.[/green]")

    console.print(f"Skipped unchanged: {len(result.skipped_paths)}")
    console.print(
        f"Snapshot updated: {result.snapshot_count} tracked file(s) in {config.state_db_path}"
    )
    return 0


@app.command()
def push(
    include: list[str] | None = typer.Option(
        None,
        "--include",
        help="Include glob pattern(s) for paths to push (repeatable).",
    ),
    exclude: list[str] | None = typer.Option(
        None,
        "--exclude",
        help="Exclude glob pattern(s) for paths to skip (repeatable).",
    ),
) -> None:
    """Upload local changes to Hugging Face and refresh the local snapshot."""
    raise typer.Exit(code=asyncio.run(_push_async(tuple(include or ()), tuple(exclude or ()))))


async def _sync_async(
    include: tuple[str, ...],
    exclude: tuple[str, ...],
    prefer: str,
) -> int:
    prefer_normalized = prefer.lower().strip()
    if prefer_normalized not in {"remote", "local"}:
        console.print("[red]Invalid --prefer value. Use 'remote' or 'local'.[/red]")
        return 1

    try:
        config = load_config()
        result = await sync_with_hf(
            config,
            include_patterns=include,
            exclude_patterns=exclude,
            prefer_conflicts=prefer_normalized,  # type: ignore[arg-type]
            console=console,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        return 1
    except Exception as exc:
        console.print(f"[red]Sync failed:[/red] {exc}")
        return 1

    _render_path_summary("Downloaded", result.downloaded_paths, "green")
    _render_path_summary("Uploaded", result.uploaded_paths, "green")
    _render_path_summary("Deleted local", result.deleted_local_paths, "yellow")
    _render_path_summary("Deleted remote", result.deleted_remote_paths, "yellow")
    _render_path_summary("Conflicts resolved", result.conflict_paths, "magenta")

    if (
        not result.downloaded_paths
        and not result.uploaded_paths
        and not result.deleted_local_paths
        and not result.deleted_remote_paths
    ):
        console.print("[green]Nothing to sync.[/green]")

    console.print(f"Skipped unchanged: {len(result.skipped_paths)}")
    console.print(
        f"Snapshot updated: {result.snapshot_count} tracked file(s) in {config.state_db_path}"
    )
    return 0


@app.command()
def sync(
    prefer: str = typer.Option(
        "remote",
        "--prefer",
        help="Conflict strategy when both local and remote changed the same path: remote or local.",
    ),
    include: list[str] | None = typer.Option(
        None,
        "--include",
        help="Include glob pattern(s) for paths to sync (repeatable).",
    ),
    exclude: list[str] | None = typer.Option(
        None,
        "--exclude",
        help="Exclude glob pattern(s) for paths to skip (repeatable).",
    ),
) -> None:
    """Bi-directional sync with simple conflict resolution."""
    raise typer.Exit(
        code=asyncio.run(_sync_async(tuple(include or ()), tuple(exclude or ()), prefer))
    )
