from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse


CONFIG_FILENAME = ".fastflow.json"
STATE_DB_FILENAME = ".ff_state.db"


@dataclass(slots=True)
class FastFlowConfig:
    repo_id: str
    token: str
    local_root: str

    @property
    def local_root_path(self) -> Path:
        return Path(self.local_root).resolve()

    @property
    def state_db_path(self) -> Path:
        return self.local_root_path / STATE_DB_FILENAME


def config_path(base_dir: Path | None = None) -> Path:
    return (base_dir or Path.cwd()).resolve() / CONFIG_FILENAME


def load_config(base_dir: Path | None = None) -> FastFlowConfig:
    path = config_path(base_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}. Run `ff init <repo_id>` first."
        )

    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    return FastFlowConfig(
        repo_id=normalize_repo_id(data["repo_id"]),
        token=data.get("token", ""),
        local_root=data["local_root"],
    )


def save_config(config: FastFlowConfig, base_dir: Path | None = None) -> Path:
    path = config_path(base_dir)
    payload = asdict(config)
    payload["repo_id"] = normalize_repo_id(str(payload["repo_id"]))
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    return path


def default_token() -> str:
    return os.getenv("HF_TOKEN", "")


def normalize_repo_id(repo_id: str) -> str:
    value = (repo_id or "").strip()
    if not value:
        return value

    # SSH forms used by Git-over-SSH (`git@huggingface.co:namespace/repo.git`)
    if value.startswith("git@huggingface.co:"):
        value = value.split(":", 1)[1].strip()
        if value.endswith(".git"):
            value = value[:-4]
        return value.strip("/")

    # SSH URL form (`ssh://git@huggingface.co/namespace/repo.git`)
    if value.startswith("ssh://"):
        parsed = urlparse(value)
        if parsed.hostname in {"huggingface.co", "www.huggingface.co", "hf.co", "www.hf.co"}:
            path = parsed.path.strip("/")
            if path.endswith(".git"):
                path = path[:-4]
            return _normalize_hf_path(path)
        return value

    if "://" not in value:
        return value.rstrip("/")

    parsed = urlparse(value)
    if parsed.hostname not in {"huggingface.co", "www.huggingface.co", "hf.co", "www.hf.co"}:
        return value

    path = parsed.path.strip("/")
    if path.endswith(".git"):
        path = path[:-4]
    return _normalize_hf_path(path)


def _normalize_hf_path(path: str) -> str:
    parts = [part for part in path.split("/") if part]
    if not parts:
        return path

    # Web URLs can be:
    # - /namespace/repo
    # - /datasets/namespace/repo
    # - /spaces/namespace/repo
    # - /models/namespace/repo
    if parts[0] in {"datasets", "spaces", "models"} and len(parts) >= 3:
        return f"{parts[1]}/{parts[2]}"
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return parts[0]
