from __future__ import annotations

import os
from pathlib import Path


SSH_PUBLIC_KEY_CANDIDATES = (
    "id_ed25519.pub",
    "id_ecdsa.pub",
    "id_rsa.pub",
    "id_dsa.pub",
)


def resolve_hf_token(config_token: str | None = None) -> str | None:
    """Resolve an HF token from env, config, or local huggingface_hub login cache."""
    for env_name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.getenv(env_name, "").strip()
        if value:
            return value

    if config_token and config_token.strip():
        return config_token.strip()

    # Prefer huggingface_hub's own token resolution so behavior matches the library.
    try:
        from huggingface_hub import get_token  # type: ignore

        value = get_token()
        if value:
            return str(value).strip() or None
    except Exception:
        pass

    # Best-effort fallback for older/newer installs or edge cases.
    for path in _token_file_candidates():
        try:
            if path.exists() and path.is_file():
                value = path.read_text(encoding="utf-8").strip()
                if value:
                    return value
        except OSError:
            continue

    return None


def _token_file_candidates() -> list[Path]:
    home = Path.home()
    hf_home = os.getenv("HF_HOME")
    candidates: list[Path] = []

    if hf_home:
        candidates.append(Path(hf_home) / "token")

    xdg_cache_home = os.getenv("XDG_CACHE_HOME")
    if xdg_cache_home:
        candidates.append(Path(xdg_cache_home) / "huggingface" / "token")

    xdg_config_home = os.getenv("XDG_CONFIG_HOME")
    if xdg_config_home:
        candidates.append(Path(xdg_config_home) / "huggingface" / "token")

    # Common defaults on Linux/macOS/Windows.
    candidates.extend(
        [
            home / ".cache" / "huggingface" / "token",
            home / ".config" / "huggingface" / "token",
            home / ".huggingface" / "token",  # legacy location
        ]
    )

    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path).lower()
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def detect_ssh_public_keys() -> list[Path]:
    """Detect common SSH public keys on Windows/Linux/macOS (best effort)."""
    home = Path.home()
    candidates = [home / ".ssh"]

    userprofile = os.getenv("USERPROFILE")
    if userprofile:
        candidates.append(Path(userprofile) / ".ssh")

    found: list[Path] = []
    seen: set[str] = set()
    for ssh_dir in candidates:
        if not ssh_dir.exists() or not ssh_dir.is_dir():
            continue

        for name in SSH_PUBLIC_KEY_CANDIDATES:
            path = ssh_dir / name
            key = str(path).lower()
            if path.exists() and path.is_file() and key not in seen:
                seen.add(key)
                found.append(path)

        # Include non-standard public keys too, but avoid noisy matches.
        try:
            for path in ssh_dir.glob("*.pub"):
                key = str(path).lower()
                if path.is_file() and key not in seen:
                    seen.add(key)
                    found.append(path)
        except OSError:
            continue

    found.sort(key=lambda p: str(p).lower())
    return found


def ssh_only_auth_hint() -> str:
    return (
        "Detected local SSH key(s), but FastFlow uses Hugging Face HTTP API (`huggingface_hub`) "
        "for transfers. SSH keys on Hugging Face are only used for Git-over-SSH and cannot "
        "authenticate HfApi/hf_hub_download. Use a Hugging Face access token (`HF_TOKEN`) or run "
        "`hf auth login` to save a local token."
    )

