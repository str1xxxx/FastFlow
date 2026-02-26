# FastFlow

FastFlow is a Python CLI for syncing large files (including binary assets like `.rpf`) with Hugging Face without Git/Git LFS.

It uses:
- `typer` for CLI
- `rich` for terminal UI/progress
- `sqlite` (`aiosqlite`) for local file state tracking
- `huggingface_hub` for upload/download/sync

## Features

- Local file state DB (`.ff_state.db`) with hashes/sizes/mtimes
- Selective sync with `--include` / `--exclude`
- `ff sync` bi-directional sync with conflict preference (`--prefer remote|local`)
- Parallel small-file transfers
- Large file uploads/downloads via `huggingface_hub` (resume/chunking support)

## Install (Linux)

### Option 1: `pipx` (recommended)

```bash
pipx install "git+https://github.com/str1xxxx/FastFlow.git"
ff --help
```

Update to the latest release:

```bash
pipx upgrade fastflow
```

### Option 2: local venv

```bash
git clone https://github.com/str1xxxx/FastFlow.git
cd FastFlow
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
ff --help
```

### Option 3: helper script

```bash
./scripts/install_linux.sh https://github.com/str1xxxx/FastFlow.git
```

## Authentication (important)

FastFlow uses the Hugging Face HTTP API (`huggingface_hub`), not Git-over-SSH.

This means:
- Hugging Face SSH keys are useful for `git clone/push` over SSH
- FastFlow still needs a Hugging Face access token for write operations

Recommended setup:

```bash
hf auth login
hf auth whoami
```

FastFlow will automatically reuse the locally saved token from `huggingface_hub`.

## Quick Start

```bash
mkdir -p ~/work/my-assets
cd ~/work/my-assets
ff init username/repo-name
ff status
ff sync
```

## Commands

- `ff init <repo_id>`
- `ff clone <repo_id> [dir]`
- `ff status`
- `ff pull`
- `ff push`
- `ff sync`

## Examples

```bash
ff init olymprp/trash
ff clone olymprp/trash my-assets
ff status
ff pull
ff push
ff sync --prefer remote
ff sync --prefer local
```

Selective sync:

```bash
ff status --include "*.rpf"
ff pull --include "dlcpacks/**" --exclude "*.tmp"
ff push --exclude ".cache/**"
ff sync --prefer local --include "mods/**"
```

## Notes

- `ff status` only shows changes and does not update the snapshot
- `ff pull`, `ff push`, and `ff sync` update `.ff_state.db` after successful transfers
- The current implementation uses Hugging Face `repo_type="model"`
- `.gitattributes` may appear during sync because Hugging Face can manage it server-side for large files

## Publish to GitHub

Before pushing:

```bash
git init
git add .
git commit -m "Initial FastFlow release"
git remote add origin https://github.com/str1xxxx/FastFlow.git
git push -u origin main
```
