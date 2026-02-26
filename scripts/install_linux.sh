#!/usr/bin/env bash
set -euo pipefail

# Installs FastFlow into an isolated CLI environment using pipx (recommended).
# Fallback to venv if pipx is not available.

REPO_URL="${1:-}"

if [[ -z "${REPO_URL}" ]]; then
  cat <<'EOF'
Usage:
  ./scripts/install_linux.sh <github_repo_url>

Example:
  ./scripts/install_linux.sh https://github.com/yourname/FastFlow.git
EOF
  exit 1
fi

if command -v pipx >/dev/null 2>&1; then
  pipx install "git+${REPO_URL}"
  echo
  echo "Installed with pipx. Run: ff --help"
  exit 0
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "python3 not found. Install Python 3.10+ and retry." >&2
  exit 1
fi

INSTALL_DIR="${HOME}/.local/share/fastflow-cli"
mkdir -p "${INSTALL_DIR}"

if [[ ! -d "${INSTALL_DIR}/.venv" ]]; then
  "${PYTHON_BIN}" -m venv "${INSTALL_DIR}/.venv"
fi

"${INSTALL_DIR}/.venv/bin/python" -m pip install --upgrade pip
"${INSTALL_DIR}/.venv/bin/python" -m pip install "git+${REPO_URL}"

cat <<EOF

Installed in: ${INSTALL_DIR}/.venv
Run:
  ${INSTALL_DIR}/.venv/bin/ff --help

Optional PATH:
  export PATH="\$PATH:${INSTALL_DIR}/.venv/bin"
EOF

