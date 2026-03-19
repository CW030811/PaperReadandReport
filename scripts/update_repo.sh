#!/usr/bin/env bash
set -euo pipefail

REMOTE="${1:-origin}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

if ! git remote get-url "${REMOTE}" >/dev/null 2>&1; then
  echo "Git remote '${REMOTE}' is not configured. Add it first with: git remote add ${REMOTE} <repo-url>" >&2
  exit 1
fi

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [[ "${BRANCH}" == "HEAD" ]]; then
  echo "Detached HEAD is not supported. Check out a branch first." >&2
  exit 1
fi

if ! git diff --quiet; then
  echo "Working tree has local changes. Commit or stash them before pulling." >&2
  exit 1
fi

git pull --rebase "${REMOTE}" "${BRANCH}"

echo "Updated '${BRANCH}' from '${REMOTE}'."
