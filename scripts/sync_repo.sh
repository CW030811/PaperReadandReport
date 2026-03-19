#!/usr/bin/env bash
set -euo pipefail

REMOTE="${1:-origin}"
MESSAGE="${2:-}"

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

git add -A

if ! git diff --cached --quiet; then
  if [[ -z "${MESSAGE}" ]]; then
    MESSAGE="sync: $(date '+%Y-%m-%d %H:%M:%S')"
  fi
  git commit -m "${MESSAGE}"
else
  echo "No local file changes to commit."
fi

git pull --rebase "${REMOTE}" "${BRANCH}"
git push "${REMOTE}" "${BRANCH}"

echo "Synced '${BRANCH}' with '${REMOTE}'."
