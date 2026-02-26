#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <tag>"
  exit 1
fi

TAG="$1"

uv version "$TAG"
git add .

if git commit -m "version bump"; then
  git tag "$TAG"
  git push origin "$TAG"
else
  echo "Commit failed (possibly due to pre-commit hooks). Aborting."
  exit 1
fi