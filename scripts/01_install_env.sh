#!/usr/bin/env bash
set -euo pipefail

REQ_FILE="requirements.txt"
PIP_FLAGS=()

for arg in "$@"; do
  case "$arg" in
    --mvp-only)
      REQ_FILE="requirements-mvp.txt"
      ;;
    --user)
      PIP_FLAGS+=("--user")
      ;;
    *)
      echo "unknown option: $arg" >&2
      exit 2
      ;;
  esac
done

if [[ "${#PIP_FLAGS[@]}" -gt 0 ]]; then
  pip install "${PIP_FLAGS[@]}" -r "$REQ_FILE"
else
  pip install -r "$REQ_FILE"
fi
