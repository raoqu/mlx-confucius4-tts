#!/usr/bin/env bash
# Clean build artifacts. With --all, also remove exported weights and golden
# vectors (regenerate them with prepare.sh).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
C4="$SCRIPT_DIR"

rm -rf "$C4/build"
echo ">> removed $C4/build"

if [[ "${1:-}" == "--all" ]]; then
  rm -rf "$SCRIPT_DIR/bin" "$C4/golden"
  echo ">> removed bin/ and c4tts/golden (run ./prepare.sh to rebuild)"
fi
