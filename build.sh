#!/usr/bin/env bash
# Build the c4tts C++/MLX engine (Release). Set C4TTS_BUILD_TESTS=ON to also
# build the parity test suite. Requires: macOS/Apple Silicon, CMake >= 3.24,
# Apple Clang, and the `mlx` Python wheel installed (its bundled C++ lib is reused).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
C4="$SCRIPT_DIR"
BUILD_TESTS="${C4TTS_BUILD_TESTS:-OFF}"

echo ">> configuring c4tts (tests=$BUILD_TESTS)"
cmake -S "$C4" -B "$C4/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DC4TTS_BUILD_TESTS="$BUILD_TESTS"

echo ">> building"
cmake --build "$C4/build" -j

echo ">> done: $C4/build/c4tts_cli"
"$C4/build/c4tts_cli" --version
