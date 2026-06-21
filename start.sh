#!/usr/bin/env bash
# Run the c4tts text-to-speech engine end to end.
#
#   ./start.sh [PROMPT_WAV] [TEXT] [OUT_WAV] [-- extra c4tts_cli flags...]
#
# Defaults: prompt=./output.wav, a bilingual sample text, out=./c4tts_out.wav.
# Auto-builds the binary and prepares the model weights on first run.
#
# Examples:
#   ./start.sh
#   ./start.sh myref.wav "你好，欢迎使用 c4tts。" out.wav
#   ./start.sh myref.wav "Hello there." out.wav -- --steps 25 --max-tokens 400
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
C4="$SCRIPT_DIR"
CLI="$C4/build/c4tts_cli"
W="$SCRIPT_DIR/bin"

# Plain text — the CLI wraps it in the model's prompt format (see --lang).
PROMPT="${1:-$SCRIPT_DIR/output.wav}"
TEXT="${2:-你好，欢迎使用 c4tts 语音合成引擎。}"
OUT="${3:-$SCRIPT_DIR/c4tts_out.wav}"
# Anything after a literal `--` is passed straight to c4tts_cli.
EXTRA=()
for ((i=1; i<=$#; i++)); do
  if [[ "${!i}" == "--" ]]; then EXTRA=("${@:i+1}"); break; fi
done

# Build if needed.
if [[ ! -x "$CLI" ]]; then
  echo ">> binary missing; building"
  "$SCRIPT_DIR/build.sh"
fi

# Prepare weights if needed.
if [[ ! -f "$W/t2s/semantic_head.weight.npy" || ! -f "$W/tokenizer/vocab.tsv" ]]; then
  echo ">> weights missing; preparing model files"
  "$SCRIPT_DIR/prepare.sh"
fi

if [[ ! -f "$PROMPT" ]]; then
  echo "error: prompt wav not found: $PROMPT" >&2
  echo "       pass a reference voice clip: ./start.sh ref.wav \"text\" out.wav" >&2
  exit 1
fi

echo ">> synthesizing -> $OUT"
"$CLI" synth --weights "$W" --prompt "$PROMPT" --text "$TEXT" --out "$OUT" "${EXTRA[@]}"
echo ">> wrote $OUT"
