#!/usr/bin/env bash
# Prepare ALL runtime model files the engine depends on, exported into
# c4tts/weights/ as a WeightStore (directory of .npy):
#   w2vbert/  campplus/  s2a/  bigvgan/  t2s/  audio/  tokenizer/
#
# Sources (downloaded to the HuggingFace cache on first run):
#   - netease-youdao/Confucius4-TTS  (t2s_model.safetensors 2.6GB, s2a_model.pt,
#                                      tokenizer.model)
#   - facebook/w2v-bert-2.0, funasr/campplus, nvidia/bigvgan_v2_22khz_80band_256x
#
# The 2.6GB T2S checkpoint may instead be placed at ./downloads/ or ~/Downloads/.
# Use --force to re-export stores that already exist.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
C4="$SCRIPT_DIR"
W="$SCRIPT_DIR/bin"
PY="${PYTHON:-python3}"
FORCE=""
[[ "${1:-}" == "--force" ]] && FORCE=1

# Locate the T2S checkpoint (explicit local copies win over an HF download).
T2S_ARG=()
for cand in "$SCRIPT_DIR/downloads/t2s_model.safetensors" "$HOME/Downloads/t2s_model.safetensors"; do
  if [[ -f "$cand" ]]; then T2S_ARG=(--t2s-path "$cand"); echo ">> using T2S checkpoint: $cand"; break; fi
done

export_store() {  # name  sentinel_file
  local name="$1" sentinel="$2"
  if [[ -z "$FORCE" && -f "$W/$name/$sentinel" ]]; then
    echo ">> [$name] already present, skipping (use --force to redo)"
    return
  fi
  echo ">> exporting [$name]"
  if [[ "$name" == "t2s" ]]; then
    "$PY" "$C4/tools/export_weights.py" t2s "${T2S_ARG[@]}"
  else
    "$PY" "$C4/tools/export_weights.py" "$name"
  fi
}

export_store w2vbert  "encoder.layers.0.ffn1.intermediate_dense.weight.npy"
export_store campplus "xvector.tdnn.linear.weight.npy"
export_store s2a      "prompt_cond.npy"
export_store bigvgan  "conv_pre.weight.npy"
export_store t2s      "semantic_head.weight.npy"
export_store audio    "mel_basis.npy"

# Tokenizer vocab (separate exporter).
if [[ -n "$FORCE" || ! -f "$W/tokenizer/vocab.tsv" ]]; then
  echo ">> exporting [tokenizer]"
  "$PY" "$C4/tools/export_tokenizer.py"
else
  echo ">> [tokenizer] already present, skipping"
fi

echo ">> all runtime model files ready under $W"
