---
license: apache-2.0
language:
  - zh
  - en
  - ja
  - ko
  - de
  - fr
  - es
  - id
  - it
  - th
  - pt
  - ru
  - ms
  - vi
library_name: c4tts
pipeline_tag: text-to-speech
base_model: netease-youdao/Confucius4-TTS
tags:
  - text-to-speech
  - tts
  - voice-cloning
  - zero-shot
  - multilingual
  - cross-lingual
  - apple-silicon
  - mlx
  - metal
  - confucius4-tts
---

# c4tts weights — Confucius4-TTS for Apple Silicon (C++/MLX)

Runtime weights for **c4tts**, a standalone C++/MLX inference engine that runs
the [Confucius4-TTS](https://huggingface.co/netease-youdao/Confucius4-TTS)
multilingual, zero-shot, cross-lingual TTS model on Apple Silicon with **no
Python dependency at run time**.

This repository hosts the model converted into c4tts's on-disk format (a
directory of `.npy` tensors that MLX memory-maps, plus the SentencePiece vocab).
It is a **repackaging** of Confucius4-TTS and its component models — see
[Provenance & licensing](#provenance--licensing). It does **not** contain new
trained weights.

- **Architecture:** speech-encoder + LLM (text→semantic, T2S) → flow-matching
  diffusion (semantic→mel, S2A) → BigVGAN vocoder (mel→waveform)
- **Output:** 22.05 kHz mono waveform
- **Languages:** 14 (zh, en, ja, ko, de, fr, es, id, it, th, pt, ru, ms, vi)
- **Capabilities:** zero-shot voice cloning from a short reference clip (no
  transcript needed), cross-lingual voice transfer
- **Precision:** weights shipped as float32; the engine can run the T2S decoder
  in int8 (`quantized_matmul`) at run time for ~2× speed at ~0.9999 logit cosine
- **Footprint:** ~4.8 GB (only the tensors the runtime actually loads — e.g.
  W2V-BERT is trimmed to the layers used for semantic-feature extraction)

## Intended use

Local, offline, multilingual TTS and zero-shot voice cloning on macOS / Apple
Silicon, via the c4tts engine — e.g. a CLI (`c4tts_cli synth`) or the built-in
HTTP server with an OpenAI-style `/v1/audio/speech` API and a web console.

**Not** intended for: real-time deepfakes, impersonation without consent, or any
use prohibited by the upstream licenses.

## How to use

These are weights for the **c4tts** engine (a C++/MLX runtime), not a
🤗 Transformers checkpoint — they won't load with `AutoModel`.

1. Build the c4tts engine for Apple Silicon (CMake + MLX).
2. Download this repo into the engine's `bin/` directory:
   ```bash
   huggingface-cli download <this-repo> --local-dir bin
   ```
   Layout: `bin/{t2s,s2a,w2vbert,campplus,bigvgan,audio}/*.npy` + `bin/tokenizer/vocab.tsv`.
3. Synthesize (text is tokenized in C++ — no Python at run time):
   ```bash
   # CLI: clone the voice in ref.wav and read the text
   ./c4tts_cli synth --prompt ref.wav --lang zh --text "你好，世界。" --out out.wav

   # or serve an OpenAI-style API + web console
   ./c4tts_cli --web --web-key <key>
   ```

Reference audio: a few seconds of clean mono speech is enough for zero-shot
cloning; pass `--lang` to select the synthesis language (cross-lingual transfer
is supported).

## Model components

| Component | Role | Source |
| --- | --- | --- |
| `t2s/` | Text→semantic LLM (GPT-2-style, 24 layers) + speaker encoder | Confucius4-TTS `t2s_model.safetensors` |
| `s2a/` | Semantic→mel flow-matching DiT + WaveNet | Confucius4-TTS `s2a_model.pt` |
| `tokenizer/vocab.tsv` | LLaMA SentencePiece BPE tokenizer | Confucius4-TTS |
| `w2vbert/` | W2V-BERT 2.0 semantic-feature encoder (layers used for extraction only) | `facebook/w2v-bert-2.0` |
| `campplus/` | CAMPPlus speaker embedding | `funasr/campplus` (`campplus_cn_common.bin`) |
| `bigvgan/` | BigVGAN v2 neural vocoder (22 kHz, 80-band, 256×) | `nvidia/bigvgan_v2_22khz_80band_256x` |
| `audio/` | Baked mel-filterbank + Hann window constants | derived |

## Provenance & licensing

This is a **derivative / repackaged** distribution. You must comply with the
license of **every** upstream component:

| Upstream | License | Link |
| --- | --- | --- |
| Confucius4-TTS (T2S, S2A, tokenizer) | Apache-2.0 | https://huggingface.co/netease-youdao/Confucius4-TTS |
| W2V-BERT 2.0 (semantic encoder) | MIT | https://huggingface.co/facebook/w2v-bert-2.0 |
| BigVGAN v2 (vocoder) | MIT | https://huggingface.co/nvidia/bigvgan_v2_22khz_80band_256x |
| CAMPPlus (speaker embedding) | see source | https://huggingface.co/funasr/campplus |

The `license: apache-2.0` tag reflects the primary model (Confucius4-TTS).
Verify the CAMPPlus license for your use case before redistribution.

## Limitations & responsible use

- Quality varies with reference-audio quality and is best on the 14 supported
  languages; out-of-distribution text/voices may degrade.
- **Voice cloning is powerful and easily misused.** Only clone voices you have
  explicit consent to use. Do not use this model to impersonate real people, to
  deceive, or to generate audio you are not authorized to produce. Consider
  disclosing AI-generated audio and watermarking outputs.
- Synthesis is non-deterministic under sampling; greedy decoding is reproducible.

## Citation

Please cite the upstream Confucius4-TTS work:

```bibtex
@misc{confucius4tts,
  title  = {Confucius4-TTS: a Multilingual and Cross-Lingual Zero-Shot TTS Engine},
  author = {NetEase Youdao},
  url    = {https://huggingface.co/netease-youdao/Confucius4-TTS}
}
```

*c4tts is an independent C++/MLX runtime; it is not affiliated with or endorsed
by NetEase Youdao, Meta, NVIDIA, or FunASR.*
