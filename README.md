# c4tts

A standalone, Apple-Silicon-optimized **C++** inference engine for
[Confucius4-TTS](../README.md) — zero-shot voice-cloning TTS with no Python
runtime dependency at inference time. Built on [MLX](https://github.com/ml-explore/mlx)
(Metal under the hood); the philosophy follows
[antirez/ds4](https://github.com/antirez/ds4): narrow focus, golden-vector
parity against the PyTorch reference, fully self-contained.

Roadmap and design: [`../docs/PLAN.md`](../docs/PLAN.md).

## Pipeline

```
prompt.wav ─┬─ resample 16k → SeamlessM4T feats → W2V-BERT[17] → normalize → semantic features ─┐
            ├─ resample 16k → Kaldi fbank → CAMPPlus ───────────────────────→ style embedding ──┤
            └─ resample 22.05k → log-mel ───────────────────────────────────→ reference mel ────┤
text → (SentencePiece) → token ids → T2S (GPT-2 + speaker enc, AR) → semantic tokens + LM latent ┤
                                                                                                 ▼
                          S2A (length regulator → DiT flow-matching + CFG ODE) → mel → BigVGAN → wav
```

## Status — runs end-to-end on real weights

The full pipeline runs in C++/MLX on Metal: prompt wav + text token ids →
W2V-BERT/CAMPPlus/reference-mel → T2S (24-layer GPT-2 AR) → S2A (DiT flow
matching + CFG) → BigVGAN → speech waveform, using the real checkpoints, with
**no Python in the synth path** (Python is used only for one-time weight export
and tokenization). Every stage is parity-verified against PyTorch.

## Parity-verified modules

Every neural stage and DSP block is ported to C++/MLX and checked against the
PyTorch reference with golden vectors (25 test suites, all green). Headline
parity (max abs error / cosine):

| Stage | Module | Parity |
|---|---|---|
| Front end | STFT log-mel, Kaldi fbank, SeamlessM4T feats, resample, WAV IO | cosine ≈ 1.0 |
| Prompt enc | **W2V-BERT 2.0 conformer**, **CAMPPlus** (real weights) | 3.5e-5 / 2.6e-5 |
| T2S | **GPT-2 stack**, ECAPA speaker enc, full Text2Semantic logits, sampling | ≤1e-6, bit-exact processors |
| S2A | DiT (RoPE/AdaLN/SwiGLU/WaveNet) + **ConditionalCFM**, full inference (real weights) | 1.6e-5 |
| Vocoder | **BigVGAN** generator (real weights) | 3.3e-5 |
| Integration | prompt encoders (real), S2A→BigVGAN (real) | cosine 1.0 |

NN primitives (linear, conv1d/2d/transpose, norms, attention, embeddings,
activations) are individually parity-tested.

## Build

Requirements: macOS / Apple Silicon, CMake ≥ 3.24, recent Apple Clang, and the
`mlx` Python wheel installed (its bundled C++ lib + CMake config are reused).

```bash
cd c4tts
cmake -S . -B build -DC4TTS_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure      # run the parity suite
./build/c4tts_cli --version
```

## Run

1. Export model weights to a `WeightStore` (one-time; reads the HF cache):
   ```bash
   python3 tools/export_weights.py all     # w2vbert, campplus, s2a, bigvgan, t2s, audio
   ```
2. Tokenize text upstream (SentencePiece `tokenizer.model`) to integer ids, one
   per line in `ids.txt`. Then:
   ```bash
   ./build/c4tts_cli synth --weights weights \
       --prompt ref.wav --tokens ids.txt --out out.wav
   ```

## Layout

```
c4tts/
├── apps/        CLI (smoke test + `synth`)
├── include/     public headers (tensor, nn, weights, audio, t2s, s2a, vocoder, campplus, w2vbert, pipeline)
├── src/         engine implementation
├── tools/       Python: export_weights.py (weights -> WeightStore), dump_golden.py (parity vectors)
└── tests/       per-module + integration parity tests (CTest)
```

## Remaining work

The neural engine, audio DSP, and end-to-end pipeline are complete and verified
on the real checkpoints. To make the `synth` path 100% self-contained (text in,
no Python at all) two non-core pieces remain: a vendored C++ **SentencePiece**
tokenizer (text → ids; currently produced upstream and fed via file) and the
multilingual **text normalizer**. Performance is Phase H: KV-cache for the T2S
AR loop (currently recomputed per step), custom Metal kernels for hotspots,
weight quantization, and memory-mapped weight loading (today the engine loads
~3.5 GB of `.npy` per run).
```
