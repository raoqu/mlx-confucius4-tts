# c4tts

A standalone, Apple-Silicon-optimized **C++** inference engine for
[Confucius4-TTS](https://huggingface.co/netease-youdao/Confucius4-TTS) — zero-shot voice-cloning TTS with no Python
runtime dependency at inference time. Built on [MLX](https://github.com/ml-explore/mlx)
(Metal under the hood); the philosophy follows
[antirez/ds4](https://github.com/antirez/ds4): narrow focus, golden-vector
parity against the PyTorch reference, fully self-contained.

Roadmap and design: [`docs/PLAN.md`](docs/PLAN.md).

## Pipeline

```
prompt.wav ─┬─ resample 16k → SeamlessM4T feats → W2V-BERT[17] → normalize → semantic features ─┐
            ├─ resample 16k → Kaldi fbank → CAMPPlus ───────────────────────→ style embedding ──┤
            └─ resample 22.05k → log-mel ───────────────────────────────────→ reference mel ────┤
text → (SentencePiece) → token ids → T2S (GPT-2 + speaker enc, AR) → semantic tokens + LM latent ┤
                                                                                                 ▼
                          S2A (length regulator → DiT flow-matching + CFG ODE) → mel → BigVGAN → wav
```

## Status — text → speech, end to end, zero Python in the synth path

The full pipeline runs in C++/MLX on Metal: a **text string** + prompt wav →
SentencePiece tokenizer → W2V-BERT/CAMPPlus/reference-mel → T2S (24-layer GPT-2
AR) → S2A (DiT flow matching + CFG) → BigVGAN → speech waveform, using the real
checkpoints. **No Python runs at inference** (Python is used only for one-time
weight/vocab export). Every stage is parity-verified against PyTorch.

```bash
./build/c4tts_cli synth --prompt ref.wav --lang zh --text "你好世界" --out out.wav
```

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

## Quick start

Convenience scripts at the repository root wrap the whole flow:

```bash
./build.sh      # compile (Release) -> build/c4tts_cli
./prepare.sh    # export ALL runtime model files into bin/
./start.sh      # text -> speech end to end (auto-builds + prepares on first run)
./clean.sh      # remove build/  (./clean.sh --all also drops bin/ + golden/)
```

`prepare.sh` downloads the checkpoints to the HuggingFace cache and exports them
to `bin/` (w2vbert, campplus, s2a, bigvgan, t2s, audio, tokenizer).
The 2.6 GB T2S checkpoint may instead be placed at
`./downloads/t2s_model.safetensors` or `~/Downloads/t2s_model.safetensors`.

Command-line example:

```bash
# ./start.sh [PROMPT_WAV] [TEXT] [OUT_WAV] [-- extra c4tts_cli flags]
./start.sh output.wav "你好，欢迎使用 c4tts 语音合成引擎。" demo_out.wav -- --steps 25

# or the binary directly (English -> --lang en):
./build/c4tts_cli synth --prompt output.wav \
    --lang en --text "Hello, this is a voice cloning demo." --out demo_out.wav
```

`--prompt` is a reference voice clip (any sample rate) whose timbre is cloned;
`--text` is plain text to speak — the CLI wraps it in the model's trained prompt
format `"You are a helpful assistant. {lang}:{text}"` automatically (pass
`--lang zh|en|ja|...`, default `zh`; `--raw-text` to skip wrapping). Flags:
`--steps N`, `--max-tokens N`, `--greedy`, `--bench`.

## Build

Requirements: macOS / Apple Silicon, CMake ≥ 3.24, recent Apple Clang, and the
`mlx` Python wheel installed (its bundled C++ lib + CMake config are reused).

```bash
cmake -S . -B build -DC4TTS_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure      # run the parity suite
./build/c4tts_cli --version
```

## Run

1. Export all runtime model files to `bin/` (one-time; reads the HF cache):
   ```bash
   ./prepare.sh                              # or: python3 tools/export_weights.py all
                                             #     python3 tools/export_tokenizer.py
   ```
2. Synthesize. `--weights` defaults to `bin/` (resolved relative to the binary),
   and `--text` is tokenized in C++:
   ```bash
   ./build/c4tts_cli synth --prompt ref.wav --lang zh --text "你好世界" --out out.wav
   ```
   `--tokens ids.txt` (one id per line) is still accepted as an alternative to
   `--text` for externally tokenized input.

## Web server & API

c4tts ships an embedded HTTP server with a single-page web console and an
OpenAI-style speech API (the same frontend page and routes as the
index-tts2-metal runtime). The model is loaded once at startup and reused for
every request.

```bash
# Web console + API at http://127.0.0.1:3456/web (admin key required)
./build/c4tts_cli --web --web-key secret

# API only, no web console
./build/c4tts_cli --server --host 0.0.0.0 --port 3456
```

Flags: `--web` (serve the console), `--server` (API only), `--web-key KEY`
(admin key for the console and its `/web/api/*` routes; env `C4TTS_WEBKEY`),
`--host` (default `127.0.0.1`), `--port` (default `3456`), `--weights DIR`
(default `bin/`), `--voice-store DIR` (default `voices/`), `--lang` (default
`zh`), `--queue-size N`.

Performance: set `C4TTS_FP16=1` (env, honored by both `synth` and the server)
to run the T2S GPT-2 projection matmuls in float16. The fp32 residual stream,
norms, attention, and sampling are preserved, so the selected tokens are
unchanged (greedy decode is bit-stable) while decode bandwidth drops. Set
`C4TTS_TIMING=1` to print per-stage timings (prompt / t2s / s2a / bigvgan).

Voices are managed in SQLite (`<store>/voices.sqlite`) with reference audio
under `<store>/samples` and c4tts voice bundles (`.pt`) under `<store>/bundles`.
A c4tts bundle embeds the source reference WAV; the conditioning is derived from
it at synth time (the on-disk format differs from index-tts2 bundles).

Key endpoints (also reachable under `/web/api/*` with the web key):

| Method | Path | Purpose |
| ------ | ---- | ------- |
| POST   | `/v1/audio/speech` | synthesize: `{input, voice, response_format:"wav", lang?, steps?, max_tokens?}` → WAV |
| GET/POST | `/v1/audio/voices` | list / create (multipart `audio_sample`, or JSON `bundle_path`) |
| GET/PATCH/DELETE | `/v1/audio/voices/{id}` | fetch / edit / delete a voice |
| GET    | `/api/voices/{id}/source-audio` | the voice's reference WAV |
| GET    | `/status` · `/health` | queue + job metrics · liveness |

```bash
# Create a voice from reference audio, then synthesize
curl -s -H 'X-MTTS-Web-Key: secret' -F name=qin -F audio_sample=@ref.wav \
  http://127.0.0.1:3456/web/api/voices
curl -s -H 'X-MTTS-Web-Key: secret' -H 'Content-Type: application/json' \
  -d '{"input":"你好世界","voice":"qin","response_format":"wav"}' \
  http://127.0.0.1:3456/web/api/speech -o out.wav
```

## Layout

```
c4tts/
├── apps/        CLI (smoke test + `synth` + `--web`/`--server`)
├── include/     public headers (tensor, nn, weights, audio, t2s, s2a, vocoder, campplus, w2vbert, pipeline, server)
├── src/         engine implementation (src/server/ = HTTP server + embedded web UI)
├── tools/       Python: export_weights.py (weights -> WeightStore), dump_golden.py (parity vectors)
└── tests/       per-module + integration parity tests (CTest)
```

## Performance

Apple M-series, 22.05 kHz output. Measured with `C4TTS_TIMING=1` and `--bench`:

| | before | after |
|---|---|---|
| Cold one-shot synth | ~30 s | **~1.7 s** |
| Warm compute (RTF) | — | **~1.2× realtime** |

Optimizations (Phase H):
- **WeightStore cache** — parsed tensors are memoized, so per-layer/per-step
  `get()` calls don't re-read the multi-GB model from disk (the dominant cost;
  ~7× end-to-end).
- **T2S KV cache** — the autoregressive loop prefills once then decodes one
  token per step (O(n) instead of O(n²) recompute).
- **mmap weight loading** — zero-copy memory-mapped `.npy` weight packs (cold
  start ~4.4 s → ~1.7 s).

Further RTF headroom (not yet done): fp16/int8 weight quantization (faster Metal
matmuls + smaller/faster load) and fused custom Metal kernels for the
attention/conv hotspots. These need perceptual-quality validation beyond the
numerical parity gates.

## Remaining work

The neural engine, audio DSP, SentencePiece tokenizer, and end-to-end pipeline
are complete and verified on the real checkpoints. Remaining items are
quality/perf, not core function:
- **Text normalizer** — the reference applies multilingual number/punctuation
  normalization + segmentation before tokenizing; c4tts currently tokenizes the
  text as given.
- **Further performance** — fp16/int8 quantization and fused Metal kernels (see
  Performance above; the cache/KV-cache/mmap wins are already in).
```
