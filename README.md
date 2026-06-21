# c4tts

A standalone, Apple-Silicon-optimized **C++** inference engine for
[Confucius4-TTS](../README.md). No Python runtime dependency at inference time.

Design goals and the step-by-step roadmap live in [`../docs/PLAN.md`](../docs/PLAN.md).
The guiding philosophy is borrowed from [antirez/ds4](https://github.com/antirez/ds4):
narrow focus, golden-vector parity against the PyTorch reference, fully
self-contained, Apple Metal first.

## Status

Phase A1 — scaffold. The build links [MLX](https://github.com/ml-explore/mlx)
(the primary compute backend) and `c4tts_cli` runs a smoke test that confirms
the Metal backend works.

## Build

Requirements: macOS on Apple Silicon, CMake ≥ 3.24, a recent Apple Clang, and
the `mlx` Python wheel installed (its bundled C++ library and CMake config are
reused — no separate MLX build needed).

```bash
cd c4tts
cmake -S . -B build
cmake --build build -j
./build/c4tts_cli --version
./build/c4tts_cli            # runs the MLX smoke test
```

If MLX is built from source instead of pip, pass `-DMLX_ROOT=/path/to/mlx`.

## Layout

```
c4tts/
├── apps/        CLI entry point
├── include/     public headers (c4::Tensor abstraction, weights, pipeline)
├── src/         engine implementation (grows per docs/PLAN.md phases)
├── kernels/     custom Metal shaders (added for hotspots)
├── tools/       Python: weight export + golden-vector capture
└── tests/       per-module numerical-parity tests
```
