// Minimal WAV (RIFF/PCM) read/write for c4tts — no external audio deps.
// Supports 16-bit integer and 32-bit float PCM; multi-channel input is
// downmixed to mono. Output is written as 16-bit PCM.

#pragma once

#include <string>

#include "c4tts/tensor.h"

namespace c4 {

struct WavData {
  Tensor samples;     // (T,) float32 mono in [-1, 1]
  int sample_rate = 0;
};

// Reads a PCM WAV file (mono downmix). Throws on unsupported/invalid files.
WavData read_wav(const std::string& path);

// Writes a mono float32 waveform (T,) as a 16-bit PCM WAV.
void write_wav(const std::string& path, const Tensor& samples, int sample_rate);

}  // namespace c4
