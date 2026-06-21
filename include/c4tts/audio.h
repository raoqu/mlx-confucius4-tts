// Audio front-end ops for c4tts.
//
// Faithful C++/MLX ports of the PyTorch reference in
// confuciustts/utils/audio_features.py. The mel filterbank and analysis window
// are precomputed constants (baked at weight-export time), matching the plan's
// "precompute mel basis / windows" note in docs/PLAN.md §2.2.

#pragma once

#include "c4tts/tensor.h"

namespace c4 {

struct MelConfig {
  int n_fft = 1024;
  int hop = 256;
  int win = 1024;  // unused directly: window is supplied precomputed
  float log_clamp = 1e-5f;
  float mag_eps = 1e-9f;  // matches sqrt(power + 1e-9) in the reference
};

// torch.nn.functional.pad(x, (pad, pad), mode="reflect") for a 1-D signal.
// Reflects without repeating the edge sample (requires pad < len(x)).
Tensor reflect_pad1d(const Tensor& x, int pad);

// Log-mel spectrogram matching mel_spectrogram() in audio_features.py.
//   audio:     (T,) float32 waveform
//   mel_basis: (n_mels, n_fft/2 + 1) librosa slaney filterbank
//   hann:      (n_fft,) analysis window
// returns (n_mels, n_frames) in the log domain.
Tensor mel_spectrogram(const Tensor& audio, const Tensor& mel_basis,
                       const Tensor& hann, const MelConfig& cfg = {});

}  // namespace c4
