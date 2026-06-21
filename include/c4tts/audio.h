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

// Windowed-sinc resampling, matching torchaudio.functional.resample defaults
// (sinc_interp_hann, lowpass_filter_width=6, rolloff=0.99).
//   wav: (T,) at orig_freq -> (ceil(T*new/orig),) at new_freq.
Tensor resample(const Tensor& wav, int orig_freq, int new_freq,
                int lowpass_filter_width = 6, float rolloff = 0.99f);

// Kaldi-compatible log-mel filterbank, matching
// torchaudio.compliance.kaldi.fbank(num_mel_bins=80, sample_frequency=16000,
// dither=0.0) with default povey window / preemphasis 0.97 / snip_edges.
//   wav: (T,) float waveform at sample_rate -> (num_frames, num_mel_bins).
// Mean subtraction is NOT applied (subtract_mean=False), matching the default.
Tensor kaldi_fbank(const Tensor& wav, int num_mel_bins = 80,
                   float sample_rate = 16000.0f);

// SeamlessM4T feature extractor for W2V-BERT (transformers
// SeamlessM4TFeatureExtractor). Kaldi-style log-mel fbank, per-mel-bin
// zero-mean/unit-var normalization (ddof=1), then stride-2 frame stacking.
//   wav: (T,) 16 kHz -> (num_frames/2, 160).
// (The 2^15 input scaling and energy offset are additive constants in the log
// domain and cancel under the per-bin mean subtraction.)
Tensor seamless_features(const Tensor& wav);

// Log-mel spectrogram matching mel_spectrogram() in audio_features.py.
//   audio:     (T,) float32 waveform
//   mel_basis: (n_mels, n_fft/2 + 1) librosa slaney filterbank
//   hann:      (n_fft,) analysis window
// returns (n_mels, n_frames) in the log domain.
Tensor mel_spectrogram(const Tensor& audio, const Tensor& mel_basis,
                       const Tensor& hann, const MelConfig& cfg = {});

}  // namespace c4
