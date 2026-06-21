#include "c4tts/audio.h"

#include <vector>

#include "mlx/mlx.h"

namespace c4 {
namespace mx = mlx::core;

Tensor reflect_pad1d(const Tensor& x, int pad) {
  const int n = static_cast<int>(x.size());
  if (pad <= 0) return x;
  // Build the gather index map for reflect padding (no edge repeat), matching
  // torch's F.pad(..., mode="reflect"). Valid for pad < n.
  std::vector<int32_t> idx(static_cast<size_t>(n + 2 * pad));
  int w = 0;
  for (int j = 0; j < pad; ++j) idx[w++] = pad - j;           // left:  [pad..1]
  for (int j = 0; j < n; ++j) idx[w++] = j;                   // middle
  for (int j = 0; j < pad; ++j) idx[w++] = n - 2 - j;         // right: [n-2..]

  int32_t* heap = new int32_t[idx.size()];
  for (size_t i = 0; i < idx.size(); ++i) heap[i] = idx[i];
  mx::array index(static_cast<void*>(heap), mx::Shape{static_cast<int>(idx.size())},
                  mx::int32, [](void* p) { delete[] static_cast<int32_t*>(p); });
  return mx::take(x, index, 0);
}

Tensor mel_spectrogram(const Tensor& audio, const Tensor& mel_basis,
                       const Tensor& hann, const MelConfig& cfg) {
  // Reference: pad = (n_fft - hop) // 2, reflect-pad, then center=False STFT.
  const int pad = (cfg.n_fft - cfg.hop) / 2;
  Tensor y = reflect_pad1d(audio, pad);

  const int lp = static_cast<int>(y.size());
  const int n_frames = 1 + (lp - cfg.n_fft) / cfg.hop;

  // Overlapping frames (n_frames, n_fft) via a strided view (no copy).
  Tensor frames = mx::as_strided(
      y, mx::Shape{n_frames, cfg.n_fft},
      mx::Strides{static_cast<int64_t>(cfg.hop), 1}, 0);

  // Window then real FFT along the frame axis (norm=Backward == torch
  // normalized=False).
  Tensor windowed = mx::multiply(frames, mx::reshape(hann, {1, cfg.n_fft}));
  Tensor spec = mx::fft::rfft(windowed, cfg.n_fft, 1);  // (n_frames, n_fft/2+1)

  // Magnitude: sqrt(re^2 + im^2 + eps), matching the reference exactly.
  Tensor re = mx::real(spec);
  Tensor im = mx::imag(spec);
  Tensor mag = mx::sqrt(
      mx::add(mx::add(mx::square(re), mx::square(im)), mx::array(cfg.mag_eps)));

  // mel_basis (n_mels, F) @ mag^T (F, n_frames) -> (n_mels, n_frames).
  Tensor mel = mx::matmul(mel_basis, mx::transpose(mag));
  return mx::log(mx::maximum(mel, mx::array(cfg.log_clamp)));
}

}  // namespace c4
