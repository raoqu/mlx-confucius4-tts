#include <cmath>
#include <limits>
#include <vector>

#include "c4tts/audio.h"
#include "mlx/mlx.h"

namespace c4 {
namespace mx = mlx::core;

namespace {
int next_pow2(int n) {
  int p = 1;
  while (p < n) p <<= 1;
  return p;
}
double mel_scale(double f) { return 1127.0 * std::log(1.0 + f / 700.0); }

// Builds an array from a host vector (copy, owned).
mx::array from_vec(const std::vector<float>& v, mx::Shape shape) {
  float* heap = new float[v.size()];
  for (size_t i = 0; i < v.size(); ++i) heap[i] = v[i];
  return mx::array(static_cast<void*>(heap), shape, mx::float32,
                   [](void* p) { delete[] static_cast<float*>(p); });
}
}  // namespace

Tensor kaldi_fbank(const Tensor& wav_in, int num_mel_bins, float sample_rate) {
  const int win = static_cast<int>(sample_rate * 0.025f);   // 25 ms -> 400
  const int shift = static_cast<int>(sample_rate * 0.010f); // 10 ms -> 160
  const int padded = next_pow2(win);                        // 512
  const int n_fft_bins = padded / 2;                        // 256
  const float preemph = 0.97f;

  Tensor wav = mx::astype(mx::flatten(wav_in), mx::float32);
  const int T = static_cast<int>(wav.size());
  const int m = 1 + (T - win) / shift;

  // Strided frames (m, win).
  Tensor frames = mx::as_strided(wav, mx::Shape{m, win},
                                 mx::Strides{static_cast<int64_t>(shift), 1}, 0);

  // remove_dc_offset: subtract per-frame mean.
  frames = mx::subtract(frames, mx::mean(frames, 1, /*keepdims=*/true));

  // preemphasis with replicate-pad on the left: x[:,j] -= 0.97 * x[:,max(0,j-1)].
  Tensor first = mx::slice(frames, mx::Shape{0, 0}, mx::Shape{m, 1});
  Tensor prev = mx::slice(frames, mx::Shape{0, 0}, mx::Shape{m, win - 1});
  Tensor offset = mx::concatenate({first, prev}, 1);  // (m, win)
  frames = mx::subtract(frames, mx::multiply(mx::array(preemph), offset));

  // Povey window: hann(win, periodic=False) ^ 0.85.
  std::vector<float> window(static_cast<size_t>(win));
  for (int n = 0; n < win; ++n) {
    double h = 0.5 - 0.5 * std::cos(2.0 * M_PI * n / (win - 1));
    window[n] = static_cast<float>(std::pow(h, 0.85));
  }
  frames = mx::multiply(frames, mx::reshape(from_vec(window, mx::Shape{win}), {1, win}));

  // Pad to padded_window_size with zeros (right).
  if (padded != win) {
    frames = mx::concatenate(
        {frames, mx::zeros({m, padded - win}, mx::float32)}, 1);
  }

  // Power spectrum: |rfft|^2 -> (m, n_fft_bins + 1).
  Tensor spec = mx::fft::rfft(frames, padded, 1);
  Tensor power = mx::add(mx::square(mx::real(spec)), mx::square(mx::imag(spec)));

  // Kaldi mel filterbank (num_mel_bins, n_fft_bins), padded with a zero column.
  const double nyquist = 0.5 * sample_rate;
  const double low_freq = 20.0, high_freq = nyquist;
  const double fft_bin_width = sample_rate / padded;
  const double mel_low = mel_scale(low_freq), mel_high = mel_scale(high_freq);
  const double mel_delta = (mel_high - mel_low) / (num_mel_bins + 1);

  std::vector<float> banks(static_cast<size_t>(num_mel_bins) * (n_fft_bins + 1), 0.0f);
  for (int b = 0; b < num_mel_bins; ++b) {
    const double left = mel_low + b * mel_delta;
    const double center = mel_low + (b + 1) * mel_delta;
    const double right = mel_low + (b + 2) * mel_delta;
    for (int i = 0; i < n_fft_bins; ++i) {
      const double mel = mel_scale(fft_bin_width * i);
      const double up = (mel - left) / (center - left);
      const double down = (right - mel) / (right - center);
      double v = std::min(up, down);
      if (v < 0) v = 0;
      banks[b * (n_fft_bins + 1) + i] = static_cast<float>(v);
    }
  }
  Tensor mel_banks = from_vec(banks, mx::Shape{num_mel_bins, n_fft_bins + 1});

  Tensor mel = mx::matmul(power, mx::transpose(mel_banks));  // (m, num_mel_bins)
  const float eps = std::numeric_limits<float>::epsilon();
  return mx::log(mx::maximum(mel, mx::array(eps)));
}

}  // namespace c4
