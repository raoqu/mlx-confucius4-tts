#include <cmath>
#include <numeric>
#include <vector>

#include "c4tts/audio.h"
#include "c4tts/nn.h"
#include "mlx/mlx.h"

namespace c4 {
namespace mx = mlx::core;

// Port of torchaudio.functional.resample (sinc_interp_hann window).
Tensor resample(const Tensor& wav_in, int orig_freq, int new_freq,
                int lowpass_filter_width, float rolloff) {
  if (orig_freq == new_freq) return wav_in;
  const int g = std::gcd(orig_freq, new_freq);
  const int of = orig_freq / g;
  const int nf = new_freq / g;

  const double base_freq = std::min(of, nf) * static_cast<double>(rolloff);
  const int width =
      static_cast<int>(std::ceil(lowpass_filter_width * of / base_freq));
  const int kw = 2 * width + of;  // idx range [-width, width+of)
  const double pi = M_PI;
  const double scale = base_freq / of;

  // kernel[i, k] for output phase i in [0,nf), input tap k in [0,kw).
  std::vector<float> kernel(static_cast<size_t>(nf) * kw);
  for (int i = 0; i < nf; ++i) {
    for (int k = 0; k < kw; ++k) {
      const int idx = k - width;  // input position offset
      double t = (-static_cast<double>(i) / nf + static_cast<double>(idx) / of) *
                 base_freq;
      if (t < -lowpass_filter_width) t = -lowpass_filter_width;
      if (t > lowpass_filter_width) t = lowpass_filter_width;
      double window = std::cos(t * pi / lowpass_filter_width / 2.0);
      window *= window;
      double tp = t * pi;
      double sinc = (tp == 0.0) ? 1.0 : std::sin(tp) / tp;
      kernel[i * kw + k] = static_cast<float>(sinc * window * scale);
    }
  }

  float* heap = new float[kernel.size()];
  for (size_t i = 0; i < kernel.size(); ++i) heap[i] = kernel[i];
  mx::array kern(static_cast<void*>(heap), mx::Shape{nf, 1, kw}, mx::float32,
                 [](void* p) { delete[] static_cast<float*>(p); });

  Tensor wav = mx::astype(mx::flatten(wav_in), mx::float32);
  const int L = static_cast<int>(wav.size());
  Tensor x = mx::reshape(wav, {1, 1, L});
  // pad (width, width + of)
  x = mx::pad(x, std::vector<std::pair<int, int>>{{0, 0}, {0, 0}, {width, width + of}},
              mx::array(0.0f), "constant");

  // conv1d stride=of -> (1, nf, n); interleave phases to the time axis.
  Tensor y = nn::conv1d(x, kern, nullptr, /*stride=*/of);  // (1, nf, n)
  const int n = y.shape(2);
  y = mx::transpose(y, {0, 2, 1});       // (1, n, nf)
  y = mx::reshape(y, {1, n * nf});       // interleaved

  const int target = static_cast<int>(
      std::ceil(static_cast<double>(new_freq) * L / orig_freq));
  Tensor out = mx::slice(y, mx::Shape{0, 0}, mx::Shape{1, target});
  return mx::reshape(out, {target});
}

}  // namespace c4
