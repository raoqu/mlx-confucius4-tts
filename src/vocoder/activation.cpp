#include "c4tts/vocoder.h"

#include <utility>
#include <vector>

#include "c4tts/nn.h"
#include "mlx/mlx.h"

namespace c4 {
namespace vocoder {
namespace mx = mlx::core;

Tensor snakebeta(const Tensor& x, const Tensor& alpha, const Tensor& beta,
                 float eps) {
  const int C = alpha.shape(0);
  Tensor a = mx::exp(mx::reshape(alpha, {1, C, 1}));
  Tensor b = mx::exp(mx::reshape(beta, {1, C, 1}));
  Tensor s = mx::sin(mx::multiply(x, a));
  Tensor recip = mx::divide(mx::array(1.0f), mx::add(b, mx::array(eps)));
  return mx::add(x, mx::multiply(recip, mx::square(s)));
}

namespace {
// Filter expanded to per-channel grouped form. torch up uses conv_transpose with
// weight (Cin=C, Cout/g=1, K); down uses conv with weight (Cout=C, Cin/g=1, K).
Tensor expand_filter(const Tensor& filt, int C, int K) {
  return mx::broadcast_to(mx::reshape(filt, {1, 1, K}), {C, 1, K});
}

Tensor edge_pad_last(const Tensor& x, int left, int right) {
  return mx::pad(x, std::vector<std::pair<int, int>>{{0, 0}, {0, 0}, {left, right}},
                 mx::array(0.0f), "edge");
}
}  // namespace

Tensor anti_aliased_snakebeta(const Tensor& x, const Tensor& alpha,
                              const Tensor& beta, const Tensor& up_filter,
                              const Tensor& down_filter, int ratio,
                              int kernel_size) {
  const int C = x.shape(1);
  const int K = kernel_size;

  // ---- UpSample1d ----
  const int up_pad = K / ratio - 1;
  const int stride = ratio;
  const int pad_left = up_pad * stride + (K - stride) / 2;
  const int pad_right = up_pad * stride + (K - stride + 1) / 2;

  Tensor u = edge_pad_last(x, up_pad, up_pad);
  Tensor uw = expand_filter(up_filter, C, K);  // (C,1,K)
  u = nn::conv_transpose1d(u, uw, nullptr, stride, /*padding=*/0, /*groups=*/C);
  u = mx::multiply(u, mx::array(static_cast<float>(ratio)));
  const int ulen = u.shape(2);
  u = mx::slice(u, mx::Shape{0, 0, pad_left},
                mx::Shape{u.shape(0), C, ulen - pad_right});

  // ---- activation ----
  u = snakebeta(u, alpha, beta);

  // ---- DownSample1d (LowPassFilter1d, even kernel) ----
  const int dl = K / 2 - 1;  // even kernel: pad_left = K/2 - 1
  const int dr = K / 2;      // pad_right = K/2
  u = edge_pad_last(u, dl, dr);
  Tensor dw = expand_filter(down_filter, C, K);
  return nn::conv1d(u, dw, nullptr, /*stride=*/ratio, /*padding=*/0,
                    /*dilation=*/1, /*groups=*/C);
}

}  // namespace vocoder
}  // namespace c4
