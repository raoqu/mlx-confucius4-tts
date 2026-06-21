#include "c4tts/nn.h"

#include <vector>

#include "mlx/mlx.h"

namespace c4 {
namespace nn {
namespace mx = mlx::core;

Tensor linear(const Tensor& x, const Tensor& weight, const Tensor* bias) {
  // weight: (out, in) -> multiply by its transpose.
  Tensor y = mx::matmul(x, mx::transpose(weight));
  if (bias) y = mx::add(y, *bias);
  return y;
}

Tensor conv1d(const Tensor& x, const Tensor& weight, const Tensor* bias,
              int stride, int padding, int dilation, int groups) {
  // PyTorch (N,Cin,L)/(Cout,Cin/g,K) -> MLX (N,L,Cin)/(Cout,K,Cin/g).
  Tensor x_ncl = mx::transpose(x, {0, 2, 1});
  Tensor w_mlx = mx::transpose(weight, {0, 2, 1});
  Tensor y = mx::conv1d(x_ncl, w_mlx, stride, padding, dilation, groups);
  if (bias) y = mx::add(y, mx::reshape(*bias, {1, 1, -1}));
  return mx::transpose(y, {0, 2, 1});  // back to (N, Cout, Lout)
}

Tensor conv_transpose1d(const Tensor& x, const Tensor& weight,
                        const Tensor* bias, int stride, int padding,
                        int groups) {
  // PyTorch (N,Cin,L)/(Cin,Cout/g,K) -> MLX (N,L,Cin)/(Cin,K,Cout/g).
  Tensor x_nlc = mx::transpose(x, {0, 2, 1});
  Tensor w_mlx = mx::transpose(weight, {0, 2, 1});
  Tensor y = mx::conv_transpose1d(x_nlc, w_mlx, stride, padding, /*dilation=*/1,
                                  /*output_padding=*/0, groups);
  if (bias) y = mx::add(y, mx::reshape(*bias, {1, 1, -1}));
  return mx::transpose(y, {0, 2, 1});  // (N, Cout, Lout)
}

Tensor layer_norm(const Tensor& x, const Tensor* weight, const Tensor* bias,
                  float eps) {
  const int axis = x.ndim() - 1;
  Tensor mean = mx::mean(x, axis, /*keepdims=*/true);
  Tensor xc = mx::subtract(x, mean);
  Tensor var = mx::mean(mx::square(xc), axis, /*keepdims=*/true);  // biased
  Tensor y = mx::divide(xc, mx::sqrt(mx::add(var, mx::array(eps))));
  if (weight) y = mx::multiply(y, *weight);
  if (bias) y = mx::add(y, *bias);
  return y;
}

Tensor group_norm(const Tensor& x, int num_groups, const Tensor* weight,
                  const Tensor* bias, float eps) {
  // x: (N, C, L). Normalize over (C/G, L) within each group, biased variance.
  const int N = x.shape(0);
  const int C = x.shape(1);
  const int L = x.shape(2);
  Tensor xg = mx::reshape(x, {N, num_groups, (C / num_groups) * L});
  Tensor mean = mx::mean(xg, 2, true);
  Tensor xc = mx::subtract(xg, mean);
  Tensor var = mx::mean(mx::square(xc), 2, true);
  Tensor y = mx::divide(xc, mx::sqrt(mx::add(var, mx::array(eps))));
  y = mx::reshape(y, {N, C, L});
  if (weight) y = mx::multiply(y, mx::reshape(*weight, {1, C, 1}));
  if (bias) y = mx::add(y, mx::reshape(*bias, {1, C, 1}));
  return y;
}

Tensor rms_norm(const Tensor& x, const Tensor& weight, float eps) {
  // Matches flow/DiT/modules.py:RMSNorm — normalize in float32, then scale.
  const int axis = x.ndim() - 1;
  Tensor xf = mx::astype(x, mx::float32);
  Tensor ms = mx::mean(mx::square(xf), axis, true);
  Tensor y = mx::multiply(xf, mx::rsqrt(mx::add(ms, mx::array(eps))));
  y = mx::astype(y, x.dtype());
  return mx::multiply(y, weight);
}

Tensor embedding(const Tensor& ids, const Tensor& weight) {
  return mx::take(weight, ids, 0);
}

Tensor interpolate_nearest(const Tensor& x, int out_len) {
  const int in_len = x.shape(x.ndim() - 1);
  std::vector<int32_t> idx(static_cast<size_t>(out_len));
  for (int i = 0; i < out_len; ++i) {
    idx[i] = static_cast<int32_t>((static_cast<int64_t>(i) * in_len) / out_len);
  }
  int32_t* heap = new int32_t[idx.size()];
  for (size_t i = 0; i < idx.size(); ++i) heap[i] = idx[i];
  mx::array index(static_cast<void*>(heap), mx::Shape{out_len}, mx::int32,
                  [](void* p) { delete[] static_cast<int32_t*>(p); });
  return mx::take(x, index, x.ndim() - 1);
}

Tensor silu(const Tensor& x) { return mx::multiply(x, mx::sigmoid(x)); }

Tensor gelu(const Tensor& x) {
  // exact gelu: 0.5 * x * (1 + erf(x / sqrt(2)))
  Tensor t = mx::erf(mx::divide(x, mx::array(1.4142135623730951f)));
  return mx::multiply(mx::multiply(x, mx::array(0.5f)),
                      mx::add(mx::array(1.0f), t));
}

Tensor mish(const Tensor& x) {
  // x * tanh(softplus(x))
  Tensor sp = mx::logaddexp(mx::array(0.0f), x);  // softplus(x) = log(1+e^x)
  return mx::multiply(x, mx::tanh(sp));
}

}  // namespace nn
}  // namespace c4
