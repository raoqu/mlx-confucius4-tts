// Neural-net primitive ops for c4tts (Phase A2 op layer, docs/PLAN.md).
//
// Thin, parity-checked wrappers over MLX that present PyTorch conventions
// (weight layouts, channel order) so the larger module ports read directly off
// the reference code. Every primitive here is covered by test_nn.

#pragma once

#include "c4tts/tensor.h"

namespace c4 {
namespace nn {

// y = x @ W^T + b. weight: (out, in) (PyTorch nn.Linear layout). bias optional.
Tensor linear(const Tensor& x, const Tensor& weight, const Tensor* bias = nullptr);

// 1-D convolution in PyTorch conventions:
//   input:  (N, C_in, L)
//   weight: (C_out, C_in/groups, K)
//   output: (N, C_out, L_out)
Tensor conv1d(const Tensor& x, const Tensor& weight, const Tensor* bias = nullptr,
              int stride = 1, int padding = 0, int dilation = 1, int groups = 1);

// 1-D transposed convolution in PyTorch conventions:
//   input:  (N, C_in, L)
//   weight: (C_in, C_out/groups, K)
//   output: (N, C_out, L_out)
Tensor conv_transpose1d(const Tensor& x, const Tensor& weight,
                        const Tensor* bias = nullptr, int stride = 1,
                        int padding = 0, int groups = 1);

// 2-D convolution in PyTorch conventions:
//   input:  (N, C_in, H, W)
//   weight: (C_out, C_in/groups, kH, kW)
//   output: (N, C_out, H_out, W_out)
Tensor conv2d(const Tensor& x, const Tensor& weight, const Tensor* bias = nullptr,
              int stride_h = 1, int stride_w = 1, int pad_h = 0, int pad_w = 0,
              int groups = 1);

// BatchNorm (eval mode) over the channel axis using running statistics:
//   y = (x - mean) / sqrt(var + eps) * gamma + beta
// channel_axis selects the channel dim (1 for (N,C,...) tensors).
Tensor batch_norm(const Tensor& x, const Tensor& mean, const Tensor& var,
                  const Tensor* gamma = nullptr, const Tensor* beta = nullptr,
                  float eps = 1e-5f, int channel_axis = 1);

// LayerNorm over the last dimension (PyTorch nn.LayerNorm with normalized_shape
// = last dim). weight/bias optional (elementwise_affine).
Tensor layer_norm(const Tensor& x, const Tensor* weight = nullptr,
                  const Tensor* bias = nullptr, float eps = 1e-5f);

// GroupNorm over channels for (N, C, L) input (PyTorch nn.GroupNorm).
Tensor group_norm(const Tensor& x, int num_groups, const Tensor* weight = nullptr,
                  const Tensor* bias = nullptr, float eps = 1e-5f);

// RMSNorm over the last dimension (x * rsqrt(mean(x^2)+eps) * weight),
// computed in float32 like flow/DiT/modules.py:RMSNorm.
Tensor rms_norm(const Tensor& x, const Tensor& weight, float eps = 1e-6f);

// Embedding lookup: weight (V, D), ids (...,) int -> (..., D).
Tensor embedding(const Tensor& ids, const Tensor& weight);

// Nearest-neighbour resize along the last axis to `out_len`, matching
// torch.nn.functional.interpolate(mode="nearest") (index = floor(i*L/out_len)).
Tensor interpolate_nearest(const Tensor& x, int out_len);

Tensor silu(const Tensor& x);
Tensor gelu(const Tensor& x);   // exact (erf) gelu, matches nn.GELU() default
Tensor mish(const Tensor& x);   // x * tanh(softplus(x)), matches nn.Mish()

}  // namespace nn
}  // namespace c4
