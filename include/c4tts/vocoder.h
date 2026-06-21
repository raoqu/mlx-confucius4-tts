// BigVGAN vocoder ops for c4tts (external/bigvgan).
//
// Faithful C++/MLX ports. Kaiser-sinc anti-aliasing filters are precomputed
// constants (baked at export time), per docs/PLAN.md §2.2. Verified by
// test_vocoder.

#pragma once

#include "c4tts/tensor.h"

namespace c4 {
namespace vocoder {

// SnakeBeta (logscale), activations.py:SnakeBeta with alpha_logscale=True:
//   x + (1/(exp(beta)+eps)) * sin(x*exp(alpha))^2
//   x: (B, C, T); alpha, beta: (C,)
Tensor snakebeta(const Tensor& x, const Tensor& alpha, const Tensor& beta,
                 float eps = 1e-9f);

// Anti-aliased activation (alias_free_activation): upsample x2 -> snakebeta ->
// downsample x2, with Kaiser-sinc FIR filters (ratio=2, kernel=12).
//   x: (B, C, T); alpha/beta: (C,); up_filter/down_filter: (K,)
Tensor anti_aliased_snakebeta(const Tensor& x, const Tensor& alpha,
                              const Tensor& beta, const Tensor& up_filter,
                              const Tensor& down_filter, int ratio = 2,
                              int kernel_size = 12);

}  // namespace vocoder
}  // namespace c4
