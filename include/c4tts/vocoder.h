// BigVGAN vocoder ops for c4tts (external/bigvgan).
//
// Faithful C++/MLX ports. Kaiser-sinc anti-aliasing filters are precomputed
// constants (baked at export time), per docs/PLAN.md §2.2. Verified by
// test_vocoder.

#pragma once

#include <string>
#include <vector>

#include "c4tts/tensor.h"
#include "c4tts/weights.h"

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

// BigVGAN generator (bigvgan.py:BigVGAN), config nvidia/bigvgan_v2_22khz_80band_256x:
// AMPBlock1 resblocks, snakebeta, 6 transposed-conv upsamples (256x total),
// clamp final (use_tanh_at_final=False). mel (B, 80, T) -> waveform (B, 1, 256T).
class BigVGAN {
 public:
  explicit BigVGAN(const WeightStore& w);
  Tensor forward(const Tensor& mel) const;

 private:
  Tensor amp_block1(const Tensor& x, const std::string& prefix,
                    int kernel_size) const;

  const WeightStore& w_;
  Tensor up_filter_, down_filter_;
  std::vector<int> upsample_rates_;
  std::vector<int> upsample_kernels_;
  std::vector<int> resblock_kernels_;
  std::vector<std::vector<int>> resblock_dilations_;
};

}  // namespace vocoder
}  // namespace c4
