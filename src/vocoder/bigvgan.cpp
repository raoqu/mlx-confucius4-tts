#include "c4tts/vocoder.h"

#include <string>

#include "c4tts/nn.h"
#include "c4tts/profile.h"
#include "mlx/mlx.h"

namespace c4 {
namespace vocoder {
namespace mx = mlx::core;

namespace {
int get_padding(int kernel, int dilation) {
  return (kernel * dilation - dilation) / 2;
}
std::string idx(const std::string& p, int i) { return p + std::to_string(i); }
}  // namespace

BigVGAN::BigVGAN(const WeightStore& w)
    : w_(w),
      up_filter_(w.get("up_filter")),
      down_filter_(w.get("down_filter")),
      upsample_rates_{4, 4, 2, 2, 2, 2},
      upsample_kernels_{8, 8, 4, 4, 4, 4},
      resblock_kernels_{3, 7, 11},
      resblock_dilations_{{1, 3, 5}, {1, 3, 5}, {1, 3, 5}} {}

// AMPBlock1: for each (c1,c2,a1,a2) layer: x = x + c2(a2(c1(a1(x)))).
Tensor BigVGAN::amp_block1(const Tensor& x_in, const std::string& prefix,
                           int kernel_size) const {
  Tensor x = x_in;
  for (int layer = 0; layer < 3; ++layer) {
    const int d = resblock_dilations_[0][layer];  // (1,3,5)
    const std::string a1 = idx(prefix + "activations.", 2 * layer) + ".act.";
    const std::string a2 = idx(prefix + "activations.", 2 * layer + 1) + ".act.";
    const std::string c1 = idx(prefix + "convs1.", layer) + ".";
    const std::string c2 = idx(prefix + "convs2.", layer) + ".";

    Tensor xt = anti_aliased_snakebeta(x, w_.get(a1 + "alpha"), w_.get(a1 + "beta"),
                                       up_filter_, down_filter_);
    Tensor c1w = w_.get(c1 + "weight"), c1b = w_.get(c1 + "bias");
    xt = nn::conv1d(xt, c1w, &c1b, /*stride=*/1, get_padding(kernel_size, d),
                    /*dilation=*/d);

    xt = anti_aliased_snakebeta(xt, w_.get(a2 + "alpha"), w_.get(a2 + "beta"),
                                up_filter_, down_filter_);
    Tensor c2w = w_.get(c2 + "weight"), c2b = w_.get(c2 + "bias");
    xt = nn::conv1d(xt, c2w, &c2b, /*stride=*/1, get_padding(kernel_size, 1),
                    /*dilation=*/1);

    x = mx::add(xt, x);
  }
  return x;
}

Tensor BigVGAN::forward(const Tensor& mel) const {
  prof::Timer bt;
  // conv_pre: Conv1d(80, 1536, 7, pad 3)
  Tensor cpw = w_.get("conv_pre.weight"), cpb = w_.get("conv_pre.bias");
  Tensor x = nn::conv1d(mel, cpw, &cpb, /*stride=*/1, /*padding=*/3);
  prof::lap("bigvgan.conv_pre", x, bt);

  const int num_kernels = static_cast<int>(resblock_kernels_.size());
  for (size_t i = 0; i < upsample_rates_.size(); ++i) {
    // Transposed-conv upsample (no anti-aliasing).
    const int u = upsample_rates_[i];
    const int k = upsample_kernels_[i];
    const std::string up = idx("ups.", static_cast<int>(i)) + ".0.";
    Tensor uw = w_.get(up + "weight"), ub = w_.get(up + "bias");
    x = nn::conv_transpose1d(x, uw, &ub, /*stride=*/u, /*padding=*/(k - u) / 2);

    // Sum the multi-receptive-field AMP blocks, then average.
    Tensor xs = mx::array(0.0f);
    for (int j = 0; j < num_kernels; ++j) {
      const std::string rb =
          idx("resblocks.", static_cast<int>(i) * num_kernels + j) + ".";
      Tensor block = amp_block1(x, rb, resblock_kernels_[j]);
      xs = (j == 0) ? block : mx::add(xs, block);
    }
    x = mx::divide(xs, mx::array(static_cast<float>(num_kernels)));
    prof::lap("bigvgan.upsample+resblocks", x, bt);
  }

  // Post: snakebeta -> conv_post (no bias) -> clamp[-1,1] (use_tanh_at_final=False)
  x = anti_aliased_snakebeta(x, w_.get("activation_post.act.alpha"),
                             w_.get("activation_post.act.beta"), up_filter_,
                             down_filter_);
  Tensor cqw = w_.get("conv_post.weight");
  x = nn::conv1d(x, cqw, /*bias=*/nullptr, /*stride=*/1, /*padding=*/3);
  Tensor out = mx::clip(x, mx::array(-1.0f), mx::array(1.0f));
  prof::lap("bigvgan.post", out, bt);
  return out;
}

}  // namespace vocoder
}  // namespace c4
