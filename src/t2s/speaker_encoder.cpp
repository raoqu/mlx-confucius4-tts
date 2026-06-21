#include <string>
#include <vector>

#include "c4tts/nn.h"
#include "c4tts/t2s.h"
#include "mlx/mlx.h"

namespace c4 {
namespace t2s {
namespace mx = mlx::core;

namespace {
// Reflect-pad the last axis by `p` (no edge repeat), matching torch reflect.
Tensor reflect_pad_last(const Tensor& x, int p) {
  if (p <= 0) return x;
  const int n = x.shape(x.ndim() - 1);
  std::vector<int32_t> idx;
  idx.reserve(static_cast<size_t>(n + 2 * p));
  for (int j = 0; j < p; ++j) idx.push_back(p - j);
  for (int j = 0; j < n; ++j) idx.push_back(j);
  for (int j = 0; j < p; ++j) idx.push_back(n - 2 - j);
  int32_t* heap = new int32_t[idx.size()];
  for (size_t i = 0; i < idx.size(); ++i) heap[i] = idx[i];
  mx::array index(static_cast<void*>(heap), mx::Shape{static_cast<int>(idx.size())},
                  mx::int32, [](void* q) { delete[] static_cast<int32_t*>(q); });
  return mx::take(x, index, x.ndim() - 1);
}
}  // namespace

SpeakerEncoder::SpeakerEncoder(const WeightStore& w, const std::string& prefix)
    : w_(w), p_(prefix) {}

Tensor SpeakerEncoder::tdnn(const Tensor& x, const std::string& p, int kernel,
                           int dilation, bool relu) const {
  Tensor wt = w_.get(p + "weight"), b = w_.get(p + "bias");
  const int pad = dilation * (kernel - 1) / 2;  // "same" (odd kernels)
  Tensor xp = reflect_pad_last(x, pad);
  Tensor y = nn::conv1d(xp, wt, &b, /*stride=*/1, /*padding=*/0, dilation);
  return relu ? mx::maximum(y, mx::array(0.0f)) : y;
}

Tensor SpeakerEncoder::se_res2net_block(const Tensor& x, const std::string& p,
                                        int kernel, int dilation) const {
  // tdnn1 (1x1) -> res2net -> tdnn2 (1x1) -> SE; residual add (no shortcut: in==out).
  Tensor h = tdnn(x, p + "tdnn1.conv.", 1, 1, true);

  // Res2Net (scale=8): chunk channels, hierarchical accumulation.
  const int scale = 8;
  const int C = h.shape(1);
  const int cs = C / scale;
  std::vector<Tensor> outs;
  Tensor prev = mx::array(0.0f);
  for (int i = 0; i < scale; ++i) {
    Tensor part = mx::slice(h, mx::Shape{0, i * cs, 0},
                            mx::Shape{h.shape(0), (i + 1) * cs, h.shape(2)});
    Tensor out_part = part;
    if (i == 1) {
      out_part = tdnn(part, p + "res2net_block.blocks.0.conv.", kernel, dilation, true);
    } else if (i >= 2) {
      out_part = tdnn(mx::add(part, prev),
                      p + "res2net_block.blocks." + std::to_string(i - 1) + ".conv.",
                      kernel, dilation, true);
    }
    outs.push_back(out_part);
    prev = out_part;
  }
  h = mx::concatenate(outs, 1);

  h = tdnn(h, p + "tdnn2.conv.", 1, 1, true);

  // Squeeze-Excitation: channel attention from time-mean.
  Tensor s = mx::mean(h, 2, /*keepdims=*/true);  // (B,C,1)
  Tensor c1w = w_.get(p + "se_block.conv1.weight"), c1b = w_.get(p + "se_block.conv1.bias");
  s = mx::maximum(nn::conv1d(s, c1w, &c1b), mx::array(0.0f));
  Tensor c2w = w_.get(p + "se_block.conv2.weight"), c2b = w_.get(p + "se_block.conv2.bias");
  s = mx::sigmoid(nn::conv1d(s, c2w, &c2b));
  h = mx::multiply(h, s);

  return mx::add(h, x);  // residual (in_channels == out_channels)
}

Tensor SpeakerEncoder::attentive_stats_pooling(const Tensor& x,
                                               const std::string& p) const {
  const int T = x.shape(2);
  const float eps = 1e-12f;

  // Unweighted mean/std over time (uniform mask).
  Tensor mean = mx::mean(x, 2, true);  // (B,C,1)
  Tensor var = mx::mean(mx::square(mx::subtract(x, mean)), 2, true);
  Tensor std = mx::sqrt(mx::maximum(var, mx::array(eps)));
  Tensor mean_e = mx::broadcast_to(mean, {x.shape(0), x.shape(1), T});
  Tensor std_e = mx::broadcast_to(std, {x.shape(0), x.shape(1), T});

  Tensor attn_in = mx::concatenate({x, mean_e, std_e}, 1);  // (B,3C,T)
  Tensor a = tdnn(attn_in, p + "tdnn.conv.", 1, 1, true);
  a = mx::tanh(a);
  Tensor cw = w_.get(p + "conv.weight"), cb = w_.get(p + "conv.bias");
  a = nn::conv1d(a, cw, &cb);            // (B,C,T)
  a = mx::softmax(a, 2);                 // attention over time

  // Weighted statistics with the attention weights.
  Tensor wmean = mx::sum(mx::multiply(a, x), 2);  // (B,C)
  Tensor wmean_e = mx::expand_dims(wmean, 2);
  Tensor wvar = mx::sum(mx::multiply(a, mx::square(mx::subtract(x, wmean_e))), 2);
  Tensor wstd = mx::sqrt(mx::maximum(wvar, mx::array(eps)));
  Tensor pooled = mx::concatenate({wmean, wstd}, 1);  // (B,2C)
  return mx::expand_dims(pooled, 2);                   // (B,2C,1)
}

Tensor SpeakerEncoder::forward(const Tensor& x_in) const {
  Tensor x = mx::transpose(x_in, {0, 2, 1});  // (B, mel, T)

  const std::vector<int> kernels = {5, 3, 3, 3, 1};
  const std::vector<int> dils = {1, 2, 3, 4, 1};

  // Block 0: plain TDNN.
  std::vector<Tensor> feats;
  Tensor h = tdnn(x, p_ + "blocks.0.conv.", kernels[0], dils[0], true);
  feats.push_back(h);

  // Blocks 1..3: SE-Res2Net.
  for (int i = 1; i <= 3; ++i) {
    h = se_res2net_block(h, p_ + "blocks." + std::to_string(i) + ".", kernels[i],
                         dils[i]);
    feats.push_back(h);
  }

  // Multi-layer feature aggregation over blocks 1..3.
  Tensor agg = mx::concatenate({feats[1], feats[2], feats[3]}, 1);
  Tensor mfa = tdnn(agg, p_ + "mfa.conv.", kernels[4], dils[4], true);

  Tensor pooled = attentive_stats_pooling(mfa, p_ + "asp.");  // (B,2C,1)
  Tensor fcw = w_.get(p_ + "fc.weight"), fcb = w_.get(p_ + "fc.bias");
  Tensor out = nn::conv1d(pooled, fcw, &fcb);  // (B, enc_dim, 1)
  return mx::squeeze(out, 2);                   // (B, enc_dim)
}

// ---- TextEmbeddingProjector ----

TextEmbeddingProjector::TextEmbeddingProjector(const WeightStore& w,
                                               const std::string& prefix)
    : w_(w), p_(prefix) {}

Tensor TextEmbeddingProjector::forward(const Tensor& text_ids) const {
  Tensor emb = nn::embedding(text_ids, w_.get(p_ + "embed.weight"));  // (B,T,4096)
  Tensor w1 = w_.get(p_ + "text_projection_fc1.weight");
  Tensor b1 = w_.get(p_ + "text_projection_fc1.bias");
  Tensor h = nn::silu(nn::linear(emb, w1, &b1));
  Tensor w2 = w_.get(p_ + "text_projection_fc2.weight");
  Tensor b2 = w_.get(p_ + "text_projection_fc2.bias");
  return nn::linear(h, w2, &b2);
}

Tensor add_learned_positions(const Tensor& x, const Tensor& pos_table) {
  const int T = x.shape(1);
  Tensor rows = mx::slice(pos_table, mx::Shape{0, 0},
                          mx::Shape{T, pos_table.shape(1)});  // (T, D)
  return mx::add(x, mx::expand_dims(rows, 0));
}

}  // namespace t2s
}  // namespace c4
