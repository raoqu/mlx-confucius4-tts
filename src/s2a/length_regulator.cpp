#include "c4tts/nn.h"
#include "c4tts/s2a.h"
#include "mlx/mlx.h"

namespace c4 {
namespace s2a {
namespace mx = mlx::core;

// ---- SemanticTokenEmbedding ------------------------------------------------

SemanticTokenEmbedding::SemanticTokenEmbedding(const WeightStore& w,
                                               const std::string& prefix)
    : emb_weight_(w.get(prefix + "embedding.weight")),
      proj_weight_(w.get(prefix + "out_project.weight")),
      proj_bias_(w.get(prefix + "out_project.bias")) {}

Tensor SemanticTokenEmbedding::forward(const Tensor& codes) const {
  // emb = Embedding(codes) -> (B, T, codebook_dim); conv expects (B, dim, T).
  Tensor emb = nn::embedding(codes, emb_weight_);          // (B, T, d)
  Tensor x = mx::transpose(emb, {0, 2, 1});                // (B, d, T)
  return nn::conv1d(x, proj_weight_, &proj_bias_);         // (B, out, T)
}

// ---- InterpolateRegulator --------------------------------------------------

InterpolateRegulator::InterpolateRegulator(const WeightStore& w,
                                           const std::string& prefix,
                                           int num_blocks, int groups)
    : w_(w), p_(prefix), num_blocks_(num_blocks), groups_(groups) {}

Tensor InterpolateRegulator::forward(const Tensor& x, int target_len) const {
  // content_in_proj: Linear over last dim (in_channels -> channels).
  Tensor cw = w_.get(p_ + "content_in_proj.weight");
  Tensor cb = w_.get(p_ + "content_in_proj.bias");
  Tensor h = nn::linear(x, cw, &cb);             // (B, T, channels)

  // interpolate(mode="nearest", size=target_len) over the time axis.
  h = mx::transpose(h, {0, 2, 1});               // (B, channels, T)
  h = nn::interpolate_nearest(h, target_len);    // (B, channels, target_len)

  // Sequential: [Conv1d(k=3,pad=1), GroupNorm, Mish] * num_blocks, then Conv1d(k=1).
  int idx = 0;
  for (int blk = 0; blk < num_blocks_; ++blk) {
    Tensor w0 = w_.get(p_ + "model." + std::to_string(idx) + ".weight");
    Tensor b0 = w_.get(p_ + "model." + std::to_string(idx) + ".bias");
    h = nn::conv1d(h, w0, &b0, /*stride=*/1, /*padding=*/1);
    idx++;
    Tensor gw = w_.get(p_ + "model." + std::to_string(idx) + ".weight");
    Tensor gb = w_.get(p_ + "model." + std::to_string(idx) + ".bias");
    h = nn::group_norm(h, groups_, &gw, &gb);
    idx++;
    h = nn::mish(h);  // model.<idx> is Mish (no params)
    idx++;
  }
  Tensor wf = w_.get(p_ + "model." + std::to_string(idx) + ".weight");
  Tensor bf = w_.get(p_ + "model." + std::to_string(idx) + ".bias");
  h = nn::conv1d(h, wf, &bf);                     // (B, out_channels, target_len)

  return mx::transpose(h, {0, 2, 1});            // (B, target_len, out_channels)
}

}  // namespace s2a
}  // namespace c4
