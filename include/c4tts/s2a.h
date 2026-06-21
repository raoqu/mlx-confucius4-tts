// S2A (semantic-to-acoustic) sub-modules for c4tts.
//
// Faithful C++/MLX ports of confuciustts/flow/. Each class loads its parameters
// from a WeightStore using the same names as the PyTorch state_dict, so weights
// exported with their original keys drop straight in. Verified by test_s2a.

#pragma once

#include <string>

#include "c4tts/tensor.h"
#include "c4tts/weights.h"

namespace c4 {
namespace s2a {

// flow/modules.py:SemanticTokenEmbedding
//   Embedding(codebook_size, codebook_dim) -> Conv1d(codebook_dim, output_dim, 1)
//   forward(codes (B,T) int) -> (B, output_dim, T)
class SemanticTokenEmbedding {
 public:
  SemanticTokenEmbedding(const WeightStore& w, const std::string& prefix);
  Tensor forward(const Tensor& codes) const;

 private:
  Tensor emb_weight_;       // (V, codebook_dim)
  Tensor proj_weight_;      // (output_dim, codebook_dim, 1)
  Tensor proj_bias_;        // (output_dim,)
};

// flow/length_regulator.py:InterpolateRegulator
//   content_in_proj: Linear(in_channels, channels)
//   model: [Conv1d(c,c,3,pad1), GroupNorm(groups,c), Mish] * len(ratios)
//          + Conv1d(c, out_channels, 1)
//   forward(x (B,T,in_channels), target_len) -> (B, target_len, out_channels)
// Mask handling is omitted for the single-sequence inference path (no padding).
class InterpolateRegulator {
 public:
  InterpolateRegulator(const WeightStore& w, const std::string& prefix,
                       int num_blocks = 4, int groups = 1);
  Tensor forward(const Tensor& x, int target_len) const;

 private:
  const WeightStore& w_;
  std::string p_;
  int num_blocks_;
  int groups_;
};

}  // namespace s2a
}  // namespace c4
