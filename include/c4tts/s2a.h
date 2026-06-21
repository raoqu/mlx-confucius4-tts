// S2A (semantic-to-acoustic) sub-modules for c4tts.
//
// Faithful C++/MLX ports of confuciustts/flow/. Each class loads its parameters
// from a WeightStore using the same names as the PyTorch state_dict, so weights
// exported with their original keys drop straight in. Verified by test_s2a.

#pragma once

#include <string>

#include "c4tts/dit.h"
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

// flow/flow_matching.py:ConditionalCFM — Euler ODE solver with classifier-free
// guidance over the DiT velocity estimator. The estimator's params live under
// "<prefix>estimator.".
class ConditionalCFM {
 public:
  ConditionalCFM(const WeightStore& w, const std::string& prefix, int hidden,
                 int num_heads, int depth, int mel_dim, int wavenet_hidden,
                 int wavenet_layers, int wavenet_kernel,
                 int wavenet_dilation_rate, int spk_dim);

  // Integrate from initial noise z to the target mel.
  //   z:      (B, mel_dim, T) initial noise
  //   t_span: (n_steps+1,) integration grid (already scheduler-transformed)
  //   mask:   (B, T) float padding mask
  //   prompt: (B, mel_dim, T_ref) reference mel (zeroed region is regenerated)
  //   mu:     (B, T, cond_dim) conditioning
  //   spks:   (B, spk_dim)
  // returns (B, mel_dim, T).
  Tensor solve_euler(Tensor z, const Tensor& t_span, const Tensor& mask,
                     const Tensor& prompt, const Tensor& mu, const Tensor& spks,
                     float cfg_rate) const;

 private:
  dit::DiT estimator_;
  int mel_dim_;
};

// flow/flow.py:MaskedDiffWithXvec — top-level S2A: semantic tokens + LM latent
// -> mel via length regulation, prompt conditioning, and the CFM ODE.
// Dimensions are fixed to the released config (hidden 512, depth 13, 8 heads,
// 80-band mel, 192-d speaker, wavenet 8 layers).
class MaskedDiffWithXvec {
 public:
  explicit MaskedDiffWithXvec(const WeightStore& w);

  // codes:      (B, T_sem) int   semantic token ids
  // lm_latent:  (B, T_sem, 1280) T2S hidden states
  // prompt_feat:(B, T_ref, 80)   reference mel
  // embedding:  (B, 192)         speaker style
  // target_len: number of mel frames to generate (excludes prompt)
  // z:          (B, 80, T_ref+target_len) initial noise (inject for determinism)
  // returns (B, 80, target_len).
  Tensor inference(const Tensor& codes, const Tensor& lm_latent,
                   const Tensor& prompt_feat, const Tensor& embedding,
                   int target_len, const Tensor& z, int n_timesteps,
                   float cfg_rate) const;

 private:
  const WeightStore& w_;
  SemanticTokenEmbedding input_embedding_;
  InterpolateRegulator length_regulator_;
  ConditionalCFM decoder_;
};

}  // namespace s2a
}  // namespace c4
