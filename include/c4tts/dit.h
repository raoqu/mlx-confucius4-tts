// DiT building blocks for c4tts (S2A estimator, flow/DiT/modules.py).
//
// Free functions implementing the math of the DiT sub-modules; the full
// DiTBlock/DiT assembly (loading from WeightStore) builds on these. Verified by
// test_dit.

#pragma once

#include <string>

#include "c4tts/tensor.h"
#include "c4tts/weights.h"

namespace c4 {
namespace dit {

// TimestepEmbedding: sinusoidal(freq_embed_dim) -> Linear -> SiLU -> Linear.
//   t: (B,) float timesteps -> (B, dim)
// w0/b0: time_mlp.0 (Linear freq_embed_dim->dim); w2/b2: time_mlp.2 (dim->dim).
Tensor timestep_embedding(const Tensor& t, const Tensor& w0, const Tensor& b0,
                          const Tensor& w2, const Tensor& b2,
                          int freq_embed_dim = 256, float scale = 1000.0f);

// AdaptiveLayerNorm: RMSNorm(x, eps) then modulate by Linear(cond)->(scale,shift).
//   x: (B, T, H), cond: (B, H). norm_weight: RMSNorm gain (H,).
//   mod_w/mod_b: modulation Linear (2H, H)/(2H,). eps defaults to AdaLN's 1e-5.
Tensor adaptive_layer_norm(const Tensor& x, const Tensor& cond,
                           const Tensor& norm_weight, const Tensor& mod_w,
                           const Tensor& mod_b, float eps = 1e-5f);

// SwiGLU FeedForward: w2(silu(w1 x) * w3 x). All Linear, no bias.
Tensor feed_forward(const Tensor& x, const Tensor& w1, const Tensor& w2,
                    const Tensor& w3);

// Rotary embedding tables: (seq_len, head_dim/2, 2) holding [cos, sin].
Tensor precompute_freqs_cis(int seq_len, int head_dim, float base = 10000.0f);

// Apply rotary embedding to x (B, T, H, head_dim) using freqs_cis (T, hd/2, 2).
// Computed in float32, matching flow/DiT/modules.py:apply_rotary_emb.
Tensor apply_rotary_emb(const Tensor& x, const Tensor& freqs_cis);

// Multi-head self-attention with RoPE (flow/DiT/modules.py:Attention).
//   x: (B, T, dim); wqkv: (3*dim, dim); wo: (dim, dim) (both no-bias).
//   freqs_cis: (T, head_dim/2, 2). mask: optional (B,1,1,T) bool (True=attend).
Tensor attention(const Tensor& x, const Tensor& wqkv, const Tensor& wo,
                 const Tensor& freqs_cis, int num_heads,
                 const Tensor* mask = nullptr);

// DiTBlock (flow/DiT/modules.py:DiTBlock): pre-AdaLN attention + SwiGLU FFN,
// with an optional U-Net skip input concatenated and projected first.
class DiTBlock {
 public:
  DiTBlock(const WeightStore& w, const std::string& prefix, int num_heads);
  Tensor forward(const Tensor& x, const Tensor& cond, const Tensor& freqs_cis,
                 const Tensor* mask = nullptr,
                 const Tensor* skip_in = nullptr) const;

 private:
  const WeightStore& w_;
  std::string p_;
  int num_heads_;
};

// FinalLayer (flow/DiT/modules.py:FinalLayer): LayerNorm (no affine, eps=1e-6)
// modulated by adaLN(SiLU->Linear)->(shift,scale), then Linear.
//   x: (B, T, H); c: (B, H).
//   ln has no params; lin_w/lin_b: linear (H,H)/(H,);
//   mod_w/mod_b: adaLN_modulation.1 Linear (2H,H)/(2H,).
Tensor final_layer(const Tensor& x, const Tensor& c, const Tensor& lin_w,
                   const Tensor& lin_b, const Tensor& mod_w, const Tensor& mod_b);

// WaveNet final stack (flow/wavenet.py:WN), weight_norm folded at export.
//   x: (B, hidden, T); x_mask: (B, 1, T); g: (B, gin, 1) conditioning.
// Convs are dilation=1, padding=2 for the default config (kernel 5).
class WaveNet {
 public:
  WaveNet(const WeightStore& w, const std::string& prefix, int hidden,
          int n_layers, int kernel_size, int dilation_rate);
  Tensor forward(const Tensor& x, const Tensor& x_mask, const Tensor& g) const;

 private:
  const WeightStore& w_;
  std::string p_;
  int hidden_, n_layers_, kernel_, dil_rate_;
};

}  // namespace dit
}  // namespace c4
