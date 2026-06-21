// DiT building blocks for c4tts (S2A estimator, flow/DiT/modules.py).
//
// Free functions implementing the math of the DiT sub-modules; the full
// DiTBlock/DiT assembly (loading from WeightStore) builds on these. Verified by
// test_dit.

#pragma once

#include "c4tts/tensor.h"

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

}  // namespace dit
}  // namespace c4
