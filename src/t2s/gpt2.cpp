#include "c4tts/t2s.h"

#include <cmath>
#include <cstdlib>

#include "c4tts/nn.h"
#include "mlx/mlx.h"

namespace c4 {
namespace t2s {
namespace mx = mlx::core;

Tensor gelu_new(const Tensor& x) {
  // 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
  const float c = 0.7978845608028654f;  // sqrt(2/pi)
  // x^3 via two multiplies (cheaper than the general exp/log-based power, and
  // this runs on the 5120-wide MLP intermediate every layer every step).
  Tensor x3 = mx::multiply(mx::multiply(x, x), x);
  Tensor inner = mx::multiply(
      mx::array(c),
      mx::add(x, mx::multiply(mx::array(0.044715f), x3)));
  return mx::multiply(mx::multiply(mx::array(0.5f), x),
                      mx::add(mx::array(1.0f), mx::tanh(inner)));
}

GPT2::GPT2(const WeightStore& w, const std::string& prefix, int n_layer,
           int n_head)
    : w_(w), p_(prefix), n_layer_(n_layer), n_head_(n_head) {
  const char* v = std::getenv("C4TTS_FP16");
  fp16_ = v && *v && std::strcmp(v, "0") != 0;
  // C4TTS_QUANT=4|8 quantizes the projection weights (takes precedence over
  // fp16). The M=1 decode GEMVs are weight-bandwidth-bound, so 8-/4-bit weights
  // cut the dominant cost further than fp16.
  if (const char* q = std::getenv("C4TTS_QUANT")) {
    const int bits = std::atoi(q);
    if (bits == 4 || bits == 8) quant_bits_ = bits;
  }
  // Affine group size. 64 is MLX's well-supported default and measured best
  // for both widths here (group 32 was empirically *worse* for 4-bit on these
  // weights). Override with C4TTS_QUANT_GROUP (must divide the projection
  // in-features 1280 and 5120, e.g. 32/64/128).
  quant_group_ = 64;
  if (const char* g = std::getenv("C4TTS_QUANT_GROUP")) {
    const int gs = std::atoi(g);
    if (gs >= 32 && (1280 % gs == 0) && (5120 % gs == 0)) quant_group_ = gs;
  }
}

// GPT-2 Conv1D projection y = x @ W (+ b). Optionally quantized (int4/int8) or
// fp16; otherwise plain fp32. W is stored (in, out); quantized_matmul wants the
// transposed (out, in) weight (it computes x @ w.T), quantized along `in`.
Tensor GPT2::proj(const Tensor& x, const std::string& name) const {
  Tensor b = w_.get(name + ".bias");
  if (quant_bits_ > 0) {
    auto it = quant_cache_.find(name);
    if (it == quant_cache_.end()) {
      Tensor wt = mx::transpose(w_.get(name + ".weight"));  // (out, in)
      it = quant_cache_.emplace(name, mx::quantize(wt, quant_group_, quant_bits_)).first;
    }
    const std::vector<Tensor>& q = it->second;  // [w_q, scales, biases]
    Tensor y = mx::quantized_matmul(x, q[0], q[1], q[2], /*transpose=*/true,
                                    quant_group_, quant_bits_);
    return mx::add(y, b);
  }
  Tensor w = w_.get(name + ".weight");
  if (!fp16_) return mx::add(mx::matmul(x, w), b);
  auto it = half_cache_.find(name);
  if (it == half_cache_.end()) {
    it = half_cache_.emplace(name, mx::astype(w, mx::float16)).first;
  }
  Tensor y = mx::matmul(mx::astype(x, mx::float16), it->second);  // fp16 GEMV
  return mx::add(mx::astype(y, mx::float32), b);                  // back to fp32
}

Tensor GPT2::block(const Tensor& x_in, const std::string& p, KVCache* cache,
                   int layer, int past_len, const Tensor* attn_mask) const {
  const int B = x_in.shape(0), T = x_in.shape(1), D = x_in.shape(2);
  const int head_dim = D / n_head_;

  // --- attention ---
  Tensor ln1w = w_.get(p + "ln_1.weight"), ln1b = w_.get(p + "ln_1.bias");
  Tensor a = nn::layer_norm(x_in, &ln1w, &ln1b);

  Tensor qkv = proj(a, p + "attn.c_attn");  // (B, T, 3D)
  auto parts = mx::split(qkv, 3, 2);
  auto to_heads = [&](const Tensor& t) {
    return mx::transpose(mx::reshape(t, {B, T, n_head_, head_dim}), {0, 2, 1, 3});
  };
  Tensor q = to_heads(parts[0]);  // (B, H, T, hd)
  Tensor k = to_heads(parts[1]);
  Tensor v = to_heads(parts[2]);

  // Prepend cached keys/values, then update the cache with the full K/V.
  if (cache) {
    if (past_len > 0) {
      k = mx::concatenate({cache->k[layer], k}, 2);
      v = mx::concatenate({cache->v[layer], v}, 2);
    }
    cache->k[layer] = k;
    cache->v[layer] = v;
  }
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  // Fused attention (one Metal kernel instead of scores-matmul + explicit
  // mask build + softmax + context-matmul). With no explicit mask, "causal"
  // aligns the T queries to the tail of the T_tot keys (correct for both
  // full-sequence prefill and cached decode). An explicit additive mask
  // (batched generation, to mask left-padding) overrides causal mode.
  Tensor ctx = attn_mask
      ? mx::fast::scaled_dot_product_attention(q, k, v, scale, "", *attn_mask)
      : mx::fast::scaled_dot_product_attention(q, k, v, scale, "causal");
  ctx = mx::reshape(mx::transpose(ctx, {0, 2, 1, 3}), {B, T, D});

  Tensor x = mx::add(x_in, proj(ctx, p + "attn.c_proj"));

  // --- mlp ---
  Tensor ln2w = w_.get(p + "ln_2.weight"), ln2b = w_.get(p + "ln_2.bias");
  Tensor m = nn::layer_norm(x, &ln2w, &ln2b);
  Tensor h = gelu_new(proj(m, p + "mlp.c_fc"));
  h = proj(h, p + "mlp.c_proj");
  return mx::add(x, h);
}

Tensor GPT2::forward(const Tensor& inputs_embeds) const {
  Tensor x = inputs_embeds;
  for (int i = 0; i < n_layer_; ++i) {
    x = block(x, p_ + "h." + std::to_string(i) + ".", nullptr, i, 0);
  }
  Tensor lfw = w_.get(p_ + "ln_f.weight"), lfb = w_.get(p_ + "ln_f.bias");
  return nn::layer_norm(x, &lfw, &lfb);
}

Tensor GPT2::forward(const Tensor& inputs_embeds, KVCache& cache,
                     int past_len, const Tensor* attn_mask) const {
  if (cache.empty()) {
    cache.k.assign(n_layer_, inputs_embeds);  // placeholder; overwritten below
    cache.v.assign(n_layer_, inputs_embeds);
  }
  Tensor x = inputs_embeds;
  for (int i = 0; i < n_layer_; ++i) {
    x = block(x, p_ + "h." + std::to_string(i) + ".", &cache, i, past_len, attn_mask);
  }
  Tensor lfw = w_.get(p_ + "ln_f.weight"), lfb = w_.get(p_ + "ln_f.bias");
  return nn::layer_norm(x, &lfw, &lfb);
}

}  // namespace t2s
}  // namespace c4
