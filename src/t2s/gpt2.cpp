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
  Tensor inner = mx::multiply(
      mx::array(c),
      mx::add(x, mx::multiply(mx::array(0.044715f), mx::power(x, mx::array(3.0f)))));
  return mx::multiply(mx::multiply(mx::array(0.5f), x),
                      mx::add(mx::array(1.0f), mx::tanh(inner)));
}

GPT2::GPT2(const WeightStore& w, const std::string& prefix, int n_layer,
           int n_head)
    : w_(w), p_(prefix), n_layer_(n_layer), n_head_(n_head) {
  const char* v = std::getenv("C4TTS_FP16");
  fp16_ = v && *v && std::strcmp(v, "0") != 0;
}

// GPT-2 Conv1D projection y = x @ W (+ b), with optional fp16 matmul.
Tensor GPT2::proj(const Tensor& x, const std::string& name) const {
  Tensor w = w_.get(name + ".weight");
  Tensor b = w_.get(name + ".bias");
  if (!fp16_) return mx::add(mx::matmul(x, w), b);
  auto it = half_cache_.find(name);
  if (it == half_cache_.end()) {
    it = half_cache_.emplace(name, mx::astype(w, mx::float16)).first;
  }
  Tensor y = mx::matmul(mx::astype(x, mx::float16), it->second);  // fp16 GEMV
  return mx::add(mx::astype(y, mx::float32), b);                  // back to fp32
}

Tensor GPT2::block(const Tensor& x_in, const std::string& p, KVCache* cache,
                   int layer, int past_len) const {
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
  // mask build + softmax + context-matmul). "causal" aligns the T queries to
  // the tail of the T_tot keys, which is exactly right for both full-sequence
  // prefill (T == T_tot) and cached decode (T new queries at absolute
  // positions past_len .. past_len+T-1 attending to past_len + T keys).
  Tensor ctx = mx::fast::scaled_dot_product_attention(q, k, v, scale, "causal");
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
                     int past_len) const {
  if (cache.empty()) {
    cache.k.assign(n_layer_, inputs_embeds);  // placeholder; overwritten below
    cache.v.assign(n_layer_, inputs_embeds);
  }
  Tensor x = inputs_embeds;
  for (int i = 0; i < n_layer_; ++i) {
    x = block(x, p_ + "h." + std::to_string(i) + ".", &cache, i, past_len);
  }
  Tensor lfw = w_.get(p_ + "ln_f.weight"), lfb = w_.get(p_ + "ln_f.bias");
  return nn::layer_norm(x, &lfw, &lfb);
}

}  // namespace t2s
}  // namespace c4
