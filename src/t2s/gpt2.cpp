#include "c4tts/t2s.h"

#include <cmath>

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

namespace {
// GPT-2 Conv1D: weight stored (in, out); y = x @ W + b.
Tensor conv1d_gpt(const Tensor& x, const Tensor& w, const Tensor& b) {
  return mx::add(mx::matmul(x, w), b);
}
}  // namespace

GPT2::GPT2(const WeightStore& w, const std::string& prefix, int n_layer,
           int n_head)
    : w_(w), p_(prefix), n_layer_(n_layer), n_head_(n_head) {}

Tensor GPT2::block(const Tensor& x_in, const std::string& p, KVCache* cache,
                   int layer, int past_len) const {
  const int B = x_in.shape(0), T = x_in.shape(1), D = x_in.shape(2);
  const int head_dim = D / n_head_;

  // --- attention ---
  Tensor ln1w = w_.get(p + "ln_1.weight"), ln1b = w_.get(p + "ln_1.bias");
  Tensor a = nn::layer_norm(x_in, &ln1w, &ln1b);

  Tensor caw = w_.get(p + "attn.c_attn.weight"), cab = w_.get(p + "attn.c_attn.bias");
  Tensor qkv = conv1d_gpt(a, caw, cab);  // (B, T, 3D)
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
  const int T_tot = k.shape(2);

  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  Tensor scores = mx::multiply(mx::matmul(q, mx::swapaxes(k, -1, -2)),
                               mx::array(scale));  // (B,H,T,T_tot)
  // Causal mask: query i (absolute pos past_len+i) may attend to key j<=that.
  Tensor rows = mx::add(mx::reshape(mx::arange(T), {T, 1}), mx::array(past_len));
  Tensor cols = mx::reshape(mx::arange(T_tot), {1, T_tot});
  Tensor causal = mx::less_equal(cols, rows);  // (T, T_tot)
  scores = mx::where(causal, scores, mx::array(-1e9f));
  Tensor attn = mx::softmax(scores, -1);
  Tensor ctx = mx::matmul(attn, v);  // (B,H,T,hd)
  ctx = mx::reshape(mx::transpose(ctx, {0, 2, 1, 3}), {B, T, D});

  Tensor cpw = w_.get(p + "attn.c_proj.weight"), cpb = w_.get(p + "attn.c_proj.bias");
  Tensor x = mx::add(x_in, conv1d_gpt(ctx, cpw, cpb));

  // --- mlp ---
  Tensor ln2w = w_.get(p + "ln_2.weight"), ln2b = w_.get(p + "ln_2.bias");
  Tensor m = nn::layer_norm(x, &ln2w, &ln2b);
  Tensor fcw = w_.get(p + "mlp.c_fc.weight"), fcb = w_.get(p + "mlp.c_fc.bias");
  Tensor h = gelu_new(conv1d_gpt(m, fcw, fcb));
  Tensor mpw = w_.get(p + "mlp.c_proj.weight"), mpb = w_.get(p + "mlp.c_proj.bias");
  h = conv1d_gpt(h, mpw, mpb);
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
