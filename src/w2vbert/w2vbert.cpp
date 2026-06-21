#include "c4tts/w2vbert.h"

#include <cmath>
#include <string>
#include <vector>

#include "c4tts/nn.h"
#include "mlx/einsum.h"
#include "mlx/mlx.h"

namespace c4 {
namespace mx = mlx::core;

W2VBert::W2VBert(const WeightStore& w, const std::string& prefix, int num_heads,
                 int left_max, int right_max, int conv_kernel)
    : w_(w),
      p_(prefix),
      num_heads_(num_heads),
      left_max_(left_max),
      right_max_(right_max),
      conv_kernel_(conv_kernel) {}

namespace {
Tensor ln(const WeightStore& w, const std::string& p, const Tensor& x) {
  Tensor g = w.get(p + "weight"), b = w.get(p + "bias");
  return nn::layer_norm(x, &g, &b, 1e-5f);
}
}  // namespace

// Wav2Vec2BertFeedForward: intermediate_dense -> swish -> output_dense.
Tensor W2VBert::feed_forward(const Tensor& x, const std::string& p) const {
  Tensor iw = w_.get(p + "intermediate_dense.weight"),
         ib = w_.get(p + "intermediate_dense.bias");
  Tensor h = nn::silu(nn::linear(x, iw, &ib));
  Tensor ow = w_.get(p + "output_dense.weight"),
         ob = w_.get(p + "output_dense.bias");
  return nn::linear(h, ow, &ob);
}

// relative_key self-attention.
Tensor W2VBert::self_attention(const Tensor& x, const std::string& p) const {
  const int B = x.shape(0), T = x.shape(1), D = x.shape(2);
  const int hd = D / num_heads_;
  auto proj = [&](const std::string& n) {
    Tensor wt = w_.get(p + n + ".weight"), b = w_.get(p + n + ".bias");
    return mx::transpose(mx::reshape(nn::linear(x, wt, &b), {B, T, num_heads_, hd}),
                         {0, 2, 1, 3});  // (B,H,T,hd)
  };
  Tensor q = proj("linear_q"), k = proj("linear_k"), v = proj("linear_v");

  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
  Tensor scores = mx::multiply(mx::matmul(q, mx::swapaxes(k, -1, -2)),
                               mx::array(scale));  // (B,H,T,T)

  // relative_key positional term: distance = clamp(r-l, -left, right) + left.
  std::vector<int32_t> dist(static_cast<size_t>(T) * T);
  for (int l = 0; l < T; ++l) {
    for (int r = 0; r < T; ++r) {
      int d = r - l;
      if (d < -left_max_) d = -left_max_;
      if (d > right_max_) d = right_max_;
      dist[l * T + r] = d + left_max_;
    }
  }
  int32_t* heap = new int32_t[dist.size()];
  for (size_t i = 0; i < dist.size(); ++i) heap[i] = dist[i];
  mx::array dist_idx(static_cast<void*>(heap), mx::Shape{T * T}, mx::int32,
                     [](void* qq) { delete[] static_cast<int32_t*>(qq); });
  Tensor pos = nn::embedding(dist_idx, w_.get(p + "distance_embedding.weight"));
  pos = mx::reshape(pos, {T, T, hd});  // (L, R, hd)
  // rel[b,h,l,r] = sum_d q[b,h,l,d] * pos[l,r,d]
  Tensor rel = mx::einsum("bhld,lrd->bhlr", {q, pos});
  scores = mx::add(scores, mx::multiply(rel, mx::array(scale)));

  Tensor probs = mx::softmax(scores, -1);
  Tensor ctx = mx::matmul(probs, v);  // (B,H,T,hd)
  ctx = mx::reshape(mx::transpose(ctx, {0, 2, 1, 3}), {B, T, D});
  Tensor ow = w_.get(p + "linear_out.weight"), ob = w_.get(p + "linear_out.bias");
  return nn::linear(ctx, ow, &ob);
}

// Conformer convolution module (GLU + causal depthwise conv).
Tensor W2VBert::conv_module(const Tensor& x, const std::string& p) const {
  Tensor h = ln(w_, p + "layer_norm.", x);     // (B,T,C)
  h = mx::transpose(h, {0, 2, 1});             // (B,C,T)

  // pointwise_conv1 (1x1, no bias) -> (B, 2C, T); GLU over channel.
  h = nn::conv1d(h, w_.get(p + "pointwise_conv1.weight"), nullptr);
  const int twoC = h.shape(1), C = twoC / 2, T = h.shape(2);
  Tensor a = mx::slice(h, mx::Shape{0, 0, 0}, mx::Shape{h.shape(0), C, T});
  Tensor b = mx::slice(h, mx::Shape{0, C, 0}, mx::Shape{h.shape(0), twoC, T});
  h = mx::multiply(a, mx::sigmoid(b));         // (B,C,T)

  // causal left-pad (kernel-1), depthwise conv (groups=C, no bias).
  h = mx::pad(h, std::vector<std::pair<int, int>>{{0, 0}, {0, 0}, {conv_kernel_ - 1, 0}},
              mx::array(0.0f), "constant");
  h = nn::conv1d(h, w_.get(p + "depthwise_conv.weight"), nullptr, 1, 0, 1, C);

  // depthwise_layer_norm over channel (applied in (B,T,C) layout).
  h = mx::transpose(h, {0, 2, 1});             // (B,T,C)
  h = ln(w_, p + "depthwise_layer_norm.", h);
  h = nn::silu(h);
  h = mx::transpose(h, {0, 2, 1});             // (B,C,T)

  h = nn::conv1d(h, w_.get(p + "pointwise_conv2.weight"), nullptr);
  return mx::transpose(h, {0, 2, 1});          // (B,T,C)
}

Tensor W2VBert::layer(const Tensor& x_in, int i) const {
  const std::string p = p_ + "encoder.layers." + std::to_string(i) + ".";

  // 1. half-step FFN1
  Tensor residual = x_in;
  Tensor h = ln(w_, p + "ffn1_layer_norm.", x_in);
  h = mx::add(mx::multiply(feed_forward(h, p + "ffn1."), mx::array(0.5f)), residual);

  // 2. self-attention
  residual = h;
  Tensor a = ln(w_, p + "self_attn_layer_norm.", h);
  h = mx::add(self_attention(a, p + "self_attn."), residual);

  // 3. convolution module
  residual = h;
  h = mx::add(residual, conv_module(h, p + "conv_module."));

  // 4. half-step FFN2 + final LN
  residual = h;
  Tensor f = ln(w_, p + "ffn2_layer_norm.", h);
  h = mx::add(mx::multiply(feed_forward(f, p + "ffn2."), mx::array(0.5f)), residual);
  return ln(w_, p + "final_layer_norm.", h);
}

Tensor W2VBert::forward(const Tensor& input_features, int num_layers) const {
  // feature_projection: LayerNorm(160) -> Linear(160, 1024)
  Tensor h = ln(w_, p_ + "feature_projection.layer_norm.", input_features);
  Tensor pw = w_.get(p_ + "feature_projection.projection.weight");
  Tensor pb = w_.get(p_ + "feature_projection.projection.bias");
  h = nn::linear(h, pw, &pb);

  for (int i = 0; i < num_layers; ++i) h = layer(h, i);
  return h;
}

}  // namespace c4
