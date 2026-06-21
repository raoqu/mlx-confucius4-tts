#include "c4tts/dit.h"

#include <cmath>

#include "c4tts/nn.h"
#include "mlx/mlx.h"

namespace c4 {
namespace dit {
namespace mx = mlx::core;

Tensor timestep_embedding(const Tensor& t, const Tensor& w0, const Tensor& b0,
                          const Tensor& w2, const Tensor& b2, int freq_embed_dim,
                          float scale) {
  // SinusPositionEmbedding(freq_embed_dim).forward(t, scale)
  const int half = freq_embed_dim / 2;
  const float k = std::log(10000.0f) / static_cast<float>(half);
  Tensor freqs = mx::exp(mx::multiply(mx::arange(static_cast<float>(half)),
                                      mx::array(-k)));  // (half,)
  Tensor emb = mx::multiply(
      mx::multiply(mx::reshape(t, {t.shape(0), 1}), mx::array(scale)),
      mx::reshape(freqs, {1, half}));  // (B, half)
  Tensor sinus = mx::concatenate({mx::cos(emb), mx::sin(emb)}, 1);  // (B, 2*half)

  // time_mlp: Linear -> SiLU -> Linear
  Tensor h = nn::linear(sinus, w0, &b0);
  h = nn::silu(h);
  return nn::linear(h, w2, &b2);
}

Tensor adaptive_layer_norm(const Tensor& x, const Tensor& cond,
                           const Tensor& norm_weight, const Tensor& mod_w,
                           const Tensor& mod_b, float eps) {
  Tensor normed = nn::rms_norm(x, norm_weight, eps);   // (B, T, H)
  Tensor mod = nn::linear(cond, mod_w, &mod_b);        // (B, 2H)
  const int H = norm_weight.shape(0);
  auto parts = mx::split(mod, 2, /*axis=*/1);          // two (B, H)
  Tensor weight = mx::expand_dims(parts[0], 1);        // (B, 1, H)
  Tensor bias = mx::expand_dims(parts[1], 1);          // (B, 1, H)
  return mx::add(mx::multiply(normed, weight), bias);
}

Tensor feed_forward(const Tensor& x, const Tensor& w1, const Tensor& w2,
                    const Tensor& w3) {
  Tensor gate = nn::silu(nn::linear(x, w1));
  Tensor up = nn::linear(x, w3);
  return nn::linear(mx::multiply(gate, up), w2);
}

Tensor precompute_freqs_cis(int seq_len, int head_dim, float base) {
  const int half = head_dim / 2;
  // freqs = 1 / base^(arange(0,head_dim,2)[:half]/head_dim)
  Tensor exps = mx::divide(
      mx::multiply(mx::arange(static_cast<float>(half)), mx::array(2.0f)),
      mx::array(static_cast<float>(head_dim)));
  Tensor freqs = mx::divide(mx::array(1.0f), mx::power(mx::array(base), exps));
  Tensor t = mx::arange(static_cast<float>(seq_len));
  Tensor outer = mx::multiply(mx::reshape(t, {seq_len, 1}),
                              mx::reshape(freqs, {1, half}));  // (seq_len, half)
  Tensor cosv = mx::cos(outer);
  Tensor sinv = mx::sin(outer);
  // stack([real, imag], -1) -> (seq_len, half, 2)
  return mx::stack({cosv, sinv}, 2);
}

Tensor apply_rotary_emb(const Tensor& x_in, const Tensor& freqs_cis) {
  // x: (B, T, H, head_dim) -> reshape (..., head_dim/2, 2)
  Tensor x = mx::astype(x_in, mx::float32);
  const int B = x.shape(0), T = x.shape(1), Hh = x.shape(2), Dh = x.shape(3);
  const int half = Dh / 2;
  Tensor xs = mx::reshape(x, {B, T, Hh, half, 2});
  auto xparts = mx::split(xs, 2, /*axis=*/4);     // two (B,T,H,half,1)
  Tensor x0 = mx::squeeze(xparts[0], 4);          // (B,T,H,half)
  Tensor x1 = mx::squeeze(xparts[1], 4);

  Tensor f = mx::reshape(freqs_cis, {1, T, 1, half, 2});
  auto fparts = mx::split(f, 2, /*axis=*/4);
  Tensor f0 = mx::squeeze(fparts[0], 4);          // (1,T,1,half)
  Tensor f1 = mx::squeeze(fparts[1], 4);

  Tensor o0 = mx::subtract(mx::multiply(x0, f0), mx::multiply(x1, f1));
  Tensor o1 = mx::add(mx::multiply(x1, f0), mx::multiply(x0, f1));
  Tensor out = mx::stack({o0, o1}, 4);            // (B,T,H,half,2)
  out = mx::reshape(out, {B, T, Hh, Dh});
  return mx::astype(out, x_in.dtype());
}

Tensor attention(const Tensor& x, const Tensor& wqkv, const Tensor& wo,
                 const Tensor& freqs_cis, int num_heads, const Tensor* mask) {
  const int B = x.shape(0), T = x.shape(1), dim = x.shape(2);
  const int head_dim = dim / num_heads;

  Tensor qkv = nn::linear(x, wqkv);          // (B, T, 3*dim)
  auto parts = mx::split(qkv, 3, /*axis=*/2);
  Tensor q = mx::reshape(parts[0], {B, T, num_heads, head_dim});
  Tensor k = mx::reshape(parts[1], {B, T, num_heads, head_dim});
  Tensor v = mx::reshape(parts[2], {B, T, num_heads, head_dim});

  q = apply_rotary_emb(q, freqs_cis);
  k = apply_rotary_emb(k, freqs_cis);

  q = mx::transpose(q, {0, 2, 1, 3});        // (B, H, T, head_dim)
  k = mx::transpose(k, {0, 2, 1, 3});
  v = mx::transpose(v, {0, 2, 1, 3});

  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  Tensor scores = mx::multiply(
      mx::matmul(q, mx::swapaxes(k, -1, -2)), mx::array(scale));  // (B,H,T,T)
  if (mask) {
    // mask (B,1,1,T) bool, True=attend; masked positions -> large negative.
    Tensor neg = mx::array(-1e9f);
    scores = mx::where(*mask, scores, neg);
  }
  Tensor attn = mx::softmax(scores, /*axis=*/-1);
  Tensor out = mx::matmul(attn, v);          // (B, H, T, head_dim)
  out = mx::reshape(mx::transpose(out, {0, 2, 1, 3}), {B, T, dim});
  return nn::linear(out, wo);
}

DiTBlock::DiTBlock(const WeightStore& w, const std::string& prefix,
                   int num_heads)
    : w_(w), p_(prefix), num_heads_(num_heads) {}

Tensor DiTBlock::forward(const Tensor& x_in, const Tensor& cond,
                         const Tensor& freqs_cis, const Tensor* mask,
                         const Tensor* skip_in) const {
  Tensor x = x_in;
  if (skip_in) {
    Tensor sw = w_.get(p_ + "skip_in_linear.weight");
    Tensor sb = w_.get(p_ + "skip_in_linear.bias");
    x = nn::linear(mx::concatenate({x, *skip_in}, -1), sw, &sb);
  }

  // h = x + attention(attention_norm(x, cond))
  Tensor an_w = w_.get(p_ + "attention_norm.norm.weight");
  Tensor an_mw = w_.get(p_ + "attention_norm.modulation.weight");
  Tensor an_mb = w_.get(p_ + "attention_norm.modulation.bias");
  Tensor normed = adaptive_layer_norm(x, cond, an_w, an_mw, an_mb);
  Tensor wqkv = w_.get(p_ + "attention.wqkv.weight");
  Tensor wo = w_.get(p_ + "attention.wo.weight");
  Tensor h = mx::add(x, attention(normed, wqkv, wo, freqs_cis, num_heads_, mask));

  // return h + feed_forward(ffn_norm(h, cond))
  Tensor fn_w = w_.get(p_ + "ffn_norm.norm.weight");
  Tensor fn_mw = w_.get(p_ + "ffn_norm.modulation.weight");
  Tensor fn_mb = w_.get(p_ + "ffn_norm.modulation.bias");
  Tensor fnormed = adaptive_layer_norm(h, cond, fn_w, fn_mw, fn_mb);
  Tensor w1 = w_.get(p_ + "feed_forward.w1.weight");
  Tensor w2 = w_.get(p_ + "feed_forward.w2.weight");
  Tensor w3 = w_.get(p_ + "feed_forward.w3.weight");
  return mx::add(h, feed_forward(fnormed, w1, w2, w3));
}

Tensor final_layer(const Tensor& x, const Tensor& c, const Tensor& lin_w,
                   const Tensor& lin_b, const Tensor& mod_w,
                   const Tensor& mod_b) {
  // adaLN_modulation = SiLU -> Linear; chunk(2) -> shift, scale.
  Tensor mod = nn::linear(nn::silu(c), mod_w, &mod_b);  // (B, 2H)
  auto parts = mx::split(mod, 2, /*axis=*/1);
  Tensor shift = mx::expand_dims(parts[0], 1);          // (B,1,H)
  Tensor scale = mx::expand_dims(parts[1], 1);
  Tensor normed = nn::layer_norm(x, nullptr, nullptr, /*eps=*/1e-6f);
  Tensor y = mx::add(mx::multiply(normed, mx::add(mx::array(1.0f), scale)), shift);
  return nn::linear(y, lin_w, &lin_b);
}

WaveNet::WaveNet(const WeightStore& w, const std::string& prefix, int hidden,
                 int n_layers, int kernel_size, int dilation_rate)
    : w_(w),
      p_(prefix),
      hidden_(hidden),
      n_layers_(n_layers),
      kernel_(kernel_size),
      dil_rate_(dilation_rate) {}

Tensor WaveNet::forward(const Tensor& x_in, const Tensor& x_mask,
                        const Tensor& g_in) const {
  Tensor x = x_in;
  Tensor output = mx::zeros_like(x);

  // cond_layer projects g to all layers at once.
  Tensor cw = w_.get(p_ + "cond_layer.weight");
  Tensor cb = w_.get(p_ + "cond_layer.bias");
  Tensor g = nn::conv1d(g_in, cw, &cb);  // (B, 2*hidden*n_layers, 1)

  for (int i = 0; i < n_layers_; ++i) {
    const int dilation =
        static_cast<int>(std::pow(static_cast<double>(dil_rate_), i));
    const int padding = (kernel_ * dilation - dilation) / 2;
    const std::string ip = p_ + "in_layers." + std::to_string(i) + ".";
    Tensor iw = w_.get(ip + "weight");
    Tensor ib = w_.get(ip + "bias");
    Tensor x_l = nn::conv1d(x, iw, &ib, 1, padding, dilation);  // (B, 2h, T)

    const int off = i * 2 * hidden_;
    Tensor g_l = mx::slice(g, mx::Shape{0, off, 0},
                           mx::Shape{g.shape(0), off + 2 * hidden_, g.shape(2)});
    // fused_add_tanh_sigmoid_multiply
    Tensor in_act = mx::add(x_l, g_l);  // broadcasts (B,2h,1) over T
    Tensor t_act = mx::tanh(
        mx::slice(in_act, mx::Shape{0, 0, 0}, mx::Shape{in_act.shape(0), hidden_, in_act.shape(2)}));
    Tensor s_act = mx::sigmoid(mx::slice(
        in_act, mx::Shape{0, hidden_, 0}, mx::Shape{in_act.shape(0), 2 * hidden_, in_act.shape(2)}));
    Tensor acts = mx::multiply(t_act, s_act);  // (B, h, T)

    const std::string rp = p_ + "res_skip_layers." + std::to_string(i) + ".";
    Tensor rw = w_.get(rp + "weight");
    Tensor rb = w_.get(rp + "bias");
    Tensor res_skip = nn::conv1d(acts, rw, &rb);

    if (i < n_layers_ - 1) {
      Tensor res = mx::slice(res_skip, mx::Shape{0, 0, 0},
                             mx::Shape{res_skip.shape(0), hidden_, res_skip.shape(2)});
      x = mx::multiply(mx::add(x, res), x_mask);
      Tensor skip = mx::slice(res_skip, mx::Shape{0, hidden_, 0},
                              mx::Shape{res_skip.shape(0), 2 * hidden_, res_skip.shape(2)});
      output = mx::add(output, skip);
    } else {
      output = mx::add(output, res_skip);
    }
  }
  return mx::multiply(output, x_mask);
}

}  // namespace dit
}  // namespace c4
