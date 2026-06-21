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

}  // namespace dit
}  // namespace c4
