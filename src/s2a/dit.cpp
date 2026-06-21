#include <string>
#include <vector>

#include "c4tts/dit.h"
#include "c4tts/nn.h"
#include "mlx/mlx.h"

namespace c4 {
namespace dit {
namespace mx = mlx::core;

DiT::DiT(const WeightStore& w, const std::string& prefix, int hidden,
         int num_heads, int depth, int mel_dim, int wavenet_hidden,
         int wavenet_layers, int wavenet_kernel, int wavenet_dilation_rate,
         int spk_dim)
    : w_(w),
      p_(prefix),
      hidden_(hidden),
      num_heads_(num_heads),
      depth_(depth),
      mel_dim_(mel_dim),
      wn_hidden_(wavenet_hidden),
      wn_layers_(wavenet_layers),
      wn_kernel_(wavenet_kernel),
      wn_dil_(wavenet_dilation_rate),
      spk_dim_(spk_dim) {}

Tensor DiT::forward(const Tensor& x_in, const Tensor& mask, const Tensor& mu,
                    const Tensor& t, const Tensor& spks,
                    const Tensor& cond_in) const {
  const int B = x_in.shape(0), T = x_in.shape(2);

  // (B, mel, T) -> (B, T, mel)
  Tensor x = mx::transpose(x_in, {0, 2, 1});
  Tensor cond = mx::transpose(cond_in, {0, 2, 1});

  // Timestep embedding for the transformer (t1) and wavenet (t2).
  Tensor t1 = timestep_embedding(t, w_.get(p_ + "t_embedder.time_mlp.0.weight"),
                                 w_.get(p_ + "t_embedder.time_mlp.0.bias"),
                                 w_.get(p_ + "t_embedder.time_mlp.2.weight"),
                                 w_.get(p_ + "t_embedder.time_mlp.2.bias"));

  // InputEmbedding: mu_projection then concat [x, cond, mu_proj, spks] -> proj.
  Tensor mu_w = w_.get(p_ + "input_embed.mu_projection.weight");
  Tensor mu_b = w_.get(p_ + "input_embed.mu_projection.bias");
  Tensor mu_proj = nn::linear(mu, mu_w, &mu_b);
  std::vector<Tensor> to_cat = {x, cond, mu_proj};
  if (spk_dim_ > 0) {
    Tensor spks_seq = mx::broadcast_to(mx::expand_dims(spks, 1),
                                       {B, T, spk_dim_});  // (B, T, spk)
    to_cat.push_back(spks_seq);
  }
  Tensor pw = w_.get(p_ + "input_embed.proj.weight");
  Tensor pb = w_.get(p_ + "input_embed.proj.bias");
  Tensor h = nn::linear(mx::concatenate(to_cat, -1), pw, &pb);  // (B, T, hidden)

  // Attention mask (B,1,1,T) bool (mask supplied as float 0/1).
  Tensor attn_mask =
      mx::reshape(mx::greater(mask, mx::array(0.0f)), {B, 1, 1, T});

  const int head_dim = hidden_ / num_heads_;
  Tensor freqs_cis = precompute_freqs_cis(T, head_dim);

  // Transformer blocks with U-Net long skips: emit i<depth/2, receive i>depth/2.
  std::vector<Tensor> skip_stack;
  for (int i = 0; i < depth_; ++i) {
    DiTBlock block(w_, p_ + "transformer_blocks." + std::to_string(i) + ".",
                   num_heads_);
    const bool receive = (i > depth_ / 2);
    const bool emit = (i < depth_ / 2);
    if (receive && !skip_stack.empty()) {
      Tensor skip_in = skip_stack.back();
      skip_stack.pop_back();
      h = block.forward(h, t1, freqs_cis, &attn_mask, &skip_in);
    } else {
      h = block.forward(h, t1, freqs_cis, &attn_mask, nullptr);
    }
    if (emit) skip_stack.push_back(h);
  }

  // transformer_norm (AdaLN).
  Tensor x_res = adaptive_layer_norm(h, t1, w_.get(p_ + "transformer_norm.norm.weight"),
                                     w_.get(p_ + "transformer_norm.modulation.weight"),
                                     w_.get(p_ + "transformer_norm.modulation.bias"));

  // Long skip connection from the input embedding's x (B,T,mel).
  if (w_.has(p_ + "skip_linear.weight")) {
    Tensor slw = w_.get(p_ + "skip_linear.weight");
    Tensor slb = w_.get(p_ + "skip_linear.bias");
    x_res = nn::linear(mx::concatenate({x_res, x}, -1), slw, &slb);
  }

  // WaveNet final path.
  Tensor c1w = w_.get(p_ + "conv1.weight");      // Linear (wh, hidden)
  Tensor c1b = w_.get(p_ + "conv1.bias");
  Tensor x_out = nn::linear(x_res, c1w, &c1b);   // (B, T, wh)
  x_out = mx::transpose(x_out, {0, 2, 1});       // (B, wh, T)

  Tensor t2 = timestep_embedding(t, w_.get(p_ + "t_embedder2.time_mlp.0.weight"),
                                 w_.get(p_ + "t_embedder2.time_mlp.0.bias"),
                                 w_.get(p_ + "t_embedder2.time_mlp.2.weight"),
                                 w_.get(p_ + "t_embedder2.time_mlp.2.bias"));
  Tensor x_mask = mx::reshape(mask, {B, 1, T});  // (B,1,T)
  Tensor g = mx::expand_dims(t2, 2);             // (B, wh, 1)

  WaveNet wavenet(w_, p_ + "wavenet.", wn_hidden_, wn_layers_, wn_kernel_, wn_dil_);
  x_out = wavenet.forward(x_out, x_mask, g);     // (B, wh, T)
  x_out = mx::transpose(x_out, {0, 2, 1});       // (B, T, wh)

  Tensor rpw = w_.get(p_ + "res_projection.weight");
  Tensor rpb = w_.get(p_ + "res_projection.bias");
  x_out = mx::add(x_out, nn::linear(x_res, rpw, &rpb));  // (B, T, wh)

  x_out = final_layer(x_out, t1, w_.get(p_ + "final_layer.linear.weight"),
                      w_.get(p_ + "final_layer.linear.bias"),
                      w_.get(p_ + "final_layer.adaLN_modulation.1.weight"),
                      w_.get(p_ + "final_layer.adaLN_modulation.1.bias"));
  x_out = mx::transpose(x_out, {0, 2, 1});       // (B, wh, T)

  // conv2: Conv1d(wh, mel, 1)
  Tensor c2w = w_.get(p_ + "conv2.weight");
  Tensor c2b = w_.get(p_ + "conv2.bias");
  return nn::conv1d(x_out, c2w, &c2b);           // (B, mel, T)
}

}  // namespace dit
}  // namespace c4
