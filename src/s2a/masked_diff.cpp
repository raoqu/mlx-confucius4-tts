#include <vector>

#include "c4tts/nn.h"
#include "c4tts/profile.h"
#include "c4tts/s2a.h"
#include "mlx/mlx.h"

namespace c4 {
namespace s2a {
namespace mx = mlx::core;

// Released s2a config (config/inference_config.yaml -> s2a_model + DiT defaults).
namespace {
constexpr int kHidden = 512;
constexpr int kHeads = 8;
constexpr int kDepth = 13;
constexpr int kMel = 80;
constexpr int kWnLayers = 8;
constexpr int kWnKernel = 5;
constexpr int kWnDil = 1;
constexpr int kSpk = 192;
}  // namespace

MaskedDiffWithXvec::MaskedDiffWithXvec(const WeightStore& w)
    : w_(w),
      input_embedding_(w, "input_embedding."),
      length_regulator_(w, "length_regulator.", /*num_blocks=*/4, /*groups=*/1),
      decoder_(w, "decoder.", kHidden, kHeads, kDepth, kMel, kHidden, kWnLayers,
               kWnKernel, kWnDil, kSpk) {}

Tensor MaskedDiffWithXvec::inference(const Tensor& codes, const Tensor& lm_latent,
                                     const Tensor& prompt_feat,
                                     const Tensor& embedding, int target_len,
                                     const Tensor& z, int n_timesteps,
                                     float cfg_rate) const {
  const int B = codes.shape(0);
  const int T_ref = prompt_feat.shape(1);
  prof::Timer et;

  // Embed semantic tokens -> (B, T_sem, 1024); concat with LM latent; project.
  Tensor sem = mx::transpose(input_embedding_.forward(codes), {0, 2, 1});  // (B,T,1024)
  Tensor combined = mx::concatenate({lm_latent, sem}, -1);                 // (B,T,2304)
  Tensor epw = w_.get("encoder_proj.weight");
  Tensor epb = w_.get("encoder_proj.bias");
  Tensor text_cond = nn::linear(combined, epw, &epb);                      // (B,T,1024)

  // Upsample to target mel length.
  Tensor cond_target = length_regulator_.forward(text_cond, target_len);   // (B,target,512)

  // Prepend the learned prompt condition token, broadcast over the prompt span.
  Tensor prompt_cond = w_.get("prompt_cond");                              // (1,1,512)
  Tensor prompt_condition =
      mx::broadcast_to(prompt_cond, {B, T_ref, prompt_cond.shape(2)});
  Tensor cat_condition = mx::concatenate({prompt_condition, cond_target}, 1);
  const int T_total = T_ref + target_len;

  // Linear t schedule (config cfm_t_scheduler="linear").
  Tensor t_span = mx::linspace(0.0f, 1.0f, n_timesteps + 1);
  Tensor mask = mx::ones({B, T_total}, mx::float32);
  Tensor prompt = mx::transpose(prompt_feat, {0, 2, 1});                   // (B,80,T_ref)
  prof::lap("s2a.encoder", cat_condition, et);

  Tensor mel_full = decoder_.solve_euler(z, t_span, mask, prompt, cat_condition,
                                         embedding, cfg_rate);             // (B,80,T_total)

  // Drop the prompt portion.
  return mx::slice(mel_full, mx::Shape{0, 0, T_ref},
                   mx::Shape{B, kMel, T_total});
}

}  // namespace s2a
}  // namespace c4
