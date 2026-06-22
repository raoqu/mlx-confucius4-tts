#include <cstdlib>
#include <string>
#include <vector>

#include "c4tts/profile.h"
#include "c4tts/s2a.h"
#include "mlx/mlx.h"

namespace c4 {
namespace s2a {
namespace mx = mlx::core;

ConditionalCFM::ConditionalCFM(const WeightStore& w, const std::string& prefix,
                               int hidden, int num_heads, int depth, int mel_dim,
                               int wavenet_hidden, int wavenet_layers,
                               int wavenet_kernel, int wavenet_dilation_rate,
                               int spk_dim)
    : estimator_(w, prefix + "estimator.", hidden, num_heads, depth, mel_dim,
                 wavenet_hidden, wavenet_layers, wavenet_kernel,
                 wavenet_dilation_rate, spk_dim),
      mel_dim_(mel_dim) {}

Tensor ConditionalCFM::solve_euler(Tensor x, const Tensor& t_span,
                                   const Tensor& mask, const Tensor& prompt,
                                   const Tensor& mu, const Tensor& spks,
                                   float cfg_rate) const {
  const int B = x.shape(0), T = x.shape(2);
  const int prompt_len = prompt.shape(2);

  // prompt_x: prompt in [0,prompt_len), zeros after — same shape as x.
  Tensor prompt_x = prompt;
  if (prompt_len < T) {
    Tensor pad = mx::zeros({B, mel_dim_, T - prompt_len}, x.dtype());
    prompt_x = mx::concatenate({prompt, pad}, 2);
  }

  // Region keep-mask: 0 over the prompt span, 1 after (zeros the prompt region).
  std::vector<float> keep(static_cast<size_t>(T), 1.0f);
  for (int i = 0; i < prompt_len && i < T; ++i) keep[i] = 0.0f;
  float* kh = new float[T];
  for (int i = 0; i < T; ++i) kh[i] = keep[i];
  Tensor region_keep =
      mx::reshape(mx::array(static_cast<void*>(kh), mx::Shape{T}, mx::float32,
                            [](void* p) { delete[] static_cast<float*>(p); }),
                  {1, 1, T});

  x = mx::multiply(x, region_keep);  // x[..., :prompt_len] = 0

  // Read the integration grid to the host so we can step dt exactly as torch.
  Tensor t_eval = mx::astype(t_span, mx::float32);
  mx::eval(t_eval);
  const float* ts = t_eval.data<float>();
  const int n = t_eval.size();  // n_steps + 1

  Tensor zero_mu = mx::zeros_like(mu);
  Tensor zero_spks = mx::zeros_like(spks);
  Tensor zero_prompt = mx::zeros_like(prompt_x);

  // Velocity field v(x, t) with classifier-free guidance (one estimator call,
  // or two when CFG is on).
  auto velocity = [&](const Tensor& xc, float tc) {
    Tensor t_arr = mx::full(mx::Shape{B}, tc);  // (B,) current time
    Tensor v = estimator_.forward(xc, mask, mu, t_arr, spks, prompt_x);
    if (cfg_rate > 0) {
      // Combine conditional and unconditional fields.
      Tensor uncond_v =
          estimator_.forward(xc, mask, zero_mu, t_arr, zero_spks, zero_prompt);
      v = mx::subtract(mx::multiply(mx::array(1.0f + cfg_rate), v),
                       mx::multiply(mx::array(cfg_rate), uncond_v));
    }
    return v;
  };

  // Integrator. Default: first-order Euler. C4TTS_SOLVER=ab2 uses 2nd-order
  // Adams-Bashforth — still one velocity eval per step, but reuses the previous
  // step's velocity for higher accuracy, so it holds quality at far fewer steps
  // on the near-straight flow-matching path (target: Euler-25 quality at ~12
  // steps, ~2x). The schedule is uniform (linspace), so dt is constant.
  const char* solver = std::getenv("C4TTS_SOLVER");
  const bool ab2 = solver && std::string(solver) == "ab2";

  float t = ts[0];
  float dt = ts[1] - ts[0];
  bool have_prev = false;
  Tensor v_prev = x;  // placeholder; overwritten after the first step
  for (int step = 1; step < n; ++step) {
    prof::Timer st;
    Tensor v = velocity(x, t);
    // AB2: x += dt*(1.5 v - 0.5 v_prev); else Euler (also AB2's first step).
    Tensor step_v = (ab2 && have_prev)
        ? mx::subtract(mx::multiply(mx::array(1.5f), v),
                       mx::multiply(mx::array(0.5f), v_prev))
        : v;
    x = mx::add(x, mx::multiply(mx::array(dt), step_v));
    v_prev = v;
    have_prev = true;
    t += dt;
    if (step < n - 1) dt = ts[step + 1] - t;
    x = mx::multiply(x, region_keep);  // re-zero prompt region each step
    mx::eval(x);  // bound the lazy graph across ODE steps
    prof::lap_cpu(cfg_rate > 0 ? "s2a.solve_step(cfg2)" : "s2a.solve_step", st);
  }
  return mx::astype(x, mx::float32);
}

}  // namespace s2a
}  // namespace c4
