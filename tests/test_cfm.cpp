// Test: ConditionalCFM.solve_euler (E4) parity vs PyTorch.
// Golden from tools/dump_golden.py cfm.

#include <cstdio>
#include <string>

#include "c4tts/npy.h"
#include "c4tts/s2a.h"
#include "c4tts/tensor.h"
#include "c4tts/weights.h"

namespace mx = mlx::core;

int main() {
  const std::string G = std::string(C4TTS_GOLDEN_DIR) + "/cfm";
  std::printf("test_cfm:\n");

  c4::WeightStore w(G);
  c4::s2a::ConditionalCFM cfm(w, "", /*hidden=*/32, /*num_heads=*/4, /*depth=*/5,
                              /*mel_dim=*/8, /*wavenet_hidden=*/32,
                              /*wavenet_layers=*/3, /*wavenet_kernel=*/5,
                              /*wavenet_dilation_rate=*/1, /*spk_dim=*/6);

  c4::Tensor z = c4::load_npy(G + "/z.npy");
  c4::Tensor mu = c4::load_npy(G + "/mu.npy");
  c4::Tensor prompt = c4::load_npy(G + "/prompt.npy");
  c4::Tensor spks = c4::load_npy(G + "/spks.npy");
  c4::Tensor mask = c4::load_npy(G + "/mask.npy");
  c4::Tensor t_span = c4::load_npy(G + "/t_span.npy");
  c4::Tensor ref = c4::load_npy(G + "/out.npy");

  c4::Tensor out = cfm.solve_euler(z, t_span, mask, prompt, mu, spks, 0.7f);
  mx::eval(out);
  auto r = c4::compare(out, ref);
  c4::print_report("solve_euler", r);

  const bool ok = out.size() == ref.size() && c4::parity_ok_deep(r) &&
                  c4::parity_ok_elementwise(r, 1e-3);
  std::printf("test_cfm: %s\n", ok ? "PASSED" : "FAILED");
  return ok ? 0 : 1;
}
