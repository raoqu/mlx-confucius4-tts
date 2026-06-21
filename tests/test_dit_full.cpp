// Test: full DiT.forward (E3c-2) parity vs PyTorch.
// Golden from tools/dump_golden.py dit_full.

#include <cstdio>
#include <string>

#include "c4tts/dit.h"
#include "c4tts/npy.h"
#include "c4tts/tensor.h"
#include "c4tts/weights.h"

namespace mx = mlx::core;

int main() {
  const std::string G = std::string(C4TTS_GOLDEN_DIR) + "/dit_full";
  std::printf("test_dit_full:\n");

  c4::WeightStore w(G);
  // Matches the dumper config (hidden==wavenet_hidden==32).
  c4::dit::DiT dit(w, "", /*hidden=*/32, /*num_heads=*/4, /*depth=*/5,
                   /*mel_dim=*/8, /*wavenet_hidden=*/32, /*wavenet_layers=*/3,
                   /*wavenet_kernel=*/5, /*wavenet_dilation_rate=*/1,
                   /*spk_dim=*/6);

  c4::Tensor x = c4::load_npy(G + "/x.npy");
  c4::Tensor mask = c4::load_npy(G + "/mask.npy");
  c4::Tensor mu = c4::load_npy(G + "/mu.npy");
  c4::Tensor t = c4::load_npy(G + "/t.npy");
  c4::Tensor spks = c4::load_npy(G + "/spks.npy");
  c4::Tensor cond = c4::load_npy(G + "/cond.npy");
  c4::Tensor ref = c4::load_npy(G + "/out.npy");

  c4::Tensor out = dit.forward(x, mask, mu, t, spks, cond);
  mx::eval(out);
  auto r = c4::compare(out, ref);
  c4::print_report("DiT.forward", r);

  const bool ok = out.size() == ref.size() && c4::parity_ok_deep(r) &&
                  c4::parity_ok_elementwise(r, 1e-3);
  std::printf("test_dit_full: %s\n", ok ? "PASSED" : "FAILED");
  return ok ? 0 : 1;
}
