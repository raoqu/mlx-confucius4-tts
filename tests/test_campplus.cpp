// Test: CAMPPlus speaker encoder (C2) parity vs PyTorch (seeded model).
// Golden from tools/dump_golden.py campplus.

#include <cstdio>
#include <string>

#include "c4tts/campplus.h"
#include "c4tts/npy.h"
#include "c4tts/tensor.h"
#include "c4tts/weights.h"

namespace mx = mlx::core;

int main() {
  const std::string G = std::string(C4TTS_GOLDEN_DIR) + "/campplus";
  std::printf("test_campplus:\n");

  c4::WeightStore w(G);
  c4::CAMPPlus model(w);

  c4::Tensor fbank = c4::load_npy(G + "/fbank.npy");
  c4::Tensor ref = c4::load_npy(G + "/emb.npy");

  c4::Tensor emb = model.forward(fbank);
  mx::eval(emb);
  auto r = c4::compare(emb, ref);
  c4::print_report("CAMPPlus.forward", r);

  const bool ok = emb.size() == ref.size() &&
                  c4::parity_ok_elementwise(r, 1e-3) && c4::parity_ok_deep(r);
  std::printf("test_campplus: %s\n", ok ? "PASSED" : "FAILED");
  return ok ? 0 : 1;
}
