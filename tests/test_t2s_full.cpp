// Test: full Text2Semantic forward logits (T2S D-3) parity vs PyTorch.
// Golden from tools/dump_golden.py t2s_full (small seeded model).

#include <cstdio>
#include <string>

#include "c4tts/npy.h"
#include "c4tts/t2s.h"
#include "c4tts/tensor.h"
#include "c4tts/weights.h"

namespace mx = mlx::core;

int main() {
  const std::string G = std::string(C4TTS_GOLDEN_DIR) + "/t2s_full";
  std::printf("test_t2s_full:\n");

  c4::WeightStore w(G);
  c4::t2s::Text2Semantic model(w, /*num_layers=*/2, /*num_heads=*/4,
                               /*start=*/38, /*stop=*/39);

  c4::Tensor text_ids = c4::load_npy(G + "/text_ids.npy");
  c4::Tensor sem = c4::load_npy(G + "/semantic_codes.npy");
  c4::Tensor cond = c4::load_npy(G + "/condition_vector.npy");
  c4::Tensor ref = c4::load_npy(G + "/logits.npy");

  c4::Tensor logits = model.forward_logits(text_ids, sem, cond);
  mx::eval(logits);
  auto r = c4::compare(logits, ref);
  c4::print_report("Text2Semantic.logits", r);

  const bool ok = logits.size() == ref.size() &&
                  c4::parity_ok_elementwise(r, 1e-3) && c4::parity_ok_deep(r);
  std::printf("test_t2s_full: %s\n", ok ? "PASSED" : "FAILED");
  return ok ? 0 : 1;
}
