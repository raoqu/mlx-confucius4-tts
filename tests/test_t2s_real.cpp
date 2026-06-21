// Test: full Text2Semantic forward logits on the REAL checkpoint (T2S D-3).
// Requires: export_weights.py t2s ; dump_golden.py t2s_real. Skips if absent.

#include <cstdio>
#include <string>
#include <sys/stat.h>

#include "c4tts/npy.h"
#include "c4tts/t2s.h"
#include "c4tts/tensor.h"
#include "c4tts/weights.h"

namespace mx = mlx::core;

static bool exists(const std::string& p) {
  struct stat st {};
  return stat(p.c_str(), &st) == 0;
}

int main() {
  const std::string W = std::string(C4TTS_WEIGHTS_DIR) + "/t2s";
  const std::string G = std::string(C4TTS_GOLDEN_DIR) + "/t2s_real";
  std::printf("test_t2s_real:\n");
  if (!exists(W + "/semantic_head.weight.npy") || !exists(G + "/logits.npy")) {
    std::printf("  SKIP: export_weights.py t2s and dump_golden.py t2s_real\n");
    return 0;
  }

  c4::WeightStore w(W);
  c4::t2s::Text2Semantic model(w, /*num_layers=*/24, /*num_heads=*/20,
                               /*start=*/8192, /*stop=*/8193);

  c4::Tensor text_ids = c4::load_npy(G + "/text_ids.npy");
  c4::Tensor sem = c4::load_npy(G + "/semantic_codes.npy");
  c4::Tensor cond = c4::load_npy(G + "/condition_vector.npy");
  c4::Tensor ref = c4::load_npy(G + "/logits.npy");

  c4::Tensor logits = model.forward_logits(text_ids, sem, cond);
  mx::eval(logits);
  auto r = c4::compare(logits, ref);
  c4::print_report("Text2Semantic.logits (real)", r);

  const bool ok = logits.size() == ref.size() && c4::parity_ok_deep(r, 0.9999) &&
                  r.max_abs_err < 5e-3;
  std::printf("test_t2s_real: %s\n", ok ? "PASSED" : "FAILED");
  return ok ? 0 : 1;
}
