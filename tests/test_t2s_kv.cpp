// Test: KV-cached generate() is consistent with the (PyTorch-verified) full
// forward_logits() — greedy codes must be the argmax of the recomputed logits.
// Requires the real t2s weights + t2s_real golden inputs. Skips if absent.

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
  std::printf("test_t2s_kv:\n");
  if (!exists(W + "/semantic_head.weight.npy") || !exists(G + "/condition_vector.npy")) {
    std::printf("  SKIP: needs real t2s weights + t2s_real golden\n");
    return 0;
  }

  c4::WeightStore w(W);
  c4::t2s::Text2Semantic model(w, 24, 20, 8192, 8193);
  c4::Tensor text_ids = c4::load_npy(G + "/text_ids.npy");
  c4::Tensor cond = c4::load_npy(G + "/condition_vector.npy");

  // Greedy generation via the KV-cached path.
  auto gen = model.generate(text_ids, cond, /*max_new_tokens=*/24, /*sample=*/false);
  const int n = gen.codes.shape(1);
  if (n == 0) { std::printf("  SKIP: no tokens generated\n"); return 0; }

  // Recompute logits over [BOS, codes] with the full (non-cached) forward and
  // check greedy self-consistency: argmax(logits[:, i]) == codes[i].
  mx::eval(gen.codes);
  std::vector<int32_t> ids(1, 8192);  // BOS
  for (int i = 0; i < n; ++i) ids.push_back(gen.codes.data<int32_t>()[i]);
  auto* h = new int32_t[ids.size()];
  for (size_t i = 0; i < ids.size(); ++i) h[i] = ids[i];
  c4::Tensor sem(static_cast<void*>(h), mx::Shape{1, (int)ids.size()}, mx::int32,
                 [](void* p) { delete[] static_cast<int32_t*>(p); });

  c4::Tensor logits = model.forward_logits(text_ids, sem, cond);  // (1, n+1, vocab)
  c4::Tensor am = mx::argmax(logits, -1);  // (1, n+1)
  mx::eval(am);

  int mism = 0;
  for (int i = 0; i < n; ++i) {
    if (am.data<int32_t>()[i] != gen.codes.data<int32_t>()[i]) ++mism;
  }
  std::printf("  generated %d codes, %d/%d match recomputed argmax\n",
              n, n - mism, n);
  const bool ok = mism == 0;
  std::printf("test_t2s_kv: %s\n", ok ? "PASSED" : "FAILED");
  return ok ? 0 : 1;
}
