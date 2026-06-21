// Test: GPT-2 transformer stack (T2S D-1) parity vs HuggingFace GPT2Model.
// Golden from tools/dump_golden.py gpt2 (position embedding zeroed).

#include <cstdio>
#include <string>

#include "c4tts/npy.h"
#include "c4tts/t2s.h"
#include "c4tts/tensor.h"
#include "c4tts/weights.h"

namespace mx = mlx::core;

int main() {
  const std::string G = std::string(C4TTS_GOLDEN_DIR) + "/gpt2";
  std::printf("test_t2s:\n");

  c4::WeightStore w(G);
  c4::t2s::GPT2 gpt(w, "", /*n_layer=*/3, /*n_head=*/4);

  c4::Tensor emb = c4::load_npy(G + "/emb.npy");
  c4::Tensor ref = c4::load_npy(G + "/out.npy");

  c4::Tensor out = gpt.forward(emb);
  mx::eval(out);
  auto r = c4::compare(out, ref);
  c4::print_report("GPT2.forward", r);

  const bool ok = out.size() == ref.size() &&
                  c4::parity_ok_elementwise(r, 1e-4) && c4::parity_ok_deep(r);
  std::printf("test_t2s: %s\n", ok ? "PASSED" : "FAILED");
  return ok ? 0 : 1;
}
