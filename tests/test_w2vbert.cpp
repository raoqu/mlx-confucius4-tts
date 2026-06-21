// Test: W2V-BERT conformer (C1) parity vs HuggingFace Wav2Vec2BertModel.
// Golden from tools/dump_golden.py w2vbert (small seeded relative_key model).

#include <cstdio>
#include <string>

#include "c4tts/npy.h"
#include "c4tts/tensor.h"
#include "c4tts/w2vbert.h"
#include "c4tts/weights.h"

namespace mx = mlx::core;

static int failures = 0;
static const std::string G = std::string(C4TTS_GOLDEN_DIR) + "/w2vbert";

static void check(const std::string& name, const c4::Tensor& out,
                  const c4::Tensor& ref) {
  mx::eval(out);
  auto r = c4::compare(out, ref);
  c4::print_report(name, r);
  if (!(out.size() == ref.size() && c4::parity_ok_elementwise(r, 1e-3) &&
        c4::parity_ok_deep(r)))
    ++failures;
}

int main() {
  std::printf("test_w2vbert:\n");
  c4::WeightStore w(G);
  // relative_key: 4 heads, left_max 4, right_max 2, conv kernel 7.
  c4::W2VBert m(w, "", /*num_heads=*/4, /*left_max=*/4, /*right_max=*/2,
                /*conv_kernel=*/7);

  c4::Tensor feats = c4::load_npy(G + "/feats.npy");
  // forward(feats, k) == reference hidden_states[k].
  check("w2vbert_hs2", m.forward(feats, 2), c4::load_npy(G + "/hs2.npy"));
  check("w2vbert_last", m.forward(feats, 3), c4::load_npy(G + "/last.npy"));

  std::printf("test_w2vbert: %s\n", failures == 0 ? "ALL PASSED" : "FAILED");
  return failures == 0 ? 0 : 1;
}
