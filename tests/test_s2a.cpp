// Test: S2A E1 (SemanticTokenEmbedding) and E2 (InterpolateRegulator) parity.
// Golden from tools/dump_golden.py s2a_lr (seeded random instances).

#include <cstdio>
#include <string>

#include "c4tts/npy.h"
#include "c4tts/s2a.h"
#include "c4tts/tensor.h"
#include "c4tts/weights.h"

namespace mx = mlx::core;

static int failures = 0;

static void check(const std::string& name, const c4::Tensor& out,
                  const c4::Tensor& ref) {
  mx::eval(out);
  auto r = c4::compare(out, ref);
  c4::print_report(name, r);
  const bool shape_ok = out.size() == ref.size();
  if (!(shape_ok && c4::parity_ok_deep(r) &&
        c4::parity_ok_elementwise(r, 1e-3)))
    ++failures;
}

int main() {
  const std::string base = std::string(C4TTS_GOLDEN_DIR);
  std::printf("test_s2a:\n");

  // E1: SemanticTokenEmbedding
  {
    c4::WeightStore w(base + "/s2a_emb");
    c4::s2a::SemanticTokenEmbedding emb(w, "");
    c4::Tensor codes = c4::load_npy(base + "/s2a_emb/codes.npy");
    c4::Tensor ref = c4::load_npy(base + "/s2a_emb/out.npy");
    check("SemanticTokenEmbedding", emb.forward(codes), ref);
  }

  // E2: InterpolateRegulator
  {
    c4::WeightStore w(base + "/s2a_lr");
    c4::s2a::InterpolateRegulator lr(w, "", /*num_blocks=*/4, /*groups=*/1);
    c4::Tensor x = c4::load_npy(base + "/s2a_lr/x.npy");
    c4::Tensor tl = c4::load_npy(base + "/s2a_lr/target_len.npy");
    mx::eval(tl);
    const int target_len = tl.data<int32_t>()[0];
    c4::Tensor ref = c4::load_npy(base + "/s2a_lr/out.npy");
    check("InterpolateRegulator", lr.forward(x, target_len), ref);
  }

  std::printf("test_s2a: %s\n", failures == 0 ? "ALL PASSED" : "FAILED");
  return failures == 0 ? 0 : 1;
}
