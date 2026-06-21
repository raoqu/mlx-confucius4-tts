// Test: BigVGAN anti-aliased SnakeBeta activation (F-1) parity vs PyTorch.
// Golden from tools/dump_golden.py vocoder_act.

#include <cstdio>
#include <string>

#include "c4tts/npy.h"
#include "c4tts/tensor.h"
#include "c4tts/vocoder.h"

namespace mx = mlx::core;

int main() {
  const std::string G = std::string(C4TTS_GOLDEN_DIR) + "/vocoder_act/";
  std::printf("test_vocoder:\n");

  c4::Tensor x = c4::load_npy(G + "x.npy");
  c4::Tensor alpha = c4::load_npy(G + "alpha.npy");
  c4::Tensor beta = c4::load_npy(G + "beta.npy");
  c4::Tensor up = c4::load_npy(G + "up_filter.npy");
  c4::Tensor down = c4::load_npy(G + "down_filter.npy");
  c4::Tensor ref = c4::load_npy(G + "out.npy");

  c4::Tensor out = c4::vocoder::anti_aliased_snakebeta(x, alpha, beta, up, down);
  mx::eval(out);
  auto r = c4::compare(out, ref);
  c4::print_report("anti_aliased_snakebeta", r);

  const bool ok = out.size() == ref.size() &&
                  c4::parity_ok_elementwise(r, 1e-4) && c4::parity_ok_deep(r);
  std::printf("test_vocoder: %s\n", ok ? "PASSED" : "FAILED");
  return ok ? 0 : 1;
}
