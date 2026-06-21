// Test: SeamlessM4T feature extractor (B4) parity vs transformers.
// Golden from tools/dump_golden.py seamless.

#include <cstdio>
#include <string>

#include "c4tts/audio.h"
#include "c4tts/npy.h"
#include "c4tts/tensor.h"

namespace mx = mlx::core;

int main() {
  const std::string G = std::string(C4TTS_GOLDEN_DIR) + "/seamless/";
  std::printf("test_seamless:\n");

  c4::Tensor wav = c4::load_npy(G + "wav.npy");
  c4::Tensor ref = c4::load_npy(G + "feats.npy");

  c4::Tensor feats = c4::seamless_features(wav);
  mx::eval(feats);
  auto r = c4::compare(feats, ref);
  c4::print_report("seamless_features", r);

  const bool ok = feats.size() == ref.size() && c4::parity_ok_deep(r, 0.9995) &&
                  r.max_abs_err < 5e-2;
  std::printf("test_seamless: %s\n", ok ? "PASSED" : "FAILED");
  return ok ? 0 : 1;
}
