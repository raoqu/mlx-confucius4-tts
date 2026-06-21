// Test: windowed-sinc resample (B1) parity vs torchaudio.functional.resample.
// Golden from tools/dump_golden.py resample.

#include <cstdio>
#include <string>

#include "c4tts/audio.h"
#include "c4tts/npy.h"
#include "c4tts/tensor.h"

namespace mx = mlx::core;

static int failures = 0;

static void check(const std::string& name, const c4::Tensor& out,
                  const c4::Tensor& ref) {
  mx::eval(out);
  auto r = c4::compare(out, ref);
  c4::print_report(name, r);
  if (!(out.size() == ref.size() && c4::parity_ok_deep(r) &&
        c4::parity_ok_elementwise(r, 1e-3)))
    ++failures;
}

int main() {
  const std::string G = std::string(C4TTS_GOLDEN_DIR) + "/resample/";
  std::printf("test_resample:\n");

  c4::Tensor wav = c4::load_npy(G + "wav.npy");
  check("resample_24k_16k", c4::resample(wav, 24000, 16000), c4::load_npy(G + "r16.npy"));
  check("resample_24k_22k", c4::resample(wav, 24000, 22050), c4::load_npy(G + "r22.npy"));

  std::printf("test_resample: %s\n", failures == 0 ? "ALL PASSED" : "FAILED");
  return failures == 0 ? 0 : 1;
}
