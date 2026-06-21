// Test: full BigVGAN generator (F-2) parity vs PyTorch on the REAL checkpoint.
//
// Requires:
//   python3 c4tts/tools/export_weights.py bigvgan   -> c4tts/weights/bigvgan
//   python3 c4tts/tools/dump_golden.py bigvgan      -> c4tts/golden/bigvgan
// Skips (passes) if weights are absent.

#include <cstdio>
#include <string>
#include <sys/stat.h>

#include "c4tts/npy.h"
#include "c4tts/tensor.h"
#include "c4tts/vocoder.h"
#include "c4tts/weights.h"

namespace mx = mlx::core;

static bool exists(const std::string& p) {
  struct stat st {};
  return stat(p.c_str(), &st) == 0;
}

int main() {
  const std::string wdir = std::string(C4TTS_WEIGHTS_DIR) + "/bigvgan";
  const std::string G = std::string(C4TTS_GOLDEN_DIR) + "/bigvgan";
  std::printf("test_bigvgan:\n");

  if (!exists(wdir + "/conv_pre.weight.npy") || !exists(G + "/wav.npy")) {
    std::printf("  SKIP: run export_weights.py bigvgan and dump_golden.py bigvgan\n");
    return 0;
  }

  c4::WeightStore w(wdir);
  c4::vocoder::BigVGAN bigvgan(w);

  c4::Tensor mel = c4::load_npy(G + "/mel.npy");
  c4::Tensor ref = c4::load_npy(G + "/wav.npy");

  c4::Tensor wav = bigvgan.forward(mel);
  mx::eval(wav);
  auto r = c4::compare(wav, ref);
  c4::print_report("BigVGAN.forward", r);

  // Deep conv vocoder; waveform in [-1,1]. Tolerate small accumulation error.
  const bool ok = wav.size() == ref.size() && c4::parity_ok_deep(r, 0.9995) &&
                  r.max_abs_err < 2e-3;
  std::printf("test_bigvgan: %s\n", ok ? "PASSED" : "FAILED");
  return ok ? 0 : 1;
}
