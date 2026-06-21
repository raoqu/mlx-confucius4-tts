// Test: Kaldi fbank (B3) parity vs torchaudio.compliance.kaldi.fbank.
// Golden from tools/dump_golden.py fbank.

#include <cstdio>
#include <string>

#include "c4tts/audio.h"
#include "c4tts/npy.h"
#include "c4tts/tensor.h"

namespace mx = mlx::core;

int main() {
  const std::string G = std::string(C4TTS_GOLDEN_DIR) + "/fbank/";
  std::printf("test_fbank:\n");

  c4::Tensor wav = c4::load_npy(G + "wav.npy");
  c4::Tensor ref = c4::load_npy(G + "fbank.npy");

  c4::Tensor fb = c4::kaldi_fbank(wav, 80, 16000.0f);
  mx::eval(fb);
  auto r = c4::compare(fb, ref);
  c4::print_report("kaldi_fbank", r);

  // Log-domain fbank; allow a slightly looser abs tol (FFT/accum order).
  const bool ok = fb.size() == ref.size() && c4::parity_ok_elementwise(r, 2e-3) &&
                  c4::parity_ok_deep(r);
  std::printf("test_fbank: %s\n", ok ? "PASSED" : "FAILED");
  return ok ? 0 : 1;
}
