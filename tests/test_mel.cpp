// Test: STFT log-mel parity (F4) against the PyTorch reference.
// Golden vectors from tools/dump_golden.py mel (under c4tts/golden/mel).

#include <cstdio>
#include <string>

#include "c4tts/audio.h"
#include "c4tts/npy.h"
#include "c4tts/tensor.h"

int main() {
  const std::string g = std::string(C4TTS_GOLDEN_DIR) + "/mel/";
  std::printf("test_mel:\n");

  c4::Tensor audio = c4::load_npy(g + "audio.npy");
  c4::Tensor mel_basis = c4::load_npy(g + "mel_basis.npy");
  c4::Tensor hann = c4::load_npy(g + "hann.npy");
  c4::Tensor ref = c4::load_npy(g + "mel_out.npy");

  c4::Tensor out = c4::mel_spectrogram(audio, mel_basis, hann);
  mlx::core::eval(out);

  auto r = c4::compare(out, ref);
  c4::print_report("mel_spectrogram", r);

  const bool shape_ok = out.ndim() == ref.ndim() && out.shape(0) == ref.shape(0) &&
                        out.shape(1) == ref.shape(1);
  // Log-domain mel: allow a slightly looser abs tol (FFT/accum order differs).
  const bool ok = shape_ok && c4::parity_ok_elementwise(r, 1e-3) &&
                  c4::parity_ok_deep(r);
  std::printf("test_mel: %s\n", ok ? "PASSED" : "FAILED");
  return ok ? 0 : 1;
}
