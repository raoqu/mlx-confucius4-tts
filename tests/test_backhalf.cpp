// Test: back-half pipeline on REAL weights — S2A inference -> BigVGAN -> wav.
// Chains the two verified real-weight stages end to end.
// Requires: export_weights.py s2a bigvgan ; dump_golden.py s2a_infer.

#include <cstdio>
#include <string>
#include <sys/stat.h>

#include "c4tts/npy.h"
#include "c4tts/s2a.h"
#include "c4tts/tensor.h"
#include "c4tts/vocoder.h"
#include "c4tts/weights.h"

namespace mx = mlx::core;

static bool exists(const std::string& p) {
  struct stat st {};
  return stat(p.c_str(), &st) == 0;
}

int main() {
  const std::string W = std::string(C4TTS_WEIGHTS_DIR);
  const std::string G = std::string(C4TTS_GOLDEN_DIR) + "/s2a_infer";
  std::printf("test_backhalf:\n");
  if (!exists(W + "/s2a/prompt_cond.npy") || !exists(W + "/bigvgan/conv_pre.weight.npy") ||
      !exists(G + "/wav.npy")) {
    std::printf("  SKIP: export s2a+bigvgan weights and dump_golden.py s2a_infer\n");
    return 0;
  }

  c4::WeightStore sw(W + "/s2a"), vw(W + "/bigvgan");
  c4::s2a::MaskedDiffWithXvec s2a(sw);
  c4::vocoder::BigVGAN bigvgan(vw);

  c4::Tensor codes = c4::load_npy(G + "/codes.npy");
  c4::Tensor lm = c4::load_npy(G + "/lm_latent.npy");
  c4::Tensor pf = c4::load_npy(G + "/prompt_feat.npy");
  c4::Tensor emb = c4::load_npy(G + "/embedding.npy");
  c4::Tensor z = c4::load_npy(G + "/z.npy");
  c4::Tensor tl = c4::load_npy(G + "/target_len.npy");
  mx::eval(tl);
  const int target_len = tl.data<int32_t>()[0];

  c4::Tensor mel = s2a.inference(codes, lm, pf, emb, target_len, z, 4, 0.7f);
  c4::Tensor wav = bigvgan.forward(mel);
  mx::eval(wav);

  auto r = c4::compare(wav, c4::load_npy(G + "/wav.npy"));
  c4::print_report("S2A->BigVGAN wav", r);
  const bool ok = wav.size() == c4::load_npy(G + "/wav.npy").size() &&
                  c4::parity_ok_deep(r, 0.999) && r.max_abs_err < 5e-3;
  std::printf("test_backhalf: %s\n", ok ? "PASSED" : "FAILED");
  return ok ? 0 : 1;
}
