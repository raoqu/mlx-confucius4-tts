// Test: full S2A MaskedDiffWithXvec.inference (E5) parity vs PyTorch, using the
// REAL exported checkpoint.
//
// Requires:
//   python3 c4tts/tools/export_weights.py s2a      -> c4tts/weights/s2a
//   python3 c4tts/tools/dump_golden.py s2a_infer   -> c4tts/golden/s2a_infer
// (Both produce gitignored artifacts.) Skips (passes) if weights are absent.

#include <cstdio>
#include <string>
#include <sys/stat.h>

#include "c4tts/npy.h"
#include "c4tts/s2a.h"
#include "c4tts/tensor.h"
#include "c4tts/weights.h"

namespace mx = mlx::core;

static bool exists(const std::string& p) {
  struct stat st {};
  return stat(p.c_str(), &st) == 0;
}

int main() {
  const std::string wdir = std::string(C4TTS_WEIGHTS_DIR) + "/s2a";
  const std::string G = std::string(C4TTS_GOLDEN_DIR) + "/s2a_infer";
  std::printf("test_s2a_inference:\n");

  if (!exists(wdir + "/prompt_cond.npy") || !exists(G + "/out.npy")) {
    std::printf("  SKIP: run export_weights.py s2a and dump_golden.py s2a_infer\n");
    return 0;
  }

  c4::WeightStore w(wdir);
  c4::s2a::MaskedDiffWithXvec model(w);

  c4::Tensor codes = c4::load_npy(G + "/codes.npy");
  c4::Tensor lm_latent = c4::load_npy(G + "/lm_latent.npy");
  c4::Tensor prompt_feat = c4::load_npy(G + "/prompt_feat.npy");
  c4::Tensor embedding = c4::load_npy(G + "/embedding.npy");
  c4::Tensor z = c4::load_npy(G + "/z.npy");
  c4::Tensor tl = c4::load_npy(G + "/target_len.npy");
  mx::eval(tl);
  const int target_len = tl.data<int32_t>()[0];
  c4::Tensor ref = c4::load_npy(G + "/out.npy");

  c4::Tensor mel = model.inference(codes, lm_latent, prompt_feat, embedding,
                                   target_len, z, /*n_timesteps=*/4, /*cfg=*/0.7f);
  mx::eval(mel);
  auto r = c4::compare(mel, ref);
  c4::print_report("MaskedDiffWithXvec.inference", r);

  // Real-weight deep stack (13-layer DiT x 8 passes): allow log-mel-scale tol.
  const bool ok = mel.size() == ref.size() && c4::parity_ok_deep(r, 0.9995) &&
                  r.max_abs_err < 5e-2;
  std::printf("test_s2a_inference: %s\n", ok ? "PASSED" : "FAILED");
  return ok ? 0 : 1;
}
