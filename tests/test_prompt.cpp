// Test: full prompt-encoder path on REAL weights (Phase C integration).
//   wav16 -> seamless_features -> W2V-BERT[17] -> normalize  == semantic
//   wav16 -> kaldi fbank -> mean-sub -> CAMPPlus            == style
//
// Requires: export_weights.py w2vbert campplus ; dump_golden.py prompt.
// Skips (passes) if weights absent.

#include <cstdio>
#include <string>
#include <sys/stat.h>

#include "c4tts/audio.h"
#include "c4tts/campplus.h"
#include "c4tts/npy.h"
#include "c4tts/tensor.h"
#include "c4tts/w2vbert.h"
#include "c4tts/weights.h"

namespace mx = mlx::core;

static bool exists(const std::string& p) {
  struct stat st {};
  return stat(p.c_str(), &st) == 0;
}

int main() {
  const std::string W = std::string(C4TTS_WEIGHTS_DIR);
  const std::string G = std::string(C4TTS_GOLDEN_DIR) + "/prompt";
  std::printf("test_prompt:\n");
  if (!exists(W + "/w2vbert/feature_projection.projection.weight.npy") ||
      !exists(G + "/semantic.npy")) {
    std::printf("  SKIP: export w2vbert/campplus weights and dump_golden.py prompt\n");
    return 0;
  }

  c4::Tensor wav = c4::load_npy(G + "/wav16.npy");
  int failures = 0;

  // --- semantic features ---
  {
    c4::WeightStore w2v(W + "/w2vbert");
    c4::W2VBert m(w2v, "", 16, 64, 8, 31);
    c4::Tensor feats = c4::seamless_features(wav);          // (T, 160)
    feats = mx::expand_dims(feats, 0);                      // (1, T, 160)
    c4::Tensor h = m.forward(feats, 17);                    // (1, T, 1024)
    c4::Tensor mean = w2v.get("semantic_mean"), std = w2v.get("semantic_std");
    c4::Tensor sem = mx::divide(mx::subtract(h, mean), std);
    mx::eval(sem);
    auto r = c4::compare(sem, c4::load_npy(G + "/semantic.npy"));
    c4::print_report("semantic", r);
    if (!(c4::parity_ok_deep(r, 0.999) && r.max_abs_err < 5e-2)) ++failures;
  }

  // --- style embedding ---
  {
    c4::WeightStore camp(W + "/campplus");
    c4::CAMPPlus m(camp);
    c4::Tensor fb = c4::kaldi_fbank(wav, 80, 16000.0f);     // (m, 80)
    fb = mx::subtract(fb, mx::mean(fb, 0, /*keepdims=*/true));
    fb = mx::expand_dims(fb, 0);                            // (1, m, 80)
    c4::Tensor style = m.forward(fb);                       // (1, 192)
    mx::eval(style);
    auto r = c4::compare(style, c4::load_npy(G + "/style.npy"));
    c4::print_report("style", r);
    if (!(c4::parity_ok_deep(r, 0.999) && r.max_abs_err < 5e-2)) ++failures;
  }

  std::printf("test_prompt: %s\n", failures == 0 ? "PASSED" : "FAILED");
  return failures == 0 ? 0 : 1;
}
