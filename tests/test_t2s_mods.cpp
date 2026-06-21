// Test: T2S D-2 modules (SpeakerEncoder ECAPA, TextEmbeddingProjector,
// LearnedPositionalEmbedding) parity vs PyTorch.
// Golden from tools/dump_golden.py t2s_mods.

#include <cstdio>
#include <string>

#include "c4tts/npy.h"
#include "c4tts/t2s.h"
#include "c4tts/tensor.h"
#include "c4tts/weights.h"

namespace mx = mlx::core;

static int failures = 0;
static const std::string G = std::string(C4TTS_GOLDEN_DIR) + "/t2s_mods";

static void check(const std::string& name, const c4::Tensor& out,
                  const c4::Tensor& ref, double max_abs = 1e-4) {
  mx::eval(out);
  auto r = c4::compare(out, ref);
  c4::print_report(name, r);
  if (!(out.size() == ref.size() && c4::parity_ok_elementwise(r, max_abs) &&
        c4::parity_ok_deep(r)))
    ++failures;
}

int main() {
  std::printf("test_t2s_mods:\n");
  c4::WeightStore w(G);

  {
    c4::t2s::SpeakerEncoder spk(w, "spk.");
    check("speaker_encoder", spk.forward(c4::load_npy(G + "/spk_x.npy")),
          c4::load_npy(G + "/spk_out.npy"), 1e-3);
  }
  {
    c4::t2s::TextEmbeddingProjector tp(w, "tp.");
    check("text_projector", tp.forward(c4::load_npy(G + "/tp_ids.npy")),
          c4::load_npy(G + "/tp_out.npy"));
  }
  {
    c4::Tensor table = c4::load_npy(G + "/pe_table.npy");
    check("learned_positions",
          c4::t2s::add_learned_positions(c4::load_npy(G + "/pe_x.npy"), table),
          c4::load_npy(G + "/pe_out.npy"));
  }

  std::printf("test_t2s_mods: %s\n", failures == 0 ? "ALL PASSED" : "FAILED");
  return failures == 0 ? 0 : 1;
}
