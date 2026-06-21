// Test: DiT building blocks (E3a) parity vs PyTorch.
// Golden from tools/dump_golden.py dit_mods.

#include <cstdio>
#include <string>

#include "c4tts/dit.h"
#include "c4tts/npy.h"
#include "c4tts/tensor.h"

namespace mx = mlx::core;

static int failures = 0;
static const std::string G = std::string(C4TTS_GOLDEN_DIR) + "/dit_mods/";
static c4::Tensor L(const std::string& n) { return c4::load_npy(G + n + ".npy"); }

static void check(const std::string& name, const c4::Tensor& out,
                  const c4::Tensor& ref) {
  mx::eval(out);
  auto r = c4::compare(out, ref);
  c4::print_report(name, r);
  if (!(out.size() == ref.size() && c4::parity_ok_elementwise(r, 1e-4)))
    ++failures;
}

int main() {
  std::printf("test_dit:\n");

  {
    c4::Tensor w0 = L("te.time_mlp.0.weight"), b0 = L("te.time_mlp.0.bias");
    c4::Tensor w2 = L("te.time_mlp.2.weight"), b2 = L("te.time_mlp.2.bias");
    check("timestep_embedding",
          c4::dit::timestep_embedding(L("te_t"), w0, b0, w2, b2), L("te_out"));
  }
  {
    c4::Tensor nw = L("aln.norm.weight"), mw = L("aln.modulation.weight"),
               mb = L("aln.modulation.bias");
    check("adaptive_layer_norm",
          c4::dit::adaptive_layer_norm(L("aln_x"), L("aln_cond"), nw, mw, mb),
          L("aln_out"));
  }
  {
    check("feed_forward",
          c4::dit::feed_forward(L("ff_x"), L("ff.w1.weight"), L("ff.w2.weight"),
                                L("ff.w3.weight")),
          L("ff_out"));
  }
  {
    // RoPE precompute table (T=7, head_dim=16) and application.
    check("precompute_freqs_cis", c4::dit::precompute_freqs_cis(7, 16),
          L("rope_fc"));
    check("apply_rotary_emb",
          c4::dit::apply_rotary_emb(L("rope_x"), L("rope_fc")), L("rope_out"));
  }

  std::printf("test_dit: %s\n", failures == 0 ? "ALL PASSED" : "FAILED");
  return failures == 0 ? 0 : 1;
}
