// Test: DiT Attention + DiTBlock (E3b) parity vs PyTorch.
// Golden from tools/dump_golden.py dit_block.

#include <cstdio>
#include <string>

#include "c4tts/dit.h"
#include "c4tts/npy.h"
#include "c4tts/tensor.h"
#include "c4tts/weights.h"

namespace mx = mlx::core;

static int failures = 0;
static const std::string G = std::string(C4TTS_GOLDEN_DIR) + "/dit_block";

static void check(const std::string& name, const c4::Tensor& out,
                  const c4::Tensor& ref) {
  mx::eval(out);
  auto r = c4::compare(out, ref);
  c4::print_report(name, r);
  if (!(out.size() == ref.size() && c4::parity_ok_elementwise(r, 1e-4)))
    ++failures;
}

int main() {
  std::printf("test_dit_block:\n");
  c4::WeightStore w(G);
  c4::Tensor fc = c4::load_npy(G + "/freqs_cis.npy");
  const int heads = 4;

  // Attention
  {
    c4::Tensor wqkv = w.get("attn.wqkv.weight");
    c4::Tensor wo = w.get("attn.wo.weight");
    c4::Tensor x = c4::load_npy(G + "/attn_x.npy");
    check("attention", c4::dit::attention(x, wqkv, wo, fc, heads),
          c4::load_npy(G + "/attn_out.npy"));
  }

  // DiTBlock
  {
    c4::dit::DiTBlock blk(w, "", heads);
    c4::Tensor x = c4::load_npy(G + "/blk_x.npy");
    c4::Tensor cond = c4::load_npy(G + "/blk_cond.npy");
    check("dit_block", blk.forward(x, cond, fc), c4::load_npy(G + "/blk_out.npy"));
  }

  std::printf("test_dit_block: %s\n", failures == 0 ? "ALL PASSED" : "FAILED");
  return failures == 0 ? 0 : 1;
}
