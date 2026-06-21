// Test: WaveNet (WN) + FinalLayer (E3c-1) parity vs PyTorch.
// Golden from tools/dump_golden.py dit_wn (weight_norm folded).

#include <cstdio>
#include <string>

#include "c4tts/dit.h"
#include "c4tts/npy.h"
#include "c4tts/tensor.h"
#include "c4tts/weights.h"

namespace mx = mlx::core;

static int failures = 0;
static const std::string G = std::string(C4TTS_GOLDEN_DIR) + "/dit_wn";

static void check(const std::string& name, const c4::Tensor& out,
                  const c4::Tensor& ref) {
  mx::eval(out);
  auto r = c4::compare(out, ref);
  c4::print_report(name, r);
  if (!(out.size() == ref.size() && c4::parity_ok_elementwise(r, 1e-4)))
    ++failures;
}

int main() {
  std::printf("test_dit_wn:\n");
  c4::WeightStore w(G);

  // WaveNet (hidden=16, n_layers=3, kernel=5, dilation_rate=1)
  {
    c4::dit::WaveNet wn(w, "", /*hidden=*/16, /*n_layers=*/3, /*kernel=*/5,
                        /*dilation_rate=*/1);
    c4::Tensor x = c4::load_npy(G + "/wn_x.npy");
    c4::Tensor mask = c4::load_npy(G + "/wn_mask.npy");
    c4::Tensor g = c4::load_npy(G + "/wn_g.npy");
    check("wavenet", wn.forward(x, mask, g), c4::load_npy(G + "/wn_out.npy"));
  }

  // FinalLayer
  {
    c4::Tensor lw = w.get("fl.linear.weight"), lb = w.get("fl.linear.bias");
    c4::Tensor mw = w.get("fl.mod.weight"), mb = w.get("fl.mod.bias");
    c4::Tensor x = c4::load_npy(G + "/fl_x.npy");
    c4::Tensor cond = c4::load_npy(G + "/fl_cond.npy");
    check("final_layer", c4::dit::final_layer(x, cond, lw, lb, mw, mb),
          c4::load_npy(G + "/fl_out.npy"));
  }

  std::printf("test_dit_wn: %s\n", failures == 0 ? "ALL PASSED" : "FAILED");
  return failures == 0 ? 0 : 1;
}
