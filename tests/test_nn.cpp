// Test: NN primitive parity (Phase A2) vs PyTorch.
// Golden vectors from tools/dump_golden.py nn (c4tts/golden/nn).

#include <cstdio>
#include <string>

#include "c4tts/nn.h"
#include "c4tts/npy.h"
#include "c4tts/tensor.h"

namespace mx = mlx::core;

static int failures = 0;
static const std::string G = std::string(C4TTS_GOLDEN_DIR) + "/nn/";

static c4::Tensor L(const std::string& n) { return c4::load_npy(G + n + ".npy"); }

static void check(const std::string& name, const c4::Tensor& out,
                  const c4::Tensor& ref, double max_abs = 1e-4) {
  mx::eval(out);
  auto r = c4::compare(out, ref);
  c4::print_report(name, r);
  if (!c4::parity_ok_elementwise(r, max_abs)) ++failures;
}

int main() {
  std::printf("test_nn:\n");

  {
    c4::Tensor w = L("linear_w"), b = L("linear_b");
    check("linear", c4::nn::linear(L("linear_x"), w, &b), L("linear_y"));
  }
  {
    c4::Tensor w = L("conv_w"), b = L("conv_b");
    check("conv1d", c4::nn::conv1d(L("conv_x"), w, &b, 2, 2, 2, 2), L("conv_y"));
  }
  {
    c4::Tensor w = L("ln_w"), b = L("ln_b");
    check("layer_norm", c4::nn::layer_norm(L("ln_x"), &w, &b), L("ln_y"));
  }
  {
    c4::Tensor w = L("gn_w"), b = L("gn_b");
    check("group_norm", c4::nn::group_norm(L("gn_x"), 4, &w, &b), L("gn_y"));
  }
  {
    c4::Tensor w = L("rms_w");
    check("rms_norm", c4::nn::rms_norm(L("rms_x"), w), L("rms_y"));
  }
  {
    c4::Tensor x = L("act_x");
    check("silu", c4::nn::silu(x), L("act_silu"));
    check("gelu", c4::nn::gelu(x), L("act_gelu"));
    check("mish", c4::nn::mish(x), L("act_mish"));
  }

  std::printf("test_nn: %s\n", failures == 0 ? "ALL PASSED" : "FAILED");
  return failures == 0 ? 0 : 1;
}
