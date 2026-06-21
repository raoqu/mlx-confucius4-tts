// Test: T2S sampling logits processors (D-4) parity vs HuggingFace.
// Golden from tools/dump_golden.py sampling.

#include <cstdio>
#include <string>
#include <vector>

#include "c4tts/npy.h"
#include "c4tts/t2s.h"
#include "c4tts/tensor.h"

namespace mx = mlx::core;
static const std::string G = std::string(C4TTS_GOLDEN_DIR) + "/sampling/";
static int failures = 0;

static std::vector<float> to_vec(const c4::Tensor& t) {
  c4::Tensor f = mx::flatten(t);
  mx::eval(f);
  const float* d = f.data<float>();
  return std::vector<float>(d, d + f.size());
}

// Compares two logit vectors, treating matching -inf positions as equal.
static void check(const std::string& name, const std::vector<float>& got,
                  const c4::Tensor& ref_t) {
  std::vector<float> ref = to_vec(ref_t);
  double max_err = 0;
  bool ok = got.size() == ref.size();
  for (size_t i = 0; ok && i < got.size(); ++i) {
    const bool gi = std::isinf(got[i]) && got[i] < 0;
    const bool ri = std::isinf(ref[i]) && ref[i] < 0;
    if (gi != ri) { ok = false; break; }
    if (!gi) max_err = std::max(max_err, (double)std::abs(got[i] - ref[i]));
  }
  std::printf("  [%s] max_err=%.3e %s\n", name.c_str(), max_err,
              (ok && max_err < 1e-5) ? "" : "<-- FAIL");
  if (!(ok && max_err < 1e-5)) ++failures;
}

int main() {
  std::printf("test_sampling:\n");
  std::vector<float> base = to_vec(c4::load_npy(G + "logits.npy"));
  c4::Tensor ids_t = c4::load_npy(G + "ids.npy");
  mx::eval(ids_t);
  std::vector<int> ids;
  for (int i = 0; i < (int)ids_t.size(); ++i) ids.push_back(ids_t.data<int32_t>()[i]);

  { auto v = base; c4::t2s::sampling::temperature(v, 0.8f); check("temperature", v, c4::load_npy(G + "temp.npy")); }
  { auto v = base; c4::t2s::sampling::repetition_penalty(v, ids, 1.3f); check("repetition_penalty", v, c4::load_npy(G + "rep.npy")); }
  { auto v = base; c4::t2s::sampling::top_k(v, 10); check("top_k", v, c4::load_npy(G + "topk.npy")); }
  { auto v = base; c4::t2s::sampling::top_p(v, 0.9f); check("top_p", v, c4::load_npy(G + "topp.npy")); }

  std::printf("test_sampling: %s\n", failures == 0 ? "ALL PASSED" : "FAILED");
  return failures == 0 ? 0 : 1;
}
