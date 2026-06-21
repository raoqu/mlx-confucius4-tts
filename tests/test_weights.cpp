// Test: WeightStore (Phase A3). Writes a couple of tensors as .npy, loads them
// back through the store, and checks names/values + missing-key behavior.

#include <cstdio>
#include <string>

#include "c4tts/npy.h"
#include "c4tts/tensor.h"
#include "c4tts/weights.h"

namespace mx = mlx::core;

static int failures = 0;
#define CHECK(cond, msg)                       \
  do {                                         \
    if (!(cond)) { std::printf("  FAIL: %s\n", msg); ++failures; } \
    else { std::printf("  ok:   %s\n", msg); } \
  } while (0)

int main() {
  std::printf("test_weights:\n");
  const std::string dir = std::string(C4TTS_FIXTURE_DIR) + "/wstore";
  // FIXTURE_DIR already exists (created by the fixture generator); make a subdir.
  ::system(("mkdir -p " + dir).c_str());

  mx::array w = mx::reshape(mx::astype(mx::arange(6.0f), mx::float32), {2, 3});
  mx::array b = mx::array({0.5f, -0.5f});
  mx::eval(w, b);
  c4::save_npy(dir + "/layer.weight.npy", w);
  c4::save_npy(dir + "/layer.bias.npy", b);

  c4::WeightStore store(dir);

  CHECK(store.has("layer.weight"), "has(layer.weight)");
  CHECK(store.has("layer.bias"), "has(layer.bias)");
  CHECK(!store.has("layer.missing"), "!has(layer.missing)");

  auto rw = c4::compare(store.get("layer.weight"), w);
  CHECK(rw.max_abs_err == 0.0, "get(layer.weight) values exact");

  c4::Tensor got = store.get("layer.weight");
  CHECK(got.ndim() == 2 && got.shape(0) == 2 && got.shape(1) == 3,
        "get(layer.weight) shape (2,3)");

  bool threw = false;
  try {
    store.get("layer.missing");
  } catch (const std::exception&) {
    threw = true;
  }
  CHECK(threw, "get(missing) throws");

  std::printf("test_weights: %s\n", failures == 0 ? "ALL PASSED" : "FAILED");
  return failures == 0 ? 0 : 1;
}
