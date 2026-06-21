// Test: npy IO round-trip (C++ <-> C++) and cross-language read (NumPy -> C++).
//
// Fixtures under C4TTS_FIXTURE_DIR are produced by tools/gen_test_fixtures.py at
// build time. Returns nonzero on any failure so CTest reports it.

#include <cstdio>
#include <string>

#include "c4tts/npy.h"
#include "c4tts/tensor.h"

namespace mx = mlx::core;

static int failures = 0;

#define CHECK(cond, msg)                                  \
  do {                                                    \
    if (!(cond)) {                                        \
      std::printf("  FAIL: %s\n", msg);                   \
      ++failures;                                         \
    } else {                                              \
      std::printf("  ok:   %s\n", msg);                   \
    }                                                     \
  } while (0)

int main() {
  const std::string fixtures = C4TTS_FIXTURE_DIR;
  const std::string tmp = std::string(C4TTS_FIXTURE_DIR) + "/_roundtrip.npy";

  std::printf("test_npy:\n");

  // 1. C++ round-trip: save then load must be bit-identical.
  {
    mx::array original = mx::reshape(
        mx::astype(mx::arange(24.0f), mx::float32), {2, 3, 4});
    mx::eval(original);
    c4::save_npy(tmp, original);
    mx::array loaded = c4::load_npy(tmp);
    auto r = c4::compare(loaded, original);
    c4::print_report("roundtrip f32", r);
    CHECK(loaded.ndim() == 3 && loaded.shape(0) == 2 && loaded.shape(1) == 3 &&
              loaded.shape(2) == 4,
          "roundtrip shape preserved");
    CHECK(r.max_abs_err == 0.0, "roundtrip values exact");
  }

  // 2. Cross-language: read NumPy-written float32 matrix.
  {
    mx::array mat = c4::load_npy(fixtures + "/mat_f32.npy");
    mx::eval(mat);
    CHECK(mat.ndim() == 2 && mat.shape(0) == 3 && mat.shape(1) == 4,
          "mat_f32 shape (3,4)");
    const float* d = mat.data<float>();
    CHECK(d[0] == 0.0f && d[5] == 5.0f && d[11] == 11.0f,
          "mat_f32 values match arange(12)");
  }

  // 3. Cross-language: float64 down-cast to float32.
  {
    mx::array vec = c4::load_npy(fixtures + "/vec_f64.npy");
    mx::eval(vec);
    const float* d = vec.data<float>();
    CHECK(vec.size() == 3 && d[0] == 1.5f && d[1] == -2.25f && d[2] == 3.125f,
          "vec_f64 down-cast to f32 values match");
  }

  // 4. Cross-language: int64 token ids down-cast to int32.
  {
    mx::array ids = c4::load_npy(fixtures + "/ids_i64.npy");
    mx::eval(ids);
    CHECK(ids.dtype() == mx::int32, "ids loaded as int32");
    const int32_t* d = ids.data<int32_t>();
    CHECK(ids.size() == 4 && d[0] == 10 && d[3] == 8193,
          "ids_i64 values match (incl. EOS 8193)");
  }

  if (failures == 0) {
    std::printf("test_npy: ALL PASSED\n");
    return 0;
  }
  std::printf("test_npy: %d FAILURE(S)\n", failures);
  return 1;
}
