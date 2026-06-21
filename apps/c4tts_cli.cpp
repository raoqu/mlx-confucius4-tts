// c4tts_cli: command-line entry point for the standalone Confucius4-TTS engine.
//
// At this stage (Phase A1, see docs/PLAN.md) it does two things:
//   1. reports its version, and
//   2. runs a tiny MLX computation on the default device so we can confirm the
//      MLX C++ backend links and executes on this machine.
//
// The full `synth(text, lang, prompt_wav) -> wav` pipeline is added over the
// later phases.

#include <cstring>
#include <iostream>

#include "mlx/mlx.h"

namespace mx = mlx::core;

namespace {
constexpr const char* kVersion = "c4tts 0.0.1 (phase A1: scaffold)";

int smoke_test() {
  // A trivial elementwise op forces MLX to allocate, dispatch to the Metal
  // backend, and evaluate. If this prints [5, 7, 9] we have a working backend.
  auto a = mx::array({1.0f, 2.0f, 3.0f});
  auto b = mx::array({4.0f, 5.0f, 6.0f});
  auto c = mx::add(a, b);
  mx::eval(c);

  const float* data = c.data<float>();
  std::cout << "mlx smoke test: [" << data[0] << ", " << data[1] << ", "
            << data[2] << "]" << std::endl;

  const bool ok = (data[0] == 5.0f && data[1] == 7.0f && data[2] == 9.0f);
  std::cout << "default device: "
            << (mx::default_device().type == mx::Device::gpu ? "gpu (Metal)"
                                                             : "cpu")
            << std::endl;
  return ok ? 0 : 1;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc >= 2 && (std::strcmp(argv[1], "--version") == 0 ||
                    std::strcmp(argv[1], "-v") == 0)) {
    std::cout << kVersion << std::endl;
    return 0;
  }

  std::cout << kVersion << std::endl;
  return smoke_test();
}
