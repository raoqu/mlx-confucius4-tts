// Test: WAV read/write round-trip (16-bit PCM quantization tolerance).

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "c4tts/tensor.h"
#include "c4tts/wav_io.h"

namespace mx = mlx::core;

int main() {
  std::printf("test_wav_io:\n");
  const std::string path = std::string(C4TTS_FIXTURE_DIR) + "/roundtrip.wav";
  const int sr = 22050, n = 4096;

  std::vector<float> sig(n);
  for (int i = 0; i < n; ++i) sig[i] = 0.5f * std::sin(2.0f * M_PI * 440.0f * i / sr);
  float* heap = new float[n];
  for (int i = 0; i < n; ++i) heap[i] = sig[i];
  mx::array wav(static_cast<void*>(heap), mx::Shape{n}, mx::float32,
                [](void* p) { delete[] static_cast<float*>(p); });

  c4::write_wav(path, wav, sr);
  c4::WavData rd = c4::read_wav(path);
  mx::eval(rd.samples);

  auto r = c4::compare(rd.samples, wav);
  c4::print_report("wav roundtrip", r);

  const bool ok = rd.sample_rate == sr && rd.samples.size() == (size_t)n &&
                  r.max_abs_err < 1e-4;  // 16-bit quantization
  std::printf("test_wav_io: %s (sr=%d)\n", ok ? "PASSED" : "FAILED",
              rd.sample_rate);
  return ok ? 0 : 1;
}
