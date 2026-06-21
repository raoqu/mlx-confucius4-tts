#include "c4tts/wav_io.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "mlx/mlx.h"

namespace c4 {
namespace mx = mlx::core;

namespace {
uint32_t rd_u32(std::ifstream& f) {
  uint32_t v = 0;
  f.read(reinterpret_cast<char*>(&v), 4);
  return v;
}
uint16_t rd_u16(std::ifstream& f) {
  uint16_t v = 0;
  f.read(reinterpret_cast<char*>(&v), 2);
  return v;
}
}  // namespace

WavData read_wav(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("wav: cannot open " + path);

  char tag[4];
  f.read(tag, 4);
  if (std::strncmp(tag, "RIFF", 4) != 0) throw std::runtime_error("wav: not RIFF");
  rd_u32(f);  // chunk size
  f.read(tag, 4);
  if (std::strncmp(tag, "WAVE", 4) != 0) throw std::runtime_error("wav: not WAVE");

  uint16_t fmt = 0, channels = 0, bits = 0;
  uint32_t rate = 0;
  std::vector<float> mono;

  while (f && f.peek() != EOF) {
    f.read(tag, 4);
    if (f.gcount() < 4) break;
    uint32_t sz = rd_u32(f);
    if (std::strncmp(tag, "fmt ", 4) == 0) {
      fmt = rd_u16(f);
      channels = rd_u16(f);
      rate = rd_u32(f);
      rd_u32(f);          // byte rate
      rd_u16(f);          // block align
      bits = rd_u16(f);
      for (uint32_t i = 16; i < sz; ++i) f.get();  // skip extra fmt bytes
    } else if (std::strncmp(tag, "data", 4) == 0) {
      const int ch = channels ? channels : 1;
      if (fmt == 1 && bits == 16) {
        const uint32_t n = sz / 2;
        std::vector<int16_t> buf(n);
        f.read(reinterpret_cast<char*>(buf.data()), sz);
        for (uint32_t i = 0; i < n; i += ch) {
          float acc = 0;
          for (int c = 0; c < ch && i + c < n; ++c) acc += buf[i + c] / 32768.0f;
          mono.push_back(acc / ch);
        }
      } else if (fmt == 3 && bits == 32) {
        const uint32_t n = sz / 4;
        std::vector<float> buf(n);
        f.read(reinterpret_cast<char*>(buf.data()), sz);
        for (uint32_t i = 0; i < n; i += ch) {
          float acc = 0;
          for (int c = 0; c < ch && i + c < n; ++c) acc += buf[i + c];
          mono.push_back(acc / ch);
        }
      } else {
        throw std::runtime_error("wav: unsupported format (fmt=" +
                                 std::to_string(fmt) + ", bits=" +
                                 std::to_string(bits) + ")");
      }
    } else {
      for (uint32_t i = 0; i < sz; ++i) f.get();  // skip unknown chunk
    }
  }

  float* heap = new float[mono.empty() ? 1 : mono.size()];
  for (size_t i = 0; i < mono.size(); ++i) heap[i] = mono[i];
  mx::array samples(static_cast<void*>(heap),
                    mx::Shape{static_cast<int>(mono.size())}, mx::float32,
                    [](void* p) { delete[] static_cast<float*>(p); });
  return WavData{samples, static_cast<int>(rate)};
}

void write_wav(const std::string& path, const Tensor& samples_in, int sample_rate) {
  Tensor s = mx::astype(mx::flatten(samples_in), mx::float32);
  mx::eval(s);
  const int n = static_cast<int>(s.size());
  const float* d = s.data<float>();

  std::vector<int16_t> pcm(n);
  for (int i = 0; i < n; ++i) {
    float v = d[i];
    if (v > 1.0f) v = 1.0f;
    if (v < -1.0f) v = -1.0f;
    pcm[i] = static_cast<int16_t>(v * 32767.0f);
  }

  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("wav: cannot write " + path);
  const uint32_t data_bytes = n * 2;
  const uint32_t byte_rate = sample_rate * 2;
  auto w32 = [&](uint32_t v) { f.write(reinterpret_cast<const char*>(&v), 4); };
  auto w16 = [&](uint16_t v) { f.write(reinterpret_cast<const char*>(&v), 2); };

  f.write("RIFF", 4);
  w32(36 + data_bytes);
  f.write("WAVE", 4);
  f.write("fmt ", 4);
  w32(16);
  w16(1);             // PCM
  w16(1);             // mono
  w32(sample_rate);
  w32(byte_rate);
  w16(2);             // block align
  w16(16);            // bits
  f.write("data", 4);
  w32(data_bytes);
  f.write(reinterpret_cast<const char*>(pcm.data()), data_bytes);
}

}  // namespace c4
