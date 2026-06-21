// Minimal NumPy .npy reader/writer for c4tts.
//
// Golden vectors and (later) exported weights are exchanged with the PyTorch
// reference as plain, uncompressed .npy files. The format is trivial to parse,
// which keeps the C++ side dependency-free (no zip/miniz needed). We support the
// dtypes that actually occur in this project: float32/float64 and int32/int64,
// loaded into / saved from MLX arrays.
//
// Reference: https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html

#pragma once

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mlx/mlx.h"

namespace c4 {
namespace mx = mlx::core;

namespace detail {

inline std::string parse_header_field(const std::string& header,
                                      const std::string& key) {
  // Finds `'key': <value>` and returns the raw value substring (up to , or }).
  const std::string needle = "'" + key + "':";
  auto pos = header.find(needle);
  if (pos == std::string::npos) {
    throw std::runtime_error("npy: missing header field '" + key + "'");
  }
  pos += needle.size();
  while (pos < header.size() && header[pos] == ' ') pos++;
  auto end = header.find_first_of(",}", pos);
  return header.substr(pos, end - pos);
}

}  // namespace detail

// Loads a .npy file into an MLX array, preserving its dtype family.
//   <f4 -> float32   <f8 -> float32 (down-cast)
//   <i4 -> int32     <i8 -> int32   (down-cast; token ids fit comfortably)
inline mx::array load_npy(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("npy: cannot open " + path);

  char magic[6];
  f.read(magic, 6);
  if (std::string(magic, 6) != "\x93NUMPY") {
    throw std::runtime_error("npy: bad magic in " + path);
  }
  uint8_t major = 0, minor = 0;
  f.read(reinterpret_cast<char*>(&major), 1);
  f.read(reinterpret_cast<char*>(&minor), 1);

  uint32_t header_len = 0;
  if (major == 1) {
    uint16_t len16 = 0;
    f.read(reinterpret_cast<char*>(&len16), 2);
    header_len = len16;
  } else {
    f.read(reinterpret_cast<char*>(&header_len), 4);
  }

  std::string header(header_len, '\0');
  f.read(header.data(), header_len);

  const std::string descr = detail::parse_header_field(header, "descr");
  const std::string fortran = detail::parse_header_field(header, "fortran_order");
  if (fortran.find("True") != std::string::npos) {
    throw std::runtime_error("npy: fortran_order not supported (" + path + ")");
  }

  // Parse shape tuple, e.g. "(3, 4,)" or "()". The tuple itself contains
  // commas, so we scan from '(' to the matching ')' rather than using the
  // generic field parser (which would stop at the first inner comma).
  auto sp_pos = header.find("'shape':");
  if (sp_pos == std::string::npos) {
    throw std::runtime_error("npy: missing shape in " + path);
  }
  auto open = header.find('(', sp_pos);
  auto close = header.find(')', open);
  std::string shape_str = header.substr(open, close - open + 1);
  std::vector<int> shape;
  std::string num;
  for (char c : shape_str) {
    if (c >= '0' && c <= '9') {
      num += c;
    } else if (!num.empty()) {
      shape.push_back(std::stoi(num));
      num.clear();
    }
  }
  int64_t count = 1;
  for (int d : shape) count *= d;
  const mx::Shape mshape(shape.begin(), shape.end());

  auto read_raw = [&](auto* dst, size_t bytes) {
    f.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(bytes));
  };

  // MLX's pointer constructor is zero-copy and owns the buffer via a deleter, so
  // we hand it a heap allocation with a matching typed deleter.
  if (descr.find("f4") != std::string::npos) {
    float* heap = new float[count];
    read_raw(heap, count * sizeof(float));
    return mx::array(static_cast<void*>(heap), mshape, mx::float32,
                     [](void* p) { delete[] static_cast<float*>(p); });
  } else if (descr.find("f8") != std::string::npos) {
    std::vector<double> tmp(count);
    read_raw(tmp.data(), count * sizeof(double));
    float* heap = new float[count];
    for (int64_t i = 0; i < count; ++i) heap[i] = static_cast<float>(tmp[i]);
    return mx::array(static_cast<void*>(heap), mshape, mx::float32,
                     [](void* p) { delete[] static_cast<float*>(p); });
  } else if (descr.find("i4") != std::string::npos) {
    int32_t* heap = new int32_t[count];
    read_raw(heap, count * sizeof(int32_t));
    return mx::array(static_cast<void*>(heap), mshape, mx::int32,
                     [](void* p) { delete[] static_cast<int32_t*>(p); });
  } else if (descr.find("i8") != std::string::npos) {
    std::vector<int64_t> tmp(count);
    read_raw(tmp.data(), count * sizeof(int64_t));
    int32_t* heap = new int32_t[count];
    for (int64_t i = 0; i < count; ++i) heap[i] = static_cast<int32_t>(tmp[i]);
    return mx::array(static_cast<void*>(heap), mshape, mx::int32,
                     [](void* p) { delete[] static_cast<int32_t*>(p); });
  }
  throw std::runtime_error("npy: unsupported dtype '" + descr + "' in " + path);
}

// Saves an MLX array to a .npy file as float32 (int32 arrays are kept as <i4).
inline void save_npy(const std::string& path, const mx::array& arr_in) {
  mx::array arr = arr_in;
  const bool is_int = (arr.dtype() == mx::int32 || arr.dtype() == mx::int64 ||
                       arr.dtype() == mx::uint32);
  if (!is_int && arr.dtype() != mx::float32) {
    arr = mx::astype(arr, mx::float32);
  }
  if (arr.dtype() == mx::int64 || arr.dtype() == mx::uint32) {
    arr = mx::astype(arr, mx::int32);
  }
  mx::eval(arr);

  const std::string descr = is_int ? "<i4" : "<f4";
  std::string shape_str = "(";
  for (int i = 0; i < arr.ndim(); ++i) {
    shape_str += std::to_string(arr.shape(i));
    if (i + 1 < arr.ndim() || arr.ndim() == 1) shape_str += ", ";
  }
  shape_str += ")";

  std::string header = "{'descr': '" + descr +
                       "', 'fortran_order': False, 'shape': " + shape_str + ", }";
  // Pad so that 10 (preamble) + header + 1 (newline) is a multiple of 64.
  size_t total = 10 + header.size() + 1;
  size_t pad = (64 - (total % 64)) % 64;
  header.append(pad, ' ');
  header += '\n';

  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("npy: cannot write " + path);
  f.write("\x93NUMPY", 6);
  const uint8_t major = 1, minor = 0;
  f.write(reinterpret_cast<const char*>(&major), 1);
  f.write(reinterpret_cast<const char*>(&minor), 1);
  const uint16_t len16 = static_cast<uint16_t>(header.size());
  f.write(reinterpret_cast<const char*>(&len16), 2);
  f.write(header.data(), header.size());

  if (is_int) {
    f.write(reinterpret_cast<const char*>(arr.data<int32_t>()),
            static_cast<std::streamsize>(arr.size() * sizeof(int32_t)));
  } else {
    f.write(reinterpret_cast<const char*>(arr.data<float>()),
            static_cast<std::streamsize>(arr.size() * sizeof(float)));
  }
}

}  // namespace c4
