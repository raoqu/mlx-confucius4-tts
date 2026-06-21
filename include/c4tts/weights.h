// Weight store for c4tts (Phase A3, docs/PLAN.md §2.2).
//
// Modules look up parameters by name through this abstraction. The initial
// backend is a directory of .npy files (one per tensor), which is trivially
// correct and lets us start porting weighted modules immediately. It can later
// be swapped for a single packed/mmap'd file (and quantized variants) without
// changing any module code, since they only see get()/has().
//
// Naming mirrors the PyTorch state_dict keys (dots kept), so a parameter
// "encoder.blocks.0.conv.weight" maps to "<dir>/encoder.blocks.0.conv.weight.npy".

#pragma once

#include <string>

#include "c4tts/tensor.h"

namespace c4 {

class WeightStore {
 public:
  // `path` is a directory containing <name>.npy files.
  explicit WeightStore(std::string path);

  bool has(const std::string& name) const;

  // Returns the tensor for `name`, throwing if absent.
  Tensor get(const std::string& name) const;

  // Returns the tensor for `name`, or `fallback` if absent.
  Tensor get_or(const std::string& name, const Tensor& fallback) const;

  const std::string& root() const { return root_; }

 private:
  std::string path_for(const std::string& name) const;
  std::string root_;
};

}  // namespace c4
