#include "c4tts/weights.h"

#include <stdexcept>
#include <sys/stat.h>

#include "c4tts/npy.h"

namespace c4 {

WeightStore::WeightStore(std::string path) : root_(std::move(path)) {
  if (!root_.empty() && root_.back() == '/') root_.pop_back();
  struct stat st {};
  if (stat(root_.c_str(), &st) != 0 || !(st.st_mode & S_IFDIR)) {
    throw std::runtime_error("WeightStore: not a directory: " + root_);
  }
}

std::string WeightStore::path_for(const std::string& name) const {
  return root_ + "/" + name + ".npy";
}

bool WeightStore::has(const std::string& name) const {
  struct stat st {};
  return stat(path_for(name).c_str(), &st) == 0;
}

Tensor WeightStore::get(const std::string& name) const {
  if (!has(name)) {
    throw std::runtime_error("WeightStore: missing weight '" + name + "' in " +
                             root_);
  }
  return load_npy(path_for(name));
}

Tensor WeightStore::get_or(const std::string& name,
                           const Tensor& fallback) const {
  return has(name) ? get(name) : fallback;
}

}  // namespace c4
