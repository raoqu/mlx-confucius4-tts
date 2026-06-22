#include "c4tts/weights.h"

#include <cstdio>
#include <cstdlib>
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
  auto it = cache_.find(name);
  if (it != cache_.end()) return it->second;
  if (!has(name)) {
    throw std::runtime_error("WeightStore: missing weight '" + name + "' in " +
                             root_);
  }
  const std::string path = path_for(name);
  // Audit hook: when C4TTS_DUMP_WEIGHTS is set, append each loaded .npy path so
  // the actually-used weight set can be diffed against bin/ (read env once).
  static const char* dump = std::getenv("C4TTS_DUMP_WEIGHTS");
  if (dump) {
    if (std::FILE* fp = std::fopen(dump, "a")) { std::fprintf(fp, "%s\n", path.c_str()); std::fclose(fp); }
  }
  Tensor t = load_npy_mmap(path);  // zero-copy for f4/i4 weight packs
  cache_.emplace(name, t);
  return t;
}

Tensor WeightStore::get_or(const std::string& name,
                           const Tensor& fallback) const {
  return has(name) ? get(name) : fallback;
}

}  // namespace c4
