#include "c4tts/profile.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <vector>

#include "mlx/mlx.h"

namespace c4 {
namespace prof {
namespace mx = mlx::core;

namespace {
struct Entry { double ms = 0.0; long n = 0; };
std::map<std::string, Entry>& table() {
  static std::map<std::string, Entry> t;
  return t;
}
}  // namespace

bool enabled() {
  static const bool on = [] {
    const char* v = std::getenv("C4TTS_PROFILE");
    return v && *v && std::string(v) != "0";
  }();
  return on;
}

void add(const std::string& name, double ms) {
  if (!enabled()) return;
  auto& e = table()[name];
  e.ms += ms;
  e.n += 1;
}

const Tensor& eval_add(const std::string& name, const Tensor& x) {
  if (!enabled()) return x;
  Timer t;
  mx::eval(x);
  add(name, t.ms());
  return x;
}

void lap(const std::string& name, const Tensor& x, Timer& timer) {
  if (!enabled()) return;
  mx::eval(x);
  add(name, timer.ms());
  timer.reset();
}

void lap_cpu(const std::string& name, Timer& timer) {
  if (!enabled()) return;
  add(name, timer.ms());
  timer.reset();
}

void report() {
  if (!enabled() || table().empty()) return;
  std::vector<std::pair<std::string, Entry>> rows(table().begin(), table().end());
  std::sort(rows.begin(), rows.end(),
            [](const auto& a, const auto& b) { return a.second.ms > b.second.ms; });
  double total = 0.0;
  for (const auto& r : rows) total += r.second.ms;
  std::fprintf(stderr, "\n=== c4tts profile (sum of component eval times = %.0f ms) ===\n", total);
  std::fprintf(stderr, "%-34s %10s %8s %12s %7s\n", "component", "total(ms)", "calls", "ms/call", "%");
  for (const auto& r : rows) {
    const auto& e = r.second;
    std::fprintf(stderr, "%-34s %10.1f %8ld %12.3f %6.1f%%\n", r.first.c_str(), e.ms, e.n,
                 e.n ? e.ms / e.n : 0.0, total > 0 ? 100.0 * e.ms / total : 0.0);
  }
  std::fprintf(stderr, "(note: profiling inserts extra GPU syncs; use C4TTS_TIMING for true wall time)\n");
  table().clear();
}

}  // namespace prof
}  // namespace c4
