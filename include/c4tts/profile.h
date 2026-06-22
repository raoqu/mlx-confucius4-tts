// Lightweight component profiler for c4tts.
//
// Enabled by the C4TTS_PROFILE env var; a no-op (and near-zero cost) otherwise.
// Because MLX is lazy/async, timing a sub-component means forcing an mx::eval()
// at its boundary — so profiling inserts extra GPU syncs and the sum of
// component times is somewhat larger than the un-instrumented total. Use the
// breakdown for relative attribution, and C4TTS_TIMING for the true wall time.
//
// Usage:
//   c4::prof::Timer t;                 // start
//   ... build a Tensor x ...
//   c4::prof::eval_add("t2s.head", x, t);   // eval(x), record elapsed under name
//
// Names use dotted hierarchy (stage.component) and are aggregated across calls
// (sum of time + invocation count). prof::report() prints a sorted table.

#pragma once

#include <chrono>
#include <string>

#include "c4tts/tensor.h"

namespace c4 {
namespace prof {

bool enabled();

// Accumulate `ms` milliseconds (and one invocation) under `name`.
void add(const std::string& name, double ms);

// eval(x) (forcing computation) and accumulate the elapsed time since `t`
// under `name`. Returns x for chaining. No-op timing when disabled.
const Tensor& eval_add(const std::string& name, const Tensor& x);

// Print the aggregated table (sorted by total time) to stderr and reset.
void report();

struct Timer {
  std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
  double ms() const {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0).count();
  }
  void reset() { t0 = std::chrono::steady_clock::now(); }
};

// Times eval(x) from `timer`, records under `name`, and resets `timer`.
void lap(const std::string& name, const Tensor& x, Timer& timer);

// Records elapsed CPU time from `timer` under `name` (no eval), resets timer.
void lap_cpu(const std::string& name, Timer& timer);

}  // namespace prof
}  // namespace c4
