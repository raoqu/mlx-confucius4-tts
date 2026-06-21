// Core tensor aliases and numerical-parity helpers for c4tts.
//
// MLX is the compute backend; `c4::Tensor` is a thin alias so module code reads
// in our own vocabulary and a future backend swap stays localized. The compare
// helpers implement the parity metrics defined in docs/PLAN.md §2.4
// (max_abs_err, cosine similarity, relative L2) used by every module test.

#pragma once

#include <cmath>
#include <cstdio>
#include <string>

#include "mlx/mlx.h"

namespace c4 {
namespace mx = mlx::core;

using Tensor = mx::array;

struct ParityReport {
  double max_abs_err = 0.0;
  double rel_l2 = 0.0;       // ||a-b||_2 / ||b||_2
  double cosine_sim = 1.0;
  int64_t size = 0;
};

// Computes parity metrics between `a` (candidate) and `b` (reference).
inline ParityReport compare(const Tensor& a_in, const Tensor& b_in) {
  Tensor a = mx::astype(mx::flatten(a_in), mx::float32);
  Tensor b = mx::astype(mx::flatten(b_in), mx::float32);
  mx::eval(a, b);

  ParityReport r;
  r.size = a.size();
  if (a.size() != b.size()) {
    r.max_abs_err = INFINITY;
    r.rel_l2 = INFINITY;
    r.cosine_sim = -1.0;
    return r;
  }

  Tensor diff = mx::subtract(a, b);
  r.max_abs_err = mx::max(mx::abs(diff)).item<float>();

  const double l2_diff = std::sqrt(mx::sum(mx::square(diff)).item<float>());
  const double l2_ref = std::sqrt(mx::sum(mx::square(b)).item<float>());
  r.rel_l2 = (l2_ref > 0) ? l2_diff / l2_ref : l2_diff;

  const double dot = mx::sum(mx::multiply(a, b)).item<float>();
  const double na = std::sqrt(mx::sum(mx::square(a)).item<float>());
  const double nb = std::sqrt(mx::sum(mx::square(b)).item<float>());
  r.cosine_sim = (na > 0 && nb > 0) ? dot / (na * nb) : 0.0;
  return r;
}

inline void print_report(const std::string& name, const ParityReport& r) {
  std::printf(
      "  [%s] n=%lld  max_abs_err=%.3e  rel_l2=%.3e  cosine=%.8f\n",
      name.c_str(), static_cast<long long>(r.size), r.max_abs_err, r.rel_l2,
      r.cosine_sim);
}

// Pass criterion for elementwise modules (linear/conv/norm): tight abs error.
inline bool parity_ok_elementwise(const ParityReport& r, double max_abs = 1e-4) {
  return r.size > 0 && r.max_abs_err < max_abs;
}

// Pass criterion for deep stacks: cosine ~1 and small relative L2.
inline bool parity_ok_deep(const ParityReport& r, double min_cos = 0.9999,
                           double max_rel_l2 = 1e-3) {
  return r.size > 0 && r.cosine_sim > min_cos && r.rel_l2 < max_rel_l2;
}

}  // namespace c4
