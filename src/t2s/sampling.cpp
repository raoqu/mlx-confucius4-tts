#include "c4tts/t2s.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <unordered_set>
#include <vector>

namespace c4 {
namespace t2s {
namespace sampling {

void temperature(std::vector<float>& logits, float temp) {
  if (temp == 1.0f || temp <= 0.0f) return;
  for (float& v : logits) v /= temp;
}

void repetition_penalty(std::vector<float>& logits,
                        const std::vector<int>& prev_tokens, float penalty) {
  if (penalty == 1.0f) return;
  std::unordered_set<int> seen(prev_tokens.begin(), prev_tokens.end());
  for (int t : seen) {
    if (t < 0 || t >= static_cast<int>(logits.size())) continue;
    float& s = logits[t];
    s = (s < 0) ? s * penalty : s / penalty;
  }
}

void top_k(std::vector<float>& logits, int k) {
  const int n = static_cast<int>(logits.size());
  if (k <= 0 || k >= n) return;
  std::vector<float> sorted(logits);
  std::nth_element(sorted.begin(), sorted.begin() + (k - 1), sorted.end(),
                   std::greater<float>());
  const float thresh = sorted[k - 1];  // k-th largest value
  const float ninf = -std::numeric_limits<float>::infinity();
  for (float& v : logits)
    if (v < thresh) v = ninf;
}

void top_p(std::vector<float>& logits, float p, int min_tokens_to_keep) {
  const int n = static_cast<int>(logits.size());
  // Indices sorted ascending by logit (matches HF: torch.sort descending=False).
  std::vector<int> idx(n);
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(),
            [&](int a, int b) { return logits[a] < logits[b]; });

  // softmax over sorted (ascending) logits.
  float maxv = *std::max_element(logits.begin(), logits.end());
  std::vector<double> probs(n);
  double sum = 0;
  for (int i = 0; i < n; ++i) {
    probs[i] = std::exp(static_cast<double>(logits[idx[i]] - maxv));
    sum += probs[i];
  }
  // cumulative probability over ascending order; remove where cumprob <= 1-p.
  const float ninf = -std::numeric_limits<float>::infinity();
  double cum = 0;
  std::vector<char> remove(n, 0);
  for (int i = 0; i < n; ++i) {
    cum += probs[i] / sum;
    if (cum <= (1.0 - p)) remove[i] = 1;
  }
  // Keep at least min_tokens_to_keep (the largest, at the end of ascending order).
  for (int i = n - min_tokens_to_keep; i < n; ++i)
    if (i >= 0) remove[i] = 0;
  for (int i = 0; i < n; ++i)
    if (remove[i]) logits[idx[i]] = ninf;
}

int argmax(const std::vector<float>& logits) {
  return static_cast<int>(
      std::max_element(logits.begin(), logits.end()) - logits.begin());
}

}  // namespace sampling
}  // namespace t2s
}  // namespace c4
