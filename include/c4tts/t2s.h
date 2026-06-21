// T2S (text-to-semantic) modules for c4tts (confuciustts/llm).
//
// The core is a GPT-2 transformer (HuggingFace GPT2Model) wrapped by
// Text2Semantic with custom embeddings/heads and an AR sampling loop. Verified
// by test_t2s.

#pragma once

#include <string>

#include "c4tts/tensor.h"
#include "c4tts/weights.h"

namespace c4 {
namespace t2s {

// GPT-2 transformer stack (transformers GPT2Model). Position embedding is the
// model's DummyPositionEmbedding (zeros), so forward only consumes inputs_embeds.
// Conv1D weights are stored (in, out) as in HuggingFace.
class GPT2 {
 public:
  GPT2(const WeightStore& w, const std::string& prefix, int n_layer,
       int n_head);

  // inputs_embeds: (B, T, D) -> last_hidden_state (B, T, D) after ln_f.
  // Full-sequence causal forward (prefill; no KV cache).
  Tensor forward(const Tensor& inputs_embeds) const;

 private:
  Tensor block(const Tensor& x, const std::string& p) const;

  const WeightStore& w_;
  std::string p_;
  int n_layer_, n_head_;
};

// GPT-2's gelu_new (tanh approximation).
Tensor gelu_new(const Tensor& x);

}  // namespace t2s
}  // namespace c4
