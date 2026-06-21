// Wav2Vec2-BERT 2.0 conformer encoder for c4tts (HuggingFace Wav2Vec2BertModel).
//
// Used to extract prompt semantic features (layer-17 hidden states) that
// condition T2S. Faithful port of the conformer block: half-step FFNs,
// relative_key self-attention, GLU convolution module. Verified by test_w2vbert.

#pragma once

#include <string>

#include "c4tts/tensor.h"
#include "c4tts/weights.h"

namespace c4 {

class W2VBert {
 public:
  W2VBert(const WeightStore& w, const std::string& prefix, int num_heads,
          int left_max, int right_max, int conv_kernel);

  // input_features: (B, T, 160). Runs `num_layers` conformer layers (no
  // attention mask / no padding) and returns the hidden states AFTER the last
  // run layer. The reference's hidden_states[k] equals forward(.., k).
  Tensor forward(const Tensor& input_features, int num_layers) const;

 private:
  Tensor feed_forward(const Tensor& x, const std::string& p) const;
  Tensor self_attention(const Tensor& x, const std::string& p) const;
  Tensor conv_module(const Tensor& x, const std::string& p) const;
  Tensor layer(const Tensor& x, int i) const;

  const WeightStore& w_;
  std::string p_;
  int num_heads_, left_max_, right_max_, conv_kernel_;
};

}  // namespace c4
