#include <vector>

#include "c4tts/nn.h"
#include "c4tts/t2s.h"
#include "mlx/mlx.h"

namespace c4 {
namespace t2s {
namespace mx = mlx::core;

Text2Semantic::Text2Semantic(const WeightStore& w, int num_layers,
                             int num_heads, int start_token, int stop_token)
    : w_(w),
      gpt_(w, "transformer.", num_layers, num_heads),
      spk_(w, "speaker_encoder."),
      text_proj_(w, "text_projector."),
      start_token_(start_token),
      stop_token_(stop_token) {}

Tensor Text2Semantic::assemble(const Tensor& text_ids,
                               const Tensor& semantic_codes,
                               const Tensor& condition_vector) const {
  // condition embedding: speaker_encoder(cond) -> (B, D) -> (B, 1, D)
  Tensor cond_emb = mx::expand_dims(spk_.forward(condition_vector), 1);

  // text embedding + learned positions
  Tensor text_emb = add_learned_positions(
      text_proj_.forward(text_ids), w_.get("text_position_embedding.embedding.weight"));

  // semantic embedding + learned positions
  Tensor sem_emb = add_learned_positions(
      nn::embedding(semantic_codes, w_.get("semantic_embedding.weight")),
      w_.get("semantic_position_embedding.embedding.weight"));

  return mx::concatenate({cond_emb, text_emb, sem_emb}, 1);
}

Tensor Text2Semantic::forward_logits(const Tensor& text_ids,
                                     const Tensor& semantic_codes,
                                     const Tensor& condition_vector) const {
  const int text_len = text_ids.shape(1);
  Tensor inputs = assemble(text_ids, semantic_codes, condition_vector);
  Tensor hidden = gpt_.forward(inputs);  // includes GPT-2 ln_f

  // final_norm(hidden[:, 1:])[:, text_len:]
  Tensor h = mx::slice(hidden, mx::Shape{0, 1, 0},
                       mx::Shape{hidden.shape(0), hidden.shape(1), hidden.shape(2)});
  Tensor fnw = w_.get("final_norm.weight"), fnb = w_.get("final_norm.bias");
  h = nn::layer_norm(h, &fnw, &fnb);
  h = mx::slice(h, mx::Shape{0, text_len, 0},
                mx::Shape{h.shape(0), h.shape(1), h.shape(2)});

  Tensor shw = w_.get("semantic_head.weight"), shb = w_.get("semantic_head.bias");
  return nn::linear(h, shw, &shb);  // (B, T_sem, vocab)
}

Tensor Text2Semantic::last_logits(const Tensor& inputs_embeds) const {
  Tensor hidden = gpt_.forward(inputs_embeds);
  const int T = hidden.shape(1);
  Tensor last = mx::slice(hidden, mx::Shape{0, T - 1, 0},
                          mx::Shape{hidden.shape(0), T, hidden.shape(2)});  // (B,1,D)
  Tensor fnw = w_.get("final_norm.weight"), fnb = w_.get("final_norm.bias");
  last = nn::layer_norm(last, &fnw, &fnb);
  Tensor shw = w_.get("semantic_head.weight"), shb = w_.get("semantic_head.bias");
  return nn::linear(last, shw, &shb);  // (B, 1, vocab)
}

Tensor Text2Semantic::generate_greedy(const Tensor& text_ids,
                                      const Tensor& condition_vector,
                                      int max_new_tokens) const {
  const int B = text_ids.shape(0);
  // Begin the semantic sequence with BOS.
  std::vector<int32_t> tokens;
  std::vector<int32_t> generated;

  Tensor sem = mx::full(mx::Shape{B, 1}, start_token_, mx::int32);  // BOS
  for (int step = 0; step < max_new_tokens; ++step) {
    Tensor inputs = assemble(text_ids, sem, condition_vector);
    Tensor logits = last_logits(inputs);            // (B, 1, vocab)
    Tensor next = mx::argmax(logits, -1);           // (B, 1)
    mx::eval(next);
    const int32_t tok = next.data<int32_t>()[0];
    if (tok == stop_token_) break;
    generated.push_back(tok);
    Tensor next_col = mx::astype(next, mx::int32);
    sem = mx::concatenate({sem, next_col}, 1);
  }

  const int n = static_cast<int>(generated.size());
  int32_t* heap = new int32_t[n > 0 ? n : 1];
  for (int i = 0; i < n; ++i) heap[i] = generated[i];
  return mx::array(static_cast<void*>(heap), mx::Shape{B, n}, mx::int32,
                   [](void* p) { delete[] static_cast<int32_t*>(p); });
}

}  // namespace t2s
}  // namespace c4
