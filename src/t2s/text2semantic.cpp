#include <algorithm>
#include <cmath>
#include <cstdint>
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

Text2Semantic::Generation Text2Semantic::generate(
    const Tensor& text_ids, const Tensor& condition_vector, int max_new_tokens,
    bool sample, float temperature, int top_k, float top_p,
    float repetition_penalty, uint64_t seed) const {
  const int B = text_ids.shape(0);
  std::vector<int> generated;
  std::vector<Tensor> latents;
  uint64_t rng = seed ? seed : 0x9E3779B97F4A7C15ull;

  Tensor sem_w = w_.get("semantic_embedding.weight");
  Tensor sem_pos = w_.get("semantic_position_embedding.embedding.weight");
  const int D = sem_w.shape(1);
  Tensor fnw = w_.get("final_norm.weight"), fnb = w_.get("final_norm.bias");
  Tensor shw = w_.get("semantic_head.weight"), shb = w_.get("semantic_head.bias");

  // cur (B,1,D) ln_f'd hidden -> semantic logits (vocab,).
  auto head = [&](const Tensor& cur) {
    Tensor h = nn::layer_norm(cur, &fnw, &fnb);
    return mx::flatten(nn::linear(h, shw, &shb));
  };

  // Prefill the KV cache over [cond, text, BOS]; cur = last position.
  Tensor bos = mx::full(mx::Shape{B, 1}, start_token_, mx::int32);
  Tensor prefix = assemble(text_ids, bos, condition_vector);
  KVCache cache;
  int past = 0;
  Tensor hidden = gpt_.forward(prefix, cache, past);
  past = prefix.shape(1);
  Tensor cur = mx::slice(hidden, mx::Shape{0, past - 1, 0},
                         mx::Shape{B, past, D});  // (B,1,D)

  for (int step = 0; step < max_new_tokens; ++step) {
    Tensor logit_t = head(cur);
    mx::eval(logit_t);
    const float* ld = logit_t.data<float>();
    std::vector<float> logits(ld, ld + logit_t.size());

    sampling::repetition_penalty(logits, generated, repetition_penalty);
    int tok;
    if (!sample) {
      tok = sampling::argmax(logits);
    } else {
      sampling::temperature(logits, temperature);
      sampling::top_k(logits, top_k);
      sampling::top_p(logits, top_p);
      float maxv = -1e30f;
      for (float v : logits) maxv = std::max(maxv, v);
      double sum = 0;
      std::vector<double> probs(logits.size());
      for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::isinf(logits[i]) ? 0.0 : std::exp((double)(logits[i] - maxv));
        sum += probs[i];
      }
      rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
      double r = (double)(rng >> 11) / (double)(1ull << 53) * sum;
      double acc = 0;
      tok = (int)logits.size() - 1;
      for (size_t i = 0; i < probs.size(); ++i) {
        acc += probs[i];
        if (r <= acc) { tok = (int)i; break; }
      }
    }
    if (tok == stop_token_) break;
    generated.push_back(tok);
    latents.push_back(cur);  // cur is the latent that predicted this token

    // Embed the new token at semantic position (step+1) (BOS occupied pos 0).
    Tensor tok_arr = mx::full(mx::Shape{B, 1}, tok, mx::int32);
    Tensor pos_row = mx::reshape(
        mx::slice(sem_pos, mx::Shape{step + 1, 0}, mx::Shape{step + 2, D}),
        {1, 1, D});
    Tensor new_emb = mx::add(nn::embedding(tok_arr, sem_w), pos_row);  // (B,1,D)
    cur = gpt_.forward(new_emb, cache, past);  // (B,1,D)
    past += 1;
  }

  const int n = static_cast<int>(generated.size());
  int32_t* heap = new int32_t[n > 0 ? n : 1];
  for (int i = 0; i < n; ++i) heap[i] = generated[i];
  Tensor codes(static_cast<void*>(heap), mx::Shape{B, n}, mx::int32,
               [](void* p) { delete[] static_cast<int32_t*>(p); });

  Tensor latent = n > 0 ? mx::concatenate(latents, 1)
                        : mx::zeros({B, 0, D}, mx::float32);
  return Generation{codes, latent};
}

}  // namespace t2s
}  // namespace c4
