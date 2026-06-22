// T2S (text-to-semantic) modules for c4tts (confuciustts/llm).
//
// The core is a GPT-2 transformer (HuggingFace GPT2Model) wrapped by
// Text2Semantic with custom embeddings/heads and an AR sampling loop. Verified
// by test_t2s.

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "c4tts/tensor.h"
#include "c4tts/weights.h"

namespace c4 {
namespace t2s {

// GPT-2 transformer stack (transformers GPT2Model). Position embedding is the
// model's DummyPositionEmbedding (zeros), so forward only consumes inputs_embeds.
// Conv1D weights are stored (in, out) as in HuggingFace.
// Per-layer key/value cache for incremental (autoregressive) decoding.
struct KVCache {
  std::vector<Tensor> k, v;  // each (B, H, T_cached, head_dim)
  bool empty() const { return k.empty(); }
};

class GPT2 {
 public:
  GPT2(const WeightStore& w, const std::string& prefix, int n_layer,
       int n_head);

  // inputs_embeds: (B, T, D) -> last_hidden_state (B, T, D) after ln_f.
  // Full-sequence causal forward (no KV cache).
  Tensor forward(const Tensor& inputs_embeds) const;

  // Incremental causal forward with a KV cache. Processes only the new tokens
  // in `inputs_embeds` (T_new), attending to `past_len` cached positions plus
  // the new ones; appends the new keys/values to `cache`. Returns the ln_f'd
  // hidden states for the new tokens (B, T_new, D).
  //
  // `attn_mask`, if non-null, is an additive attention mask broadcastable to
  // (B, H, T_new, T_total) (0 where allowed, large-negative where masked); it
  // replaces the implicit "causal" mode. Used by batched generation to mask the
  // left-padding of variable-length prompts.
  Tensor forward(const Tensor& inputs_embeds, KVCache& cache, int past_len,
                 const Tensor* attn_mask = nullptr) const;

  int n_layer() const { return n_layer_; }

 private:
  // Unified block: plain when cache==nullptr, else uses/updates cache[layer].
  // `attn_mask` (additive) overrides causal masking when non-null.
  Tensor block(const Tensor& x, const std::string& p, KVCache* cache, int layer,
               int past_len, const Tensor* attn_mask = nullptr) const;

  // Projection y = x @ W (+ b) for the four bandwidth-heavy GPT-2 projections.
  // When fp16 compute is enabled (C4TTS_FP16), the matmul runs in float16
  // (the weight read is the bottleneck for M=1 decode GEMVs, so half precision
  // ~halves it) while the fp32 residual stream / norms are preserved: x is
  // cast to fp16, multiplied by a cached fp16 weight, and the result cast back
  // to fp32 before the bias add. With fp16 disabled this is a plain fp32 matmul.
  Tensor proj(const Tensor& x, const std::string& name) const;

  const WeightStore& w_;
  std::string p_;
  int n_layer_, n_head_;
  bool fp16_ = false;
  int quant_bits_ = 0;   // 0 = off; 4 or 8 quantizes the projection weights
  int quant_group_ = 64; // affine group size (finer for 4-bit fidelity)
  // Cached fp16 copies of projection weights, keyed by weight name.
  mutable std::unordered_map<std::string, Tensor> half_cache_;
  // Cached quantized projection weights: name -> [w_q, scales, biases].
  mutable std::unordered_map<std::string, std::vector<Tensor>> quant_cache_;
};

// GPT-2's gelu_new (tanh approximation).
Tensor gelu_new(const Tensor& x);

// Autoregressive sampling logits processors (HuggingFace-compatible), operating
// in-place on a host logits vector. Used by Text2Semantic generation.
namespace sampling {
void temperature(std::vector<float>& logits, float temp);
void repetition_penalty(std::vector<float>& logits,
                        const std::vector<int>& prev_tokens, float penalty);
void top_k(std::vector<float>& logits, int k);
void top_p(std::vector<float>& logits, float p, int min_tokens_to_keep = 1);
int argmax(const std::vector<float>& logits);
}  // namespace sampling

// ECAPA-TDNN speaker encoder (llm/speaker_encoder.py:Qwen3TTSSpeakerEncoder).
// Produces the T2S condition embedding from prompt semantic features.
//   input: (B, T, mel_dim) -> (B, enc_dim)   (forward, no L2 normalize)
class SpeakerEncoder {
 public:
  SpeakerEncoder(const WeightStore& w, const std::string& prefix);
  Tensor forward(const Tensor& x) const;

 private:
  // "same"-padding reflect TDNN conv (+ optional ReLU).
  Tensor tdnn(const Tensor& x, const std::string& p, int kernel, int dilation,
              bool relu) const;
  Tensor se_res2net_block(const Tensor& x, const std::string& p, int kernel,
                          int dilation) const;
  Tensor attentive_stats_pooling(const Tensor& x, const std::string& p) const;

  const WeightStore& w_;
  std::string p_;
};

// llm/text_encoder.py:TextEmbeddingProjector — Embedding -> fc1 -> SiLU -> fc2.
//   text_ids: (B, T) int -> (B, T, output_size)
class TextEmbeddingProjector {
 public:
  TextEmbeddingProjector(const WeightStore& w, const std::string& prefix);
  Tensor forward(const Tensor& text_ids) const;

 private:
  const WeightStore& w_;
  std::string p_;
};

// llm/position_embeddings.py:LearnedPositionalEmbedding — add learned pos emb.
//   x: (B, T, D); adds embedding rows [0, T).
Tensor add_learned_positions(const Tensor& x, const Tensor& pos_table);

// Text2Semantic (llm/llm.py): the full T2S model — assembles
// [condition, text, semantic] embeddings, runs GPT-2, and projects to semantic
// logits. Drives autoregressive semantic-token generation.
class Text2Semantic {
 public:
  Text2Semantic(const WeightStore& w, int num_layers, int num_heads,
                int start_token = 8192, int stop_token = 8193);

  // Training-style forward: logits over the semantic positions.
  //   text_ids: (B, T_text) int; semantic_codes: (B, T_sem) int (incl BOS/EOS);
  //   condition_vector: (B, T_cond, spk_dim) -> (B, T_sem, semantic_vocab).
  Tensor forward_logits(const Tensor& text_ids, const Tensor& semantic_codes,
                        const Tensor& condition_vector) const;

  // Greedy autoregressive generation (no KV cache; recomputes per step).
  // Returns the generated semantic token ids (B, T_gen), excluding BOS/EOS.
  Tensor generate_greedy(const Tensor& text_ids, const Tensor& condition_vector,
                         int max_new_tokens) const;

  struct Generation {
    Tensor codes;   // (B, T_gen) int — generated semantic tokens (no BOS/EOS)
    Tensor latent;  // (B, T_gen, D) — GPT-2 hidden states over those tokens
  };

  // Autoregressive generation with logits processors, returning codes AND the
  // LM latent (matching llm.py generate(return_latent=True)). `sample`=false is
  // greedy (deterministic); otherwise top-k/top-p/temperature sampling.
  Generation generate(const Tensor& text_ids, const Tensor& condition_vector,
                      int max_new_tokens, bool sample = false,
                      float temperature = 1.0f, int top_k = 0,
                      float top_p = 1.0f, float repetition_penalty = 1.0f,
                      uint64_t seed = 0) const;

  // Batched generation of N independent segments that share the same condition
  // (e.g. the segments of one long utterance). Variable-length prompts are
  // left-padded and decoded as one batch, turning the M=1 decode GEMVs into
  // M=N GEMMs — large throughput win on the dispatch-bound T2S decode for long
  // text. Returns one Generation per input segment (same order). `seed` is
  // offset per segment for independent sampling.
  std::vector<Generation> generate_batch(
      const std::vector<std::vector<int>>& text_ids_list,
      const Tensor& condition_vector, int max_new_tokens, bool sample = false,
      float temperature = 1.0f, int top_k = 0, float top_p = 1.0f,
      float repetition_penalty = 1.0f, uint64_t seed = 0) const;

 private:
  // Builds [cond, text, sem] inputs_embeds for the given semantic prefix.
  Tensor assemble(const Tensor& text_ids, const Tensor& semantic_codes,
                  const Tensor& condition_vector) const;
  // Logits for the last position only (used by the AR loop).
  Tensor last_logits(const Tensor& inputs_embeds) const;

  const WeightStore& w_;
  GPT2 gpt_;
  SpeakerEncoder spk_;
  TextEmbeddingProjector text_proj_;
  int start_token_, stop_token_;
};

}  // namespace t2s
}  // namespace c4
