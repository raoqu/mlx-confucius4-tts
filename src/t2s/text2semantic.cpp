#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

std::vector<Text2Semantic::Generation> Text2Semantic::generate_batch(
    const std::vector<std::vector<int>>& text_ids_list,
    const Tensor& condition_vector, int max_new_tokens, bool sample,
    float temperature, int top_k, float top_p, float repetition_penalty,
    uint64_t seed) const {
  const int N = static_cast<int>(text_ids_list.size());
  std::vector<Generation> out;
  if (N == 0) return out;

  Tensor sem_w = w_.get("semantic_embedding.weight");
  Tensor sem_pos = w_.get("semantic_position_embedding.embedding.weight");
  const int D = sem_w.shape(1);
  Tensor text_pos = w_.get("text_position_embedding.embedding.weight");
  Tensor fnw = w_.get("final_norm.weight"), fnb = w_.get("final_norm.bias");
  Tensor shw = w_.get("semantic_head.weight"), shb = w_.get("semantic_head.bias");

  // Shared condition embedding (1,1,D) and BOS (sem pos 0) embedding (1,1,D).
  Tensor cond_emb = mx::expand_dims(spk_.forward(condition_vector), 1);
  Tensor bos_id = mx::full(mx::Shape{1, 1}, start_token_, mx::int32);
  Tensor bos_emb = add_learned_positions(nn::embedding(bos_id, sem_w), sem_pos);

  // Per-segment prefixes [cond, text+textpos, BOS], left-padded to a common L.
  std::vector<Tensor> prefixes;
  std::vector<int> lens;
  for (int b = 0; b < N; ++b) {
    const auto& tids = text_ids_list[b];
    const int T = static_cast<int>(tids.size());
    int32_t* heap = new int32_t[T > 0 ? T : 1];
    for (int i = 0; i < T; ++i) heap[i] = tids[i];
    Tensor text_ids(static_cast<void*>(heap), mx::Shape{1, T}, mx::int32,
                    [](void* p) { delete[] static_cast<int32_t*>(p); });
    Tensor text_emb = add_learned_positions(text_proj_.forward(text_ids), text_pos);
    prefixes.push_back(mx::concatenate({cond_emb, text_emb, bos_emb}, 1));
    lens.push_back(1 + T + 1);
  }
  const int L = *std::max_element(lens.begin(), lens.end());

  std::vector<int> pad_len(N);
  std::vector<Tensor> padded;
  for (int b = 0; b < N; ++b) {
    pad_len[b] = L - lens[b];
    Tensor p = prefixes[b];
    if (pad_len[b] > 0) {
      p = mx::concatenate({mx::zeros({1, pad_len[b], D}, p.dtype()), p}, 1);
    }
    padded.push_back(p);
  }
  Tensor batch = mx::concatenate(padded, 0);  // (N, L, D)

  // Host-built additive attention masks.
  const float NEG = -1e9f;
  auto make_array = [](std::vector<float>&& buf, const mx::Shape& shape) {
    const int64_t n = static_cast<int64_t>(buf.size());
    float* heap = new float[n > 0 ? n : 1];
    std::copy(buf.begin(), buf.end(), heap);
    return mx::array(static_cast<void*>(heap), shape, mx::float32,
                     [](void* p) { delete[] static_cast<float*>(p); });
  };

  // Prefill mask (N,1,L,L): query i may attend key j iff j<=i (causal) and
  // j>=pad_len[b] (key not in the left-pad); the diagonal is always allowed so
  // padded query rows are not fully masked.
  std::vector<float> pm(static_cast<size_t>(N) * L * L, 0.0f);
  for (int b = 0; b < N; ++b)
    for (int i = 0; i < L; ++i)
      for (int j = 0; j < L; ++j) {
        const bool ok = (j <= i) && (j >= pad_len[b] || j == i);
        pm[(static_cast<size_t>(b) * L + i) * L + j] = ok ? 0.0f : NEG;
      }
  Tensor prefill_mask = mx::reshape(make_array(std::move(pm), {N, L, L}), {N, 1, L, L});

  // Precompute the decode key-validity mask base once (column j is valid iff
  // j >= pad_len[b]; generated positions j>=L are always valid). Each step
  // slices [0:Tkv] device-side instead of rebuilding+uploading a host mask.
  const int max_kv = L + max_new_tokens;
  std::vector<float> dmb(static_cast<size_t>(N) * max_kv);
  for (int b = 0; b < N; ++b)
    for (int j = 0; j < max_kv; ++j)
      dmb[static_cast<size_t>(b) * max_kv + j] = (j < pad_len[b]) ? NEG : 0.0f;
  Tensor decode_base = make_array(std::move(dmb), {N, max_kv});

  KVCache cache;
  int past = 0;
  Tensor hidden = gpt_.forward(batch, cache, past, &prefill_mask);  // (N,L,D)
  past = L;
  Tensor cur = mx::slice(hidden, mx::Shape{0, L - 1, 0}, mx::Shape{N, L, D});  // (N,1,D)

  auto head = [&](const Tensor& c) {
    Tensor h = nn::layer_norm(c, &fnw, &fnb);
    return nn::linear(h, shw, &shb);  // (N,1,vocab)
  };

  std::vector<std::vector<int>> gen(N);
  std::vector<std::vector<Tensor>> lat(N);
  std::vector<bool> done(N, false);
  int n_done = 0;
  std::vector<uint64_t> rng(N);
  for (int b = 0; b < N; ++b)
    rng[b] = (seed ? seed : 0x9E3779B97F4A7C15ull) + static_cast<uint64_t>(b) * 0x100000001b3ull;

  for (int step = 0; step < max_new_tokens && n_done < N; ++step) {
    Tensor logits_t = head(cur);  // (N,1,vocab)
    mx::eval(logits_t);
    const int vocab = logits_t.shape(2);
    const float* ld = logits_t.data<float>();

    std::vector<int> toks(N, 0);
    for (int b = 0; b < N; ++b) {
      if (done[b]) continue;
      std::vector<float> logits(ld + static_cast<size_t>(b) * vocab,
                                ld + static_cast<size_t>(b + 1) * vocab);
      sampling::repetition_penalty(logits, gen[b], repetition_penalty);
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
        uint64_t& r = rng[b];
        r ^= r << 13; r ^= r >> 7; r ^= r << 17;
        double rr = (double)(r >> 11) / (double)(1ull << 53) * sum;
        double acc = 0;
        tok = (int)logits.size() - 1;
        for (size_t i = 0; i < probs.size(); ++i) {
          acc += probs[i];
          if (rr <= acc) { tok = (int)i; break; }
        }
      }
      if (tok == stop_token_) { done[b] = true; ++n_done; continue; }
      gen[b].push_back(tok);
      lat[b].push_back(mx::slice(cur, mx::Shape{b, 0, 0}, mx::Shape{b + 1, 1, D}));
      toks[b] = tok;
    }
    if (n_done >= N) break;

    int32_t* th = new int32_t[N];
    for (int b = 0; b < N; ++b) th[b] = toks[b];
    Tensor tok_arr(static_cast<void*>(th), mx::Shape{N, 1}, mx::int32,
                   [](void* p) { delete[] static_cast<int32_t*>(p); });
    Tensor pos_row = mx::reshape(
        mx::slice(sem_pos, mx::Shape{step + 1, 0}, mx::Shape{step + 2, D}), {1, 1, D});
    Tensor new_emb = mx::add(nn::embedding(tok_arr, sem_w), pos_row);  // (N,1,D)

    // Decode mask (N,1,1,Tkv): slice the precomputed base [0:Tkv].
    const int Tkv = past + 1;
    Tensor decode_mask =
        mx::reshape(mx::slice(decode_base, mx::Shape{0, 0}, mx::Shape{N, Tkv}), {N, 1, 1, Tkv});

    cur = gpt_.forward(new_emb, cache, past, &decode_mask);  // (N,1,D)
    past += 1;
    // Bound MLX's reclaimable buffer pool during the long batched decode (the
    // batched KV cache is N x larger than single-stream, so an unbounded pool
    // can OOM on long segments). The live KV cache / latents survive (they are
    // referenced, not pooled buffers).
    if ((step & 31) == 31) mx::clear_cache();
  }

  for (int b = 0; b < N; ++b) {
    const int ng = static_cast<int>(gen[b].size());
    int32_t* ch = new int32_t[ng > 0 ? ng : 1];
    for (int i = 0; i < ng; ++i) ch[i] = gen[b][i];
    Tensor codes(static_cast<void*>(ch), mx::Shape{1, ng}, mx::int32,
                 [](void* p) { delete[] static_cast<int32_t*>(p); });
    Tensor latent = ng > 0 ? mx::concatenate(lat[b], 1) : mx::zeros({1, 0, D}, mx::float32);
    out.push_back(Generation{codes, latent});
  }
  return out;
}

}  // namespace t2s
}  // namespace c4
