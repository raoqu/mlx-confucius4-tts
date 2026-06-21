// End-to-end ConfuciusTTS pipeline for c4tts.
//
// Composes the verified stages into the full zero-shot TTS engine:
//   prompt wav -> {W2V-BERT semantic feats, CAMPPlus style, reference mel}
//   text token ids -> T2S (semantic tokens + LM latent)
//   -> S2A (mel via flow matching) -> BigVGAN (waveform).
//
// Mirrors confuciustts/cli/inference.py:ConfuciusTTS. Text tokenization and
// normalization are upstream (SentencePiece); this engine consumes token ids.

#pragma once

#include <string>
#include <vector>

#include "c4tts/campplus.h"
#include "c4tts/s2a.h"
#include "c4tts/t2s.h"
#include "c4tts/tensor.h"
#include "c4tts/vocoder.h"
#include "c4tts/w2vbert.h"
#include "c4tts/weights.h"

namespace c4 {

struct SynthOptions {
  int n_timesteps = 25;
  float inference_cfg_rate = 0.7f;
  int max_new_tokens = 1000;
  bool sample = true;
  float temperature = 0.8f;
  int top_k = 30;
  float top_p = 0.8f;
  float repetition_penalty = 10.0f;
  uint64_t seed = 1234;
  float length_ratio = 1.72f;  // target mel frames per semantic token
};

class Pipeline {
 public:
  // `weights_root` contains subdirs: w2vbert/, campplus/, s2a/, t2s/, audio/.
  explicit Pipeline(const std::string& weights_root);

  // Reference conditioning extracted once from the prompt audio.
  struct Prompt {
    Tensor semantic;  // (1, T, 1024) normalized W2V-BERT[17] features
    Tensor style;     // (1, 192) CAMPPlus style embedding
    Tensor ref_mel;   // (1, T_mel, 80) reference mel
  };
  Prompt make_prompt(const std::string& prompt_wav_path) const;

  // Full synthesis: returns a mono 22.05 kHz waveform (T,).
  Tensor synth(const std::string& prompt_wav_path,
               const std::vector<int>& text_token_ids,
               const SynthOptions& opt = {}) const;

  int sample_rate() const { return 22050; }

 private:
  std::string root_;
  WeightStore w2v_w_, camp_w_, s2a_w_, t2s_w_, audio_w_, bigvgan_w_;
  W2VBert w2vbert_;
  CAMPPlus campplus_;
  t2s::Text2Semantic t2s_;
  s2a::MaskedDiffWithXvec s2a_;
  vocoder::BigVGAN bigvgan_;
};

}  // namespace c4
