#include "c4tts/pipeline.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "c4tts/audio.h"
#include "c4tts/wav_io.h"
#include "mlx/mlx.h"

namespace c4 {
namespace mx = mlx::core;

namespace {
// Lightweight stage timer, enabled with C4TTS_TIMING=1.
struct Stage {
  const char* name;
  std::chrono::steady_clock::time_point t0;
  bool on;
  explicit Stage(const char* n)
      : name(n), t0(std::chrono::steady_clock::now()),
        on(std::getenv("C4TTS_TIMING") != nullptr) {}
  void done(const mx::array& a) const {
    if (!on) return;
    mx::eval(a);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "  [timing] %-12s %6lld ms\n", name, (long long)ms);
  }
};
}  // namespace

Pipeline::Pipeline(const std::string& weights_root)
    : root_(weights_root),
      w2v_w_(weights_root + "/w2vbert"),
      camp_w_(weights_root + "/campplus"),
      s2a_w_(weights_root + "/s2a"),
      t2s_w_(weights_root + "/t2s"),
      audio_w_(weights_root + "/audio"),
      bigvgan_w_(weights_root + "/bigvgan"),
      w2vbert_(w2v_w_, "", /*heads=*/16, /*left=*/64, /*right=*/8, /*kernel=*/31),
      campplus_(camp_w_),
      t2s_(t2s_w_, /*num_layers=*/24, /*num_heads=*/20),
      s2a_(s2a_w_),
      bigvgan_(bigvgan_w_) {}

Pipeline::Prompt Pipeline::make_prompt(const std::string& path) const {
  WavData wav = read_wav(path);
  Tensor w16 = resample(wav.samples, wav.sample_rate, 16000);
  Tensor w22 = resample(wav.samples, wav.sample_rate, 22050);

  // Semantic features: seamless -> W2V-BERT[17] -> normalize.
  Tensor feats = mx::expand_dims(seamless_features(w16), 0);  // (1, T, 160)
  Tensor h = w2vbert_.forward(feats, 17);                     // (1, T, 1024)
  Tensor sem = mx::divide(mx::subtract(h, w2v_w_.get("semantic_mean")),
                          w2v_w_.get("semantic_std"));

  // Style embedding: kaldi fbank -> mean-sub -> CAMPPlus.
  Tensor fb = kaldi_fbank(w16, 80, 16000.0f);
  fb = mx::subtract(fb, mx::mean(fb, 0, /*keepdims=*/true));
  Tensor style = campplus_.forward(mx::expand_dims(fb, 0));   // (1, 192)

  // Reference mel at 22.05 kHz: (1, T_mel, 80).
  Tensor mel = mel_spectrogram(w22, audio_w_.get("mel_basis"), audio_w_.get("hann"));
  Tensor ref_mel = mx::expand_dims(mx::transpose(mel, {1, 0}), 0);

  return Prompt{sem, style, ref_mel};
}

Tensor Pipeline::synth(const std::string& path,
                       const std::vector<int>& text_token_ids,
                       const SynthOptions& opt) const {
  Stage tp("prompt");
  Prompt p = make_prompt(path);
  tp.done(p.semantic);
  return synth(p, text_token_ids, opt);
}

Tensor Pipeline::synth(const Prompt& p,
                       const std::vector<int>& text_token_ids,
                       const SynthOptions& opt) const {
  // Text token ids -> (1, T_text) int32.
  const int n = static_cast<int>(text_token_ids.size());
  int32_t* heap = new int32_t[n > 0 ? n : 1];
  for (int i = 0; i < n; ++i) heap[i] = text_token_ids[i];
  Tensor text_ids(static_cast<void*>(heap), mx::Shape{1, n}, mx::int32,
                  [](void* q) { delete[] static_cast<int32_t*>(q); });

  // T2S: text + semantic conditioning -> semantic tokens + LM latent.
  Stage tt("t2s");
  auto gen = t2s_.generate(text_ids, p.semantic, opt.max_new_tokens, opt.sample,
                           opt.temperature, opt.top_k, opt.top_p,
                           opt.repetition_penalty, opt.seed);
  tt.done(gen.latent);
  const int n_codes = gen.codes.shape(1);
  const int target_len = static_cast<int>(n_codes * opt.length_ratio);

  // Initial flow-matching noise (B=1, 80, T_ref + target). Seeded for
  // reproducibility.
  const int T_ref = p.ref_mel.shape(1);
  Tensor z = mx::random::normal({1, 80, T_ref + target_len}, mx::random::key(opt.seed));

  // S2A -> mel; BigVGAN -> waveform.
  Stage ts("s2a");
  Tensor mel = s2a_.inference(gen.codes, gen.latent, p.ref_mel, p.style,
                              target_len, z, opt.n_timesteps,
                              opt.inference_cfg_rate);
  ts.done(mel);
  Stage tv("bigvgan");
  Tensor wav = bigvgan_.forward(mel);  // (1, 1, T*256)
  tv.done(wav);
  return mx::flatten(wav);
}

}  // namespace c4
