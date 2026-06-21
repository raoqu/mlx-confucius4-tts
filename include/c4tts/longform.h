// Long-form synthesis helpers for c4tts.
//
// The T2S model has fixed trained sequence lengths (semantic position
// embedding 1520, text position embedding 520) and a per-call max_new_tokens
// cap, so long input must be split into sentence-sized segments, each
// synthesized independently with the same voice conditioning, then merged.
//
// Faithful ports of confuciustts/frontend/text_normalizer.py:segment_text and
// confuciustts/utils/audio_post.py:cross_fade_concat.

#pragma once

#include <functional>
#include <string>
#include <vector>

#include "c4tts/tensor.h"

namespace c4 {

// Splits `text` at sentence punctuation into segments of at most `max_tokens`
// "length" units, merging short trailing pieces. Length is character count for
// Chinese (lang=="zh"); for other languages it uses `token_len` if provided
// (e.g. the tokenizer's token count), else falls back to character count.
std::vector<std::string> segment_text(
    const std::string& text,
    const std::string& lang,
    const std::function<int(const std::string&)>& token_len = {},
    int max_tokens = 80,
    int min_tokens = 60,
    int merge_threshold = 20);

// Concatenates per-segment waveforms (each 1-D, T samples) with a short
// fade-out / silence / fade-in between consecutive segments. Returns a 1-D
// waveform. A single chunk is returned unchanged.
Tensor cross_fade_concat(const std::vector<Tensor>& chunks,
                         int sample_rate,
                         float silence_duration = 0.3f);

}  // namespace c4
