// SentencePiece BPE tokenizer for c4tts (LLaMA-style, used by Confucius4-TTS).
//
// Reproduces sentencepiece's BPE encode() bit-exactly: identity normalizer,
// add_dummy_prefix, whitespace -> U+2581 metaspace, highest-score adjacent
// merges, and byte_fallback (<0xXX>) for out-of-vocab characters. The vocab is
// exported by tools/export_tokenizer.py.

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace c4 {

class Tokenizer {
 public:
  // Loads the exported vocab.tsv (one line per id: "<type>\t<score>\t<piece>").
  explicit Tokenizer(const std::string& vocab_tsv_path);

  // Encodes text to token ids (matches spm.encode). Optionally prepends BOS (1)
  // / appends EOS (2), matching the LlamaTokenizer special-token config.
  std::vector<int> encode(const std::string& text, bool add_bos = false,
                          bool add_eos = false) const;

  int bos_id() const { return 1; }
  int eos_id() const { return 2; }
  int vocab_size() const { return static_cast<int>(id2score_.size()); }

 private:
  std::unordered_map<std::string, int> piece2id_;
  std::vector<float> id2score_;
  int byte2id_[256];  // <0xXX> piece ids for byte fallback
};

}  // namespace c4
