// Test: SentencePiece BPE tokenizer parity vs spm.encode().
// Requires: export_tokenizer.py (vocab + golden). Skips if vocab absent.

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "c4tts/tokenizer.h"

static bool exists(const std::string& p) {
  struct stat st {};
  return stat(p.c_str(), &st) == 0;
}

int main() {
  const std::string vocab = std::string(C4TTS_WEIGHTS_DIR) + "/tokenizer/vocab.tsv";
  const std::string cases = std::string(C4TTS_GOLDEN_DIR) + "/tokenizer/cases.tsv";
  std::printf("test_tokenizer:\n");
  if (!exists(vocab) || !exists(cases)) {
    std::printf("  SKIP: run tools/export_tokenizer.py\n");
    return 0;
  }

  c4::Tokenizer tok(vocab);
  std::ifstream f(cases);
  std::string line;
  int failures = 0, n = 0;
  while (std::getline(f, line)) {
    auto tab = line.find('\t');
    if (tab == std::string::npos) continue;
    std::string text = line.substr(0, tab);
    std::vector<int> ref;
    std::stringstream ss(line.substr(tab + 1));
    std::string tk;
    while (std::getline(ss, tk, ',')) if (!tk.empty()) ref.push_back(std::stoi(tk));

    std::vector<int> got = tok.encode(text);
    ++n;
    bool ok = got == ref;
    std::printf("  [%s] %s (got %zu, ref %zu ids)\n",
                ok ? "ok " : "FAIL", text.substr(0, 40).c_str(), got.size(), ref.size());
    if (!ok) {
      ++failures;
      std::printf("      got:");
      for (size_t i = 0; i < got.size() && i < 12; ++i) std::printf(" %d", got[i]);
      std::printf("\n      ref:");
      for (size_t i = 0; i < ref.size() && i < 12; ++i) std::printf(" %d", ref[i]);
      std::printf("\n");
    }
  }
  std::printf("test_tokenizer: %s (%d cases)\n",
              failures == 0 ? "ALL PASSED" : "FAILED", n);
  return failures == 0 ? 0 : 1;
}
