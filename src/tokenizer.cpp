#include "c4tts/tokenizer.h"

#include <cstdio>
#include <fstream>
#include <queue>
#include <sstream>
#include <stdexcept>

namespace c4 {

namespace {
const std::string kMeta = "\xe2\x96\x81";  // U+2581 "▁"

std::string unescape(const std::string& s) {
  std::string out;
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '\\' && i + 1 < s.size()) {
      if (s[i + 1] == 't') { out += '\t'; ++i; continue; }
      if (s[i + 1] == 'n') { out += '\n'; ++i; continue; }
    }
    out += s[i];
  }
  return out;
}

// Length of the UTF-8 character starting at byte b.
int utf8_len(unsigned char b) {
  if (b < 0x80) return 1;
  if ((b >> 5) == 0x6) return 2;
  if ((b >> 4) == 0xE) return 3;
  if ((b >> 3) == 0x1E) return 4;
  return 1;
}
}  // namespace

Tokenizer::Tokenizer(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("tokenizer: cannot open " + path);
  for (int i = 0; i < 256; ++i) byte2id_[i] = -1;

  std::string line;
  int id = 0;
  while (std::getline(f, line)) {
    // "<type>\t<score>\t<piece>"
    auto t1 = line.find('\t');
    auto t2 = line.find('\t', t1 + 1);
    if (t1 == std::string::npos || t2 == std::string::npos) {
      id2score_.push_back(0.0f);
      ++id;
      continue;
    }
    float score = std::stof(line.substr(t1 + 1, t2 - t1 - 1));
    std::string piece = unescape(line.substr(t2 + 1));
    piece2id_[piece] = id;
    id2score_.push_back(score);

    // Byte pieces "<0xXX>" -> byte value, for fallback.
    if (piece.size() == 6 && piece[0] == '<' && piece[1] == '0' &&
        piece[2] == 'x' && piece[5] == '>') {
      int v = std::stoi(piece.substr(3, 2), nullptr, 16);
      byte2id_[v] = id;
    }
    ++id;
  }
}

std::vector<int> Tokenizer::encode(const std::string& text, bool add_bos,
                                   bool add_eos) const {
  // Normalize: whitespace -> metaspace (U+2581). Matches HuggingFace
  // LlamaTokenizer with legacy=False, which does NOT add a dummy prefix — the
  // first token keeps its leading metaspace only if the text starts with a space.
  std::string norm;
  for (char c : text) norm += (c == ' ') ? kMeta : std::string(1, c);

  // Initial symbols: one per UTF-8 character.
  struct Sym { std::string piece; int prev, next; bool alive; };
  std::vector<Sym> sym;
  for (size_t i = 0; i < norm.size();) {
    int l = utf8_len(static_cast<unsigned char>(norm[i]));
    sym.push_back({norm.substr(i, l), (int)sym.size() - 1, (int)sym.size() + 1, true});
    i += l;
  }
  if (sym.empty()) {
    std::vector<int> out;
    if (add_bos) out.push_back(bos_id());
    if (add_eos) out.push_back(eos_id());
    return out;
  }
  sym.back().next = -1;

  // Candidate merge: highest score first, then leftmost.
  struct Cand { float score; int left; int len; };  // len = combined byte length (staleness check)
  struct Cmp {
    bool operator()(const Cand& a, const Cand& b) const {
      if (a.score != b.score) return a.score < b.score;  // max-heap on score
      return a.left > b.left;                              // tie -> smaller left
    }
  };
  std::priority_queue<Cand, std::vector<Cand>, Cmp> pq;

  auto try_push = [&](int left) {
    if (left < 0) return;
    int right = sym[left].next;
    if (right < 0) return;
    auto it = piece2id_.find(sym[left].piece + sym[right].piece);
    if (it == piece2id_.end()) return;
    pq.push({id2score_[it->second], left,
             (int)(sym[left].piece.size() + sym[right].piece.size())});
  };
  for (int i = 0; i < (int)sym.size(); ++i) try_push(i);

  while (!pq.empty()) {
    Cand c = pq.top();
    pq.pop();
    int left = c.left;
    if (!sym[left].alive) continue;
    int right = sym[left].next;
    if (right < 0 || !sym[right].alive) continue;
    // Staleness: combined length must still match this candidate.
    if ((int)(sym[left].piece.size() + sym[right].piece.size()) != c.len) continue;
    if (piece2id_.find(sym[left].piece + sym[right].piece) == piece2id_.end()) continue;

    // Merge right into left.
    sym[left].piece += sym[right].piece;
    sym[right].alive = false;
    sym[left].next = sym[right].next;
    if (sym[right].next >= 0) sym[sym[right].next].prev = left;

    try_push(left);             // (left, new next)
    try_push(sym[left].prev);   // (prev, left)
  }

  // Emit ids, with byte fallback for unknown symbols.
  std::vector<int> out;
  if (add_bos) out.push_back(bos_id());
  for (int i = 0; i >= 0; i = sym[i].next) {
    if (!sym[i].alive) continue;
    auto it = piece2id_.find(sym[i].piece);
    if (it != piece2id_.end()) {
      out.push_back(it->second);
    } else {
      for (unsigned char b : sym[i].piece) {
        int bid = byte2id_[b];
        out.push_back(bid >= 0 ? bid : 0 /*unk*/);
      }
    }
  }
  if (add_eos) out.push_back(eos_id());
  return out;
}

}  // namespace c4
