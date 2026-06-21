#include "c4tts/longform.h"

#include <algorithm>
#include <cstdint>

#include "mlx/mlx.h"

namespace c4 {
namespace mx = mlx::core;

namespace {

// Splits a UTF-8 string into its code points (each returned as a substring).
std::vector<std::string> utf8_chars(const std::string& s) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i < s.size()) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        size_t len = 1;
        if (c >= 0xF0) len = 4;
        else if (c >= 0xE0) len = 3;
        else if (c >= 0xC0) len = 2;
        if (i + len > s.size()) len = 1;  // truncated; treat byte as one char
        out.push_back(s.substr(i, len));
        i += len;
    }
    return out;
}

bool contains(const std::vector<std::string>& set, const std::string& c) {
    return std::find(set.begin(), set.end(), c) != set.end();
}

}  // namespace

std::vector<std::string> segment_text(const std::string& text,
                                      const std::string& lang,
                                      const std::function<int(const std::string&)>& token_len,
                                      int max_tokens, int min_tokens, int merge_threshold) {
    const bool zh = (lang == "zh");
    // "Length" of a piece: characters for zh, tokenizer tokens otherwise.
    auto calc_length = [&](const std::string& t) -> int {
        if (zh || !token_len) return static_cast<int>(utf8_chars(t).size());
        return token_len(t);
    };
    auto should_merge = [&](const std::string& t) {
        return calc_length(t) < merge_threshold;
    };

    const std::vector<std::string> punctuation =
        zh ? std::vector<std::string>{"。", "？", "！", "；", "：", ".", "?", "!", ";"}
           : std::vector<std::string>{".", "?", "!", ";", ":"};
    const std::vector<std::string> quotes = {"“", "”"};  // “ ”

    std::vector<std::string> chars = utf8_chars(text);
    // Ensure the text ends on a sentence boundary.
    if (!chars.empty() && !contains(punctuation, chars.back())) {
        chars.push_back(zh ? "。" : ".");
    }

    // Split at each punctuation mark, keeping the mark with the preceding text.
    std::vector<std::string> segments;
    size_t start = 0;
    for (size_t i = 0; i < chars.size(); ++i) {
        if (!contains(punctuation, chars[i])) continue;
        if (i > start) {
            std::string seg;
            for (size_t j = start; j < i; ++j) seg += chars[j];
            seg += chars[i];
            if (i + 1 < chars.size() && contains(quotes, chars[i + 1])) {
                seg += chars[i + 1];
                start = i + 2;
            } else {
                start = i + 1;
            }
            segments.push_back(seg);
        } else {
            start = i + 1;
        }
    }

    // A single over-long segment (no internal punctuation) is hard-split.
    if (segments.size() == 1 && calc_length(segments[0]) > max_tokens) {
        std::vector<std::string> sc = utf8_chars(segments[0]);
        if (!sc.empty() && contains(punctuation, sc.back())) sc.pop_back();
        segments.clear();
        for (size_t i = 0; i < sc.size(); i += static_cast<size_t>(max_tokens)) {
            std::string chunk;
            for (size_t j = i; j < sc.size() && j < i + static_cast<size_t>(max_tokens); ++j)
                chunk += sc[j];
            segments.push_back(chunk);
        }
    }

    // Greedily merge consecutive segments up to max_tokens.
    std::vector<std::string> final_segments;
    std::string current;
    for (const auto& seg : segments) {
        if (calc_length(current + seg) > max_tokens && calc_length(current) > min_tokens) {
            final_segments.push_back(current);
            current.clear();
        }
        current += seg;
    }
    if (!current.empty()) {
        if (should_merge(current) && !final_segments.empty()) {
            final_segments.back() += current;
        } else {
            final_segments.push_back(current);
        }
    }

    // For zh, drop a trailing sentence terminator (matches the reference).
    if (zh) {
        const std::vector<std::string> strip = {"。", "；", "：", ".", ";"};
        for (auto& seg : final_segments) {
            std::vector<std::string> sc = utf8_chars(seg);
            if (!sc.empty() && contains(strip, sc.back())) {
                seg.clear();
                for (size_t j = 0; j + 1 < sc.size(); ++j) seg += sc[j];
            }
        }
    }

    // Drop punctuation-only / empty segments.
    std::vector<std::string> result;
    for (const auto& seg : final_segments) {
        bool has_content = false;
        for (const auto& ch : utf8_chars(seg)) {
            if (!contains(punctuation, ch) && ch != " " && ch != "“" && ch != "”") {
                has_content = true;
                break;
            }
        }
        if (has_content) result.push_back(seg);
    }
    if (result.empty() && !text.empty()) result.push_back(text);  // never return nothing
    return result;
}

Tensor cross_fade_concat(const std::vector<Tensor>& chunks, int sample_rate,
                         float silence_duration) {
    if (chunks.empty()) return mx::zeros({0}, mx::float32);
    if (chunks.size() == 1) return chunks[0];

    // Pull each chunk to host as a flat float vector (waveforms are small).
    auto to_host = [](const Tensor& t) {
        Tensor f = mx::astype(mx::flatten(t), mx::float32);
        mx::eval(f);
        const float* d = f.data<float>();
        return std::vector<float>(d, d + f.size());
    };

    const int total_n = static_cast<int>(silence_duration * sample_rate);
    const int fade_n = total_n / 3;
    const int silence_n = fade_n;

    std::vector<float> merged = to_host(chunks[0]);
    for (size_t c = 1; c < chunks.size(); ++c) {
        std::vector<float> next = to_host(chunks[c]);
        // Fade out the tail of merged.
        const int fout = std::min<int>(fade_n, static_cast<int>(merged.size()));
        for (int i = 0; i < fout; ++i) {
            const float w = 1.0f - static_cast<float>(i) / std::max(1, fout - 1);
            merged[merged.size() - fout + i] *= w;
        }
        // Silence gap.
        merged.insert(merged.end(), static_cast<size_t>(std::max(0, silence_n)), 0.0f);
        // Fade in the head of next.
        const int fin = std::min<int>(fade_n, static_cast<int>(next.size()));
        for (int i = 0; i < fin; ++i) {
            const float w = static_cast<float>(i) / std::max(1, fin - 1);
            next[i] *= w;
        }
        merged.insert(merged.end(), next.begin(), next.end());
    }

    const int n = static_cast<int>(merged.size());
    float* heap = new float[n > 0 ? n : 1];
    std::copy(merged.begin(), merged.end(), heap);
    return mx::array(static_cast<void*>(heap), mx::Shape{n}, mx::float32,
                     [](void* p) { delete[] static_cast<float*>(p); });
}

}  // namespace c4
