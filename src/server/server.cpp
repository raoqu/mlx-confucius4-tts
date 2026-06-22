// ---------------------------------------------------------------------------
// c4tts HTTP server + web admin.
//
// Ported from the index-tts2-metal runtime server so c4tts exposes the same
// web console and the same HTTP API:
//
//   OpenAI-style endpoints:
//     POST   /v1/audio/speech
//     GET    /v1/audio/voices              POST /v1/audio/voices
//     GET/PATCH/DELETE /v1/audio/voices/{id}
//     GET/POST /v1/audio/voice_consents    GET/PATCH/DELETE /v1/audio/voice_consents/{id}
//
//   Local web management endpoints (served under /web/api/*):
//     GET    /status
//     GET    /voices                       POST /voices
//     GET/PATCH/DELETE /voices/{id}
//     GET    /voices/{id}/source-audio
//     POST   /speech                        POST /login
//
// Voice metadata lives in sqlite (voices.sqlite); uploaded reference audio is
// kept under <store>/samples and c4tts voice bundles under <store>/bundles.
// Synthesis runs through a resident c4::Pipeline loaded once at startup.
//
// A c4tts voice bundle (.pt) is a c4tts-native container: a short magic
// header followed by the source reference WAV bytes. The conditioning
// (W2V-BERT semantics, CAMPPlus style, reference mel) is derived from that
// audio at synth time by the pipeline, so the bundle only needs to carry the
// reference waveform.
// ---------------------------------------------------------------------------

#include "c4tts/server.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sqlite3.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "c4tts/lang_tokens.h"
#include "c4tts/longform.h"
#include "c4tts/pipeline.h"
#include "c4tts/tokenizer.h"
#include "c4tts/wav_io.h"
#include "mlx/mlx.h"

namespace c4 {
namespace server {
namespace {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

constexpr const char* kBundleMagic = "C4TTSVB1\n";  // 9 bytes
constexpr int kSampleRate = 22050;

// ---------------------------------------------------------------------------
// Small string / JSON / HTTP helpers (ported verbatim in spirit from the
// reference server; c4tts has no JSON dependency so we parse the few fields we
// need by hand).
// ---------------------------------------------------------------------------

std::string json_escape(const std::string& value) {
    std::string out;
    out.reserve(value.size() + 8);
    for (const char c : value) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

std::string env_string(const char* name, const std::string& fallback) {
    if (const char* v = std::getenv(name)) {
        if (*v) return v;
    }
    return fallback;
}

uint32_t env_u32(const char* name, uint32_t fallback, uint32_t min_value, uint32_t max_value) {
    if (const char* v = std::getenv(name)) {
        const long n = std::strtol(v, nullptr, 10);
        if (n >= static_cast<long>(min_value) && n <= static_cast<long>(max_value)) {
            return static_cast<uint32_t>(n);
        }
    }
    return fallback;
}

std::string now_epoch_string() {
    return std::to_string(static_cast<long long>(std::time(nullptr)));
}

std::string make_id(const std::string& prefix) {
    static std::atomic<uint64_t> counter{0};
    const auto n = counter.fetch_add(1);
    const auto t = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    std::ostringstream out;
    out << prefix << "_" << std::hex << t << n;
    return out.str();
}

std::string lower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
}

std::string content_type_from_head(const std::string& head) {
    const std::string lower = lower_copy(head);
    const size_t p = lower.find("content-type:");
    if (p == std::string::npos) return {};
    size_t start = p + 13;
    while (start < head.size() && std::isspace(static_cast<unsigned char>(head[start]))) ++start;
    size_t end = head.find("\r\n", start);
    if (end == std::string::npos) end = head.size();
    return head.substr(start, end - start);
}

std::string header_value(const std::string& head, const std::string& name) {
    const std::string lower = lower_copy(head);
    std::string needle = lower_copy(name) + ":";
    const size_t p = lower.find(needle);
    if (p == std::string::npos) return {};
    size_t start = p + needle.size();
    while (start < head.size() && std::isspace(static_cast<unsigned char>(head[start]))) ++start;
    size_t end = head.find("\r\n", start);
    if (end == std::string::npos) end = head.size();
    return head.substr(start, end - start);
}

std::string route_path_from_head(const std::string& head, std::string& method) {
    const size_t eol = head.find("\r\n");
    const std::string first = eol == std::string::npos ? head : head.substr(0, eol);
    const size_t sp1 = first.find(' ');
    const size_t sp2 = sp1 == std::string::npos ? std::string::npos : first.find(' ', sp1 + 1);
    if (sp1 == std::string::npos || sp2 == std::string::npos) {
        method.clear();
        return {};
    }
    method = first.substr(0, sp1);
    std::string path = first.substr(sp1 + 1, sp2 - sp1 - 1);
    const size_t q = path.find('?');
    if (q != std::string::npos) path.resize(q);
    return path;
}

std::string json_string_field(const std::string& body, const std::string& key) {
    const std::string needle = "\"" + key + "\"";
    size_t pos = body.find(needle);
    while (pos != std::string::npos) {
        size_t p = pos + needle.size();
        while (p < body.size() && std::isspace(static_cast<unsigned char>(body[p]))) ++p;
        if (p < body.size() && body[p] == ':') {
            ++p;
            while (p < body.size() && std::isspace(static_cast<unsigned char>(body[p]))) ++p;
            if (p < body.size() && body[p] == '"') {
                ++p;
                std::string out;
                while (p < body.size() && body[p] != '"') {
                    char c = body[p];
                    if (c == '\\' && p + 1 < body.size()) {
                        const char e = body[p + 1];
                        p += 2;
                        switch (e) {
                            case 'n': out += '\n'; break;
                            case 'r': out += '\r'; break;
                            case 't': out += '\t'; break;
                            case 'b': out += '\b'; break;
                            case 'f': out += '\f'; break;
                            case 'u': {
                                if (p + 4 <= body.size()) {
                                    const unsigned cp = static_cast<unsigned>(
                                        std::strtoul(body.substr(p, 4).c_str(), nullptr, 16));
                                    p += 4;
                                    if (cp < 0x80) {
                                        out += static_cast<char>(cp);
                                    } else if (cp < 0x800) {
                                        out += static_cast<char>(0xC0 | (cp >> 6));
                                        out += static_cast<char>(0x80 | (cp & 0x3F));
                                    } else {
                                        out += static_cast<char>(0xE0 | (cp >> 12));
                                        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                                        out += static_cast<char>(0x80 | (cp & 0x3F));
                                    }
                                }
                                break;
                            }
                            default: out += e; break;
                        }
                    } else {
                        out += c;
                        ++p;
                    }
                }
                return out;
            }
        }
        pos = body.find(needle, pos + 1);
    }
    return {};
}

bool json_bool_field(const std::string& body, const std::string& key, bool fallback) {
    const std::string needle = "\"" + key + "\"";
    size_t pos = body.find(needle);
    while (pos != std::string::npos) {
        size_t p = pos + needle.size();
        while (p < body.size() && std::isspace(static_cast<unsigned char>(body[p]))) ++p;
        if (p < body.size() && body[p] == ':') {
            ++p;
            while (p < body.size() && std::isspace(static_cast<unsigned char>(body[p]))) ++p;
            if (body.compare(p, 4, "true") == 0) return true;
            if (body.compare(p, 5, "false") == 0) return false;
            return fallback;
        }
        pos = body.find(needle, pos + 1);
    }
    return fallback;
}

long json_int_field(const std::string& body, const std::string& key, long fallback) {
    const std::string needle = "\"" + key + "\"";
    size_t pos = body.find(needle);
    while (pos != std::string::npos) {
        size_t p = pos + needle.size();
        while (p < body.size() && std::isspace(static_cast<unsigned char>(body[p]))) ++p;
        if (p < body.size() && body[p] == ':') {
            ++p;
            while (p < body.size() && std::isspace(static_cast<unsigned char>(body[p]))) ++p;
            char* end = nullptr;
            const long n = std::strtol(body.c_str() + p, &end, 10);
            if (end != body.c_str() + p) return n;
            return fallback;
        }
        pos = body.find(needle, pos + 1);
    }
    return fallback;
}

// Accepts both {"voice":"id"} and {"voice":{"id":"..."}}.
std::string json_voice_field(const std::string& body) {
    const std::string direct = json_string_field(body, "voice");
    if (!direct.empty()) return direct;
    const size_t p = body.find("\"voice\"");
    if (p == std::string::npos) return {};
    const size_t id_pos = body.find("\"id\"", p);
    if (id_pos == std::string::npos) return {};
    const size_t end_obj = body.find('}', p);
    if (end_obj != std::string::npos && id_pos > end_obj) return {};
    return json_string_field(body.substr(id_pos), "id");
}

// ---------------------------------------------------------------------------
// Multipart form parsing (for audio uploads).
// ---------------------------------------------------------------------------

struct MultipartPart {
    std::string name;
    std::string filename;
    std::string content_type;
    std::string data;
};

std::string trim_crlf(std::string s) {
    while (!s.empty() && (s.back() == '\r' || s.back() == '\n')) s.pop_back();
    return s;
}

std::vector<MultipartPart> parse_multipart(const std::string& content_type, const std::string& body) {
    std::vector<MultipartPart> parts;
    const std::string key = "boundary=";
    const size_t bp = content_type.find(key);
    if (bp == std::string::npos) return parts;
    std::string boundary = content_type.substr(bp + key.size());
    const size_t semicolon = boundary.find(';');
    if (semicolon != std::string::npos) boundary.resize(semicolon);
    if (boundary.size() >= 2 && boundary.front() == '"' && boundary.back() == '"') {
        boundary = boundary.substr(1, boundary.size() - 2);
    }
    const std::string marker = "--" + boundary;
    size_t pos = body.find(marker);
    while (pos != std::string::npos) {
        pos += marker.size();
        if (body.compare(pos, 2, "--") == 0) break;
        if (body.compare(pos, 2, "\r\n") == 0) pos += 2;
        const size_t header_end = body.find("\r\n\r\n", pos);
        if (header_end == std::string::npos) break;
        const std::string headers = body.substr(pos, header_end - pos);
        size_t data_start = header_end + 4;
        size_t next = body.find(marker, data_start);
        if (next == std::string::npos) break;
        std::string data = trim_crlf(body.substr(data_start, next - data_start));
        MultipartPart part;
        part.data = std::move(data);
        const std::string lower = lower_copy(headers);
        const size_t cd = lower.find("content-disposition:");
        if (cd != std::string::npos) {
            const size_t line_end = lower.find("\r\n", cd);
            const std::string line = headers.substr(cd, (line_end == std::string::npos ? headers.size() : line_end) - cd);
            auto attr = [&](const std::string& k) -> std::string {
                const std::string nd = k + "=\"";
                const size_t a = line.find(nd);
                if (a == std::string::npos) return {};
                const size_t s = a + nd.size();
                const size_t e = line.find('"', s);
                return e == std::string::npos ? std::string{} : line.substr(s, e - s);
            };
            part.name = attr("name");
            part.filename = attr("filename");
        }
        const size_t ct = lower.find("content-type:");
        if (ct != std::string::npos) {
            size_t s = ct + 13;
            while (s < headers.size() && std::isspace(static_cast<unsigned char>(headers[s]))) ++s;
            size_t e = headers.find("\r\n", s);
            if (e == std::string::npos) e = headers.size();
            part.content_type = headers.substr(s, e - s);
        }
        if (!part.name.empty()) parts.push_back(std::move(part));
        pos = next;
    }
    return parts;
}

std::string multipart_value(const std::vector<MultipartPart>& parts, const std::string& name) {
    for (const auto& p : parts) {
        if (p.name == name) return p.data;
    }
    return {};
}

std::optional<MultipartPart> multipart_file(const std::vector<MultipartPart>& parts,
                                            const std::vector<std::string>& names) {
    for (const auto& n : names) {
        for (const auto& p : parts) {
            if (p.name == n && (!p.filename.empty() || !p.data.empty())) return p;
        }
    }
    return std::nullopt;
}

// ---------------------------------------------------------------------------
// Socket I/O.
// ---------------------------------------------------------------------------

bool send_all(int fd, const char* data, size_t len) {
    size_t sent = 0;
    while (sent < len) {
        const ssize_t n = ::send(fd, data + sent, len - sent, 0);
        if (n <= 0) return false;
        sent += static_cast<size_t>(n);
    }
    return true;
}

void send_response(int fd, int code, const std::string& status,
                   const std::string& content_type, const std::string& body) {
    std::ostringstream head;
    head << "HTTP/1.1 " << code << " " << status << "\r\n"
         << "Content-Type: " << content_type << "\r\n"
         << "Content-Length: " << body.size() << "\r\n"
         << "Connection: close\r\n\r\n";
    const std::string h = head.str();
    send_all(fd, h.data(), h.size());
    send_all(fd, body.data(), body.size());
}

void send_json_error(int fd, int code, const std::string& status, const std::string& message) {
    send_response(fd, code, status, "application/json",
                  "{\"error\":{\"message\":\"" + json_escape(message) + "\"}}");
}

bool read_request(int fd, std::string& head, std::string& body) {
    std::string buf;
    char chunk[8192];
    size_t header_end = std::string::npos;
    while (header_end == std::string::npos) {
        const ssize_t n = ::recv(fd, chunk, sizeof(chunk), 0);
        if (n <= 0) return false;
        buf.append(chunk, static_cast<size_t>(n));
        header_end = buf.find("\r\n\r\n");
        if (buf.size() > (1u << 20)) return false;
    }
    head = buf.substr(0, header_end);
    body = buf.substr(header_end + 4);
    size_t content_length = 0;
    const std::string lower = lower_copy(head);
    const size_t cl = lower.find("content-length:");
    if (cl != std::string::npos) {
        content_length = static_cast<size_t>(std::strtoull(lower.c_str() + cl + 15, nullptr, 10));
    }
    if (content_length > (128u << 20)) return false;
    while (body.size() < content_length) {
        const ssize_t n = ::recv(fd, chunk, sizeof(chunk), 0);
        if (n <= 0) return false;
        body.append(chunk, static_cast<size_t>(n));
    }
    if (body.size() > content_length) body.resize(content_length);
    return true;
}

// ---------------------------------------------------------------------------
// File helpers + WAV duration + c4tts voice bundle container.
// ---------------------------------------------------------------------------

void save_file_bytes(const std::string& path, const std::string& data) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("failed to write file: " + path);
    out.write(data.data(), static_cast<std::streamsize>(data.size()));
}

std::string read_file_bytes(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

bool path_is_under_dir(const fs::path& path, const fs::path& dir) {
    std::error_code ec;
    const fs::path cp = fs::weakly_canonical(path, ec);
    if (ec) return false;
    const fs::path cd = fs::weakly_canonical(dir, ec);
    if (ec) return false;
    auto dit = cd.begin();
    auto pit = cp.begin();
    for (; dit != cd.end(); ++dit, ++pit) {
        if (pit == cp.end() || *pit != *dit) return false;
    }
    return pit != cp.end();
}

bool remove_path_under_dir(const std::string& path, const std::string& dir, const char* label) {
    if (path.empty() || dir.empty()) return false;
    const fs::path p(path);
    std::error_code ec;
    if (!fs::exists(p, ec) || ec) return false;
    if (!path_is_under_dir(p, fs::path(dir))) return false;
    if (fs::is_directory(p, ec)) {
        fs::remove_all(p, ec);
    } else {
        fs::remove(p, ec);
    }
    if (ec) {
        std::cerr << "warning: failed to delete " << label << ": " << path
                  << " (" << ec.message() << ")" << std::endl;
        return false;
    }
    return true;
}

std::string safe_ext(const std::string& filename, const std::string& fallback) {
    const size_t dot = filename.rfind('.');
    if (dot == std::string::npos || dot + 1 >= filename.size()) return fallback;
    std::string ext = filename.substr(dot);
    if (ext.size() > 8) return fallback;
    for (char c : ext) {
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '.')) return fallback;
    }
    return ext;
}

uint16_t read_le16(const uint8_t* p) { return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8); }
uint32_t read_le32(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

double wav_duration_seconds_from_bytes(const std::string& bytes) {
    if (bytes.size() < 44 || bytes.compare(0, 4, "RIFF") != 0 || bytes.compare(8, 4, "WAVE") != 0) {
        return 0.0;
    }
    const auto* data = reinterpret_cast<const uint8_t*>(bytes.data());
    uint16_t channels = 0, bits = 0;
    uint32_t sample_rate = 0, data_bytes = 0;
    size_t pos = 12;
    while (pos + 8 <= bytes.size()) {
        const std::string id(bytes.data() + pos, bytes.data() + pos + 4);
        const uint32_t size = read_le32(data + pos + 4);
        const size_t cdata = pos + 8;
        if (cdata + size > bytes.size()) break;
        if (id == "fmt " && size >= 16) {
            channels = read_le16(data + cdata + 2);
            sample_rate = read_le32(data + cdata + 4);
            bits = read_le16(data + cdata + 14);
        } else if (id == "data") {
            data_bytes = size;
        }
        pos = cdata + size + (size & 1u);
    }
    const double bps = static_cast<double>(sample_rate) * channels * bits / 8.0;
    return bps > 0.0 ? static_cast<double>(data_bytes) / bps : 0.0;
}

bool bytes_are_wav(const std::string& bytes) {
    return bytes.size() >= 12 && bytes.compare(0, 4, "RIFF") == 0 && bytes.compare(8, 4, "WAVE") == 0;
}

const std::string kBundleMagicStr = kBundleMagic;

// Writes a c4tts voice bundle: magic header + raw source WAV bytes.
void write_voice_bundle(const std::string& path, const std::string& wav_bytes) {
    save_file_bytes(path, kBundleMagicStr + wav_bytes);
}

// Extracts the source WAV bytes from a c4tts voice bundle. Tolerates importing
// a plain .wav file as a bundle (returns the file unchanged in that case).
std::string voice_bundle_source_wav(const std::string& bundle_path) {
    const std::string raw = read_file_bytes(bundle_path);
    if (raw.size() >= kBundleMagicStr.size() &&
        raw.compare(0, kBundleMagicStr.size(), kBundleMagicStr) == 0) {
        return raw.substr(kBundleMagicStr.size());
    }
    if (bytes_are_wav(raw)) return raw;
    return {};
}

bool path_is_voice_bundle(const std::string& path) {
    std::error_code ec;
    if (!fs::is_regular_file(fs::path(path), ec) || ec) return false;
    return !voice_bundle_source_wav(path).empty();
}

// ---------------------------------------------------------------------------
// Synthesis engine: a resident pipeline + tokenizer, loaded once.
// ---------------------------------------------------------------------------

struct Engine {
    Pipeline pipeline;
    Tokenizer tokenizer;
    std::string default_lang;
    size_t lru_capacity;  // per-voice conditioning cache capacity (0 disables)

    Engine(const std::string& weights_dir, const std::string& lang, size_t lru_cap)
        : pipeline(weights_dir),
          tokenizer(weights_dir + "/tokenizer/vocab.tsv"),
          default_lang(lang),
          lru_capacity(lru_cap) {}

    // Synthesizes `text` with the reference WAV at `prompt_wav_path`. The
    // extracted conditioning (W2V-BERT semantics, CAMPPlus style, ref mel) is
    // cached per voice in an LRU keyed by `voice_key` + the wav's size/mtime,
    // so repeated requests for the same voice skip feature extraction. Writes a
    // 16-bit PCM WAV to `out_wav`. Single-worker-thread only (no locking).
    void synth_to_file(const std::string& voice_key,
                       const std::string& prompt_wav_path,
                       const std::string& text,
                       const std::string& lang,
                       int steps,
                       int max_tokens,
                       const std::string& out_wav) {
        SynthOptions opt;
        if (steps > 0) opt.n_timesteps = steps;
        if (max_tokens > 0) opt.max_new_tokens = max_tokens;
        const std::string l = lang.empty() ? default_lang : lang;

        // Split long input into model-sized segments (the conditioning is the
        // same for every segment, so it's extracted once and reused).
        auto token_len = [&](const std::string& t) {
            return static_cast<int>(tokenizer.encode(t, false, false).size());
        };
        const std::vector<std::string> segments = segment_text(text, l, token_len);

        if (lru_capacity == 0) {
            Pipeline::Prompt p = pipeline.make_prompt(prompt_wav_path);
            synth_segments(p, segments, l, opt, out_wav);
        } else {
            synth_segments(prompt_for(voice_key, prompt_wav_path), segments, l, opt, out_wav);
        }
    }

private:
    // Synthesizes each text segment with the shared conditioning and writes the
    // cross-faded concatenation.
    void synth_segments(const Pipeline::Prompt& prompt,
                        const std::vector<std::string>& segments,
                        const std::string& lang,
                        const SynthOptions& opt,
                        const std::string& out_wav) {
        std::vector<Tensor> waves;
        waves.reserve(segments.size());
        for (const auto& seg : segments) {
            std::vector<int> ids =
                tokenizer.encode(format_tts_text(seg, lang), /*add_bos=*/true, /*add_eos=*/true);
            if (ids.empty()) continue;
            waves.push_back(pipeline.synth(prompt, ids, opt));
        }
        if (waves.empty()) throw std::runtime_error("tokenizer produced no token ids");
        write_wav(out_wav, cross_fade_concat(waves, kSampleRate), kSampleRate);
    }

    // Cache key folds in the wav's size+mtime so editing a voice's audio (same
    // id) invalidates the entry.
    static std::string cache_key(const std::string& voice_key, const std::string& wav) {
        std::error_code ec;
        const auto sz = static_cast<unsigned long long>(fs::file_size(wav, ec));
        const auto wt = static_cast<long long>(
            fs::last_write_time(wav, ec).time_since_epoch().count());
        return voice_key + "|" + std::to_string(sz) + "|" + std::to_string(wt);
    }

    const Pipeline::Prompt& prompt_for(const std::string& voice_key,
                                       const std::string& wav) {
        const std::string key = cache_key(voice_key, wav);
        auto it = index_.find(key);
        if (it != index_.end()) {
            lru_.splice(lru_.begin(), lru_, it->second);  // move to front (MRU)
            return it->second->second;
        }
        lru_.emplace_front(key, pipeline.make_prompt(wav));
        index_[key] = lru_.begin();
        while (lru_.size() > lru_capacity) {
            index_.erase(lru_.back().first);
            lru_.pop_back();
        }
        return lru_.front().second;
    }

    std::list<std::pair<std::string, Pipeline::Prompt>> lru_;
    std::unordered_map<std::string, decltype(lru_)::iterator> index_;
};

// ---------------------------------------------------------------------------
// Work dispatcher: a single worker thread drains a bounded queue so the
// pipeline (and Metal) is only ever touched by one thread at a time. Mirrors
// the reference server's queue + status reporting.
// ---------------------------------------------------------------------------

struct WorkEvent {
    uint64_t id = 0;
    std::string kind;
    std::string label;
    std::string status;
    std::string error;
    double elapsed_seconds = 0.0;
    std::string finished_at;
    double audio_seconds = 0.0;
    double rtf = 0.0;
};

class WorkDispatcher {
public:
    explicit WorkDispatcher(uint32_t max_waiting) : max_waiting_(std::max<uint32_t>(1, max_waiting)) {}

    bool submit(const std::string& kind,
                const std::string& label,
                const std::function<bool(std::string&, WorkEvent&)>& fn,
                std::string& error) {
        auto item = std::make_shared<Item>();
        item->id = next_id_.fetch_add(1) + 1;
        item->kind = kind;
        item->label = label;
        item->fn = fn;
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (queue_.size() >= max_waiting_) {
                ++rejected_;
                error = kind + " queue is full";
                push_event_locked({item->id, kind, label, "rejected", error, 0.0, now_epoch_string(), 0.0, 0.0});
                return false;
            }
            queue_.push_back(item);
            ++submitted_;
            cv_.notify_one();
        }
        std::unique_lock<std::mutex> done_lock(item->mu);
        item->done_cv.wait(done_lock, [&] { return item->done; });
        error = item->error;
        return item->ok;
    }

    void run_forever() {
        for (;;) {
            std::shared_ptr<Item> item;
            {
                std::unique_lock<std::mutex> lock(mu_);
                cv_.wait(lock, [&] { return !queue_.empty(); });
                item = queue_.front();
                queue_.pop_front();
                running_ = true;
                current_id_ = item->id;
                current_kind_ = item->kind;
                current_label_ = item->label;
                current_started_ = Clock::now();
            }
            std::string error;
            bool ok = false;
            WorkEvent event{item->id, item->kind, item->label, "", "", 0.0, "", 0.0, 0.0};
            const auto start = Clock::now();
            try {
                ok = item->fn(error, event);
            } catch (const std::exception& e) {
                error = e.what();
                ok = false;
            }
            const double elapsed = std::chrono::duration<double>(Clock::now() - start).count();
            event.status = ok ? "completed" : "failed";
            event.error = error;
            event.elapsed_seconds = elapsed;
            event.finished_at = now_epoch_string();
            if (event.audio_seconds > 0.0) event.rtf = elapsed / event.audio_seconds;
            {
                std::lock_guard<std::mutex> lock(mu_);
                running_ = false;
                current_id_ = 0;
                current_kind_.clear();
                current_label_.clear();
                if (ok) ++completed_; else ++failed_;
                push_event_locked(event);
            }
            {
                std::lock_guard<std::mutex> done_lock(item->mu);
                item->ok = ok;
                item->error = error;
                item->done = true;
            }
            item->done_cv.notify_all();
        }
    }

    std::string status_json(const std::string& weights_dir, const std::string& voice_store,
                            bool web_enabled, bool web_auth_required) {
        std::lock_guard<std::mutex> lock(mu_);
        std::ostringstream out;
        out << "{\"status\":\"ok\","
            << "\"model_bundle\":\"" << json_escape(weights_dir) << "\","
            << "\"voice_store\":\"" << json_escape(voice_store) << "\","
            << "\"web_enabled\":" << (web_enabled ? "true" : "false") << ","
            << "\"web_auth_required\":" << (web_auth_required ? "true" : "false") << ","
            << "\"queue\":{\"max_waiting\":" << max_waiting_
            << ",\"waiting\":" << queue_.size()
            << ",\"running\":" << (running_ ? "true" : "false");
        if (running_) {
            const double elapsed = std::chrono::duration<double>(Clock::now() - current_started_).count();
            out << ",\"current\":{\"id\":" << current_id_
                << ",\"kind\":\"" << json_escape(current_kind_) << "\""
                << ",\"label\":\"" << json_escape(current_label_) << "\""
                << ",\"elapsed_seconds\":" << elapsed << "}";
        } else {
            out << ",\"current\":null";
        }
        out << "},\"totals\":{\"submitted\":" << submitted_
            << ",\"completed\":" << completed_
            << ",\"failed\":" << failed_
            << ",\"rejected\":" << rejected_ << "},\"recent\":[";
        for (size_t i = 0; i < recent_.size(); ++i) {
            if (i) out << ",";
            const auto& e = recent_[i];
            out << "{\"id\":" << e.id
                << ",\"kind\":\"" << json_escape(e.kind) << "\""
                << ",\"label\":\"" << json_escape(e.label) << "\""
                << ",\"status\":\"" << json_escape(e.status) << "\""
                << ",\"error\":\"" << json_escape(e.error) << "\""
                << ",\"elapsed_seconds\":" << e.elapsed_seconds
                << ",\"audio_seconds\":" << e.audio_seconds
                << ",\"rtf\":" << e.rtf
                << ",\"finished_at\":\"" << json_escape(e.finished_at) << "\"}";
        }
        out << "]}";
        return out.str();
    }

private:
    struct Item {
        uint64_t id = 0;
        std::string kind;
        std::string label;
        std::function<bool(std::string&, WorkEvent&)> fn;
        std::mutex mu;
        std::condition_variable done_cv;
        bool done = false;
        bool ok = false;
        std::string error;
    };

    void push_event_locked(const WorkEvent& event) {
        recent_.push_front(event);
        while (recent_.size() > 32) recent_.pop_back();
    }

    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<std::shared_ptr<Item>> queue_;
    std::deque<WorkEvent> recent_;
    std::atomic<uint64_t> next_id_{0};
    size_t max_waiting_ = 16;
    bool running_ = false;
    uint64_t current_id_ = 0;
    std::string current_kind_;
    std::string current_label_;
    Clock::time_point current_started_{};
    uint64_t submitted_ = 0;
    uint64_t completed_ = 0;
    uint64_t failed_ = 0;
    uint64_t rejected_ = 0;
};

// ---------------------------------------------------------------------------
// SQLite voice store (same schema/layout as the reference server).
// ---------------------------------------------------------------------------

struct VoiceRecord {
    std::string id;
    std::string name;
    std::string description;
    std::string bundle_path;
    std::string sample_path;
    double source_audio_seconds = 0.0;
    std::string source;
    std::string created_at;
    std::string updated_at;
};

struct ConsentRecord {
    std::string id;
    std::string name;
    std::string language;
    std::string recording_path;
    std::string created_at;
    std::string updated_at;
};

class VoiceStore {
public:
    explicit VoiceStore(std::string root) : root_(std::move(root)) {
        fs::create_directories(root_);
        fs::create_directories(fs::path(root_) / "samples");
        fs::create_directories(fs::path(root_) / "bundles");
        const std::string db_path = (fs::path(root_) / "voices.sqlite").string();
        if (sqlite3_open(db_path.c_str(), &db_) != SQLITE_OK) {
            throw std::runtime_error("failed to open sqlite voice store: " + db_path);
        }
        exec("PRAGMA journal_mode=WAL");
        exec("CREATE TABLE IF NOT EXISTS voices ("
             "id TEXT PRIMARY KEY, name TEXT, description TEXT, bundle_path TEXT NOT NULL,"
             "sample_path TEXT, source_audio_seconds REAL DEFAULT 0, source TEXT,"
             "created_at TEXT, updated_at TEXT, deleted INTEGER DEFAULT 0)");
        exec_ignore_duplicate_column("ALTER TABLE voices ADD COLUMN source_audio_seconds REAL DEFAULT 0");
        exec("CREATE TABLE IF NOT EXISTS voice_consents ("
             "id TEXT PRIMARY KEY, name TEXT, language TEXT, recording_path TEXT,"
             "created_at TEXT, updated_at TEXT, deleted INTEGER DEFAULT 0)");
    }

    ~VoiceStore() {
        if (db_) sqlite3_close(db_);
    }

    std::string root() const { return root_; }
    std::string samples_dir() const { return (fs::path(root_) / "samples").string(); }
    std::string bundles_dir() const { return (fs::path(root_) / "bundles").string(); }

    void exec(const std::string& sql) {
        char* err = nullptr;
        if (sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &err) != SQLITE_OK) {
            std::string message = err ? err : "sqlite error";
            sqlite3_free(err);
            throw std::runtime_error(message);
        }
    }

    void exec_ignore_duplicate_column(const std::string& sql) {
        char* err = nullptr;
        if (sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &err) != SQLITE_OK) {
            std::string message = err ? err : "sqlite error";
            sqlite3_free(err);
            if (message.find("duplicate column name") == std::string::npos) {
                throw std::runtime_error(message);
            }
        }
    }

    void insert_voice(const VoiceRecord& v) {
        std::lock_guard<std::mutex> lock(mu_);
        sqlite3_stmt* st = nullptr;
        prepare("INSERT OR REPLACE INTO voices"
                "(id,name,description,bundle_path,sample_path,source_audio_seconds,source,created_at,updated_at,deleted)"
                " VALUES(?,?,?,?,?,?,?,?,?,0)", &st);
        bind(st, 1, v.id); bind(st, 2, v.name); bind(st, 3, v.description);
        bind(st, 4, v.bundle_path); bind(st, 5, v.sample_path);
        sqlite3_bind_double(st, 6, v.source_audio_seconds);
        bind(st, 7, v.source);
        bind(st, 8, v.created_at); bind(st, 9, v.updated_at);
        step_done(st);
    }

    std::optional<VoiceRecord> get_voice(const std::string& id_or_name) {
        std::lock_guard<std::mutex> lock(mu_);
        sqlite3_stmt* st = nullptr;
        prepare("SELECT id,name,description,bundle_path,sample_path,source_audio_seconds,source,created_at,updated_at"
                " FROM voices WHERE deleted=0 AND (id=? OR name=?) LIMIT 1", &st);
        bind(st, 1, id_or_name);
        bind(st, 2, id_or_name);
        std::optional<VoiceRecord> out;
        if (sqlite3_step(st) == SQLITE_ROW) out = row_voice(st);
        sqlite3_finalize(st);
        return out;
    }

    std::vector<VoiceRecord> list_voices() {
        std::lock_guard<std::mutex> lock(mu_);
        sqlite3_stmt* st = nullptr;
        prepare("SELECT id,name,description,bundle_path,sample_path,source_audio_seconds,source,created_at,updated_at"
                " FROM voices WHERE deleted=0 ORDER BY created_at DESC", &st);
        std::vector<VoiceRecord> out;
        while (sqlite3_step(st) == SQLITE_ROW) out.push_back(row_voice(st));
        sqlite3_finalize(st);
        return out;
    }

    bool update_voice(const std::string& id, const VoiceRecord& patch) {
        auto current = get_voice(id);
        if (!current) return false;
        VoiceRecord v = *current;
        if (!patch.name.empty()) v.name = patch.name;
        if (!patch.description.empty()) v.description = patch.description;
        if (!patch.bundle_path.empty()) v.bundle_path = patch.bundle_path;
        if (!patch.sample_path.empty()) v.sample_path = patch.sample_path;
        if (patch.source_audio_seconds > 0.0) v.source_audio_seconds = patch.source_audio_seconds;
        if (!patch.source.empty()) v.source = patch.source;
        v.updated_at = now_epoch_string();
        insert_voice(v);
        return true;
    }

    bool delete_voice(const std::string& id) {
        std::lock_guard<std::mutex> lock(mu_);
        sqlite3_stmt* st = nullptr;
        prepare("UPDATE voices SET deleted=1, updated_at=? WHERE id=? AND deleted=0", &st);
        bind(st, 1, now_epoch_string());
        bind(st, 2, id);
        step_done(st);
        return sqlite3_changes(db_) > 0;
    }

    void insert_consent(const ConsentRecord& c) {
        std::lock_guard<std::mutex> lock(mu_);
        sqlite3_stmt* st = nullptr;
        prepare("INSERT OR REPLACE INTO voice_consents"
                "(id,name,language,recording_path,created_at,updated_at,deleted)"
                " VALUES(?,?,?,?,?,?,0)", &st);
        bind(st, 1, c.id); bind(st, 2, c.name); bind(st, 3, c.language);
        bind(st, 4, c.recording_path); bind(st, 5, c.created_at); bind(st, 6, c.updated_at);
        step_done(st);
    }

    std::optional<ConsentRecord> get_consent(const std::string& id) {
        std::lock_guard<std::mutex> lock(mu_);
        sqlite3_stmt* st = nullptr;
        prepare("SELECT id,name,language,recording_path,created_at,updated_at"
                " FROM voice_consents WHERE deleted=0 AND id=? LIMIT 1", &st);
        bind(st, 1, id);
        std::optional<ConsentRecord> out;
        if (sqlite3_step(st) == SQLITE_ROW) out = row_consent(st);
        sqlite3_finalize(st);
        return out;
    }

    std::vector<ConsentRecord> list_consents() {
        std::lock_guard<std::mutex> lock(mu_);
        sqlite3_stmt* st = nullptr;
        prepare("SELECT id,name,language,recording_path,created_at,updated_at"
                " FROM voice_consents WHERE deleted=0 ORDER BY created_at DESC", &st);
        std::vector<ConsentRecord> out;
        while (sqlite3_step(st) == SQLITE_ROW) out.push_back(row_consent(st));
        sqlite3_finalize(st);
        return out;
    }

    bool update_consent(const std::string& id, const ConsentRecord& patch) {
        auto current = get_consent(id);
        if (!current) return false;
        ConsentRecord c = *current;
        if (!patch.name.empty()) c.name = patch.name;
        if (!patch.language.empty()) c.language = patch.language;
        if (!patch.recording_path.empty()) c.recording_path = patch.recording_path;
        c.updated_at = now_epoch_string();
        insert_consent(c);
        return true;
    }

    bool delete_consent(const std::string& id) {
        std::lock_guard<std::mutex> lock(mu_);
        sqlite3_stmt* st = nullptr;
        prepare("UPDATE voice_consents SET deleted=1, updated_at=? WHERE id=? AND deleted=0", &st);
        bind(st, 1, now_epoch_string());
        bind(st, 2, id);
        step_done(st);
        return sqlite3_changes(db_) > 0;
    }

    // Resolves a voice id/name (or a direct bundle/wav path) to a c4tts voice
    // bundle path usable for synthesis.
    std::string resolve_voice_bundle(const std::string& voice) {
        if (voice.empty()) return {};
        if (auto rec = get_voice(voice)) {
            if (path_is_voice_bundle(rec->bundle_path)) return rec->bundle_path;
            if (!rec->sample_path.empty() && path_is_voice_bundle(rec->sample_path)) return rec->sample_path;
        }
        if (path_is_voice_bundle(voice)) return voice;
        return {};
    }

private:
    void prepare(const std::string& sql, sqlite3_stmt** st) {
        if (sqlite3_prepare_v2(db_, sql.c_str(), -1, st, nullptr) != SQLITE_OK) {
            throw std::runtime_error(sqlite3_errmsg(db_));
        }
    }

    static void bind(sqlite3_stmt* st, int index, const std::string& value) {
        sqlite3_bind_text(st, index, value.c_str(), -1, SQLITE_TRANSIENT);
    }

    static std::string col(sqlite3_stmt* st, int index) {
        const unsigned char* v = sqlite3_column_text(st, index);
        return v ? reinterpret_cast<const char*>(v) : "";
    }

    static VoiceRecord row_voice(sqlite3_stmt* st) {
        return {col(st, 0), col(st, 1), col(st, 2), col(st, 3), col(st, 4),
                sqlite3_column_double(st, 5), col(st, 6), col(st, 7), col(st, 8)};
    }

    static ConsentRecord row_consent(sqlite3_stmt* st) {
        return {col(st, 0), col(st, 1), col(st, 2), col(st, 3), col(st, 4), col(st, 5)};
    }

    static void step_done(sqlite3_stmt* st) {
        const int rc = sqlite3_step(st);
        if (rc != SQLITE_DONE) {
            std::string msg = sqlite3_errmsg(sqlite3_db_handle(st));
            sqlite3_finalize(st);
            throw std::runtime_error(msg);
        }
        sqlite3_finalize(st);
    }

    std::string root_;
    sqlite3* db_ = nullptr;
    std::mutex mu_;
};

std::string voice_json(const VoiceRecord& v) {
    std::ostringstream out;
    out << "{\"id\":\"" << json_escape(v.id) << "\","
        << "\"object\":\"audio.voice\","
        << "\"name\":\"" << json_escape(v.name) << "\","
        << "\"description\":\"" << json_escape(v.description) << "\","
        << "\"bundle_path\":\"" << json_escape(v.bundle_path) << "\","
        << "\"sample_path\":\"" << json_escape(v.sample_path) << "\","
        << "\"source_audio_seconds\":" << v.source_audio_seconds << ","
        << "\"source\":\"" << json_escape(v.source) << "\","
        << "\"created_at\":\"" << json_escape(v.created_at) << "\","
        << "\"updated_at\":\"" << json_escape(v.updated_at) << "\"}";
    return out.str();
}

std::string consent_json(const ConsentRecord& c, const std::string& voice_id = "") {
    std::ostringstream out;
    out << "{\"id\":\"" << json_escape(c.id) << "\","
        << "\"object\":\"audio.voice_consent\","
        << "\"name\":\"" << json_escape(c.name) << "\","
        << "\"language\":\"" << json_escape(c.language) << "\","
        << "\"recording_path\":\"" << json_escape(c.recording_path) << "\",";
    if (!voice_id.empty()) out << "\"voice_id\":\"" << json_escape(voice_id) << "\",";
    out << "\"created_at\":\"" << json_escape(c.created_at) << "\","
        << "\"updated_at\":\"" << json_escape(c.updated_at) << "\"}";
    return out.str();
}

// ---------------------------------------------------------------------------
// Server configuration + request handlers.
// ---------------------------------------------------------------------------

struct ServerConfig {
    std::string host = "127.0.0.1";
    uint16_t port = 3456;
    std::string weights_dir = "bin";
    std::string voice_store_dir = "voices";
    uint32_t queue_size = 16;
    uint32_t lru_capacity = 3;  // per-voice conditioning cache (0 disables)
    bool web_enabled = false;
    std::string web_key;
    std::string lang = "zh";
    bool verbose = false;
};

bool web_key_authorized(const ServerConfig& cfg, const std::string& head, const std::string& body) {
    if (!cfg.web_enabled || cfg.web_key.empty()) return true;
    if (header_value(head, "x-mtts-web-key") == cfg.web_key) return true;
    const std::string auth = header_value(head, "authorization");
    const std::string prefix = "Bearer ";
    if (auth.size() > prefix.size() && auth.compare(0, prefix.size(), prefix) == 0 &&
        auth.substr(prefix.size()) == cfg.web_key) {
        return true;
    }
    return json_string_field(body, "key") == cfg.web_key;
}

// Materializes a usable on-disk reference WAV for a resolved voice bundle and
// returns its path. Prefers an existing sample file; otherwise extracts the
// bundle's embedded WAV into a cache file under the voice store.
std::string reference_wav_for_voice(VoiceStore& store, const VoiceRecord& voice) {
    if (!voice.sample_path.empty() && fs::exists(voice.sample_path) && bytes_are_wav(read_file_bytes(voice.sample_path))) {
        return voice.sample_path;
    }
    const std::string wav = voice_bundle_source_wav(voice.bundle_path);
    if (wav.empty()) return {};
    const std::string cache_dir = (fs::path(store.root()) / "cache").string();
    fs::create_directories(cache_dir);
    const std::string cache_wav = (fs::path(cache_dir) / (voice.id + ".wav")).string();
    if (!fs::exists(cache_wav)) save_file_bytes(cache_wav, wav);
    return cache_wav;
}

double source_audio_seconds_for_voice(const std::string& bundle_path, const std::string& sample_path) {
    if (!sample_path.empty()) {
        const double s = wav_duration_seconds_from_bytes(read_file_bytes(sample_path));
        if (s > 0.0) return s;
    }
    return wav_duration_seconds_from_bytes(voice_bundle_source_wav(bundle_path));
}

// Creates a voice from reference audio on disk: validates the WAV, keeps a
// sample copy, and writes a c4tts voice bundle embedding the source audio.
std::optional<VoiceRecord> create_voice_from_audio(VoiceStore& store,
                                                   const std::string& name,
                                                   const std::string& description,
                                                   const std::string& audio_path,
                                                   std::string& error) {
    const std::string wav_bytes = read_file_bytes(audio_path);
    if (!bytes_are_wav(wav_bytes)) {
        error = "reference audio is not a readable WAV file";
        return std::nullopt;
    }
    const std::string voice_id = make_id("voice");
    const std::string out_bundle = (fs::path(store.bundles_dir()) / (voice_id + ".pt")).string();
    try {
        write_voice_bundle(out_bundle, wav_bytes);
    } catch (const std::exception& e) {
        error = e.what();
        return std::nullopt;
    }
    const std::string now = now_epoch_string();
    const double source_seconds = wav_duration_seconds_from_bytes(wav_bytes);
    VoiceRecord v{voice_id,
                  name.empty() ? voice_id : name,
                  description,
                  out_bundle,
                  audio_path,
                  source_seconds,
                  "clone",
                  now,
                  now};
    store.insert_voice(v);
    return v;
}

void send_voice_list(int fd, VoiceStore& store) {
    const auto voices = store.list_voices();
    std::ostringstream out;
    out << "{\"object\":\"list\",\"data\":[";
    for (size_t i = 0; i < voices.size(); ++i) {
        if (i) out << ",";
        out << voice_json(voices[i]);
    }
    out << "]}";
    send_response(fd, 200, "OK", "application/json", out.str());
}

void send_voice_source_audio(int fd, VoiceStore& store, const std::string& id) {
    const auto voice = store.get_voice(id);
    if (!voice) {
        send_json_error(fd, 404, "Not Found", "voice not found");
        return;
    }
    std::string bytes = voice_bundle_source_wav(voice->bundle_path);
    if (bytes.empty() && !voice->sample_path.empty()) bytes = read_file_bytes(voice->sample_path);
    if (bytes.empty()) {
        send_json_error(fd, 404, "Not Found", "source audio not found in voice bundle");
        return;
    }
    send_response(fd, 200, "OK", "audio/wav", bytes);
}

void remove_voice_files(VoiceStore& store, const VoiceRecord& voice) {
    bool bundle_still_used = false;
    for (const auto& active : store.list_voices()) {
        if (active.id != voice.id && active.bundle_path == voice.bundle_path) {
            bundle_still_used = true;
            break;
        }
    }
    if (!bundle_still_used) {
        remove_path_under_dir(voice.bundle_path, store.bundles_dir(), "voice bundle");
    }
    remove_path_under_dir(voice.sample_path, store.samples_dir(), "voice sample");
    remove_path_under_dir((fs::path(store.root()) / "cache" / (voice.id + ".wav")).string(),
                          (fs::path(store.root()) / "cache").string(), "voice cache");
}

void send_consent_list(int fd, VoiceStore& store) {
    const auto consents = store.list_consents();
    std::ostringstream out;
    out << "{\"object\":\"list\",\"data\":[";
    for (size_t i = 0; i < consents.size(); ++i) {
        if (i) out << ",";
        out << consent_json(consents[i]);
    }
    out << "]}";
    send_response(fd, 200, "OK", "application/json", out.str());
}

bool create_voice_from_request(int fd,
                               const std::string& head,
                               const std::string& body,
                               VoiceStore& store) {
    const std::string ct = content_type_from_head(head);
    std::string name, description, bundle_path, sample_path, consent_id;
    if (ct.find("multipart/form-data") != std::string::npos) {
        const auto parts = parse_multipart(ct, body);
        name = multipart_value(parts, "name");
        description = multipart_value(parts, "description");
        consent_id = multipart_value(parts, "consent");
        if (auto file = multipart_file(parts, {"audio_sample", "recording", "file", "audio"})) {
            const std::string id = make_id("sample");
            sample_path = (fs::path(store.samples_dir()) / (id + safe_ext(file->filename, ".wav"))).string();
            save_file_bytes(sample_path, file->data);
        }
    } else {
        name = json_string_field(body, "name");
        description = json_string_field(body, "description");
        bundle_path = json_string_field(body, "bundle_path");
        sample_path = json_string_field(body, "sample_path");
        consent_id = json_string_field(body, "consent");
    }
    if (sample_path.empty() && !consent_id.empty()) {
        if (auto c = store.get_consent(consent_id)) sample_path = c->recording_path;
    }
    if (!bundle_path.empty()) {
        if (store.resolve_voice_bundle(bundle_path).empty()) {
            send_json_error(fd, 400, "Bad Request", "bundle_path is not a usable voice bundle");
            return true;
        }
        const std::string now = now_epoch_string();
        const std::string voice_id = make_id("voice");
        const double source_seconds = source_audio_seconds_for_voice(bundle_path, sample_path);
        VoiceRecord v{voice_id, name.empty() ? voice_id : name, description, bundle_path,
                      sample_path, source_seconds, "import", now, now};
        store.insert_voice(v);
        send_response(fd, 200, "OK", "application/json", voice_json(v));
        return true;
    }
    if (sample_path.empty()) {
        send_json_error(fd, 400, "Bad Request",
                        "missing audio_sample, recording, sample_path, consent, or bundle_path");
        return true;
    }
    std::string error;
    auto v = create_voice_from_audio(store, name, description, sample_path, error);
    if (!v) {
        send_json_error(fd, 500, "Internal Server Error", error);
        return true;
    }
    send_response(fd, 200, "OK", "application/json", voice_json(*v));
    return true;
}

void handle_speech(int fd,
                   const std::string& body,
                   VoiceStore& store,
                   const ServerConfig& cfg,
                   Engine& engine,
                   WorkDispatcher& dispatcher,
                   uint64_t request_id) {
    const auto received = Clock::now();
    const std::string input = json_string_field(body, "input");
    const std::string voice = json_voice_field(body);
    const std::string response_format = json_string_field(body, "response_format");
    std::string lang = json_string_field(body, "lang");
    if (lang.empty()) lang = json_string_field(body, "language");
    const int steps = static_cast<int>(json_int_field(body, "steps", 0));
    const int max_tokens = static_cast<int>(json_int_field(body, "max_tokens", 0));

    if (input.empty()) {
        send_json_error(fd, 400, "Bad Request", "missing required field: input");
        return;
    }
    if (!response_format.empty() && response_format != "wav") {
        send_json_error(fd, 400, "Bad Request", "only response_format=wav is supported");
        return;
    }
    if (voice.empty()) {
        send_json_error(fd, 400, "Bad Request", "missing required field: voice");
        return;
    }
    const auto voice_rec = store.get_voice(voice);
    if (!voice_rec) {
        send_json_error(fd, 400, "Bad Request", "voice not found");
        return;
    }
    const std::string prompt_wav = reference_wav_for_voice(store, *voice_rec);
    if (prompt_wav.empty()) {
        send_json_error(fd, 400, "Bad Request", "voice has no usable reference audio");
        return;
    }

    // Keep synthesis scratch alongside the voice store (not in the CWD) so all
    // runtime data lives under one directory next to the app.
    const std::string tmp_dir = (fs::path(cfg.voice_store_dir) / ".tmp").string();
    fs::create_directories(tmp_dir);
    const std::string out_wav = (fs::path(tmp_dir) / ("req_" + std::to_string(request_id) + ".wav")).string();

    if (cfg.verbose) {
        std::cerr << ">> request " << request_id << ": voice=" << voice_rec->id
                  << " input=" << input.substr(0, 64) << (input.size() > 64 ? "..." : "") << std::endl;
    }

    std::string error;
    const bool ok = dispatcher.submit("tts", input, [&](std::string& task_error, WorkEvent& event) {
        // Runs on the single worker thread, so the resident pipeline is only
        // ever touched by one thread at a time.
        try {
            engine.synth_to_file(voice_rec->id, prompt_wav, input, lang, steps, max_tokens, out_wav);
        } catch (const std::exception& e) {
            // Free MLX's reclaimable buffer cache even on failure (held voices
            // in the LRU survive — they are referenced, not cached buffers).
            mx::clear_cache();
            task_error = e.what();
            return false;
        }
        const std::string wav_bytes = read_file_bytes(out_wav);
        event.audio_seconds = wav_bytes.size() > 44
            ? static_cast<double>(wav_bytes.size() - 44) / 2.0 / kSampleRate
            : 0.0;
        // Release the per-request scratch buffers MLX cached for this sequence
        // length so memory doesn't climb as output lengths vary across requests.
        mx::clear_cache();
        return true;
    }, error);

    if (!ok) {
        send_json_error(fd, error == "tts queue is full" ? 429 : 500,
                        error == "tts queue is full" ? "Too Many Requests" : "Internal Server Error",
                        error);
        return;
    }

    const std::string wav_bytes = read_file_bytes(out_wav);
    {
        std::error_code ec;
        fs::remove(out_wav, ec);
    }
    const double total = std::chrono::duration<double>(Clock::now() - received).count();
    const double audio_seconds = wav_bytes.size() > 44
        ? static_cast<double>(wav_bytes.size() - 44) / 2.0 / kSampleRate
        : 0.0;
    if (cfg.verbose) {
        std::cerr << ">> request " << request_id << " done: audio=" << audio_seconds
                  << "s total=" << total << "s rtf="
                  << (audio_seconds > 0 ? total / audio_seconds : 0.0) << std::endl;
    }
    send_response(fd, 200, "OK", "audio/wav", wav_bytes);
}

bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

void handle_connection(int fd,
                       VoiceStore& store,
                       const ServerConfig& cfg,
                       Engine& engine,
                       WorkDispatcher& dispatcher,
                       uint64_t request_id) {
    std::string head, body;
    if (!read_request(fd, head, body)) {
        ::close(fd);
        return;
    }
    std::string method;
    const std::string path = route_path_from_head(head, method);

    try {
        std::string effective_path = path;
        if (path == "/web" || path == "/web/" || path == "/") {
            if (!cfg.web_enabled) {
                send_json_error(fd, 404, "Not Found", "web UI is not enabled");
            } else if (method != "GET") {
                send_json_error(fd, 405, "Method Not Allowed", "method not allowed");
            } else {
                send_response(fd, 200, "OK", "text/html; charset=utf-8", web_index_html());
            }
            ::close(fd);
            return;
        }
        if (path == "/web/api/login") {
            if (!cfg.web_enabled) {
                send_json_error(fd, 404, "Not Found", "web UI is not enabled");
            } else if (method != "POST") {
                send_json_error(fd, 405, "Method Not Allowed", "method not allowed");
            } else if (!web_key_authorized(cfg, head, body)) {
                send_json_error(fd, 401, "Unauthorized", "invalid web key");
            } else {
                send_response(fd, 200, "OK", "application/json", "{\"ok\":true}");
            }
            ::close(fd);
            return;
        }
        if (path.rfind("/web/api/", 0) == 0) {
            if (!cfg.web_enabled) {
                send_json_error(fd, 404, "Not Found", "web UI is not enabled");
                ::close(fd);
                return;
            }
            if (!web_key_authorized(cfg, head, body)) {
                send_json_error(fd, 401, "Unauthorized", "invalid web key");
                ::close(fd);
                return;
            }
            effective_path = path.substr(std::string("/web/api").size());
            if (effective_path.empty()) effective_path = "/";
        }

        if (method == "GET" && (effective_path == "/health" || effective_path == "/v1/health")) {
            send_response(fd, 200, "OK", "application/json", "{\"status\":\"ok\"}");
        } else if (method == "GET" && (effective_path == "/api/status" || effective_path == "/status")) {
            send_response(fd, 200, "OK", "application/json",
                          dispatcher.status_json(cfg.weights_dir, cfg.voice_store_dir, cfg.web_enabled,
                                                 cfg.web_enabled && !cfg.web_key.empty()));
        } else if (method == "POST" && (effective_path == "/v1/audio/speech" || effective_path == "/speech")) {
            handle_speech(fd, body, store, cfg, engine, dispatcher, request_id);
        } else if (method == "GET" &&
                   (effective_path == "/api/voices" || effective_path == "/v1/audio/voices" || effective_path == "/voices")) {
            send_voice_list(fd, store);
        } else if (method == "POST" &&
                   (effective_path == "/api/voices" || effective_path == "/v1/audio/voices" || effective_path == "/voices")) {
            create_voice_from_request(fd, head, body, store);
        } else if (method == "GET" &&
                   (effective_path.rfind("/api/voices/", 0) == 0 ||
                    effective_path.rfind("/v1/audio/voices/", 0) == 0 ||
                    effective_path.rfind("/voices/", 0) == 0) &&
                   ends_with(effective_path, "/source-audio")) {
            const std::string prefix = effective_path.rfind("/api/voices/", 0) == 0
                ? "/api/voices/"
                : (effective_path.rfind("/v1/audio/voices/", 0) == 0 ? "/v1/audio/voices/" : "/voices/");
            std::string id = effective_path.substr(prefix.size());
            id.resize(id.size() - std::string("/source-audio").size());
            send_voice_source_audio(fd, store, id);
        } else if (effective_path.rfind("/api/voices/", 0) == 0 ||
                   effective_path.rfind("/v1/audio/voices/", 0) == 0 ||
                   effective_path.rfind("/voices/", 0) == 0) {
            const std::string prefix = effective_path.rfind("/api/voices/", 0) == 0
                ? "/api/voices/"
                : (effective_path.rfind("/v1/audio/voices/", 0) == 0 ? "/v1/audio/voices/" : "/voices/");
            const std::string id = effective_path.substr(prefix.size());
            if (method == "GET") {
                auto v = store.get_voice(id);
                if (!v) send_json_error(fd, 404, "Not Found", "voice not found");
                else send_response(fd, 200, "OK", "application/json", voice_json(*v));
            } else if (method == "PATCH" || method == "PUT") {
                VoiceRecord patch;
                patch.name = json_string_field(body, "name");
                patch.description = json_string_field(body, "description");
                patch.bundle_path = json_string_field(body, "bundle_path");
                patch.sample_path = json_string_field(body, "sample_path");
                if (!patch.bundle_path.empty() || !patch.sample_path.empty()) {
                    const auto current = store.get_voice(id);
                    const std::string bundle_for_duration = !patch.bundle_path.empty()
                        ? patch.bundle_path : (current ? current->bundle_path : "");
                    const std::string sample_for_duration = !patch.sample_path.empty()
                        ? patch.sample_path : (current ? current->sample_path : "");
                    patch.source_audio_seconds = source_audio_seconds_for_voice(bundle_for_duration, sample_for_duration);
                }
                if (!patch.bundle_path.empty() && store.resolve_voice_bundle(patch.bundle_path).empty()) {
                    send_json_error(fd, 400, "Bad Request", "bundle_path is not a usable voice bundle");
                } else if (!store.update_voice(id, patch)) {
                    send_json_error(fd, 404, "Not Found", "voice not found");
                } else {
                    send_response(fd, 200, "OK", "application/json", voice_json(*store.get_voice(id)));
                }
            } else if (method == "DELETE") {
                auto v = store.get_voice(id);
                if (!v || !store.delete_voice(id)) {
                    send_json_error(fd, 404, "Not Found", "voice not found");
                } else {
                    remove_voice_files(store, *v);
                    send_response(fd, 200, "OK", "application/json", "{\"deleted\":true}");
                }
            } else {
                send_json_error(fd, 405, "Method Not Allowed", "method not allowed");
            }
        } else if (method == "GET" &&
                   (effective_path == "/v1/audio/voice_consents" || effective_path == "/voice_consents")) {
            send_consent_list(fd, store);
        } else if (method == "POST" &&
                   (effective_path == "/v1/audio/voice_consents" || effective_path == "/voice_consents")) {
            const std::string ct = content_type_from_head(head);
            std::string name, language, recording_path;
            if (ct.find("multipart/form-data") != std::string::npos) {
                const auto parts = parse_multipart(ct, body);
                name = multipart_value(parts, "name");
                language = multipart_value(parts, "language");
                if (auto file = multipart_file(parts, {"recording", "audio_sample", "file", "audio"})) {
                    const std::string sid = make_id("sample");
                    recording_path = (fs::path(store.samples_dir()) / (sid + safe_ext(file->filename, ".wav"))).string();
                    save_file_bytes(recording_path, file->data);
                }
            } else {
                name = json_string_field(body, "name");
                language = json_string_field(body, "language");
                recording_path = json_string_field(body, "recording_path");
            }
            if (recording_path.empty()) {
                send_json_error(fd, 400, "Bad Request", "missing recording, audio_sample, or recording_path");
            } else {
                const std::string now = now_epoch_string();
                ConsentRecord c{make_id("consent"), name, language, recording_path, now, now};
                store.insert_consent(c);
                std::string voice_id;
                if (json_bool_field(body, "create_voice", true)) {
                    std::string error;
                    auto v = create_voice_from_audio(store, name, "created from voice consent", recording_path, error);
                    if (!v) {
                        send_json_error(fd, 500, "Internal Server Error", error);
                        ::close(fd);
                        return;
                    }
                    voice_id = v->id;
                }
                send_response(fd, 200, "OK", "application/json", consent_json(c, voice_id));
            }
        } else if (effective_path.rfind("/v1/audio/voice_consents/", 0) == 0 ||
                   effective_path.rfind("/voice_consents/", 0) == 0) {
            const std::string prefix = effective_path.rfind("/v1/audio/voice_consents/", 0) == 0
                ? "/v1/audio/voice_consents/" : "/voice_consents/";
            const std::string id = effective_path.substr(prefix.size());
            if (method == "GET") {
                auto c = store.get_consent(id);
                if (!c) send_json_error(fd, 404, "Not Found", "voice consent not found");
                else send_response(fd, 200, "OK", "application/json", consent_json(*c));
            } else if (method == "PATCH" || method == "PUT") {
                ConsentRecord patch;
                patch.name = json_string_field(body, "name");
                patch.language = json_string_field(body, "language");
                patch.recording_path = json_string_field(body, "recording_path");
                if (!store.update_consent(id, patch)) send_json_error(fd, 404, "Not Found", "voice consent not found");
                else send_response(fd, 200, "OK", "application/json", consent_json(*store.get_consent(id)));
            } else if (method == "DELETE") {
                if (!store.delete_consent(id)) send_json_error(fd, 404, "Not Found", "voice consent not found");
                else send_response(fd, 200, "OK", "application/json", "{\"deleted\":true}");
            } else {
                send_json_error(fd, 405, "Method Not Allowed", "method not allowed");
            }
        } else {
            send_json_error(fd, 404, "Not Found", "unknown endpoint");
        }
    } catch (const std::exception& e) {
        send_json_error(fd, 500, "Internal Server Error", e.what());
    }
    ::close(fd);
}

}  // namespace

int run_server(const std::string& host,
               uint16_t port,
               const std::string& weights_dir,
               const std::string& voice_store_dir,
               uint32_t queue_size,
               uint32_t lru_capacity,
               bool web_enabled,
               const std::string& web_key,
               const std::string& lang,
               bool verbose) {
    ::signal(SIGPIPE, SIG_IGN);

    ServerConfig cfg;
    cfg.host = host;
    cfg.port = port;
    cfg.weights_dir = weights_dir;
    cfg.voice_store_dir = voice_store_dir.empty()
        ? env_string("C4TTS_VOICE_STORE", "voices") : voice_store_dir;
    cfg.queue_size = queue_size == 0 ? env_u32("C4TTS_QUEUE_SIZE", 16, 1, 10000) : queue_size;
    cfg.lru_capacity = lru_capacity == UINT32_MAX
        ? env_u32("C4TTS_LRU_CACHE", 3, 0, 64) : lru_capacity;
    cfg.web_enabled = web_enabled;
    cfg.web_key = web_key.empty() ? env_string("C4TTS_WEBKEY", "") : web_key;
    cfg.lang = lang.empty() ? "zh" : lang;
    cfg.verbose = verbose;

    // Bound MLX's reclaimable Metal buffer cache so RSS doesn't climb as
    // per-request scratch buffers accumulate across varying sequence lengths.
    mx::set_cache_limit(256ull << 20);  // 256 MB

    std::cerr << ">> c4tts: loading weights from " << cfg.weights_dir << " ..." << std::endl;
    std::unique_ptr<Engine> engine;
    try {
        engine = std::make_unique<Engine>(cfg.weights_dir, cfg.lang, cfg.lru_capacity);
    } catch (const std::exception& e) {
        std::cerr << "error: failed to load model from " << cfg.weights_dir << ": " << e.what() << std::endl;
        return 1;
    }

    VoiceStore store(cfg.voice_store_dir);
    WorkDispatcher dispatcher(cfg.queue_size);

    const int listener = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listener < 0) {
        std::cerr << "error: socket() failed" << std::endl;
        return 1;
    }
    int one = 1;
    ::setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
        std::cerr << "error: invalid --host address: " << host << std::endl;
        return 1;
    }
    if (::bind(listener, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        std::cerr << "error: bind " << host << ":" << port << " failed (port in use?)" << std::endl;
        return 1;
    }
    if (::listen(listener, 64) != 0) {
        std::cerr << "error: listen() failed" << std::endl;
        return 1;
    }
    // Synthesis scratch lives under <voice_store>/.tmp (created per request), so
    // no CWD-relative outputs/ directory is needed.
    std::cerr << ">> c4tts server listening on http://" << host << ":" << port << std::endl;
    std::cerr << ">> endpoints: POST /v1/audio/speech, /v1/audio/voices, /api/voices" << std::endl;
    if (cfg.web_enabled) {
        std::cerr << ">> web admin: http://" << host << ":" << port << "/web"
                  << (cfg.web_key.empty() ? "  (no web key configured)" : "") << std::endl;
    }
    std::cerr << ">> voice store: " << cfg.voice_store_dir << " queue_size=" << cfg.queue_size
              << " lrucache=" << cfg.lru_capacity << std::endl;

    std::atomic<uint64_t> request_counter{0};
    std::thread accept_thread([&] {
        for (;;) {
            const int fd = ::accept(listener, nullptr, nullptr);
            if (fd < 0) continue;
            const uint64_t request_id = request_counter.fetch_add(1) + 1;
            std::thread([fd, &store, &cfg, &engine, &dispatcher, request_id] {
                handle_connection(fd, store, cfg, *engine, dispatcher, request_id);
            }).detach();
        }
    });
    accept_thread.detach();
    dispatcher.run_forever();
    return 0;
}

}  // namespace server
}  // namespace c4
