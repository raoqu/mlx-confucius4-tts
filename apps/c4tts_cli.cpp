// c4tts_cli: command-line entry point for the standalone Confucius4-TTS engine.
//
// Subcommands:
//   (none) / --version      report version (and run an MLX backend smoke test)
//   synth                    full TTS: prompt wav + token ids -> waveform wav
//
// Text tokenization (SentencePiece) and normalization run upstream; `synth`
// consumes integer token ids (one per line) so the C++ engine is Python-free.

#include <mach-o/dyld.h>

#include <climits>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "c4tts/lang_tokens.h"
#include "c4tts/pipeline.h"
#include "c4tts/server.h"
#include "c4tts/tokenizer.h"
#include "c4tts/wav_io.h"
#include "mlx/mlx.h"

namespace mx = mlx::core;

namespace {
constexpr const char* kVersion = "c4tts 0.1.0";

int smoke_test() {
  auto a = mx::array({1.0f, 2.0f, 3.0f});
  auto b = mx::array({4.0f, 5.0f, 6.0f});
  auto c = mx::add(a, b);
  mx::eval(c);
  const float* d = c.data<float>();
  std::cout << "mlx smoke test: [" << d[0] << ", " << d[1] << ", " << d[2] << "]"
            << " on "
            << (mx::default_device().type == mx::Device::gpu ? "gpu (Metal)" : "cpu")
            << std::endl;
  return (d[0] == 5 && d[1] == 7 && d[2] == 9) ? 0 : 1;
}

std::string arg(int argc, char** argv, const std::string& key,
                const std::string& dflt = "") {
  for (int i = 1; i < argc - 1; ++i)
    if (key == argv[i]) return argv[i + 1];
  return dflt;
}

bool has_flag(int argc, char** argv, const std::string& key) {
  for (int i = 1; i < argc; ++i)
    if (key == argv[i]) return true;
  return false;
}

// Directory holding the executable (resolved via the Mach-O image path), or
// empty on failure. Lets defaults anchor to the install location, not the CWD.
std::string exe_dir() {
  char buf[PATH_MAX];
  uint32_t sz = sizeof(buf);
  if (_NSGetExecutablePath(buf, &sz) == 0) {
    std::string p(buf);
    auto slash = p.find_last_of('/');
    if (slash != std::string::npos) return p.substr(0, slash);
  }
  return "";
}

// Default model directory: <repo>/bin, resolved relative to the executable
// (<repo>/build/c4tts_cli) so it works regardless of the current dir.
std::string default_weights_dir() {
  const std::string d = exe_dir();
  return d.empty() ? "bin" : d + "/../bin";
}

// Default voice store: <repo>/voices, anchored to the executable so voices
// persist regardless of the working directory the server is launched from.
std::string default_voice_store() {
  const std::string d = exe_dir();
  return d.empty() ? "voices" : d + "/../voices";
}

int synth(int argc, char** argv) {
  std::string weights = arg(argc, argv, "--weights");
  if (weights.empty()) weights = arg(argc, argv, "--weight");  // tolerate singular
  if (weights.empty()) weights = default_weights_dir();
  const std::string prompt = arg(argc, argv, "--prompt");
  const std::string tokens = arg(argc, argv, "--tokens");
  const std::string text = arg(argc, argv, "--text");
  const std::string out = arg(argc, argv, "--out", "output.wav");
  if (prompt.empty() || (tokens.empty() && text.empty())) {
    std::cerr << "usage: c4tts_cli synth --weights DIR --prompt ref.wav "
                 "(--text \"...\" | --tokens ids.txt) --out out.wav\n";
    return 2;
  }

  const std::string lang = arg(argc, argv, "--lang", "zh");
  bool raw = false;
  for (int i = 1; i < argc; ++i)
    if (std::strcmp(argv[i], "--raw-text") == 0) raw = true;

  std::vector<int> ids;
  if (!text.empty()) {
    // Wrap with the model's trained prompt format unless --raw-text is given,
    // then tokenize in C++ (SentencePiece BPE) with BOS+EOS (LlamaTokenizer cfg).
    const std::string vocab_path = weights + "/tokenizer/vocab.tsv";
    std::ifstream vf(vocab_path);
    if (!vf.good()) {
      std::cerr << "c4tts: cannot find " << vocab_path
                << "\n       check --weights points at the export dir (e.g. "
                   "c4tts/weights) and run ./prepare.sh\n";
      return 2;
    }
    const std::string formatted = raw ? text : c4::format_tts_text(text, lang);
    c4::Tokenizer tok(vocab_path);
    ids = tok.encode(formatted, /*add_bos=*/true, /*add_eos=*/true);
  } else {
    std::ifstream tf(tokens);
    int id;
    while (tf >> id) ids.push_back(id);
  }
  if (ids.empty()) {
    std::cerr << "c4tts: no token ids\n";
    return 2;
  }

  c4::SynthOptions opt;
  const std::string mt = arg(argc, argv, "--max-tokens");
  if (!mt.empty()) opt.max_new_tokens = std::stoi(mt);
  const std::string st = arg(argc, argv, "--steps");
  if (!st.empty()) opt.n_timesteps = std::stoi(st);
  for (int i = 1; i < argc; ++i)
    if (std::strcmp(argv[i], "--greedy") == 0) opt.sample = false;

  bool bench = false;
  for (int i = 1; i < argc; ++i)
    if (std::strcmp(argv[i], "--bench") == 0) bench = true;

  std::cout << "c4tts: loading weights from " << weights << " ...\n";
  c4::Pipeline pipe(weights);
  if (bench) {
    // Warm the weight cache so the timed run reflects compute, not first-load.
    std::cout << "c4tts: warmup pass ...\n";
    pipe.synth(prompt, ids, opt);
  }
  std::cout << "c4tts: synthesizing (" << ids.size() << " text tokens, "
            << "max_new=" << opt.max_new_tokens << ", steps=" << opt.n_timesteps
            << ", " << (opt.sample ? "sampling" : "greedy") << ") ...\n";
  c4::Tensor wav = pipe.synth(prompt, ids, opt);
  c4::write_wav(out, wav, pipe.sample_rate());
  std::cout << "c4tts: wrote " << out << "\n";
  return 0;
}

// HTTP server + web admin. Mirrors index-tts2-metal's --web/--server modes:
//   c4tts_cli --server [--host H] [--port P] [--weights DIR] [--voice-store DIR]
//   c4tts_cli --web    [--web-key KEY] [--host H] [--port P] ...
// --web implies --server and additionally serves the web console at /web.
int serve(int argc, char** argv) {
  std::string weights = arg(argc, argv, "--weights");
  if (weights.empty()) weights = arg(argc, argv, "--weight");
  if (weights.empty()) weights = default_weights_dir();

  const std::string host = arg(argc, argv, "--host", "127.0.0.1");
  std::string port_str = arg(argc, argv, "--port");
  uint16_t port = 3456;
  if (!port_str.empty()) {
    const long n = std::stol(port_str);
    if (n < 1 || n > 65535) {
      std::cerr << "c4tts: --port must be 1-65535 (got " << n << ")\n";
      return 2;
    }
    port = static_cast<uint16_t>(n);
  }

  std::string voice_store = arg(argc, argv, "--voice-store");
  if (voice_store.empty()) voice_store = arg(argc, argv, "--voice_store");
  if (voice_store.empty()) voice_store = default_voice_store();  // stable, CWD-independent

  // --web-key (alias --webkey) supplies the admin key.
  std::string web_key = arg(argc, argv, "--web-key");
  if (web_key.empty()) web_key = arg(argc, argv, "--webkey");

  std::string queue_str = arg(argc, argv, "--queue-size");
  if (queue_str.empty()) queue_str = arg(argc, argv, "--queue_size");
  uint32_t queue_size = 0;
  if (!queue_str.empty()) queue_size = static_cast<uint32_t>(std::stoul(queue_str));

  // --lrucache N: per-voice conditioning cache capacity (0 disables). Unset
  // (UINT32_MAX) falls back to the C4TTS_LRU_CACHE env / default of 3.
  std::string lru_str = arg(argc, argv, "--lrucache");
  uint32_t lru_capacity = UINT32_MAX;
  if (!lru_str.empty()) lru_capacity = static_cast<uint32_t>(std::stoul(lru_str));

  const std::string lang = arg(argc, argv, "--lang", "zh");
  const bool web_enabled = has_flag(argc, argv, "--web");
  const bool verbose = has_flag(argc, argv, "--verbose") || has_flag(argc, argv, "-V");

  return c4::server::run_server(host, port, weights, voice_store, queue_size,
                                lru_capacity, web_enabled, web_key, lang, verbose);
}
}  // namespace

int main(int argc, char** argv) {
  // Default the product entrypoints to int8 weight quantization for the T2S
  // decode (~2x on the dominant stage, logit cosine 0.9999, tokens preserved).
  // Honored only if unset, so the user can override: C4TTS_QUANT=0 -> fp32,
  // C4TTS_QUANT=4 -> 4-bit. The library/golden tests construct modules directly
  // (not via main), so they keep the faithful fp32 default.
  if (!std::getenv("C4TTS_QUANT")) ::setenv("C4TTS_QUANT", "8", /*overwrite=*/0);

  if (argc >= 2 && std::strcmp(argv[1], "synth") == 0) return synth(argc, argv);
  // Server mode: `serve` subcommand, or --web/--server anywhere in the args.
  if ((argc >= 2 && std::strcmp(argv[1], "serve") == 0) ||
      has_flag(argc, argv, "--web") || has_flag(argc, argv, "--server")) {
    return serve(argc, argv);
  }
  if (argc >= 2 && (std::strcmp(argv[1], "--version") == 0 ||
                    std::strcmp(argv[1], "-v") == 0)) {
    std::cout << kVersion << std::endl;
    return 0;
  }
  std::cout << kVersion << std::endl;
  return smoke_test();
}
