// c4tts_cli: command-line entry point for the standalone Confucius4-TTS engine.
//
// Subcommands:
//   (none) / --version      report version (and run an MLX backend smoke test)
//   synth                    full TTS: prompt wav + token ids -> waveform wav
//
// Text tokenization (SentencePiece) and normalization run upstream; `synth`
// consumes integer token ids (one per line) so the C++ engine is Python-free.

#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "c4tts/pipeline.h"
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

int synth(int argc, char** argv) {
  const std::string weights = arg(argc, argv, "--weights", "weights");
  const std::string prompt = arg(argc, argv, "--prompt");
  const std::string tokens = arg(argc, argv, "--tokens");
  const std::string text = arg(argc, argv, "--text");
  const std::string out = arg(argc, argv, "--out", "output.wav");
  if (prompt.empty() || (tokens.empty() && text.empty())) {
    std::cerr << "usage: c4tts_cli synth --weights DIR --prompt ref.wav "
                 "(--text \"...\" | --tokens ids.txt) --out out.wav\n";
    return 2;
  }

  std::vector<int> ids;
  if (!text.empty()) {
    // Tokenize in C++ (SentencePiece BPE), BOS+EOS per the LlamaTokenizer config.
    c4::Tokenizer tok(weights + "/tokenizer/vocab.tsv");
    ids = tok.encode(text, /*add_bos=*/true, /*add_eos=*/true);
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

  std::cout << "c4tts: loading weights from " << weights << " ...\n";
  c4::Pipeline pipe(weights);
  std::cout << "c4tts: synthesizing (" << ids.size() << " text tokens, "
            << "max_new=" << opt.max_new_tokens << ", steps=" << opt.n_timesteps
            << ", " << (opt.sample ? "sampling" : "greedy") << ") ...\n";
  c4::Tensor wav = pipe.synth(prompt, ids, opt);
  c4::write_wav(out, wav, pipe.sample_rate());
  std::cout << "c4tts: wrote " << out << "\n";
  return 0;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc >= 2 && std::strcmp(argv[1], "synth") == 0) return synth(argc, argv);
  if (argc >= 2 && (std::strcmp(argv[1], "--version") == 0 ||
                    std::strcmp(argv[1], "-v") == 0)) {
    std::cout << kVersion << std::endl;
    return 0;
  }
  std::cout << kVersion << std::endl;
  return smoke_test();
}
