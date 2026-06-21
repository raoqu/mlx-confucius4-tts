// HTTP server + web admin for c4tts.
//
// Mirrors the index-tts2-metal runtime server: the same single-page web
// console (served at /web) and the same HTTP API (OpenAI-style /v1/audio/*
// plus local /web/api/* management routes), backed by the same sqlite voice
// store layout. Synthesis runs through a resident c4::Pipeline that is loaded
// once at startup and reused for every request.
//
// Note: the c4tts voice "bundle" (.pt) is a c4tts-native container that embeds
// the source reference audio — the on-disk format differs from index-tts2's
// bundles, but the management mechanism (sqlite + samples/ + bundles/) is the
// same.

#pragma once

#include <cstdint>
#include <string>

namespace c4 {
namespace server {

// Returns the embedded web admin HTML (defined in web_assets.cpp).
const char* web_index_html();

// Runs the blocking HTTP server. `weights_dir` is the c4tts model weights
// root (the same directory passed to `c4tts_cli synth --weights`).
// `voice_store_dir` holds voices.sqlite + samples/ + bundles/ (default
// "voices"). When `web_enabled` is true the /web console is served; when
// `web_key` is non-empty the console and its /web/api/* routes require it.
// `lang` is the default synthesis language code used to wrap input text in
// the model's prompt format. Returns a process exit code.
// `lru_capacity` bounds the per-voice conditioning cache (0 disables it; pass
// UINT32_MAX to fall back to the C4TTS_LRU_CACHE env / default of 3).
int run_server(const std::string& host,
               uint16_t port,
               const std::string& weights_dir,
               const std::string& voice_store_dir,
               uint32_t queue_size,
               uint32_t lru_capacity,
               bool web_enabled,
               const std::string& web_key,
               const std::string& lang,
               bool verbose = false);

}  // namespace server
}  // namespace c4
