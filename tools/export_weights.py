#!/usr/bin/env python3
"""Export Confucius4-TTS checkpoints to a c4tts WeightStore (directory of .npy).

Performs the inference-time static transforms the C++ engine relies on:
  - folds weight_norm (weight_g/weight_v -> weight),
  - drops the WeightNormConv1d ".conv" wrapper level,
  - skips recomputable buffers (e.g. DiT freqs_cis).

Names are kept identical to the PyTorch state_dict otherwise, so C++ modules
load by the same prefixes.

Usage (run from repo root):
    python3 c4tts/tools/export_weights.py s2a            # -> c4tts/weights/s2a
"""
import argparse
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Outputs live under the c4tts repo root; the PyTorch reference packages
# (confuciustts/, external/, checkpoints/) live in the upstream Confucius4-TTS
# repo, which may be this repo's parent (nested) or a sibling. Make both
# importable so weight export works in either layout.
_REF = os.environ.get("CONFUCIUS_SRC", os.path.dirname(REPO_ROOT))
for _p in (REPO_ROOT, _REF):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

DEFAULT_OUT = os.path.join(REPO_ROOT, "bin")
# Recomputable buffers we never need to ship: DiT RoPE table, and the fixed
# Kaiser-sinc anti-aliasing filters (baked separately for the vocoder).
SKIP_SUFFIXES = ("freqs_cis", ".filter")


def _save(out_dir, name, arr):
    np.save(os.path.join(out_dir, name + ".npy"), np.ascontiguousarray(arr))


def _fold_weight_norm(g, v):
    # weight_norm dim=0: w = v * g / ||v||, norm over all dims except 0.
    dims = tuple(range(1, v.dim()))
    norm = v.pow(2).sum(dim=dims, keepdim=True).sqrt()
    return (v * g / norm)


def export_state_dict(sd, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    keys = set(sd.keys())
    for k in sd:
        if any(k.endswith(s) for s in SKIP_SUFFIXES):
            continue
        if k.endswith(".weight_g"):
            continue  # handled with its _v partner
        if k.endswith(".weight_v"):
            g = sd[k[: -len("weight_v")] + "weight_g"]
            w = _fold_weight_norm(g, sd[k])
            base = k[: -len(".weight_v")]
            if base.endswith(".conv"):  # WeightNormConv1d wrapper (s2a wavenet)
                base = base[: -len(".conv")]
            _save(out_dir, base + ".weight", w.detach().cpu().numpy())
            n += 1
            continue
        # Plain tensor; collapse the WeightNormConv1d ".conv." wrapper only for
        # the s2a wavenet (its conv.bias). Other modules (e.g. the T2S speaker
        # encoder TDNNs) use a real ".conv." sublayer that must be preserved.
        name = k.replace(".conv.", ".") if (".conv." in k and "wavenet" in k) else k
        _save(out_dir, name, sd[k].detach().cpu().numpy())
        n += 1
    return n


def export_s2a(out_base):
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("netease-youdao/Confucius4-TTS", filename="s2a_model.pt")
    sd = torch.load(p, map_location="cpu", weights_only=False)
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()
    out_dir = os.path.join(out_base, "s2a")
    n = export_state_dict(sd, out_dir)
    print(f"[s2a] exported {n} tensors -> {out_dir}")


def export_bigvgan(out_base):
    from huggingface_hub import hf_hub_download
    from external.bigvgan.alias_free_activation.torch.filter import (
        kaiser_sinc_filter1d,
    )
    repo = "nvidia/bigvgan_v2_22khz_80band_256x"
    f = hf_hub_download(repo, filename="bigvgan_generator.pt")
    ckpt = torch.load(f, map_location="cpu", weights_only=False)
    sd = ckpt["generator"] if "generator" in ckpt else ckpt
    out_dir = os.path.join(out_base, "bigvgan")
    n = export_state_dict(sd, out_dir)
    # Bake the (ratio=2, kernel=12) Kaiser-sinc anti-aliasing filters once.
    up = kaiser_sinc_filter1d(0.5 / 2, 0.6 / 2, 12).view(-1).numpy()
    down = kaiser_sinc_filter1d(0.5 / 2, 0.6 / 2, 12).view(-1).numpy()
    _save(out_dir, "up_filter", up)
    _save(out_dir, "down_filter", down)
    print(f"[bigvgan] exported {n} tensors (+filters) -> {out_dir}")


def export_w2vbert(out_base):
    from transformers import Wav2Vec2BertModel
    m = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")
    sd = m.state_dict()
    out_dir = os.path.join(out_base, "w2vbert")
    # Only the feature projection + conformer layers are needed.
    sd = {k: v for k, v in sd.items()
          if k.startswith("feature_projection.") or k.startswith("encoder.layers.")}
    n = export_state_dict(sd, out_dir)
    # Also export the layer-17 normalization stats (wav2vec2bert_stats.pt),
    # found under checkpoints/ of this repo or the upstream reference repo.
    stats_path = next(
        (p for p in (os.path.join(REPO_ROOT, "checkpoints", "wav2vec2bert_stats.pt"),
                     os.path.join(_REF, "checkpoints", "wav2vec2bert_stats.pt"))
         if os.path.exists(p)), None)
    if stats_path is None:
        raise FileNotFoundError("wav2vec2bert_stats.pt not found (checkpoints/)")
    stats = torch.load(stats_path, map_location="cpu")
    _save(out_dir, "semantic_mean", stats["mean"].numpy())
    _save(out_dir, "semantic_std", torch.sqrt(stats["var"]).numpy())
    print(f"[w2vbert] exported {n} tensors (+stats) -> {out_dir}")


def export_campplus(out_base):
    from huggingface_hub import hf_hub_download
    f = hf_hub_download("funasr/campplus", filename="campplus_cn_common.bin")
    sd = torch.load(f, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    out_dir = os.path.join(out_base, "campplus")
    n = export_state_dict(sd, out_dir)
    print(f"[campplus] exported {n} tensors -> {out_dir}")


def export_t2s(out_base, path=None):
    import safetensors.torch
    # Prefer an explicit local file, then a manual download dir, then HF cache.
    f = path
    if f is None:
        local = os.path.join(REPO_ROOT, "downloads", "t2s_model.safetensors")
        if os.path.exists(local):
            f = local
        else:
            from huggingface_hub import hf_hub_download
            f = hf_hub_download("netease-youdao/Confucius4-TTS",
                                filename="t2s_model.safetensors")
    sd = safetensors.torch.load_file(f, device="cpu")
    out_dir = os.path.join(out_base, "t2s")
    n = export_state_dict(sd, out_dir)
    print(f"[t2s] exported {n} tensors from {f} -> {out_dir}")


def export_audio(out_base):
    # Baked mel constants for the 22.05 kHz reference-mel front end.
    from librosa.filters import mel as librosa_mel_fn
    out_dir = os.path.join(out_base, "audio")
    os.makedirs(out_dir, exist_ok=True)
    basis = librosa_mel_fn(sr=22050, n_fft=1024, n_mels=80, fmin=0, fmax=None)
    _save(out_dir, "mel_basis", basis.astype("float32"))
    _save(out_dir, "hann", torch.hann_window(1024).numpy().astype("float32"))
    print(f"[audio] mel_basis + hann -> {out_dir}")


EXPORTERS = {
    "s2a": export_s2a,
    "bigvgan": export_bigvgan,
    "w2vbert": export_w2vbert,
    "campplus": export_campplus,
    "t2s": export_t2s,
    "audio": export_audio,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", choices=list(EXPORTERS) + ["all"])
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--t2s-path", default=None,
                    help="explicit path to t2s_model.safetensors")
    args = ap.parse_args()
    targets = list(EXPORTERS) if args.model == "all" else [args.model]
    for t in targets:
        if t == "t2s":
            export_t2s(args.out, args.t2s_path)
        else:
            EXPORTERS[t](args.out)


if __name__ == "__main__":
    main()
