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

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DEFAULT_OUT = os.path.join(REPO_ROOT, "c4tts", "weights")
SKIP_SUFFIXES = ("freqs_cis",)


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
            name = k[: -len(".conv.weight_v")] + ".weight"  # drop ".conv"
            _save(out_dir, name, w.detach().cpu().numpy())
            n += 1
            continue
        # Plain tensor; collapse a ".conv." wrapper if present (WeightNormConv1d).
        name = k.replace(".conv.", ".") if ".conv." in k else k
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


EXPORTERS = {
    "s2a": export_s2a,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", choices=list(EXPORTERS) + ["all"])
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    targets = list(EXPORTERS) if args.model == "all" else [args.model]
    for t in targets:
        EXPORTERS[t](args.out)


if __name__ == "__main__":
    main()
