#!/usr/bin/env python3
"""Capture golden vectors from the PyTorch reference for c4tts parity tests.

Each subcommand dumps a module's inputs, outputs and any precomputed constants
(mel filterbank, windows, ...) as plain .npy files under <out>/<module>/, which
the corresponding C++ test loads and compares against (see docs/PLAN.md §2.4).

Run from the repository root so `confuciustts` is importable:

    python3 c4tts/tools/dump_golden.py mel
    python3 c4tts/tools/dump_golden.py all
"""
import argparse
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DEFAULT_OUT = os.path.join(REPO_ROOT, "c4tts", "golden")

# Audio params (config/inference_config.yaml -> audio:)
SR = 22050
N_FFT = 1024
HOP = 256
WIN = 1024
N_MELS = 80
FMIN = 0
FMAX = None


def _save(out_dir, name, arr):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, name + ".npy"), np.ascontiguousarray(arr))


def dump_mel(out_base):
    """STFT log-mel (F4): confuciustts.utils.audio_features.mel_spectrogram."""
    from librosa.filters import mel as librosa_mel_fn
    from confuciustts.utils.audio_features import mel_spectrogram

    out_dir = os.path.join(out_base, "mel")
    torch.manual_seed(0)
    # ~0.37s of audio so we exercise several frames.
    wav = (torch.randn(1, 8192) * 0.1).float()

    mel = mel_spectrogram(wav, SR, N_FFT, HOP, WIN, N_MELS, FMIN, FMAX)  # (1,80,T)

    # Precomputed constants the C++ side consumes (baked at export time).
    mel_basis = librosa_mel_fn(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX)
    hann = torch.hann_window(WIN).numpy()

    _save(out_dir, "audio", wav.squeeze(0).numpy())   # (T_samples,)
    _save(out_dir, "mel_out", mel.squeeze(0).numpy())  # (80, T_frames)
    _save(out_dir, "mel_basis", mel_basis.astype(np.float32))  # (80, 513)
    _save(out_dir, "hann", hann.astype(np.float32))            # (1024,)
    print(f"[mel] audio={tuple(wav.shape)} -> mel={tuple(mel.shape)} ; "
          f"basis={mel_basis.shape} ; saved to {out_dir}")


def dump_nn(out_base):
    """NN primitives (Phase A2): linear, conv1d, norms, activations."""
    import torch.nn.functional as F

    out_dir = os.path.join(out_base, "nn")
    torch.manual_seed(0)

    # linear: x (2,3,16) @ W(8,16) + b(8)
    x = torch.randn(2, 3, 16)
    w = torch.randn(8, 16)
    b = torch.randn(8)
    _save(out_dir, "linear_x", x.numpy())
    _save(out_dir, "linear_w", w.numpy())
    _save(out_dir, "linear_b", b.numpy())
    _save(out_dir, "linear_y", F.linear(x, w, b).numpy())

    # conv1d with stride/padding/dilation/groups
    cx = torch.randn(2, 8, 20)
    cw = torch.randn(12, 4, 3)  # groups=2 -> in/groups = 4
    cb = torch.randn(12)
    cy = F.conv1d(cx, cw, cb, stride=2, padding=2, dilation=2, groups=2)
    _save(out_dir, "conv_x", cx.numpy())
    _save(out_dir, "conv_w", cw.numpy())
    _save(out_dir, "conv_b", cb.numpy())
    _save(out_dir, "conv_y", cy.numpy())

    # layer_norm over last dim
    lx = torch.randn(2, 5, 16)
    lw = torch.randn(16)
    lb = torch.randn(16)
    ly = F.layer_norm(lx, (16,), lw, lb, eps=1e-5)
    _save(out_dir, "ln_x", lx.numpy())
    _save(out_dir, "ln_w", lw.numpy())
    _save(out_dir, "ln_b", lb.numpy())
    _save(out_dir, "ln_y", ly.numpy())

    # group_norm (N,C,L), groups=4
    gx = torch.randn(2, 16, 10)
    gw = torch.randn(16)
    gb = torch.randn(16)
    gy = F.group_norm(gx, 4, gw, gb, eps=1e-5)
    _save(out_dir, "gn_x", gx.numpy())
    _save(out_dir, "gn_w", gw.numpy())
    _save(out_dir, "gn_b", gb.numpy())
    _save(out_dir, "gn_y", gy.numpy())

    # rms_norm (repo's custom impl, float32)
    rx = torch.randn(2, 5, 16)
    rw = torch.randn(16)
    ms = rx.float().pow(2).mean(-1, keepdim=True)
    ry = (rx.float() * torch.rsqrt(ms + 1e-6)).type_as(rx) * rw
    _save(out_dir, "rms_x", rx.numpy())
    _save(out_dir, "rms_w", rw.numpy())
    _save(out_dir, "rms_y", ry.numpy())

    # activations
    ax = torch.randn(4, 32)
    _save(out_dir, "act_x", ax.numpy())
    _save(out_dir, "act_silu", F.silu(ax).numpy())
    _save(out_dir, "act_gelu", F.gelu(ax).numpy())
    _save(out_dir, "act_mish", F.mish(ax).numpy())

    print(f"[nn] linear/conv1d/layer_norm/group_norm/rms_norm/acts -> {out_dir}")


DUMPERS = {
    "mel": dump_mel,
    "nn": dump_nn,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("module", choices=list(DUMPERS) + ["all"])
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    targets = list(DUMPERS) if args.module == "all" else [args.module]
    for t in targets:
        DUMPERS[t](args.out)


if __name__ == "__main__":
    main()
