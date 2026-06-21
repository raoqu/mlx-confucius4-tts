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


DUMPERS = {
    "mel": dump_mel,
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
