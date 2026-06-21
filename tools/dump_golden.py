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


def _save_state_dict(out_dir, module):
    for k, v in module.state_dict().items():
        _save(out_dir, k, v.detach().cpu().numpy())


def dump_s2a_lr(out_base):
    """S2A E1+E2: SemanticTokenEmbedding and InterpolateRegulator.

    Uses seeded, randomly-initialized instances of the real classes so module
    math is verified independently of the trained checkpoint.
    """
    from confuciustts.flow.modules import SemanticTokenEmbedding
    from confuciustts.flow.length_regulator import InterpolateRegulator

    # --- E1: SemanticTokenEmbedding ---
    out_dir = os.path.join(out_base, "s2a_emb")
    torch.manual_seed(1)
    emb = SemanticTokenEmbedding(codebook_size=8192, codebook_dim=8, output_dim=1024)
    emb.eval()
    codes = torch.randint(0, 8192, (1, 20))
    with torch.no_grad():
        y = emb(codes)  # (1, 1024, 20)
    _save_state_dict(out_dir, emb)
    _save(out_dir, "codes", codes.numpy().astype("int64"))
    _save(out_dir, "out", y.numpy())
    print(f"[s2a_emb] codes{tuple(codes.shape)} -> {tuple(y.shape)} ; {out_dir}")

    # --- E2: InterpolateRegulator ---
    out_dir = os.path.join(out_base, "s2a_lr")
    torch.manual_seed(2)
    lr = InterpolateRegulator(channels=512, sampling_ratios=(1, 1, 1, 1),
                              out_channels=512, groups=1, in_channels=1024)
    lr.eval()
    x = torch.randn(1, 20, 1024)
    target_len = 34
    with torch.no_grad():
        out, _ = lr(x, torch.tensor([target_len]))  # (1, 34, 512)
    _save_state_dict(out_dir, lr)
    _save(out_dir, "x", x.numpy())
    _save(out_dir, "target_len", np.array([target_len], dtype="int64"))
    _save(out_dir, "out", out.numpy())
    print(f"[s2a_lr] x{tuple(x.shape)} -> {tuple(out.shape)} ; {out_dir}")


def dump_dit_mods(out_base):
    """DiT building blocks (E3a): TimestepEmbedding, AdaptiveLayerNorm,
    FeedForward, and RoPE precompute/apply."""
    from confuciustts.flow.DiT.modules import (
        TimestepEmbedding, AdaptiveLayerNorm, FeedForward,
        precompute_freqs_cis, apply_rotary_emb,
    )

    out_dir = os.path.join(out_base, "dit_mods")
    H = 64

    # TimestepEmbedding
    torch.manual_seed(3)
    te = TimestepEmbedding(H).eval()
    t = torch.rand(2)
    with torch.no_grad():
        te_out = te(t)
    _save(out_dir, "te.time_mlp.0.weight", te.time_mlp[0].weight.detach().numpy())
    _save(out_dir, "te.time_mlp.0.bias", te.time_mlp[0].bias.detach().numpy())
    _save(out_dir, "te.time_mlp.2.weight", te.time_mlp[2].weight.detach().numpy())
    _save(out_dir, "te.time_mlp.2.bias", te.time_mlp[2].bias.detach().numpy())
    _save(out_dir, "te_t", t.numpy())
    _save(out_dir, "te_out", te_out.numpy())

    # AdaptiveLayerNorm
    torch.manual_seed(4)
    aln = AdaptiveLayerNorm(H).eval()
    ax = torch.randn(2, 7, H)
    acond = torch.randn(2, H)
    with torch.no_grad():
        aln_out = aln(ax, acond)
    _save(out_dir, "aln.norm.weight", aln.norm.weight.detach().numpy())
    _save(out_dir, "aln.modulation.weight", aln.modulation.weight.detach().numpy())
    _save(out_dir, "aln.modulation.bias", aln.modulation.bias.detach().numpy())
    _save(out_dir, "aln_x", ax.numpy())
    _save(out_dir, "aln_cond", acond.numpy())
    _save(out_dir, "aln_out", aln_out.numpy())

    # FeedForward (SwiGLU)
    torch.manual_seed(5)
    ff = FeedForward(H, H * 3).eval()
    fx = torch.randn(2, 7, H)
    with torch.no_grad():
        ff_out = ff(fx)
    _save(out_dir, "ff.w1.weight", ff.w1.weight.detach().numpy())
    _save(out_dir, "ff.w2.weight", ff.w2.weight.detach().numpy())
    _save(out_dir, "ff.w3.weight", ff.w3.weight.detach().numpy())
    _save(out_dir, "ff_x", fx.numpy())
    _save(out_dir, "ff_out", ff_out.numpy())

    # RoPE: precompute + apply (head_dim=16, T=7, heads=4)
    T, heads, head_dim = 7, 4, 16
    fc = precompute_freqs_cis(T, head_dim, base=10000, dtype=torch.float32)
    rx = torch.randn(2, T, heads, head_dim)
    rope_out = apply_rotary_emb(rx, fc)
    _save(out_dir, "rope_fc", fc.numpy())
    _save(out_dir, "rope_x", rx.numpy())
    _save(out_dir, "rope_out", rope_out.numpy())

    print(f"[dit_mods] TimestepEmbedding/AdaLN/FeedForward/RoPE -> {out_dir}")


DUMPERS = {
    "mel": dump_mel,
    "nn": dump_nn,
    "s2a_lr": dump_s2a_lr,
    "dit_mods": dump_dit_mods,
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
