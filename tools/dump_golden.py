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


def dump_dit_block(out_base):
    """DiT E3b: Attention and DiTBlock."""
    from confuciustts.flow.DiT.modules import (
        Attention, DiTBlock, precompute_freqs_cis,
    )

    out_dir = os.path.join(out_base, "dit_block")
    dim, heads, T = 64, 4, 9
    head_dim = dim // heads
    fc = precompute_freqs_cis(T, head_dim, base=10000, dtype=torch.float32)
    _save(out_dir, "freqs_cis", fc.numpy())

    # Attention (no mask)
    torch.manual_seed(6)
    attn = Attention(dim, heads).eval()
    ax = torch.randn(2, T, dim)
    with torch.no_grad():
        ao = attn(ax, None, fc)
    _save(out_dir, "attn.wqkv.weight", attn.wqkv.weight.detach().numpy())
    _save(out_dir, "attn.wo.weight", attn.wo.weight.detach().numpy())
    _save(out_dir, "attn_x", ax.numpy())
    _save(out_dir, "attn_out", ao.numpy())

    # DiTBlock (no skip, no mask)
    torch.manual_seed(7)
    blk = DiTBlock(dim, heads, dim * 3).eval()
    bx = torch.randn(2, T, dim)
    cond = torch.randn(2, dim)
    with torch.no_grad():
        bo = blk(bx, cond, None, fc, None)
    _save_state_dict(out_dir, blk)  # keys: attention.*, feed_forward.*, *_norm.*
    _save(out_dir, "blk_x", bx.numpy())
    _save(out_dir, "blk_cond", cond.numpy())
    _save(out_dir, "blk_out", bo.numpy())

    print(f"[dit_block] Attention/DiTBlock -> {out_dir}")


def dump_dit_wn(out_base):
    """DiT E3c-1: WaveNet (WN, weight_norm folded) and FinalLayer."""
    from confuciustts.flow.wavenet import WN
    from confuciustts.flow.DiT.modules import FinalLayer

    out_dir = os.path.join(out_base, "dit_wn")
    hidden, n_layers, kernel, gin, T = 16, 3, 5, 16, 11

    torch.manual_seed(8)
    wn = WN(hidden, kernel, 1, n_layers, gin).eval()
    x = torch.randn(2, hidden, T)
    xmask = torch.ones(2, 1, T)
    g = torch.randn(2, gin, 1)
    with torch.no_grad():
        y = wn(x, xmask, g)

    # Fold weight_norm by reading the effective `.conv.weight`/`.conv.bias`.
    def fold(mod, name):
        _save(out_dir, name + ".weight", mod.conv.weight.detach().numpy())
        _save(out_dir, name + ".bias", mod.conv.bias.detach().numpy())

    fold(wn.cond_layer, "cond_layer")
    for i in range(n_layers):
        fold(wn.in_layers[i], f"in_layers.{i}")
        fold(wn.res_skip_layers[i], f"res_skip_layers.{i}")
    _save(out_dir, "wn_x", x.numpy())
    _save(out_dir, "wn_mask", xmask.numpy())
    _save(out_dir, "wn_g", g.numpy())
    _save(out_dir, "wn_out", y.numpy())

    # FinalLayer
    torch.manual_seed(9)
    fl = FinalLayer(hidden).eval()
    fx = torch.randn(2, 7, hidden)
    fcond = torch.randn(2, hidden)
    with torch.no_grad():
        fo = fl(fx, fcond)
    _save(out_dir, "fl.linear.weight", fl.linear.weight.detach().numpy())
    _save(out_dir, "fl.linear.bias", fl.linear.bias.detach().numpy())
    _save(out_dir, "fl.mod.weight", fl.adaLN_modulation[1].weight.detach().numpy())
    _save(out_dir, "fl.mod.bias", fl.adaLN_modulation[1].bias.detach().numpy())
    _save(out_dir, "fl_x", fx.numpy())
    _save(out_dir, "fl_cond", fcond.numpy())
    _save(out_dir, "fl_out", fo.numpy())

    print(f"[dit_wn] WaveNet/FinalLayer -> {out_dir}")


def dump_dit_full(out_base):
    """DiT E3c-2: full DiT.forward (small config) with WaveNet final layer."""
    from confuciustts.flow.DiT.dit import DiT

    out_dir = os.path.join(out_base, "dit_full")
    hidden, heads, depth, mel, mu_dim, spk = 32, 4, 5, 8, 16, 6
    # The reference ties FinalLayer/wavenet width to hidden_dim (real config has
    # estimator_hidden_dim == wavenet_hidden_dim == 512), so keep them equal.
    wn_hidden, wn_kernel, wn_dil, wn_layers = 32, 5, 1, 3

    torch.manual_seed(11)
    dit = DiT(hidden_dim=hidden, num_heads=heads, depth=depth, mel_dim=mel,
              mu_dim=mu_dim, spk_dim=spk, long_skip_connection=True,
              ff_intermediate_size=hidden * 3, final_layer="wavenet",
              wavenet_hidden_dim=wn_hidden, wavenet_kernel_size=wn_kernel,
              wavenet_dilation_rate=wn_dil, wavenet_num_layers=wn_layers).eval()

    B, T = 1, 10
    x = torch.randn(B, mel, T)
    mask = torch.ones(B, T, dtype=torch.bool)
    mu = torch.randn(B, T, mu_dim)
    t = torch.rand(B)
    spks = torch.randn(B, spk)
    cond = torch.randn(B, mel, T)
    with torch.no_grad():
        y = dit(x, mask, mu, t, spks, cond)

    # All params except the buffer and wavenet (folded separately below).
    for k, v in dit.state_dict().items():
        if k == "freqs_cis" or k.startswith("wavenet."):
            continue
        _save(out_dir, k, v.detach().cpu().numpy())

    def fold(mod, name):
        _save(out_dir, name + ".weight", mod.conv.weight.detach().numpy())
        _save(out_dir, name + ".bias", mod.conv.bias.detach().numpy())

    fold(dit.wavenet.cond_layer, "wavenet.cond_layer")
    for i in range(wn_layers):
        fold(dit.wavenet.in_layers[i], f"wavenet.in_layers.{i}")
        fold(dit.wavenet.res_skip_layers[i], f"wavenet.res_skip_layers.{i}")

    _save(out_dir, "x", x.numpy())
    _save(out_dir, "mask", mask.float().numpy())
    _save(out_dir, "mu", mu.numpy())
    _save(out_dir, "t", t.numpy())
    _save(out_dir, "spks", spks.numpy())
    _save(out_dir, "cond", cond.numpy())
    _save(out_dir, "out", y.numpy())
    print(f"[dit_full] DiT.forward x{tuple(x.shape)} -> {tuple(y.shape)} ; {out_dir}")


DUMPERS = {
    "mel": dump_mel,
    "nn": dump_nn,
    "s2a_lr": dump_s2a_lr,
    "dit_mods": dump_dit_mods,
    "dit_block": dump_dit_block,
    "dit_wn": dump_dit_wn,
    "dit_full": dump_dit_full,
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
