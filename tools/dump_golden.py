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

    # conv2d (N,C,H,W), groups=1, stride (2,1), padding 1
    c2x = torch.randn(1, 3, 8, 6)
    c2w = torch.randn(5, 3, 3, 3)
    c2b = torch.randn(5)
    c2y = F.conv2d(c2x, c2w, c2b, stride=(2, 1), padding=1)
    _save(out_dir, "conv2d_x", c2x.numpy())
    _save(out_dir, "conv2d_w", c2w.numpy())
    _save(out_dir, "conv2d_b", c2b.numpy())
    _save(out_dir, "conv2d_y", c2y.numpy())

    # batch_norm (eval) on (N,C,T) with running stats
    bn = torch.nn.BatchNorm1d(8)
    bn.running_mean.normal_()
    bn.running_var.uniform_(0.5, 1.5)
    bn.weight.data.normal_()
    bn.bias.data.normal_()
    bn.eval()
    bnx = torch.randn(2, 8, 10)
    with torch.no_grad():
        bny = bn(bnx)
    _save(out_dir, "bn_x", bnx.numpy())
    _save(out_dir, "bn_mean", bn.running_mean.numpy())
    _save(out_dir, "bn_var", bn.running_var.numpy())
    _save(out_dir, "bn_w", bn.weight.detach().numpy())
    _save(out_dir, "bn_b", bn.bias.detach().numpy())
    _save(out_dir, "bn_y", bny.numpy())

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


def dump_cfm(out_base):
    """S2A E4: ConditionalCFM.solve_euler (Euler ODE + CFG)."""
    from confuciustts.flow.flow_matching import ConditionalCFM

    out_dir = os.path.join(out_base, "cfm")
    hidden, heads, depth, mel, cond_dim, style = 32, 4, 5, 8, 16, 6

    torch.manual_seed(12)
    cfm = ConditionalCFM(
        sigma_min=1e-6, training_cfg_rate=0.2, inference_cfg_rate=0.7,
        t_scheduler="linear", hidden_dim=hidden, num_heads=heads, depth=depth,
        mel_dim=mel, cond_dim=cond_dim, style_dim=style,
        long_skip_connection=True, ff_intermediate_size=hidden * 3,
        final_layer="wavenet", wavenet_hidden_dim=hidden, wavenet_kernel_size=5,
        wavenet_dilation_rate=1, wavenet_num_layers=3).eval()

    b, T, Tref = 1, 10, 4
    mu = torch.randn(b, T, cond_dim)
    prompt = torch.randn(b, mel, Tref)
    spks = torch.randn(b, style)
    x_lens = torch.tensor([T])
    n_timesteps = 6
    cfg_rate = 0.7
    torch.manual_seed(99)
    z = torch.randn(b, mel, T)
    t_span = torch.linspace(0, 1, n_timesteps + 1)  # linear schedule

    with torch.no_grad():
        out = cfm.solve_euler(z.clone(), t_span=t_span, x_lens=x_lens,
                              prompt=prompt, mu=mu, spks=spks, cfg_rate=cfg_rate)

    for k, v in cfm.state_dict().items():
        if "freqs_cis" in k or ".wavenet." in k:
            continue
        _save(out_dir, k, v.detach().cpu().numpy())

    def fold(mod, name):
        _save(out_dir, name + ".weight", mod.conv.weight.detach().numpy())
        _save(out_dir, name + ".bias", mod.conv.bias.detach().numpy())

    wn = cfm.estimator.wavenet
    fold(wn.cond_layer, "estimator.wavenet.cond_layer")
    for i in range(3):
        fold(wn.in_layers[i], f"estimator.wavenet.in_layers.{i}")
        fold(wn.res_skip_layers[i], f"estimator.wavenet.res_skip_layers.{i}")

    _save(out_dir, "z", z.numpy())
    _save(out_dir, "mu", mu.numpy())
    _save(out_dir, "prompt", prompt.numpy())
    _save(out_dir, "spks", spks.numpy())
    _save(out_dir, "mask", torch.ones(b, T).numpy())
    _save(out_dir, "t_span", t_span.numpy())
    _save(out_dir, "out", out.numpy())
    print(f"[cfm] solve_euler n={n_timesteps} -> {tuple(out.shape)} ; {out_dir}")


def dump_s2a_inference(out_base):
    """S2A E5: full MaskedDiffWithXvec.inference with the REAL checkpoint."""
    from confuciustts.flow.flow import MaskedDiffWithXvec, MaskedDiffWithXvecConfig
    from huggingface_hub import hf_hub_download

    out_dir = os.path.join(out_base, "s2a_infer")
    cfg = MaskedDiffWithXvecConfig(input_size=512, output_size=80,
                                   spk_embed_dim=192, semantic_embed_dim=1024,
                                   lm_latent_dim=1280, estimator_mlp_ratio=3.0)
    model = MaskedDiffWithXvec(cfg)
    sd = torch.load(hf_hub_download("netease-youdao/Confucius4-TTS",
                                    filename="s2a_model.pt"),
                    map_location="cpu", weights_only=False)
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()
    model.load_state_dict(sd)
    model.eval()

    torch.manual_seed(7)
    T_sem, T_ref, target = 8, 6, 12
    codes = torch.randint(0, 8192, (1, T_sem))
    lm_latent = torch.randn(1, T_sem, 1280)
    prompt_feat = torch.randn(1, T_ref, 80)
    embedding = torch.randn(1, 192)
    n_timesteps, cfg_rate = 4, 0.7

    # Capture the initial noise the decoder will draw (first randn after seed).
    t_total = T_ref + target
    torch.manual_seed(123)
    z = torch.randn(1, 80, t_total)
    torch.manual_seed(123)
    with torch.no_grad():
        mel = model.inference(codes, lm_latent, prompt_feat, embedding,
                              torch.tensor([target]), n_timesteps=n_timesteps,
                              inference_cfg_rate=cfg_rate)

    _save(out_dir, "codes", codes.numpy().astype("int64"))
    _save(out_dir, "lm_latent", lm_latent.numpy())
    _save(out_dir, "prompt_feat", prompt_feat.numpy())
    _save(out_dir, "embedding", embedding.numpy())
    _save(out_dir, "z", z.numpy())
    _save(out_dir, "target_len", np.array([target], dtype="int64"))
    _save(out_dir, "out", mel.numpy())
    print(f"[s2a_infer] inference -> mel {tuple(mel.shape)} ; {out_dir}")


def dump_vocoder_act(out_base):
    """BigVGAN F-1: anti-aliased SnakeBeta activation."""
    from external.bigvgan.activations import SnakeBeta
    from external.bigvgan.alias_free_activation.torch.act import Activation1d

    out_dir = os.path.join(out_base, "vocoder_act")
    C = 16
    torch.manual_seed(20)
    act = Activation1d(activation=SnakeBeta(C, alpha_logscale=True)).eval()
    with torch.no_grad():
        act.act.alpha.copy_(torch.randn(C) * 0.3)
        act.act.beta.copy_(torch.randn(C) * 0.3)
    x = torch.randn(1, C, 13)
    with torch.no_grad():
        y = act(x)

    _save(out_dir, "alpha", act.act.alpha.detach().numpy())
    _save(out_dir, "beta", act.act.beta.detach().numpy())
    _save(out_dir, "up_filter", act.upsample.filter.detach().view(-1).numpy())
    _save(out_dir, "down_filter",
          act.downsample.lowpass.filter.detach().view(-1).numpy())
    _save(out_dir, "x", x.numpy())
    _save(out_dir, "out", y.numpy())
    print(f"[vocoder_act] Activation1d(SnakeBeta) x{tuple(x.shape)} -> "
          f"{tuple(y.shape)} ; {out_dir}")


def dump_bigvgan(out_base):
    """BigVGAN F-2: full generator (mel -> waveform) on the REAL checkpoint."""
    from external.bigvgan.bigvgan import BigVGAN

    out_dir = os.path.join(out_base, "bigvgan")
    model = BigVGAN.from_pretrained("nvidia/bigvgan_v2_22khz_80band_256x",
                                    use_cuda_kernel=False)
    model.eval()
    torch.manual_seed(30)
    mel = torch.randn(1, 80, 10)
    with torch.no_grad():
        wav = model(mel)  # (1, 1, 2560)
    _save(out_dir, "mel", mel.numpy())
    _save(out_dir, "wav", wav.numpy())
    print(f"[bigvgan] mel{tuple(mel.shape)} -> wav {tuple(wav.shape)} ; {out_dir}")


def dump_gpt2(out_base):
    """T2S D-1: GPT-2 transformer stack (with zeroed position embedding)."""
    from transformers import GPT2Config, GPT2Model

    out_dir = os.path.join(out_base, "gpt2")
    torch.manual_seed(40)
    cfg = GPT2Config(vocab_size=100, n_positions=64, n_embd=64, n_layer=3,
                     n_head=4, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
    m = GPT2Model(cfg).eval()
    m.wpe.weight.data.zero_()  # matches DummyPositionEmbedding (zeros)

    emb = torch.randn(1, 9, 64)
    with torch.no_grad():
        out = m(inputs_embeds=emb).last_hidden_state

    for k, v in m.state_dict().items():
        if k.startswith("wpe") or k.startswith("wte"):
            continue
        if k.endswith(".attn.bias") or k.endswith(".attn.masked_bias"):
            continue  # causal-mask buffers (not c_attn.bias); C++ builds its own
        _save(out_dir, k, v.detach().cpu().numpy())
    _save(out_dir, "emb", emb.numpy())
    _save(out_dir, "out", out.numpy())
    print(f"[gpt2] inputs_embeds{tuple(emb.shape)} -> {tuple(out.shape)} ; {out_dir}")


def dump_t2s_mods(out_base):
    """T2S D-2: SpeakerEncoder (ECAPA), TextEmbeddingProjector, position emb."""
    from confuciustts.llm.speaker_encoder import (
        Qwen3TTSSpeakerEncoder, Qwen3TTSSpeakerEncoderConfig)
    from confuciustts.llm.text_encoder import TextEmbeddingProjector
    from confuciustts.llm.position_embeddings import LearnedPositionalEmbedding

    out_dir = os.path.join(out_base, "t2s_mods")

    # --- SpeakerEncoder (small config, same architecture as the real one) ---
    torch.manual_seed(41)
    cfg = Qwen3TTSSpeakerEncoderConfig(
        mel_dim=64, enc_dim=80, enc_channels=[64, 64, 64, 64, 128],
        enc_kernel_sizes=[5, 3, 3, 3, 1], enc_dilations=[1, 2, 3, 4, 1],
        enc_attention_channels=16, enc_res2net_scale=8, enc_se_channels=16)
    spk = Qwen3TTSSpeakerEncoder(cfg).eval()
    sx = torch.randn(1, 15, 64)
    with torch.no_grad():
        sy = spk(sx)  # (1, 80)
    for k, v in spk.state_dict().items():
        _save(out_dir, "spk." + k, v.detach().cpu().numpy())
    _save(out_dir, "spk_x", sx.numpy())
    _save(out_dir, "spk_out", sy.numpy())

    # --- TextEmbeddingProjector ---
    torch.manual_seed(42)
    tp = TextEmbeddingProjector(vocab_size=50, embed_dim=32, output_size=16).eval()
    tids = torch.randint(0, 50, (1, 7))
    with torch.no_grad():
        ty = tp(tids)
    for k, v in tp.state_dict().items():
        _save(out_dir, "tp." + k, v.detach().cpu().numpy())
    _save(out_dir, "tp_ids", tids.numpy().astype("int64"))
    _save(out_dir, "tp_out", ty.numpy())

    # --- LearnedPositionalEmbedding ---
    torch.manual_seed(43)
    pe = LearnedPositionalEmbedding(20, 16).eval()
    px = torch.randn(1, 9, 16)
    with torch.no_grad():
        py = pe(px)
    _save(out_dir, "pe_table", pe.embedding.weight.detach().numpy())
    _save(out_dir, "pe_x", px.numpy())
    _save(out_dir, "pe_out", py.numpy())

    print(f"[t2s_mods] SpeakerEncoder/TextProjector/PosEmb -> {out_dir}")


def dump_t2s_full(out_base):
    """T2S D-3: full Text2Semantic.forward logits (small seeded model)."""
    from confuciustts.llm.llm import Text2Semantic, Text2SemanticConfig

    out_dir = os.path.join(out_base, "t2s_full")
    torch.manual_seed(44)
    cfg = Text2SemanticConfig(
        num_layers=2, model_dim=64, num_heads=4, vocab_size=50,
        semantic_vocab_size=40, text_embedding_dim=32, speaker_embedding_dim=48,
        start_semantic_token=38, stop_semantic_token=39)
    model = Text2Semantic(cfg).eval()

    B, T_text, T_sem, T_cond = 1, 5, 7, 9
    text_ids = torch.randint(0, 50, (B, T_text))
    semantic_codes = torch.randint(0, 38, (B, T_sem))
    condition_vector = torch.randn(B, T_cond, 48)
    attn = torch.ones(B, 1 + T_text + T_sem, dtype=torch.bool)
    with torch.no_grad():
        out = model(text_inputs=text_ids, semantic_codes=semantic_codes,
                    condition_vector=condition_vector, attention_mask=attn)
    logits = out.logits  # (B, T_sem, vocab)

    for k, v in model.state_dict().items():
        if k.endswith(".attn.bias") or k.endswith(".attn.masked_bias"):
            continue
        if k.startswith("transformer.wpe") or k.startswith("transformer.wte"):
            continue
        _save(out_dir, k, v.detach().cpu().numpy())
    _save(out_dir, "text_ids", text_ids.numpy().astype("int64"))
    _save(out_dir, "semantic_codes", semantic_codes.numpy().astype("int64"))
    _save(out_dir, "condition_vector", condition_vector.numpy())
    _save(out_dir, "logits", logits.numpy())
    print(f"[t2s_full] logits {tuple(logits.shape)} ; {out_dir}")


def dump_campplus(out_base):
    """CAMPPlus speaker encoder (C2): fbank -> style embedding (seeded model)."""
    from external.campplus import CAMPPlus

    out_dir = os.path.join(out_base, "campplus")
    torch.manual_seed(45)
    model = CAMPPlus(feat_dim=80, embedding_size=512).eval()
    # Randomize BN running stats so eval-mode normalization is exercised.
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            m.running_mean.normal_(0, 0.5)
            m.running_var.uniform_(0.5, 1.5)

    fbank = torch.randn(1, 150, 80)  # T=150 exercises seg_pooling partial window
    with torch.no_grad():
        emb = model(fbank)  # (1, 512)
    for k, v in model.state_dict().items():
        if k.endswith("num_batches_tracked"):
            continue
        _save(out_dir, k, v.detach().cpu().numpy())
    _save(out_dir, "fbank", fbank.numpy())
    _save(out_dir, "emb", emb.numpy())
    print(f"[campplus] fbank{tuple(fbank.shape)} -> {tuple(emb.shape)} ; {out_dir}")


def dump_w2vbert(out_base):
    """W2V-BERT conformer (C1): small seeded Wav2Vec2BertModel."""
    from transformers import Wav2Vec2BertConfig, Wav2Vec2BertModel

    out_dir = os.path.join(out_base, "w2vbert")
    torch.manual_seed(46)
    cfg = Wav2Vec2BertConfig(
        hidden_size=64, num_hidden_layers=3, num_attention_heads=4,
        intermediate_size=128, feature_projection_input_dim=32,
        conv_depthwise_kernel_size=7, position_embeddings_type="relative_key",
        left_max_position_embeddings=4, right_max_position_embeddings=2,
        add_adapter=False, hidden_act="swish", hidden_dropout=0.0,
        attention_dropout=0.0, feat_proj_dropout=0.0, activation_dropout=0.0,
        conformer_conv_dropout=0.0, layerdrop=0.0)
    m = Wav2Vec2BertModel(cfg).eval()

    feats = torch.randn(1, 12, 32)
    with torch.no_grad():
        out = m(input_features=feats, output_hidden_states=True)

    for k, v in m.state_dict().items():
        if not (k.startswith("feature_projection.") or k.startswith("encoder.layers.")):
            continue
        _save(out_dir, k, v.detach().cpu().numpy())
    _save(out_dir, "feats", feats.numpy())
    _save(out_dir, "hs2", out.hidden_states[2].numpy())   # after layer 1
    _save(out_dir, "last", out.last_hidden_state.numpy())  # after layer 2
    print(f"[w2vbert] feats{tuple(feats.shape)} -> last {tuple(out.last_hidden_state.shape)} ; {out_dir}")


def dump_fbank(out_base):
    """Kaldi fbank (B3): torchaudio.compliance.kaldi.fbank parity."""
    import torchaudio

    out_dir = os.path.join(out_base, "fbank")
    torch.manual_seed(47)
    wav = torch.randn(1, 6000) * 0.1  # ~0.375 s at 16 kHz
    fb = torchaudio.compliance.kaldi.fbank(
        wav, num_mel_bins=80, sample_frequency=16000, dither=0.0)
    _save(out_dir, "wav", wav.squeeze(0).numpy())
    _save(out_dir, "fbank", fb.numpy())
    print(f"[fbank] wav{tuple(wav.shape)} -> {tuple(fb.shape)} ; {out_dir}")


def dump_seamless(out_base):
    """SeamlessM4T feature extractor (B4): W2V-BERT 160-dim input features."""
    from transformers import SeamlessM4TFeatureExtractor

    out_dir = os.path.join(out_base, "seamless")
    fe = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
    torch.manual_seed(48)
    wav = (torch.randn(8000) * 0.1).numpy()  # 0.5 s at 16 kHz
    feats = fe(wav, sampling_rate=16000, return_tensors="np")["input_features"][0]
    _save(out_dir, "wav", wav)
    _save(out_dir, "feats", feats)
    print(f"[seamless] wav{wav.shape} -> {feats.shape} ; {out_dir}")


def dump_resample(out_base):
    """Resample (B1): torchaudio.functional.resample parity."""
    import torchaudio

    out_dir = os.path.join(out_base, "resample")
    torch.manual_seed(49)
    wav = torch.randn(4000) * 0.1  # source at 24 kHz
    r16 = torchaudio.functional.resample(wav, 24000, 16000)
    r22 = torchaudio.functional.resample(wav, 24000, 22050)
    _save(out_dir, "wav", wav.numpy())
    _save(out_dir, "r16", r16.numpy())
    _save(out_dir, "r22", r22.numpy())
    print(f"[resample] 24k{tuple(wav.shape)} -> 16k{tuple(r16.shape)} 22k{tuple(r22.shape)} ; {out_dir}")


def dump_prompt(out_base):
    """Real-weight prompt encoders (C integration): semantic + style from wav."""
    import torchaudio
    from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertModel
    from huggingface_hub import hf_hub_download
    from external.campplus import CAMPPlus

    out_dir = os.path.join(out_base, "prompt")
    fe = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
    w2v = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0").eval()
    stats = torch.load(os.path.join(REPO_ROOT, "checkpoints", "wav2vec2bert_stats.pt"),
                       map_location="cpu")
    mean, std = stats["mean"], torch.sqrt(stats["var"])
    camp = CAMPPlus(feat_dim=80, embedding_size=192)
    camp.load_state_dict(torch.load(
        hf_hub_download("funasr/campplus", filename="campplus_cn_common.bin"),
        map_location="cpu"), strict=False)
    camp.eval()

    torch.manual_seed(50)
    wav16 = torch.randn(1, 8000) * 0.1

    inputs = fe(wav16.squeeze(0).numpy(), sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        feats = w2v(input_features=inputs["input_features"],
                    attention_mask=inputs.get("attention_mask"),
                    output_hidden_states=True).hidden_states[17]
    sem = (feats - mean) / std

    fbank = torchaudio.compliance.kaldi.fbank(
        wav16, num_mel_bins=80, sample_frequency=16000, dither=0.0)
    fbank = fbank - fbank.mean(0, keepdim=True)
    with torch.no_grad():
        style = camp(fbank.unsqueeze(0))

    _save(out_dir, "wav16", wav16.squeeze(0).numpy())
    _save(out_dir, "semantic", sem.numpy())
    _save(out_dir, "style", style.numpy())
    print(f"[prompt] semantic{tuple(sem.shape)} style{tuple(style.shape)} ; {out_dir}")


DUMPERS = {
    "mel": dump_mel,
    "nn": dump_nn,
    "s2a_lr": dump_s2a_lr,
    "dit_mods": dump_dit_mods,
    "dit_block": dump_dit_block,
    "dit_wn": dump_dit_wn,
    "dit_full": dump_dit_full,
    "cfm": dump_cfm,
    "s2a_infer": dump_s2a_inference,
    "vocoder_act": dump_vocoder_act,
    "bigvgan": dump_bigvgan,
    "gpt2": dump_gpt2,
    "t2s_mods": dump_t2s_mods,
    "t2s_full": dump_t2s_full,
    "campplus": dump_campplus,
    "w2vbert": dump_w2vbert,
    "fbank": dump_fbank,
    "seamless": dump_seamless,
    "resample": dump_resample,
    "prompt": dump_prompt,
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
