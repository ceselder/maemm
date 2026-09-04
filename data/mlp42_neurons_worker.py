"""Layer-42 MLP neuron analysis for Qwen/Qwen3.6-27B — container-side logic (launched by modal_mlp42_neurons.py).

Question: take the layer-42 MLP neurons (the 17,408 SwiGLU units feeding down_proj), find the SPARSELY
activating ones, project each through down_proj into the residual stream (its output direction = the
corresponding COLUMN of down_proj.weight) and ask whether those directions have structure: are they close to
SAE features / BSF block subspaces / cluster-probe directions, do they cluster among themselves, and can our
activation->text inverter verbalize them?

Conventions (match the rest of the suite): forward = [BOS 248044] + 256 content tokens of a TRAIN window of
/data/acts27b (rows < ceil(0.95 * n_seq)); position 0 (BOS/sink) is dropped from every statistic. The layer-42
read of the suite (mxf.inject.read_resid on decoder layer 42's output) is resid_post = the stream AFTER layer
42's MLP write, so the neurons studied here write directly into the very stream our directions live in.

A SwiGLU neuron value a = silu(gate) * up is SIGNED, so every statistic is kept for |a| with the sign of the
extreme value recorded as the neuron's polarity; the direction a neuron writes when it fires is
polarity * unit(down_proj[:, i]).

Stage "stats"  -> /data/mlp42/neuron_stats.npz, down_proj_cols.f16, sel_windows.npz, peak_context.npz, sae_match.npz
Stage "dirs"   -> /data/mlp42/dirs_analysis.npz   (BSF block-subspace fraction, cluster-probe NN cosine, controls)
Stage "verbalize" -> /data/mlp42/verbalize_<tag>.jsonl + .json summary (inverter generations + clean-base scoring)
"""
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from mxf.config import D_MODEL, INJECT_LAYER, READ_LAYER, STEER_COEFF
from mxf.inject import get_layer, hooked, make_inject_hook
from mxf.prompts import build_prompt_ids

BOS = 248044
ACTS = "/data/acts27b"
OUT = "/data/mlp42"
SAE_PT = "/data/sae/ae.pt"
MAXACTS_PT = "/data/sae/maxacts.pt"
BSF_DIR = "/data/bsf27b_1b"
BANK = "/data/banks/everything"
TOPK = 32                    # max-activating contexts kept per neuron
HIST_LO, HIST_BPD, HIST_DEC = -4.0, 16, 7   # log10|a| histogram: 16 bins/decade over [1e-4, 1e3) + under/overflow
NB = 2 + HIST_BPD * HIST_DEC
NORM_FILTER_MULT = 10.0      # scoring: drop re-encoded tokens with norm > 10x batch median (== eval_universal)
T0 = time.time()


def log(msg):
    print(f"[mlp42 +{time.time() - T0:6.0f}s] {msg}", flush=True)


class _Stop(Exception):
    pass


def hist_edges():
    """Bin b in [1, NB-2] covers |a| in [10^(LO + (b-1)/BPD), 10^(LO + b/BPD)); bin 0 = |a| < 1e-4; bin NB-1 = >= 1e3."""
    return 10.0 ** (HIST_LO + np.arange(NB - 1) / HIST_BPD)


def load_windows(n_windows, win_len, seed):
    meta = json.load(open(f"{ACTS}/meta.json"))
    NS, T = int(meta["n_seq"]), int(meta["seq_len"])
    n_train = int(math.ceil(NS * 0.95))
    toks = np.memmap(f"{ACTS}/toks.i32", np.int32, "r", shape=(NS, T))
    rng = np.random.default_rng(seed)
    sel = np.sort(rng.choice(n_train, n_windows, replace=False))
    ids = np.ascontiguousarray(toks[sel, :win_len]).astype(np.int32)
    return sel, ids, n_train


@torch.no_grad()
def forward_capture(model, ids, layer, mlp):
    """One forward through layers 0..READ_LAYER. Returns (a [B,T,d_ff] bf16 = down_proj input, h [B,T,d] resid_post)."""
    cap = {}

    def pre(_m, inp):
        cap["a"] = inp[0]

    def post(_m, _i, out):
        cap["h"] = out[0] if isinstance(out, tuple) else out
        raise _Stop

    h1 = mlp.down_proj.register_forward_pre_hook(pre)
    h2 = layer.register_forward_hook(post)
    try:
        model(input_ids=ids, attention_mask=torch.ones_like(ids))
    except _Stop:
        pass
    finally:
        h1.remove(); h2.remove()
    return cap["a"], cap["h"]


# ----------------------------------------------------------------------------------------------------------------
# stage: stats
# ----------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def run_stats(model, tok, sae, n_windows=4000, win_len=256, batch=16, seed=0, dev="cuda:0", tmp="/root/mlp42_tmp"):
    os.makedirs(OUT, exist_ok=True); os.makedirs(tmp, exist_ok=True)
    layer = get_layer(model, READ_LAYER)
    mlp = layer.mlp
    W = mlp.down_proj.weight                       # [d, d_ff]  (nn.Linear stores [out, in])
    N = W.shape[1]
    assert W.shape[0] == D_MODEL, W.shape
    wcols = W.T.float().contiguous()               # [N, d]  neuron i writes a_i * wcols[i]
    wnorm = wcols.norm(dim=1)
    wunit = wcols / wnorm[:, None]
    mu = torch.from_numpy(np.load(f"{ACTS}/whiten_mu.npy").astype(np.float32)).to(dev)
    sel, ids_np, n_train = load_windows(n_windows, win_len, seed)
    n_tok = n_windows * win_len
    log(f"stats: N={N} d_ff neurons, {n_windows} windows x {win_len} tokens = {n_tok} tokens (train rows < {n_train}), "
        f"gate/up {tuple(mlp.gate_proj.weight.shape)} down {tuple(W.shape)} act={type(mlp.act_fn).__name__}")

    # accumulators
    sum_a = torch.zeros(N, dtype=torch.float64, device=dev); sum_a2 = torch.zeros_like(sum_a)
    n_zero = torch.zeros(N, dtype=torch.int64, device=dev)
    max_pos = torch.full((N,), -float("inf"), device=dev); min_neg = torch.full((N,), float("inf"), device=dev)
    hist_pos = torch.zeros(NB * N, dtype=torch.int64, device=dev); hist_neg = torch.zeros_like(hist_pos)
    topv = torch.zeros(TOPK, N, device=dev); topi = torch.full((TOPK, N), -1, dtype=torch.int64, device=dev)
    ratio_max = torch.zeros(N, device=dev)
    cos_max_c = torch.full((N,), -1.0, device=dev); cos_arg_c = torch.zeros(N, dtype=torch.int64, device=dev)
    cos_max_u = torch.full((N,), -1.0, device=dev); cos_arg_u = torch.zeros(N, dtype=torch.int64, device=dev)
    hnorm_all = np.zeros(n_tok, np.float32)
    Fd = sae.d_sae
    sum_f = torch.zeros(Fd, dtype=torch.float64, device=dev); sum_f2 = torch.zeros_like(sum_f)
    sum_af = torch.zeros(N, Fd, dtype=torch.float32, device=dev)          # [N, F] co-activation (9.1 GB)
    nfire_f = torch.zeros(Fd, dtype=torch.int64, device=dev)
    hstore = np.memmap(f"{tmp}/h.f16", np.float16, "w+", shape=(n_tok, D_MODEL))
    ar = torch.arange(N, device=dev)
    bos_col = np.full((batch, 1), BOS, np.int32)
    t_loop = time.time()
    for b0 in range(0, n_windows, batch):
        b1 = min(b0 + batch, n_windows); B = b1 - b0
        ids = torch.from_numpy(np.concatenate([bos_col[:B], ids_np[b0:b1]], 1).astype(np.int64)).to(dev)
        a_bf, h_bf = forward_capture(model, ids, layer, mlp)
        a = a_bf[:, 1:].reshape(-1, N).float()                              # drop BOS/sink position
        h = h_bf[:, 1:].reshape(-1, D_MODEL).float()
        g0 = b0 * win_len                                                    # global token index of row 0
        hstore[g0:g0 + a.shape[0]] = h.to(torch.float16).cpu().numpy()
        hn = h.norm(dim=1)
        hnorm_all[g0:g0 + a.shape[0]] = hn.cpu().numpy()
        # moments / extremes
        sum_a += a.sum(0, dtype=torch.float64); sum_a2 += (a * a).sum(0, dtype=torch.float64)
        n_zero += (a == 0).sum(0)
        max_pos = torch.maximum(max_pos, a.max(0).values); min_neg = torch.minimum(min_neg, a.min(0).values)
        absa = a.abs()
        # log-histogram of |a|, sign-resolved
        bins = ((torch.log10(absa.clamp_min(1e-30)) - HIST_LO) * HIST_BPD).floor().long().add_(1).clamp_(0, NB - 1)
        flat = bins * N + ar
        hist_pos += torch.bincount(flat[a > 0], minlength=NB * N)
        hist_neg += torch.bincount(flat[a < 0], minlength=NB * N)
        del bins, flat
        # running top-k |a| contexts (signed values kept)
        v, i = absa.topk(TOPK, dim=0)
        cv = torch.cat([topv, a.gather(0, i)]); ci = torch.cat([topi, i + g0])
        keep = cv.abs().topk(TOPK, dim=0).indices
        topv = cv.gather(0, keep); topi = ci.gather(0, keep)
        # neuron write norm as a fraction of the residual norm at the same token
        ratio_max = torch.maximum(ratio_max, (absa * wnorm[None, :] / hn[:, None]).max(0).values)
        # presence of the neuron's direction in the real residual (centered + uncentered), max over tokens
        for hh, cm, ca in ((F.normalize(h - mu, dim=1), cos_max_c, cos_arg_c), (F.normalize(h, dim=1), cos_max_u, cos_arg_u)):
            c = hh @ wunit.T                                                 # [tokens, N]
            v2, i2 = c.max(0)
            upd = v2 > cm
            cm[upd] = v2[upd]; ca[upd] = i2[upd] + g0
        # SAE co-activation (pre-topk post-ReLU encoder acts, the suite's scoring convention)
        f = torch.relu((h - sae.b_dec) @ sae.W_enc + sae.b_enc)             # [tokens, F]
        sum_f += f.sum(0, dtype=torch.float64); sum_f2 += (f * f).sum(0, dtype=torch.float64)
        nfire_f += (f > 0).sum(0)
        sum_af.addmm_(a.T, f)
        del f, a, h, absa, a_bf, h_bf
        if (b0 // batch) % 25 == 0:
            el = time.time() - t_loop
            log(f"stats {b1}/{n_windows} windows ({b1 * win_len / max(el, 1e-6):.0f} tok/s)")
    hstore.flush()
    tokens_per_s = n_tok / (time.time() - t_loop)
    hp = hist_pos.view(NB, N); hng = hist_neg.view(NB, N)
    mean = (sum_a / n_tok); var = (sum_a2 / n_tok - mean * mean).clamp_min(0)
    max_abs = torch.maximum(max_pos, -min_neg)
    polarity = torch.where(max_pos >= -min_neg, torch.ones(N, device=dev), -torch.ones(N, device=dev))
    hn_t = torch.from_numpy(hnorm_all)
    log(f"stats pass done: {tokens_per_s:.0f} tok/s | resid norm median {hn_t.median():.1f} mean {hn_t.mean():.1f} | "
        f"max|a| median {max_abs.median():.3f} p90 {max_abs.quantile(0.9):.3f} max {max_abs.max():.3f} | "
        f"||w|| median {wnorm.median():.4f} | ratio_max median {ratio_max.median():.4f}")

    # ---- SAE matches: (1) direction cosine NN (decoder rows, encoder cols), (2) co-activation Pearson correlation
    Wd = F.normalize(sae.W_dec, dim=1)                                       # [F, d] unit rows
    We = F.normalize(sae.W_enc.T, dim=1)                                     # [F, d] unit encoder dirs
    signed = wunit * polarity[:, None]                                       # direction written when the neuron fires
    nn_dec_cos = torch.zeros(N, device=dev); nn_dec_id = torch.zeros(N, dtype=torch.int64, device=dev)
    nn_enc_cos = torch.zeros(N, device=dev); nn_enc_id = torch.zeros(N, dtype=torch.int64, device=dev)
    nn_dec_abs = torch.zeros(N, device=dev); nn_enc_abs = torch.zeros(N, device=dev)
    for c0 in range(0, N, 1024):
        s = signed[c0:c0 + 1024]
        cd = s @ Wd.T; ce = s @ We.T                                        # [c, F] signed cos
        v, i = cd.max(1); nn_dec_cos[c0:c0 + 1024] = v; nn_dec_id[c0:c0 + 1024] = i
        v, i = ce.max(1); nn_enc_cos[c0:c0 + 1024] = v; nn_enc_id[c0:c0 + 1024] = i
        nn_dec_abs[c0:c0 + 1024] = cd.abs().max(1).values; nn_enc_abs[c0:c0 + 1024] = ce.abs().max(1).values
        del cd, ce
    # controls for the SAE NN test: random unit dirs, and the SAE's own enc<->dec pairing (same feature)
    g = torch.Generator(device=dev).manual_seed(seed)
    R = F.normalize(torch.randn(2048, D_MODEL, generator=g, device=dev), dim=1)
    rand_dec_abs = torch.cat([(R[c0:c0 + 512] @ Wd.T).abs().max(1).values for c0 in range(0, 2048, 512)])
    rand_enc_abs = torch.cat([(R[c0:c0 + 512] @ We.T).abs().max(1).values for c0 in range(0, 2048, 512)])
    enc_dec_pair = (Wd * We).sum(1)                                          # [F] cos(dec_f, enc_f)
    # nearest OTHER decoder row for a random sample of features (how close are SAE features to each other)
    fs = torch.from_numpy(np.random.default_rng(seed).choice(Fd, 2048, replace=False)).to(dev)
    cdd = Wd[fs] @ Wd.T; cdd[torch.arange(2048, device=dev), fs] = 0.0
    sae_self_nn_abs = cdd.abs().max(1).values
    del cdd
    # Pearson correlation neuron-activation x feature-activation over all tokens
    mean_f = sum_f / n_tok; var_f = (sum_f2 / n_tok - mean_f * mean_f).clamp_min(0)
    std_a = var.sqrt().float(); std_f = var_f.sqrt().float()
    corr_top_v = torch.full((N, 5), -2.0, device=dev); corr_top_i = torch.zeros(N, 5, dtype=torch.int64, device=dev)
    corr_at_nn_dec = torch.zeros(N, device=dev); corr_at_nn_enc = torch.zeros(N, device=dev)
    fchunk = 8192
    for f0 in range(0, Fd, fchunk):
        f1 = min(f0 + fchunk, Fd)
        cov = sum_af[:, f0:f1] / n_tok - mean.float()[:, None] * mean_f.float()[None, f0:f1]
        corr = cov / (std_a[:, None] * std_f[None, f0:f1] + 1e-12)
        corr[:, std_f[f0:f1] == 0] = 0.0
        corr = corr * polarity[:, None]                                      # sign relative to the firing polarity
        cv = torch.cat([corr_top_v, corr], 1)
        ci = torch.cat([corr_top_i, torch.arange(f0, f1, device=dev)[None, :].expand(N, -1)], 1)
        k = cv.topk(5, dim=1).indices
        corr_top_v = cv.gather(1, k); corr_top_i = ci.gather(1, k)
        m = (nn_dec_id >= f0) & (nn_dec_id < f1)
        corr_at_nn_dec[m] = corr[m, nn_dec_id[m] - f0]
        m = (nn_enc_id >= f0) & (nn_enc_id < f1)
        corr_at_nn_enc[m] = corr[m, nn_enc_id[m] - f0]
        del cov, corr, cv, ci
    del sum_af
    torch.cuda.empty_cache()
    log(f"SAE match: NN dec |cos| median {nn_dec_abs.median():.3f} p90 {nn_dec_abs.quantile(0.9):.3f} max {nn_dec_abs.max():.3f} "
        f"| random ctrl median {rand_dec_abs.median():.3f} p90 {rand_dec_abs.quantile(0.9):.3f} "
        f"| enc-dec same-feature cos median {enc_dec_pair.median():.3f} | top corr median {corr_top_v[:, 0].median():.3f} "
        f"p90 {corr_top_v[:, 0].quantile(0.9):.3f}")

    # ---- at each neuron's peak token: is the neuron's write a large share of the residual? which SAE features are on?
    peak_g = topi[0].cpu().numpy()
    hp_rows = torch.from_numpy(np.asarray(hstore[np.sort(peak_g)]).astype(np.float32)).to(dev)
    order = np.argsort(peak_g); inv = np.empty_like(order); inv[order] = np.arange(N)
    hp_rows = hp_rows[torch.from_numpy(inv).to(dev)]                         # back in neuron order
    peak_a = topv[0]
    peak_cos_c = F.cosine_similarity(hp_rows - mu, signed, dim=1)
    peak_cos_u = F.cosine_similarity(hp_rows, signed, dim=1)
    peak_frac = peak_a.abs() * wnorm / hp_rows.norm(dim=1)
    fpk = torch.relu((hp_rows - sae.b_dec) @ sae.W_enc + sae.b_enc)         # [N, F]
    pk_v, pk_i = fpk.topk(5, dim=1)
    nn_dec_act_at_peak = fpk.gather(1, nn_dec_id[:, None])[:, 0]
    nn_dec_rank_at_peak = (fpk > nn_dec_act_at_peak[:, None]).sum(1) + 1
    del fpk
    log(f"peak: cos(h-mu, dir) median {peak_cos_c.median():.3f} p90 {peak_cos_c.quantile(0.9):.3f} | write/||h|| median "
        f"{peak_frac.median():.3f} p90 {peak_frac.quantile(0.9):.3f} max {peak_frac.max():.3f} | NN-dec feature rank at peak "
        f"median {nn_dec_rank_at_peak.float().median():.0f}, rank1 frac {(nn_dec_rank_at_peak == 1).float().mean():.3f}")

    edges = hist_edges()
    np.savez(f"{OUT}/neuron_stats.npz",
             n_tok=n_tok, n_windows=n_windows, win_len=win_len, seed=seed, sel_windows=sel, N=N,
             sum_a=sum_a.cpu().numpy(), sum_a2=sum_a2.cpu().numpy(), mean=mean.cpu().numpy(), std=var.sqrt().cpu().numpy(),
             n_zero=n_zero.cpu().numpy(), max_pos=max_pos.cpu().numpy(), min_neg=min_neg.cpu().numpy(),
             max_abs=max_abs.cpu().numpy(), polarity=polarity.cpu().numpy(),
             hist_pos=hp.cpu().numpy(), hist_neg=hng.cpu().numpy(), hist_edges=edges, hist_lo=HIST_LO, hist_bpd=HIST_BPD,
             wnorm=wnorm.cpu().numpy(), ratio_max=ratio_max.cpu().numpy(),
             cos_max_c=cos_max_c.cpu().numpy(), cos_arg_c=cos_arg_c.cpu().numpy(),
             cos_max_u=cos_max_u.cpu().numpy(), cos_arg_u=cos_arg_u.cpu().numpy(),
             topv=topv.cpu().numpy(), topi=topi.cpu().numpy(),
             resid_norm_mean=float(hn_t.mean()), resid_norm_median=float(hn_t.median()),
             tokens_per_s=tokens_per_s)
    np.savez(f"{OUT}/sel_windows.npz", sel_windows=sel, ids=ids_np, bos=BOS, n_train=n_train)
    np.save(f"{OUT}/hnorm.f32.npy", hnorm_all)
    wcols.to(torch.float16).cpu().numpy().tofile(f"{OUT}/down_proj_cols.f16")
    np.savez(f"{OUT}/sae_match.npz",
             nn_dec_cos=nn_dec_cos.cpu().numpy(), nn_dec_id=nn_dec_id.cpu().numpy(), nn_dec_abs=nn_dec_abs.cpu().numpy(),
             nn_enc_cos=nn_enc_cos.cpu().numpy(), nn_enc_id=nn_enc_id.cpu().numpy(), nn_enc_abs=nn_enc_abs.cpu().numpy(),
             rand_dec_abs=rand_dec_abs.cpu().numpy(), rand_enc_abs=rand_enc_abs.cpu().numpy(),
             enc_dec_pair=enc_dec_pair.cpu().numpy(), sae_self_nn_abs=sae_self_nn_abs.cpu().numpy(), sae_self_ids=fs.cpu().numpy(),
             corr_top_v=corr_top_v.cpu().numpy(), corr_top_i=corr_top_i.cpu().numpy(),
             corr_at_nn_dec=corr_at_nn_dec.cpu().numpy(), corr_at_nn_enc=corr_at_nn_enc.cpu().numpy(),
             sae_mean=mean_f.cpu().numpy(), sae_std=std_f.cpu().numpy(), sae_nfire=nfire_f.cpu().numpy(),
             d_sae=Fd)
    np.savez(f"{OUT}/peak_context.npz",
             peak_g=peak_g, peak_a=peak_a.cpu().numpy(), peak_cos_c=peak_cos_c.cpu().numpy(), peak_cos_u=peak_cos_u.cpu().numpy(),
             peak_frac=peak_frac.cpu().numpy(), peak_sae_top_v=pk_v.cpu().numpy(), peak_sae_top_i=pk_i.cpu().numpy(),
             nn_dec_act_at_peak=nn_dec_act_at_peak.cpu().numpy(), nn_dec_rank_at_peak=nn_dec_rank_at_peak.cpu().numpy())
    meta = {"model": "Qwen/Qwen3.6-27B", "layer": READ_LAYER, "d_model": D_MODEL, "d_ff": int(N),
            "mlp_modules": {"gate_proj": list(mlp.gate_proj.weight.shape), "up_proj": list(mlp.up_proj.weight.shape),
                            "down_proj": list(W.shape), "act_fn": type(mlp.act_fn).__name__},
            "neuron_value": "a_i = act_fn(gate_proj(x))_i * up_proj(x)_i = input i of down_proj (signed)",
            "neuron_direction": "down_proj.weight[:, i] (column i); polarity = sign of the extreme corpus value",
            "windows": {"n": n_windows, "len": win_len, "source": f"{ACTS}/toks.i32 train rows < {n_train}", "seed": seed,
                        "bos_prepended": BOS, "position0_dropped": True},
            "n_tokens": n_tok, "tokens_per_s": tokens_per_s,
            "hist": {"kind": "log10|a| sign-resolved", "lo": HIST_LO, "bins_per_decade": HIST_BPD, "n_bins": NB},
            "resid_norm": {"mean": float(hn_t.mean()), "median": float(hn_t.median())},
            "files": ["neuron_stats.npz", "down_proj_cols.f16 (fp16 [d_ff, d_model] = down_proj.weight.T)", "sel_windows.npz",
                      "hnorm.f32.npy", "sae_match.npz", "peak_context.npz"],
            "created": time.time()}
    json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=1)
    log("stats stage saved")
    return {"wunit": wunit, "signed": signed, "polarity": polarity, "R": R, "Wd_sample": Wd[fs], "sample_feats": fs}


# ----------------------------------------------------------------------------------------------------------------
# stage: dirs (BSF block subspaces + cluster probes; SAE handled in stats)
# ----------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def run_dirs(signed, R, Wd_sample, dev="cuda:0"):
    """signed [N, d] polarity-signed unit neuron dirs; R random unit controls; Wd_sample SAE decoder dirs (feature ref)."""
    N = signed.shape[0]
    out = {}
    # ---- BSF: fraction of a direction's (whitened) energy inside its best block subspace
    sasa_meta = json.load(open(f"{BSF_DIR}/meta.json"))
    Q = torch.load(f"{BSF_DIR}/blocks_Q.pt", map_location="cpu", weights_only=False)["Q"].float()     # [G, b, d]
    G, b, d = Q.shape
    assert d == D_MODEL
    Qf = Q.view(G * b, d).to(dev)
    zca = torch.from_numpy(np.load(f"{BSF_DIR}/whiten_zca.npy").astype(np.float32)).to(dev)
    log(f"BSF: G={G} b={b} k={sasa_meta.get('k')} EV={sasa_meta.get('ev_final_mean20')}")

    def bsf_frac(X, Qf):
        y = F.normalize(X @ zca, dim=1)                                    # whitened-space direction (difference vector)
        fr, gi = [], []
        for c0 in range(0, X.shape[0], 512):
            z = (Qf @ y[c0:c0 + 512].T).view(G, b, -1)                      # [G, b, c]
            e = (z * z).sum(1)                                               # [G, c] energy in each block subspace
            v, i = e.max(0)
            fr.append(v); gi.append(i)
        return torch.cat(fr), torch.cat(gi)

    for name, X in (("neuron", signed), ("random", R), ("sae_dec", Wd_sample)):
        fr, gi = bsf_frac(X, Qf)
        out[f"bsf_frac_{name}"] = fr.cpu().numpy(); out[f"bsf_block_{name}"] = gi.cpu().numpy()
        log(f"BSF top-block energy fraction [{name}]: median {fr.median():.3f} p90 {fr.quantile(0.9):.3f} max {fr.max():.3f}")
    # chance level for a b-dim subspace of a d-dim space: b/d, and max over G random subspaces ~ larger
    out["bsf_chance_single"] = b / d
    del Qf, Q
    torch.cuda.empty_cache()

    # ---- cluster probes: nearest-neighbour cosine to the 100k probe directions of the training bank
    rows = []
    with open(f"{BANK}/records.jsonl") as fh:
        for i, line in enumerate(fh):
            if '"family": "cluster"' in line:
                r = json.loads(line)
                assert r["family"] == "cluster" and int(r["vec_idx"]) == i
                rows.append(i)
    rows = np.array(rows, np.int64)
    sz = os.path.getsize(f"{BANK}/vecs.f32")
    V = np.memmap(f"{BANK}/vecs.f32", np.float32, "r", shape=(sz // (4 * D_MODEL), D_MODEL))
    P = F.normalize(torch.from_numpy(np.asarray(V[rows]).astype(np.float32)).to(dev), dim=1)   # [100k, d]
    log(f"probes: {len(rows)} cluster rows loaded")
    for name, X in (("neuron", signed), ("random", R), ("sae_dec", Wd_sample)):
        mx, mi = [], []
        for c0 in range(0, X.shape[0], 1024):
            c = X[c0:c0 + 1024] @ P.T
            v, i = c.abs().max(1); mx.append(v); mi.append(i)
        mx = torch.cat(mx); mi = torch.cat(mi)
        out[f"probe_nn_abs_{name}"] = mx.cpu().numpy(); out[f"probe_nn_row_{name}"] = rows[mi.cpu().numpy()]
        log(f"probe NN |cos| [{name}]: median {mx.median():.3f} p90 {mx.quantile(0.9):.3f} max {mx.max():.3f}")
    del P
    # ---- neuron-neuron nearest neighbour (off-diagonal)
    C = signed @ signed.T
    C.fill_diagonal_(0.0)
    v, i = C.abs().max(1)
    out["neuron_nn_abs"] = v.cpu().numpy(); out["neuron_nn_id"] = i.cpu().numpy()
    Cr = R @ R.T; Cr.fill_diagonal_(0.0)
    out["random_nn_abs_within2048"] = Cr.abs().max(1).values.cpu().numpy()
    log(f"neuron-neuron NN |cos|: median {v.median():.3f} p90 {v.quantile(0.9):.3f} max {v.max():.3f} (N={N})")
    del C, Cr
    np.savez(f"{OUT}/dirs_analysis.npz", **out)
    log("dirs stage saved")


# ----------------------------------------------------------------------------------------------------------------
# stage: verbalize (inverter generations, clean-base scoring incl. neuron fire-back)
# ----------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def reencode(texts, actor, tok, dev, layer, mlp, neuron_ids, sbatch=32):
    """Clean-base re-encode (== eval_universal._reencode: right pad, BOS sink, 10x-median norm filter) that ALSO returns
    the layer-42 MLP neuron values of each row's paired neuron(s). neuron_ids: list of int (k=1) or list of k-lists.
    Yields (s, h [b,T,d], keep [b,T], a_sel [b,k,T], last5 [b,T])."""
    prev = tok.padding_side; tok.padding_side = "right"
    sink = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id   # == eval_universal._reencode
    try:
        for s in range(0, len(texts), sbatch):
            batch = [t if t.strip() else " " for t in texts[s:s + sbatch]]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=95, add_special_tokens=False).to(dev)
            B = enc["input_ids"].shape[0]
            ids = torch.cat([torch.full((B, 1), sink, device=dev, dtype=enc["input_ids"].dtype), enc["input_ids"]], 1)
            am = torch.cat([torch.ones((B, 1), device=dev, dtype=enc["attention_mask"].dtype), enc["attention_mask"]], 1)
            cap = {}

            def pre(_m, inp):
                cap["a"] = inp[0]

            def post(_m, _i, out):
                cap["h"] = (out[0] if isinstance(out, tuple) else out).float()
                raise _Stop

            h1 = mlp.down_proj.register_forward_pre_hook(pre); h2 = layer.register_forward_hook(post)
            try:
                with actor.disable_adapter():
                    actor(input_ids=ids, attention_mask=am)
            except _Stop:
                pass
            finally:
                h1.remove(); h2.remove()
            h = cap["h"]
            nid = torch.as_tensor(neuron_ids[s:s + B], device=dev, dtype=torch.long)          # [b, k] neurons per row
            if nid.dim() == 1:
                nid = nid[:, None]
            a_sel = cap["a"].float()[torch.arange(B, device=dev)[:, None], :, nid]           # [b, k, T]
            keep = am.bool().clone(); keep[:, 0] = False
            nrm = h.norm(dim=-1)
            med = nrm.masked_fill(~keep, float("nan")).nanmedian(dim=1, keepdim=True).values
            keep = keep & (nrm <= NORM_FILTER_MULT * med)
            L = am.sum(1)                                                       # incl. sink
            pos = torch.arange(h.shape[1], device=dev)[None, :]
            last5 = keep & (pos >= (L - 5)[:, None])
            yield s, h, keep, a_sel, last5
    finally:
        tok.padding_side = prev


@torch.no_grad()
def run_verbalize(base, tok, adapter, sets, sae, tag, dev="cuda:0", bo=4, temp=1.0, max_new=48, min_new=16, gen_chunk=128,
                  gen_seed=1234):
    """sets: list of dicts {name, ids (list[int]), dirs [n, d] unit tensor, neuron (list[int] or None), sae_feat (list or None),
    ref_max (list[float] corpus peak per row or None)}. Writes /data/mlp42/verbalize_<tag>.jsonl + summary json."""
    from peft import PeftModel
    mu = torch.from_numpy(np.load(f"{ACTS}/whiten_mu.npy").astype(np.float32)).to(dev)
    actor = PeftModel.from_pretrained(base, adapter, is_trainable=False); actor.eval()
    sub = get_layer(actor, INJECT_LAYER)
    layer = get_layer(actor, READ_LAYER); mlp = layer.mlp
    prompt_ids, mpos = build_prompt_ids(tok); marker = mpos[0]
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/verbalize_{tag}.jsonl"
    fh = open(path, "w")
    summary = {}
    for S in sets:
        name, dirs = S["name"], S["dirs"].to(dev).float()
        n = dirs.shape[0]
        fb = max(1, gen_chunk // bo)
        recs = []
        t0 = time.time()
        with torch.random.fork_rng(devices=[dev]):
            torch.manual_seed(gen_seed)
            for s in range(0, n, fb):
                rows = [i for i in range(s, min(s + fb, n)) for _ in range(bo)]
                vecs = [dirs[i:i + 1] for i in rows]
                hook = make_inject_hook(vecs, [[marker]] * len(rows), STEER_COEFF, dev, torch.bfloat16, mode="add")
                ids = torch.tensor([list(prompt_ids)] * len(rows), device=dev)
                with hooked(sub, hook):
                    gen = actor.generate(ids, do_sample=True, temperature=temp, top_p=1.0, top_k=0, min_p=0.0,
                                         max_new_tokens=max_new, min_new_tokens=min_new, pad_token_id=tok.pad_token_id)
                texts = tok.batch_decode(gen[:, len(prompt_ids):], skip_special_tokens=True)
                rdirs = dirs[rows]
                nids = [S["neuron"][i] if S.get("neuron") is not None else 0 for i in rows]
                nids = [n if isinstance(n, (list, tuple)) else [n] for n in nids]           # k members per row
                for s2, h, keep, a_sel, last5 in reencode(texts, actor, tok, dev, layer, mlp, nids):
                    b = h.shape[0]
                    d = rdirs[s2:s2 + b]
                    hn = F.normalize(h, dim=-1); hc = F.normalize(h - mu, dim=-1)
                    cos_u = torch.einsum("btd,bd->bt", hn, d); cos_c = torch.einsum("btd,bd->bt", hc, d)
                    cu_all = cos_u.masked_fill(~keep, -1.0).max(1).values
                    cu_l5 = cos_u.masked_fill(~last5, -1.0).max(1).values
                    cc_all = cos_c.masked_fill(~keep, -1.0).max(1).values
                    cc_l5 = cos_c.masked_fill(~last5, -1.0).max(1).values
                    n_keep = keep.sum(1)
                    if S.get("neuron") is not None:
                        def _k(v):
                            return v if isinstance(v, (list, tuple)) else [v]
                        pol = torch.tensor([_k(S["polarity"][i]) for i in rows[s2:s2 + b]], device=dev, dtype=torch.float32)   # [b,k]
                        rmx = torch.tensor([_k(S["ref_max"][i]) for i in rows[s2:s2 + b]], device=dev, dtype=torch.float32)    # [b,k]
                        av_k = (a_sel * pol[:, :, None]).masked_fill(~keep[:, None, :], -float("inf")).max(2).values          # [b,k]
                        av5_k = (a_sel * pol[:, :, None]).masked_fill(~last5[:, None, :], -float("inf")).max(2).values
                        na_k = av_k / rmx.clamp_min(1e-6)
                        av, av_l5 = av_k[:, 0], av5_k[:, 0]
                    else:
                        av = av_l5 = torch.full((b,), float("nan"), device=dev)
                    if S.get("sae_feat") is not None:
                        feats = [S["sae_feat"][i] for i in rows[s2:s2 + b]]
                        per = sae.encode_features(h, feats)
                        bi = torch.arange(b, device=dev)
                        fa = per[bi, :, bi].masked_fill(~keep, -1.0).max(1).values.clamp(min=0.0)
                    else:
                        fa = torch.full((b,), float("nan"), device=dev)
                    for j in range(b):
                        i = rows[s2 + j]
                        r = {"set": name, "row": i, "id": int(S["ids"][i]), "sample": (s2 + j) % bo, "text": texts[s2 + j],
                             "n_tok": int(n_keep[j]), "cos_all": float(cu_all[j]), "cos_last5": float(cu_l5[j]),
                             "cosc_all": float(cc_all[j]), "cosc_last5": float(cc_l5[j])}
                        if S.get("neuron") is not None:
                            k_members = _k(S["neuron"][i])
                            na_row = na_k[j].tolist()
                            # norm_act = the WEAKEST member's fire-back (k=1: the neuron itself); *_max = the strongest member
                            r.update({"members": [int(m) for m in k_members], "neuron_act": float(av[j]), "neuron_act_last5": float(av_l5[j]),
                                      "corpus_max": _k(S["ref_max"][i])[0], "norm_act_members": na_row,
                                      "norm_act": float(min(na_row)), "norm_act_max": float(max(na_row))})
                        if S.get("sae_feat") is not None:
                            rm = float(S["ref_max"][i])
                            r.update({"sae_act": float(fa[j]), "corpus_peak": rm, "norm_act": float(fa[j]) / rm if rm > 0 else float("nan")})
                        recs.append(r); fh.write(json.dumps(r) + "\n")
                fh.flush()
                log(f"verbalize[{name}] {min(s + fb, n)}/{n} dirs ({time.time() - t0:.0f}s)")
        # best-of-bo aggregates
        by = {}
        for r in recs:
            by.setdefault(r["row"], []).append(r)
        def best(key):
            return float(np.mean([max(x[key] for x in v) for v in by.values()]))
        summ = {"n_dirs": n, "bo": bo, "cos_all_bo": best("cos_all"), "cos_last5_bo": best("cos_last5"),
                "cosc_all_bo": best("cosc_all"), "cosc_last5_bo": best("cosc_last5"),
                "cos_all_mean1": float(np.mean([r["cos_all"] for r in recs])),
                "cos_last5_mean1": float(np.mean([r["cos_last5"] for r in recs]))}
        if "norm_act" in recs[0]:
            na = np.array([max(x["norm_act"] for x in v) for v in by.values()], np.float64)
            summ.update({"norm_act_bo": float(np.nanmean(na)), "fired10_bo": float(np.nanmean(na > 0.10)),
                         "fired25_bo": float(np.nanmean(na > 0.25)), "fired50_bo": float(np.nanmean(na > 0.50)),
                         "beat_corpus_bo": float(np.nanmean(na > 1.0)),
                         "norm_act_mean1": float(np.nanmean([r["norm_act"] for r in recs]))})
            if len(recs[0].get("members", [0])) > 1:
                nx = np.array([max(x["norm_act_max"] for x in v) for v in by.values()], np.float64)
                summ.update({"k": len(recs[0]["members"]), "any_fired10_bo": float(np.nanmean(nx > 0.10)),
                             "any_fired25_bo": float(np.nanmean(nx > 0.25)), "any_fired50_bo": float(np.nanmean(nx > 0.50))})
        if "sae_act" in recs[0]:
            sa = np.array([max(x["sae_act"] for x in v) for v in by.values()], np.float64)
            summ["sae_fired_bo"] = float(np.mean(sa > 1.0))
        summary[name] = summ
        log(f"verbalize[{name}] summary: {json.dumps(summ)}")
    fh.close()
    json.dump({"tag": tag, "adapter": adapter, "bo": bo, "temp": temp, "max_new": max_new, "min_new": min_new,
               "gen_seed": gen_seed, "sets": summary}, open(f"{OUT}/verbalize_{tag}.json", "w"), indent=1)
    log(f"verbalize saved -> {path}")
    return summary
