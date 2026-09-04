#!/usr/bin/env python3
"""Layer-42 MLP neurons of Qwen3.6-27B: local analysis + every figure/number for the report.

Reads the /data/mlp42 dump written by data/modal_mlp42_neurons.py (download with
    modal volume get maemm-data /mlp42 <raw>/mlp42
) and writes to the report folder:
    data/*.json          exact numbers behind every figure and table (replot-ready)
    <figure>.png + .pdf  every figure in both formats
    data/neuron_sets.json  the sparse / dense neuron id lists (input to the verbalize Modal stage)

Usage:
    python scripts/mlp42_neurons_analyze.py --raw ~/shared/reports/maemm-mlp-neurons/raw/mlp42 \
        --report ~/shared/reports/maemm-mlp-neurons [--verbalize <raw>/mlp42/verbalize_rlA.jsonl]

Sparsity definition (see --rel-thr / --sparse-max / --dense-min): a neuron "fires" at a token when |a| exceeds
rel_thr x its own corpus max |a| (default 10%). Its firing frequency is the fraction of the 1.02M content tokens
where that happens (read off the 16-bins/decade log-histogram of |a|, log-uniform interpolation inside the
straddling bin). SPARSE = frequency < sparse_max (default 1e-3 = 0.1% of tokens) AND a non-trivial write
(max |a|*||w_down|| in the top 75% of neurons — i.e. we drop the weakest quarter). DENSE = frequency >= dense_min
(default 3e-2). Both thresholds are reported alongside the full frequency distribution, so the reader can
re-cut.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# palette (dataviz skill reference instance): categorical slots in fixed order + chrome
C_SPARSE, C_DENSE, C_SAE, C_ALL = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
C_RANDOM, C_GRID, C_MUTED, C_INK = "#898781", "#e1e0d9", "#898781", "#0b0b0b"
D_MODEL = 5120
Q = np.linspace(0.005, 0.995, 199)          # quantile grid for ECDF-style curves (stored in data/*.json)


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_GRID)
    ax.tick_params(colors=C_MUTED, labelsize=8)
    ax.yaxis.label.set_color(C_INK); ax.xaxis.label.set_color(C_INK)
    ax.grid(True, color=C_GRID, lw=0.6); ax.set_axisbelow(True)


def save(fig, report, stem):
    fig.savefig(report / f"{stem}.png", dpi=170, bbox_inches="tight")
    fig.savefig(report / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def qcurve(x):
    x = np.asarray(x, np.float64); x = x[np.isfinite(x)]
    return np.quantile(x, Q).tolist() if len(x) else []


def summ(x):
    x = np.asarray(x, np.float64); x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"n": 0}
    return {"n": int(len(x)), "mean": float(x.mean()), "median": float(np.median(x)), "p10": float(np.quantile(x, 0.1)),
            "p90": float(np.quantile(x, 0.9)), "p99": float(np.quantile(x, 0.99)), "max": float(x.max()), "min": float(x.min())}


def freq_above(hist, edges, lo, bpd, thr):
    """hist [NB, N] counts of |a| per log bin (bin 0 underflow, bin NB-1 overflow); thr [N] or scalar.
    Returns per-neuron count of tokens with |a| >= thr (log-uniform interpolation inside the straddling bin)."""
    NB, N = hist.shape
    thr = np.broadcast_to(np.asarray(thr, np.float64), (N,))
    pos = (np.log10(np.maximum(thr, 1e-30)) - lo) * bpd            # continuous bin coordinate: bin b covers [b-1, b)
    b = np.floor(pos).astype(np.int64) + 1
    frac_above = 1.0 - (pos - np.floor(pos))                        # share of bin b above thr (log-uniform)
    b = np.clip(b, 0, NB - 1)
    csum = np.cumsum(hist[::-1], axis=0)[::-1]                      # csum[b] = counts in bins >= b
    above_full = np.where(b + 1 < NB, csum[np.minimum(b + 1, NB - 1), np.arange(N)], 0)
    straddle = hist[b, np.arange(N)] * frac_above
    out = above_full + straddle
    out[thr <= 0] = hist.sum(0)[thr <= 0]
    return out


def decode_ctx(tok, ids_row, pos, left=14, right=6):
    lo, hi = max(0, pos - left), min(len(ids_row), pos + right + 1)
    pre = tok.decode(ids_row[lo:pos].tolist()); pk = tok.decode(ids_row[pos:pos + 1].tolist()); post = tok.decode(ids_row[pos + 1:hi].tolist())
    return (pre + "⟦" + pk + "⟧" + post).replace("\n", "⏎")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--verbalize", default=None, help="verbalize_<tag>.jsonl (optional; adds the verbalization figure/table)")
    ap.add_argument("--rel-thr", type=float, default=0.10)
    ap.add_argument("--sparse-max", type=float, default=1e-3)
    ap.add_argument("--dense-min", type=float, default=3e-2)
    ap.add_argument("--n-sparse", type=int, default=256)
    ap.add_argument("--n-dense", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sae-examples", default=None, help="HF ceselder/qwen36-27b-sae-l42 data/examples.parquet (decoded SAE windows)")
    a = ap.parse_args()
    raw, report = Path(a.raw), Path(a.report)
    (report / "data").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    st = np.load(raw / "neuron_stats.npz"); sm = np.load(raw / "sae_match.npz"); pk = np.load(raw / "peak_context.npz")
    da = np.load(raw / "dirs_analysis.npz"); sw = np.load(raw / "sel_windows.npz")
    meta = json.load(open(raw / "meta.json"))
    N, n_tok = int(st["N"]), int(st["n_tok"])
    hist = (st["hist_pos"] + st["hist_neg"]).astype(np.float64)
    edges, lo, bpd = st["hist_edges"], float(st["hist_lo"]), int(st["hist_bpd"])
    max_abs, wnorm, pol = st["max_abs"].astype(np.float64), st["wnorm"].astype(np.float64), st["polarity"]
    write_max = max_abs * wnorm                                        # max ||a w|| the neuron ever writes
    ratio_max = st["ratio_max"].astype(np.float64)
    resid_med = float(st["resid_norm_median"])
    cols = np.fromfile(raw / "down_proj_cols.f16", np.float16).reshape(N, D_MODEL).astype(np.float32)
    U = cols / np.linalg.norm(cols, axis=1, keepdims=True) * pol[:, None].astype(np.float32)

    # ------------------------------------------------------------------ firing frequency + bands
    freq_rel = freq_above(hist, edges, lo, bpd, a.rel_thr * max_abs) / n_tok
    freq_rel25 = freq_above(hist, edges, lo, bpd, 0.25 * max_abs) / n_tok
    freq_rel50 = freq_above(hist, edges, lo, bpd, 0.5 * max_abs) / n_tok
    # absolute threshold: the |a| value exceeded by 1% of ALL neuron-token pairs (global scale reference)
    tot = hist.sum(1); cum = np.cumsum(tot[::-1])[::-1] / tot.sum()
    b99 = int(np.argmax(cum <= 0.01))
    abs_thr = float(edges[max(b99 - 1, 0)])
    freq_abs = freq_above(hist, edges, lo, bpd, abs_thr) / n_tok
    band_edges = [0, 1e-4, 1e-3, 1e-2, 1e-1, 1.01]
    band_names = ["<0.01%", "0.01–0.1%", "0.1–1%", "1–10%", "≥10%"]
    band_idx = np.digitize(freq_rel, band_edges[1:-1])
    write_floor = float(np.quantile(write_max, 0.25))
    nontrivial = write_max >= write_floor
    sparse_mask = (freq_rel < a.sparse_max) & nontrivial
    dense_mask = freq_rel >= a.dense_min
    sparse_ids = np.flatnonzero(sparse_mask); dense_ids = np.flatnonzero(dense_mask)
    print(f"N={N} tokens={n_tok} | rel_thr={a.rel_thr} | bands {dict(zip(band_names, np.bincount(band_idx, minlength=5).tolist()))}")
    print(f"sparse (freq<{a.sparse_max} & write>=p25 {write_floor:.3g}): {len(sparse_ids)} | dense (freq>={a.dense_min}): {len(dense_ids)}")
    print(f"abs thr (top-1% of all |a|) = {abs_thr:.3g}; max|a| median {np.median(max_abs):.3g}; ||w|| median {np.median(wnorm):.3g}; "
          f"resid median {resid_med:.1f}; ratio_max median {np.median(ratio_max):.3g} p99 {np.quantile(ratio_max, .99):.3g}")
    sel_sparse = np.sort(rng.choice(sparse_ids, min(a.n_sparse, len(sparse_ids)), replace=False))
    sel_dense = np.sort(rng.choice(dense_ids, min(a.n_dense, len(dense_ids)), replace=False))
    json.dump({"rel_thr": a.rel_thr, "sparse_max": a.sparse_max, "dense_min": a.dense_min, "write_floor_p25": write_floor,
               "n_sparse": int(len(sparse_ids)), "n_dense": int(len(dense_ids)),
               "sparse_ids": sparse_ids.tolist(), "dense_ids": dense_ids.tolist(),
               "verbalize_sparse": sel_sparse.tolist(), "verbalize_dense": sel_dense.tolist()},
              open(report / "data" / "neuron_sets.json", "w"))

    bands = {"band_edges": band_edges, "band_names": band_names, "counts": np.bincount(band_idx, minlength=5).tolist(),
             "rel_thr": a.rel_thr, "n_neurons": N, "n_tokens": n_tok,
             "freq_rel_hist": None, "abs_thr_top1pct": abs_thr,
             "freq_rel_summary": summ(freq_rel), "freq_rel25_summary": summ(freq_rel25), "freq_rel50_summary": summ(freq_rel50),
             "freq_abs_summary": summ(freq_abs), "max_abs_summary": summ(max_abs), "write_max_summary": summ(write_max),
             "ratio_max_summary": summ(ratio_max), "wnorm_summary": summ(wnorm), "resid_norm_median": resid_med,
             "n_polarity_neg": int((pol < 0).sum()), "n_zero_neurons": int((max_abs <= 0).sum()),
             "sparse_def": f"freq_rel<{a.sparse_max} & write_max>=p25", "n_sparse": int(len(sparse_ids)),
             "dense_def": f"freq_rel>={a.dense_min}", "n_dense": int(len(dense_ids)),
             "sparse_write_max_summary": summ(write_max[sparse_mask]), "dense_write_max_summary": summ(write_max[dense_mask]),
             "sparse_ratio_max_summary": summ(ratio_max[sparse_mask]), "dense_ratio_max_summary": summ(ratio_max[dense_mask])}
    fl = np.log10(np.clip(freq_rel, 1e-7, 1)); hb = np.linspace(-7, 0, 57)
    hc, _ = np.histogram(fl, bins=hb); bands["freq_rel_hist"] = {"log10_bin_edges": hb.tolist(), "counts": hc.tolist()}
    hc25, _ = np.histogram(np.log10(np.clip(freq_rel25, 1e-7, 1)), bins=hb); bands["freq_rel25_hist"] = hc25.tolist()
    hca, _ = np.histogram(np.log10(np.clip(freq_abs, 1e-7, 1)), bins=hb); bands["freq_abs_hist"] = hca.tolist()
    json.dump(bands, open(report / "data" / "neuron_bands.json", "w"), indent=1)

    # ---- F1 headline: frequency distribution + (frequency vs write share) scatter
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))
    ax = axs[0]
    ax.bar(hb[:-1], hc, width=np.diff(hb), align="edge", color=C_ALL, alpha=0.85, lw=0)
    for e in band_edges[1:-1]:
        ax.axvline(np.log10(e), color=C_MUTED, lw=0.8)
    for i, nm in enumerate(band_names):
        x0 = np.log10(max(band_edges[i], 1e-7)); x1 = np.log10(band_edges[i + 1])
        ax.text((x0 + x1) / 2, ax.get_ylim()[1] * 0.97, f"{nm}\n{bands['counts'][i]:,}", ha="center", va="top", fontsize=8, color=C_INK)
    ax.set_xlabel(f"fraction of tokens where |a| ≥ {a.rel_thr:.0%} of the neuron's own max  (log10)")
    ax.set_ylabel("neurons")
    ax.set_title(f"Firing frequency spans >5 decades: {bands['counts'][0] + bands['counts'][1]:,} of {N:,} layer-42 neurons "
                 f"fire on <0.1% of tokens\n(1.02M FineFineWeb tokens, 4000×256-token windows; SwiGLU value a, |a| vs own max)", fontsize=9.5)
    style(ax)
    ax = axs[1]
    for m, c, lab in ((~sparse_mask & ~dense_mask, C_MUTED, "other"), (dense_mask, C_DENSE, f"dense (≥{a.dense_min:.0%}), n={dense_mask.sum()}"),
                      (sparse_mask, C_SPARSE, f"sparse (<{a.sparse_max:.1%}, write ≥ p25), n={sparse_mask.sum()}")):
        ax.scatter(np.clip(freq_rel[m], 1e-7, 1), ratio_max[m], s=4, color=c, alpha=0.5, lw=0, label=lab)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(f"firing frequency (|a| ≥ {a.rel_thr:.0%} of own max)"); ax.set_ylabel("max over tokens of ‖a·w_down‖ / ‖resid_post‖")
    ax.set_title("Sparse neurons are not weak: their single-token write can reach a sizeable share of the residual norm\n"
                 "(each dot = one neuron; y = largest neuron-write / residual-norm ratio seen at any token)", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, markerscale=3)
    style(ax)
    fig.tight_layout(); save(fig, report, "fig1_firing_frequency")

    # ------------------------------------------------------------------ SAE nearest-neighbour + co-activation
    nn_dec_abs, nn_enc_abs = sm["nn_dec_abs"].astype(np.float64), sm["nn_enc_abs"].astype(np.float64)
    nn_dec_cos, nn_enc_cos = sm["nn_dec_cos"].astype(np.float64), sm["nn_enc_cos"].astype(np.float64)
    rand_dec, rand_enc = sm["rand_dec_abs"].astype(np.float64), sm["rand_enc_abs"].astype(np.float64)
    enc_dec_pair = sm["enc_dec_pair"].astype(np.float64); sae_self = sm["sae_self_nn_abs"].astype(np.float64)
    corr_top = sm["corr_top_v"][:, 0].astype(np.float64); corr_at_nn_dec = sm["corr_at_nn_dec"].astype(np.float64)
    corr_top_id = sm["corr_top_i"][:, 0]
    same_feat = (corr_top_id == sm["nn_dec_id"])
    sae_json = {"description": "nearest SAE feature by direction cosine (polarity-signed unit down_proj column vs unit decoder rows / unit "
                               "encoder columns of the 131072-feature layer-42 SAE) and by activation Pearson correlation over the 1.02M tokens",
                "sets": {}}
    for name, m in (("sparse", sparse_mask), ("dense", dense_mask), ("all", np.ones(N, bool))):
        sae_json["sets"][name] = {"n": int(m.sum()), "nn_dec_abs": summ(nn_dec_abs[m]), "nn_enc_abs": summ(nn_enc_abs[m]),
                                  "nn_dec_signed": summ(nn_dec_cos[m]), "nn_enc_signed": summ(nn_enc_cos[m]),
                                  "corr_top": summ(corr_top[m]), "corr_at_nn_dec": summ(corr_at_nn_dec[m]),
                                  "frac_corr_top_is_nn_dec": float(same_feat[m].mean()),
                                  "frac_nn_dec_abs_gt_0.3": float((nn_dec_abs[m] > 0.3).mean()), "frac_nn_dec_abs_gt_0.5": float((nn_dec_abs[m] > 0.5).mean()),
                                  "frac_corr_top_gt_0.3": float((corr_top[m] > 0.3).mean()), "frac_corr_top_gt_0.5": float((corr_top[m] > 0.5).mean()),
                                  "q_nn_dec_abs": qcurve(nn_dec_abs[m]), "q_nn_enc_abs": qcurve(nn_enc_abs[m]), "q_corr_top": qcurve(corr_top[m])}
    sae_json["controls"] = {"random_unit_dirs_nn_dec_abs": summ(rand_dec), "random_unit_dirs_nn_enc_abs": summ(rand_enc),
                            "q_random_nn_dec_abs": qcurve(rand_dec), "expected_max_abs_cos_random": float(np.sqrt(2 * np.log(131072) / D_MODEL)),
                            "sae_enc_dec_same_feature_cos": summ(enc_dec_pair), "q_sae_enc_dec_pair": qcurve(enc_dec_pair),
                            "sae_dec_nearest_other_dec_abs": summ(sae_self), "q_sae_self_nn_abs": qcurve(sae_self)}
    json.dump(sae_json, open(report / "data" / "sae_match.json", "w"), indent=1)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axs[0]
    for x, c, lab in ((nn_dec_abs[sparse_mask], C_SPARSE, f"sparse neurons (n={sparse_mask.sum()})"), (nn_dec_abs[dense_mask], C_DENSE, f"dense neurons (n={dense_mask.sum()})"),
                      (rand_dec, C_RANDOM, "random unit vectors (n=2048)"), (sae_self, C_SAE, "SAE feature → nearest OTHER feature")):
        ax.plot(np.quantile(x, Q), Q, color=c, lw=2, label=lab)
    ax.set_xlabel("max |cos| with any of 131,072 SAE decoder directions"); ax.set_ylabel("fraction of directions ≤ x")
    ax.set_title("Neuron write directions are only mildly closer to SAE features than random vectors\n(ECDF of nearest-neighbour |cos|; 5120-d residual space)", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, loc="lower right"); style(ax)
    ax = axs[1]
    for x, c, lab in ((corr_top[sparse_mask], C_SPARSE, "sparse neurons"), (corr_top[dense_mask], C_DENSE, "dense neurons"), (corr_top, C_ALL, "all neurons")):
        ax.plot(np.quantile(x, Q), Q, color=c, lw=2, label=lab)
    ax.set_xlabel("max Pearson corr(neuron value, SAE feature activation) over all features"); ax.set_ylabel("fraction ≤ x")
    ax.set_title("Co-activation tells the same story: few neurons track a single SAE feature\n(1.02M tokens, pre-topk ReLU encoder activations)", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, loc="lower right"); style(ax)
    ax = axs[2]
    for m, c, lab in ((dense_mask, C_DENSE, "dense"), (sparse_mask, C_SPARSE, "sparse")):
        ax.scatter(nn_dec_abs[m], corr_top[m], s=5, color=c, alpha=0.5, lw=0, label=lab)
    ax.set_xlabel("nearest SAE decoder |cos|"); ax.set_ylabel("top co-activation corr")
    ax.set_title("Direction-cosine match and activation-correlation match agree only in the tail\n(each dot = one neuron)", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, markerscale=3); style(ax)
    fig.tight_layout(); save(fig, report, "fig2_sae_match")

    # ------------------------------------------------------------------ BSF + probes + neuron-neuron
    bsf_n, bsf_r, bsf_s = da["bsf_frac_neuron"].astype(np.float64), da["bsf_frac_random"].astype(np.float64), da["bsf_frac_sae_dec"].astype(np.float64)
    pr_n, pr_r, pr_s = da["probe_nn_abs_neuron"].astype(np.float64), da["probe_nn_abs_random"].astype(np.float64), da["probe_nn_abs_sae_dec"].astype(np.float64)
    nn_n, nn_r = da["neuron_nn_abs"].astype(np.float64), da["random_nn_abs_within2048"].astype(np.float64)
    other = {"description": "BSF: energy fraction of the ZCA-whitened direction inside its best 8-dim block subspace (32768 blocks); "
                            "probes: max |cos| with the 100k cluster-probe directions of the training bank; neuron-neuron: max off-diagonal |cos|",
             "bsf": {"sparse": summ(bsf_n[sparse_mask]), "dense": summ(bsf_n[dense_mask]), "all": summ(bsf_n), "random": summ(bsf_r), "sae_dec": summ(bsf_s),
                     "chance_single_block": float(da["bsf_chance_single"]),
                     "q": {"sparse": qcurve(bsf_n[sparse_mask]), "dense": qcurve(bsf_n[dense_mask]), "random": qcurve(bsf_r), "sae_dec": qcurve(bsf_s)}},
             "probe": {"sparse": summ(pr_n[sparse_mask]), "dense": summ(pr_n[dense_mask]), "all": summ(pr_n), "random": summ(pr_r), "sae_dec": summ(pr_s),
                       "q": {"sparse": qcurve(pr_n[sparse_mask]), "dense": qcurve(pr_n[dense_mask]), "random": qcurve(pr_r), "sae_dec": qcurve(pr_s)}},
             "neuron_nn": {"sparse": summ(nn_n[sparse_mask]), "dense": summ(nn_n[dense_mask]), "all": summ(nn_n), "random_within_2048": summ(nn_r),
                           "q": {"sparse": qcurve(nn_n[sparse_mask]), "dense": qcurve(nn_n[dense_mask]), "all": qcurve(nn_n), "random": qcurve(nn_r)}}}
    json.dump(other, open(report / "data" / "bsf_probe_nn.json", "w"), indent=1)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, (xs, xl, ttl) in zip(axs, (
            (((bsf_n[sparse_mask], C_SPARSE, "sparse neurons"), (bsf_n[dense_mask], C_DENSE, "dense neurons"), (bsf_r, C_RANDOM, "random unit vectors"), (bsf_s, C_SAE, "SAE decoder directions")),
             "energy fraction inside the best BSF block subspace (8 of 5120 dims)",
             "Neuron directions sit in BSF block subspaces no more than random vectors do\n(BSF = 32,768 × 8-dim block featurizer; whitened space)"),
            (((pr_n[sparse_mask], C_SPARSE, "sparse neurons"), (pr_n[dense_mask], C_DENSE, "dense neurons"), (pr_r, C_RANDOM, "random unit vectors"), (pr_s, C_SAE, "SAE decoder directions")),
             "max |cos| with any of 100k cluster-probe directions",
             "Cluster-probe directions are slightly closer to neuron writes than to random vectors\n(same probe bank the inverter trains on)"),
            (((nn_n[sparse_mask], C_SPARSE, "sparse neurons"), (nn_n[dense_mask], C_DENSE, "dense neurons"), (nn_n, C_ALL, "all neurons"), (nn_r, C_RANDOM, "random (2048 vectors)")),
             "max |cos| with any OTHER layer-42 neuron direction",
             "Neuron write directions are nearly mutually orthogonal: nearest neighbour |cos| stays small\n(17,408 directions in 5120-d)"))):
        for x, c, lab in xs:
            ax.plot(np.quantile(x, Q), Q, color=c, lw=2, label=lab)
        ax.set_xlabel(xl); ax.set_ylabel("fraction ≤ x"); ax.set_title(ttl, fontsize=9.5)
        ax.legend(frameon=False, fontsize=8, loc="lower right"); style(ax)
    fig.tight_layout(); save(fig, report, "fig3_bsf_probe_neighbours")

    # ------------------------------------------------------------------ structure among the sparse directions
    from scipy.cluster.hierarchy import fcluster, linkage
    from sklearn.manifold import TSNE
    Us = U[sparse_ids]
    Cs = Us @ Us.T; iu = np.triu_indices(len(sparse_ids), 1); off = Cs[iu]
    R = rng.standard_normal((len(sparse_ids), D_MODEL)).astype(np.float32); R /= np.linalg.norm(R, axis=1, keepdims=True)
    offr = (R @ R.T)[iu]
    Ud = U[dense_ids]; Cd = Ud @ Ud.T; offd = Cd[np.triu_indices(len(dense_ids), 1)]
    link = linkage(Us, method="average", metric="cosine")
    lab = fcluster(link, t=0.6, criterion="distance")               # clusters of dirs with mean cosine >= 0.4
    sizes = np.bincount(lab)
    big = np.flatnonzero(sizes >= 3)
    n_in_clusters = int(sizes[big].sum()) if len(big) else 0
    tsne = TSNE(n_components=2, perplexity=min(30, max(5, len(sparse_ids) // 20)), init="pca", random_state=a.seed, metric="cosine")
    emb = tsne.fit_transform(Us) if len(sparse_ids) >= 10 else np.zeros((len(sparse_ids), 2))
    hb2 = np.linspace(-0.5, 1.0, 76)
    pw = {"n_sparse": int(len(sparse_ids)), "n_pairs": int(len(off)),
          "pairwise_cos": {"sparse": summ(off), "random": summ(offr), "dense": summ(offd),
                           "frac_abs_gt_0.2": {"sparse": float((np.abs(off) > 0.2).mean()), "random": float((np.abs(offr) > 0.2).mean()), "dense": float((np.abs(offd) > 0.2).mean())},
                           "frac_abs_gt_0.4": {"sparse": float((np.abs(off) > 0.4).mean()), "random": float((np.abs(offr) > 0.4).mean()), "dense": float((np.abs(offd) > 0.4).mean())},
                           "hist_bin_edges": hb2.tolist(), "hist_sparse": np.histogram(off, bins=hb2)[0].tolist(),
                           "hist_random": np.histogram(offr, bins=hb2)[0].tolist(), "hist_dense": np.histogram(offd, bins=hb2)[0].tolist()},
          "clustering": {"method": "average-linkage, cosine distance, cut at distance 0.6 (cos ≥ 0.4)", "n_clusters_size_ge3": int(len(big)),
                         "n_neurons_in_clusters_size_ge3": n_in_clusters, "cluster_sizes_ge3": sorted(sizes[big].tolist(), reverse=True)},
          "tsne": {"x": emb[:, 0].tolist(), "y": emb[:, 1].tolist(), "neuron_id": sparse_ids.tolist(), "cluster": lab.tolist(),
                   "log10_freq": np.log10(np.clip(freq_rel[sparse_ids], 1e-7, 1)).tolist()}}
    json.dump(pw, open(report / "data" / "pairwise_structure.json", "w"), indent=1)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axs[0]
    ax.hist(offr, bins=hb2, color=C_RANDOM, alpha=0.6, lw=0, density=True, label="random unit vectors")
    ax.hist(offd, bins=hb2, color=C_DENSE, alpha=0.6, lw=0, density=True, label="dense neurons")
    ax.hist(off, bins=hb2, color=C_SPARSE, alpha=0.6, lw=0, density=True, label="sparse neurons")
    ax.set_yscale("log"); ax.set_xlabel("pairwise cosine between neuron write directions"); ax.set_ylabel("density (log)")
    ax.set_title(f"Pairwise cosines of sparse-neuron directions are near-random: only {pw['pairwise_cos']['frac_abs_gt_0.2']['sparse']:.2%} of pairs have |cos|>0.2\n"
                 f"({len(sparse_ids)} sparse dirs → {len(off):,} pairs vs matched random vectors)", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8); style(ax)
    ax = axs[1]
    inclu = np.isin(lab, big)
    ax.scatter(emb[~inclu, 0], emb[~inclu, 1], s=8, color=C_MUTED, alpha=0.6, lw=0, label="singleton / pair")
    cmap = plt.get_cmap("tab20")
    for j, c in enumerate(big[np.argsort(-sizes[big])][:20]):
        m = lab == c
        ax.scatter(emb[m, 0], emb[m, 1], s=14, color=cmap(j % 20), lw=0, label=f"cluster {c} (n={m.sum()})" if j < 8 else None)
    ax.set_title(f"t-SNE of sparse-neuron directions: {len(big)} small clusters (≥3 members, cos≥0.4) hold {n_in_clusters} of {len(sparse_ids)} neurons\n"
                 "(cosine metric; grey = no partner above the cut)", fontsize=9.5)
    ax.set_xticks([]); ax.set_yticks([]); ax.legend(frameon=False, fontsize=7, loc="best", markerscale=1.5); style(ax)
    fig.tight_layout(); save(fig, report, "fig4_sparse_structure")

    # ------------------------------------------------------------------ presence in real activations (ceiling for verbalization)
    cos_max_c, cos_max_u = st["cos_max_c"].astype(np.float64), st["cos_max_u"].astype(np.float64)
    pk_cos_c, pk_frac = pk["peak_cos_c"].astype(np.float64), pk["peak_frac"].astype(np.float64)
    pres = {"description": "max over the 1.02M corpus tokens of cos(h_t - mu, dir) [centered] and cos(h_t, dir) [uncentered, the scorer's convention]; "
                           "at the neuron's own peak token: cos(h - mu, dir) and ||a w|| / ||h||",
            "cos_max_c": {"sparse": summ(cos_max_c[sparse_mask]), "dense": summ(cos_max_c[dense_mask]), "all": summ(cos_max_c)},
            "cos_max_u": {"sparse": summ(cos_max_u[sparse_mask]), "dense": summ(cos_max_u[dense_mask]), "all": summ(cos_max_u)},
            "peak_cos_c": {"sparse": summ(pk_cos_c[sparse_mask]), "dense": summ(pk_cos_c[dense_mask]), "all": summ(pk_cos_c)},
            "peak_frac": {"sparse": summ(pk_frac[sparse_mask]), "dense": summ(pk_frac[dense_mask]), "all": summ(pk_frac)},
            "nn_dec_rank_at_peak": {"sparse": summ(pk["nn_dec_rank_at_peak"][sparse_mask].astype(np.float64)),
                                    "dense": summ(pk["nn_dec_rank_at_peak"][dense_mask].astype(np.float64)),
                                    "frac_rank1_sparse": float((pk["nn_dec_rank_at_peak"][sparse_mask] == 1).mean()),
                                    "frac_rank1_dense": float((pk["nn_dec_rank_at_peak"][dense_mask] == 1).mean())},
            "q": {"cos_max_u_sparse": qcurve(cos_max_u[sparse_mask]), "cos_max_u_dense": qcurve(cos_max_u[dense_mask]),
                  "cos_max_c_sparse": qcurve(cos_max_c[sparse_mask]), "cos_max_c_dense": qcurve(cos_max_c[dense_mask]),
                  "peak_frac_sparse": qcurve(pk_frac[sparse_mask]), "peak_frac_dense": qcurve(pk_frac[dense_mask])}}
    json.dump(pres, open(report / "data" / "presence.json", "w"), indent=1)

    # ------------------------------------------------------------------ verbalization (optional)
    verb = None
    if a.verbalize and os.path.exists(a.verbalize):
        recs = [json.loads(l) for l in open(a.verbalize)]
        by = {}
        for r in recs:
            by.setdefault((r["set"], r["row"]), []).append(r)
        verb = {"source": a.verbalize, "sets": {}}
        for sname in ("neuron_sparse", "neuron_dense", "sae", "random"):
            groups = [v for (s, _), v in by.items() if s == sname]
            if not groups:
                continue
            def bo(key):
                return np.array([max(x.get(key, np.nan) for x in g) for g in groups], np.float64)
            d = {"n_dirs": len(groups), "n_samples": sum(len(g) for g in groups),
                 "cos_last5_bo": summ(bo("cos_last5")), "cos_all_bo": summ(bo("cos_all")), "cosc_last5_bo": summ(bo("cosc_last5")),
                 "cos_last5_mean_single": float(np.mean([x["cos_last5"] for g in groups for x in g])),
                 "q_cos_last5_bo": qcurve(bo("cos_last5")), "q_cos_all_bo": qcurve(bo("cos_all"))}
            if "norm_act" in groups[0][0]:
                na = bo("norm_act")
                d.update({"norm_act_bo": summ(na), "fired10_bo": float(np.nanmean(na > 0.10)), "fired25_bo": float(np.nanmean(na > 0.25)),
                          "fired50_bo": float(np.nanmean(na > 0.50)), "beat_corpus_bo": float(np.nanmean(na > 1.0)), "q_norm_act_bo": qcurve(na)})
            if "sae_act" in groups[0][0]:
                d["fired_sae_gt1_bo"] = float(np.mean(bo("sae_act") > 1.0))
            if sname.startswith("neuron"):
                ids_v = np.array([g[0]["id"] for g in groups])
                d["ceiling_cos_max_u"] = summ(cos_max_u[ids_v]); d["ceiling_cos_max_c"] = summ(cos_max_c[ids_v])
                d["ratio_verbalized_over_corpus_ceiling"] = float(np.mean(bo("cos_all") / np.maximum(cos_max_u[ids_v], 1e-6)))
            verb["sets"][sname] = d
        json.dump(verb, open(report / "data" / "verbalize.json", "w"), indent=1)
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
        names = [s for s in ("neuron_sparse", "neuron_dense", "sae", "random") if s in verb["sets"]]
        labels = {"neuron_sparse": "sparse\nneurons", "neuron_dense": "dense\nneurons", "sae": "SAE\nfeatures", "random": "random\n(control)"}
        colors = {"neuron_sparse": C_SPARSE, "neuron_dense": C_DENSE, "sae": C_SAE, "random": C_RANDOM}
        ax = axs[0]
        vals = [verb["sets"][s]["cos_last5_bo"]["mean"] for s in names]
        ax.bar(range(len(names)), vals, color=[colors[s] for s in names], width=0.6, lw=0)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8, color=C_INK)
        ax.set_xticks(range(len(names))); ax.set_xticklabels([labels[s] for s in names], fontsize=8)
        ax.set_ylabel("best-of-4 max cos over last 5 tokens"); ax.axhline(1 / np.sqrt(D_MODEL), color=C_MUTED, lw=0.8)
        ax.set_title("Inverter reproduces neuron directions far above the random-vector floor\n(RL-A adapter, T=1, ≤48 tokens, 4 samples/direction, clean-base layer-42 scoring)", fontsize=9.5)
        style(ax)
        ax = axs[1]
        for s in ("neuron_sparse", "neuron_dense"):
            if s in verb["sets"]:
                ids_v = np.array([g[0]["id"] for (sn, _), g in by.items() if sn == s])
                ax.scatter(cos_max_u[ids_v], [max(x["cos_all"] for x in g) for (sn, _), g in by.items() if sn == s], s=9, color=colors[s], alpha=0.7, lw=0, label=labels[s].replace("\n", " "))
        lim = [0, max(0.5, float(np.nanmax(cos_max_u[np.concatenate([sel_sparse, sel_dense])])) * 1.05)]
        ax.plot(lim, lim, color=C_MUTED, lw=0.8); ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("ceiling: max cos(h_t, dir) over 1.02M real corpus tokens"); ax.set_ylabel("verbalized: best-of-4 max cos(h_t, dir) over all tokens")
        ax.set_title("Verbalizations land near what real text can achieve for each neuron direction\n(diagonal = matches the corpus maximum)", fontsize=9.5)
        ax.legend(frameon=False, fontsize=8, markerscale=2); style(ax)
        ax = axs[2]
        keys = [("fired10_bo", "≥10% of\ncorpus max"), ("fired25_bo", "≥25%"), ("fired50_bo", "≥50%"), ("beat_corpus_bo", "beats\ncorpus max")]
        w = 0.38
        for j, s in enumerate(("neuron_sparse", "neuron_dense")):
            if s in verb["sets"]:
                vals = [verb["sets"][s][k] for k, _ in keys]
                ax.bar(np.arange(len(keys)) + (j - 0.5) * w, vals, width=w - 0.04, color=colors[s], lw=0, label=labels[s].replace("\n", " "))
                for i, v in enumerate(vals):
                    ax.text(i + (j - 0.5) * w, v + 0.01, f"{v:.2f}", ha="center", fontsize=7.5, color=C_INK)
        if "sae" in verb["sets"] and "fired_sae_gt1_bo" in verb["sets"]["sae"]:
            ax.axhline(verb["sets"]["sae"]["fired_sae_gt1_bo"], color=C_SAE, lw=1.2)
            ax.text(len(keys) - 0.5, verb["sets"]["sae"]["fired_sae_gt1_bo"] + 0.01, f"SAE fired (act>1): {verb['sets']['sae']['fired_sae_gt1_bo']:.2f}", ha="right", fontsize=8, color=C_SAE)
        ax.set_xticks(range(len(keys))); ax.set_xticklabels([k for _, k in keys], fontsize=8); ax.set_ylim(0, 1)
        ax.set_ylabel("fraction of neurons (best of 4 generations)")
        ax.set_title("Fire-back: does the neuron itself activate on the generated text?\n(neuron value on the clean base, relative to its corpus max)", fontsize=9.5)
        ax.legend(frameon=False, fontsize=8); style(ax)
        fig.tight_layout(); save(fig, report, "fig5_verbalization")

    # ------------------------------------------------------------------ examples: sparse neurons, contexts, SAE NN, verbalizations
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    ids_all = sw["ids"]; win_len = ids_all.shape[1]
    topv, topi = st["topv"], st["topi"]
    sae_ex = None
    if a.sae_examples and os.path.exists(a.sae_examples):
        import pandas as pd
        df = pd.read_parquet(a.sae_examples)
        sae_ex = df[df["rank"] == 0].set_index("feature")["window"].to_dict()
    verb_by = {}
    if a.verbalize and os.path.exists(a.verbalize):
        for r in (json.loads(l) for l in open(a.verbalize)):
            if r["set"] == "neuron_sparse":
                verb_by.setdefault(r["id"], []).append(r)
    # pick: the sparse neurons we verbalized, ordered by best verbalization cosine if available, else by write share
    cand = sel_sparse.tolist()
    if verb_by:
        cand = sorted(cand, key=lambda i: -max((r["cos_last5"] for r in verb_by.get(i, [])), default=-1))
        show = cand[:6] + cand[len(cand) // 2 - 2: len(cand) // 2 + 2] + cand[-3:]
    else:
        show = sorted(cand, key=lambda i: -ratio_max[i])[:12]
    examples = []
    for i in show:
        ctxs = []
        for k in range(3):
            g = int(topi[k, i]); wnd, pos = g // win_len, g % win_len
            ctxs.append({"value": float(topv[k, i]), "window": int(sw["sel_windows"][wnd]), "pos": int(pos), "text": decode_ctx(tok, ids_all[wnd], pos)})
        e = {"neuron": int(i), "polarity": int(pol[i]), "freq_rel": float(freq_rel[i]), "max_abs": float(max_abs[i]), "write_max": float(write_max[i]),
             "ratio_max": float(ratio_max[i]), "cos_max_u": float(cos_max_u[i]), "contexts": ctxs,
             "nn_sae_dec": {"feature": int(sm["nn_dec_id"][i]), "cos": float(nn_dec_cos[i]), "corr": float(corr_at_nn_dec[i]),
                            "example": sae_ex.get(int(sm["nn_dec_id"][i])) if sae_ex else None},
             "top_corr_sae": {"feature": int(corr_top_id[i]), "corr": float(corr_top[i]), "example": sae_ex.get(int(corr_top_id[i])) if sae_ex else None},
             "cluster_members": []}
        if i in sparse_ids:
            j = int(np.flatnonzero(sparse_ids == i)[0])
            if sizes[lab[j]] >= 2:
                e["cluster_members"] = [int(x) for x in sparse_ids[lab == lab[j]] if x != i][:6]
        if verb_by.get(i):
            vs = sorted(verb_by[i], key=lambda r: -r["cos_last5"])
            e["verbalizations"] = [{"text": r["text"], "cos_last5": r["cos_last5"], "cos_all": r["cos_all"], "norm_act": r.get("norm_act")} for r in vs]
        examples.append(e)
    # cluster spot-check: contexts of the 4 biggest clusters
    clusters = []
    for c in big[np.argsort(-sizes[big])][:6]:
        mem = sparse_ids[lab == c]
        rows = []
        for i in mem[:6]:
            g = int(topi[0, i]); wnd, pos = g // win_len, g % win_len
            rows.append({"neuron": int(i), "top_context": decode_ctx(tok, ids_all[wnd], pos), "freq_rel": float(freq_rel[i])})
        sub = U[mem]; cc = sub @ sub.T
        clusters.append({"cluster": int(c), "size": int(len(mem)), "mean_pairwise_cos": float(cc[np.triu_indices(len(mem), 1)].mean()), "members": rows})
    json.dump({"examples": examples, "clusters": clusters}, open(report / "data" / "examples.json", "w"), indent=1, ensure_ascii=False)

    # ------------------------------------------------------------------ headline numbers
    head = {"n_neurons": N, "n_tokens": n_tok, "d_ff": meta["d_ff"], "layer": meta["layer"], "tokens_per_s": meta.get("tokens_per_s"),
            "bands": dict(zip(band_names, bands["counts"])), "n_sparse": int(len(sparse_ids)), "n_dense": int(len(dense_ids)),
            "sparse_def": bands["sparse_def"], "dense_def": bands["dense_def"],
            "sae_nn_dec_abs_median": {"sparse": float(np.median(nn_dec_abs[sparse_mask])), "dense": float(np.median(nn_dec_abs[dense_mask])), "random": float(np.median(rand_dec))},
            "sae_nn_dec_abs_p90": {"sparse": float(np.quantile(nn_dec_abs[sparse_mask], .9)), "dense": float(np.quantile(nn_dec_abs[dense_mask], .9)), "random": float(np.quantile(rand_dec, .9))},
            "frac_nn_dec_abs_gt_0.5": {"sparse": float((nn_dec_abs[sparse_mask] > .5).mean()), "dense": float((nn_dec_abs[dense_mask] > .5).mean())},
            "corr_top_median": {"sparse": float(np.median(corr_top[sparse_mask])), "dense": float(np.median(corr_top[dense_mask]))},
            "frac_corr_top_gt_0.5": {"sparse": float((corr_top[sparse_mask] > .5).mean()), "dense": float((corr_top[dense_mask] > .5).mean())},
            "bsf_frac_median": {"sparse": float(np.median(bsf_n[sparse_mask])), "dense": float(np.median(bsf_n[dense_mask])), "random": float(np.median(bsf_r)), "sae_dec": float(np.median(bsf_s))},
            "probe_nn_median": {"sparse": float(np.median(pr_n[sparse_mask])), "dense": float(np.median(pr_n[dense_mask])), "random": float(np.median(pr_r)), "sae_dec": float(np.median(pr_s))},
            "pairwise_frac_abs_gt_0.2": pw["pairwise_cos"]["frac_abs_gt_0.2"], "n_clusters_ge3": pw["clustering"]["n_clusters_size_ge3"],
            "n_in_clusters_ge3": pw["clustering"]["n_neurons_in_clusters_size_ge3"],
            "cos_max_u_median": {"sparse": float(np.median(cos_max_u[sparse_mask])), "dense": float(np.median(cos_max_u[dense_mask]))},
            "peak_frac_median": {"sparse": float(np.median(pk_frac[sparse_mask])), "dense": float(np.median(pk_frac[dense_mask]))},
            "verbalize": {s: {k: v for k, v in d.items() if not k.startswith("q_")} for s, d in verb["sets"].items()} if verb else None}
    json.dump(head, open(report / "data" / "headline.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in head.items() if k != "verbalize"}, indent=1))
    if verb:
        for s, d in verb["sets"].items():
            print(s, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items() if not k.startswith("q_") and not isinstance(v, dict)},
                  {k: round(v["mean"], 4) for k, v in d.items() if isinstance(v, dict) and "mean" in v})


if __name__ == "__main__":
    main()
