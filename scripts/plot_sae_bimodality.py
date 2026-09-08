"""How bimodal is the inverter's per-SAE-feature fidelity? Figures + replot-ready data for the
report ~/shared/reports/maemm-sae-bimodality/ from the evaluator's per-direction dumps
(eval/eval_ckpt_daemon.py --dump-per-dir -> /data/eval_ckpt/<tag>/perdir_ckpt_<k>.json).

Input: three dumps (SFT-only LoRA, RL LoRA, full fine-tune), each holding for the SAME 512 held-out SAE
features the best-of-4 normalized activation norm_act = act(best sample) / corpus max (plus fired / rank /
best text) and for every cosine family the best-of-4 max-token cosine per direction.

Figures (PNG + PDF, same stem) and data/*.json with every number shown:
  sae_normact_hist         per-feature norm_act histogram, one panel per checkpoint, shared bins, mean lines
  sae_normact_ecdf         the same as ECDFs on one axis
  sae_normact_fractions    missed (<0.1) / middle / nailed (>0.9) stacked bars + BC / dip annotations
  realact_cos_contrast     the real-activation family's per-direction cosine (unimodal contrast)
  sae_normact_scatter      per-feature SFT vs RL and SFT LoRA vs full FT, bottom decile highlighted

Usage:
    python scripts/plot_sae_bimodality.py --sft perdir_ckpt_10107.json --rl perdir_ckpt_150.json \
        --fullft perdir_ckpt_2441.json --out ~/shared/reports/maemm-sae-bimodality
"""
import argparse
import json
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# categorical slots 1-3 of the validated reference palette (all-pairs safe), chrome from the same sheet
COLORS = {"sft": "#2a78d6", "rl": "#eb6834", "fullft": "#1baf7a"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
LABELS = {"sft": "SFT-only LoRA (23M real-activation examples)",
          "rl": "RL LoRA (best checkpoint, step 150)",
          "fullft": "Full fine-tune (10M examples)"}
SHORT = {"sft": "SFT LoRA", "rl": "RL LoRA", "fullft": "Full FT"}
ORDER = ["sft", "rl", "fullft"]
MISS, NAIL = 0.10, 0.90
BC_THRESH = 5.0 / 9.0

plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white"})


def bimodality_coefficient(x):
    """BC = (g^2 + 1) / (k + 3 (n-1)^2 / ((n-2)(n-3))) with the SAMPLE skewness g and SAMPLE excess kurtosis k
    (bias-corrected, as in SAS / Pfister et al. 2013); > 5/9 suggests bimodality (uniform = 5/9, normal = 1/3)."""
    x = np.asarray(x, np.float64)
    n = len(x)
    g = stats.skew(x, bias=False)
    k = stats.kurtosis(x, bias=False)          # excess
    return float((g ** 2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))), float(g), float(k)


def dip_test(x):
    try:
        import diptest
    except ImportError:
        return None
    dip, p = diptest.diptest(np.asarray(x, np.float64))
    return {"dip": float(dip), "p": float(p)}


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"={p:.3f}"


def describe(x):
    x = np.asarray(x, np.float64)
    bc, g, k = bimodality_coefficient(x)
    out = {"n": int(len(x)), "mean": float(x.mean()), "median": float(np.median(x)), "std": float(x.std(ddof=1)),
           "frac_missed_lt_0.1": float(np.mean(x < MISS)), "frac_middle_0.1_to_0.9": float(np.mean((x >= MISS) & (x <= NAIL))),
           "frac_nailed_gt_0.9": float(np.mean(x > NAIL)), "frac_gt_1": float(np.mean(x > 1.0)),
           "skewness": g, "excess_kurtosis": k, "bimodality_coefficient": bc, "bc_threshold": BC_THRESH, "bc_bimodal": bool(bc > BC_THRESH),
           "quantiles": {q: float(np.quantile(x, q)) for q in (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)}}
    bc1, g1, k1 = bimodality_coefficient(np.minimum(x, 1.0))   # fidelity saturated at the corpus max: removes the right tail
    out.update({"bimodality_coefficient_clip1": bc1, "skewness_clip1": g1, "excess_kurtosis_clip1": k1, "bc_clip1_bimodal": bool(bc1 > BC_THRESH),
                "max": float(x.max())})
    d = dip_test(x)
    if d is not None:
        out["hartigan_dip"] = d["dip"]; out["hartigan_dip_p"] = d["p"]
    return out


def load(path):
    d = json.load(open(path))
    s = d["perdir"]["sae"]
    return {"path": path, "ckpt": d["ckpt"], "ckpt_step": d["ckpt_step"], "tag": d.get("tag"), "aggregates": d.get("aggregates", {}),
            "protocol": d.get("protocol", {}), "feature": np.array(s["feature"]), "norm_act": np.array(s["norm_act"], np.float64),
            "best_act": np.array(s["best_act"], np.float64), "corpus_peak": np.array(s["corpus_peak"], np.float64),
            "rank": np.array(s.get("rank", [np.nan] * len(s["feature"])), np.float64), "fired": np.array(s["fired"]),
            "best_text": s.get("best_text", [""] * len(s["feature"])), "act_bo": s.get("act_bo"),
            "cos": {f: np.array(v, np.float64) for f, v in d["perdir"]["cos"].items()},
            "sae_cos": np.array(s["cos"], np.float64) if "cos" in s else None}


def savefig(fig, out, stem):
    fig.savefig(os.path.join(out, stem + ".png"), dpi=170, bbox_inches="tight")
    fig.savefig(os.path.join(out, stem + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.tick_params(length=0)


def hist_panels(D, key_fn, xlabel, title, stem, out, bins, mean_lines=True, xlim=None, annotate=None, label_left=False):
    """One row per checkpoint, shared bins/x, counts on y, a solid mean line, optional per-panel annotation text."""
    fig, axes = plt.subplots(len(ORDER), 1, figsize=(8.2, 6.6), sharex=True, sharey=True)
    data = {"bins": [float(b) for b in bins], "counts": {}, "means": {}}
    for ax, k in zip(axes, ORDER):
        x = key_fn(D[k])
        cnt, _ = np.histogram(np.clip(x, bins[0], bins[-1] - 1e-9), bins=bins)
        data["counts"][k] = [int(c) for c in cnt]; data["means"][k] = float(x.mean())
        ax.bar(bins[:-1], cnt, width=np.diff(bins), align="edge", color=COLORS[k], edgecolor="white", linewidth=0.6)
        if mean_lines:
            ax.axvline(x.mean(), color=INK, lw=1.2)
            ax.text(x.mean() + 0.01 * (bins[-1] - bins[0]), ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] > 0 else 1, f"mean {x.mean():.2f}",
                    color=INK, fontsize=9, va="top")
        ax.text(0.01 if label_left else 0.99, 0.95, LABELS[k] + (f"\n{annotate(D[k], x)}" if annotate else ""), transform=ax.transAxes,
                ha="left" if label_left else "right", va="top", fontsize=9, color=INK2)
        style(ax)
        ax.set_ylabel("features" if "sae" in stem else "directions")
    axes[-1].set_xlabel(xlabel)
    if xlim:
        axes[-1].set_xlim(*xlim)
    fig.suptitle(title, fontsize=11.5, fontweight="bold", x=0.02, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    savefig(fig, out, stem)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", required=True); ap.add_argument("--rl", required=True); ap.add_argument("--fullft", required=True)
    ap.add_argument("--out", required=True, help="report folder (figures + data/*.json go here)")
    ap.add_argument("--ref", default=None, help="optional json {key: {metric: value}} of the daemons' own aggregates for the reproduction check")
    a = ap.parse_args()
    out = a.out; os.makedirs(os.path.join(out, "data"), exist_ok=True)
    D = {"sft": load(a.sft), "rl": load(a.rl), "fullft": load(a.fullft)}
    for k in ORDER:   # the raw dumps ship with the report (texts included)
        dst = os.path.join(out, "data", f"perdir_{k}.json")
        if os.path.abspath(D[k]["path"]) != os.path.abspath(dst):
            shutil.copy(D[k]["path"], dst)
    if a.ref:
        dst = os.path.join(out, "data", "reference_aggregates.json")
        if os.path.abspath(a.ref) != os.path.abspath(dst):
            shutil.copy(a.ref, dst)
    feats = D["sft"]["feature"]
    assert all((D[k]["feature"] == feats).all() for k in ORDER), "the three dumps score different SAE features"
    n = len(feats)

    # ---- per-feature table + summary statistics ----
    per_feature = {"feature": [int(f) for f in feats], "corpus_peak": D["sft"]["corpus_peak"].tolist(),
                   **{f"norm_act_{k}": D[k]["norm_act"].tolist() for k in ORDER},
                   **{f"best_act_{k}": D[k]["best_act"].tolist() for k in ORDER},
                   **{f"rank_{k}": [None if np.isnan(r) else int(r) for r in D[k]["rank"]] for k in ORDER},
                   **{f"fired_{k}": [int(v) for v in D[k]["fired"]] for k in ORDER}}
    json.dump(per_feature, open(os.path.join(out, "data", "sae_normact_per_feature.json"), "w"))

    statsd = {"sae_norm_act": {k: describe(D[k]["norm_act"]) for k in ORDER},
              "realact_cos": {k: describe(D[k]["cos"]["realact"]) for k in ORDER},
              "all_cos_families": {k: {f: describe(v) for f, v in D[k]["cos"].items()} for k in ORDER},
              "thresholds": {"missed_lt": MISS, "nailed_gt": NAIL, "bc_bimodal_gt": BC_THRESH},
              "diptest_available": dip_test(np.arange(10.0)) is not None,
              "checkpoints": {k: {"ckpt": D[k]["ckpt"], "ckpt_step": D[k]["ckpt_step"], "tag": D[k]["tag"], "label": LABELS[k]} for k in ORDER}}
    json.dump(statsd, open(os.path.join(out, "data", "bimodality_stats.json"), "w"), indent=1)
    S = statsd["sae_norm_act"]

    # ---- aggregates: recomputed from the dump vs the evaluator's own numbers (and the reference daemons') ----
    agg = {}
    for k in ORDER:
        na, best, rk = D[k]["norm_act"], D[k]["best_act"], D[k]["rank"]
        fams = [f for f in D[k]["cos"] if f != "random"]
        rec = {"eval/sae/norm_act": float(na.mean()), "eval/sae/unverbalized_frac": float(np.mean(best <= 1.0)),
               "eval/sae/unverbalized_p10": float(np.mean(na < 0.1)), "eval/sae/fired": float(np.mean(best > 1.0)),
               "eval/sae/rank1_frac": float(np.mean(rk == 1)), "eval/mean_all": float(np.mean([D[k]["cos"][f].mean() for f in fams])),
               **{f"eval/{f}/cos": float(D[k]["cos"][f].mean()) for f in D[k]["cos"]}}
        agg[k] = {"from_dump": rec, "evaluator_this_run": {m: D[k]["aggregates"].get(m) for m in rec}}
    if a.ref:
        ref = json.load(open(a.ref))
        for k in ORDER:
            agg[k]["reference_daemon"] = ref.get(k, {})
    json.dump(agg, open(os.path.join(out, "data", "aggregates_check.json"), "w"), indent=1)

    # ---- (i) histogram ----
    hi = max(1.5, float(np.ceil(max(D[k]["norm_act"].max() for k in ORDER) * 10) / 10))
    hi = min(hi, 2.0)   # 95th percentile of the RL checkpoint is ~1.8; the few larger values are clipped into the last bin (said in the x-label)
    bins = np.round(np.arange(0, hi + 1e-9, 0.05), 3)
    frac_txt = lambda d, x: (f"missed <0.1: {np.mean(x < MISS):.0%}   nailed >0.9: {np.mean(x > NAIL):.0%}"
                             + (f"   dip test p{fmt_p(dip_test(x)['p'])}" if dip_test(x) else ""))
    hd = hist_panels(D, lambda d: d["norm_act"], "per-feature fidelity: best-of-4 activation of the target SAE feature / its corpus max (values above the axis end are clipped into the last bin)",
                     "Per-feature SAE fidelity splits into a spike of misses near 0 and a broad hump of hits around the corpus max\n"
                     f"({n} held-out SAE features, best of 4 samples at T=1, Qwen3.6-27B activation-to-text inverter, three checkpoints)",
                     "sae_normact_hist", out, bins, annotate=frac_txt)
    json.dump(hd, open(os.path.join(out, "data", "sae_normact_hist.json"), "w"))

    # ---- (ii) ECDF ----
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ecdf = {}
    ytext = {"sft": 0.06, "fullft": -0.06, "rl": -0.04}   # label offsets: SFT and full FT sit within 2pp of each other at x=0.1
    for k in ORDER:
        x = np.sort(D[k]["norm_act"]); y = np.arange(1, n + 1) / n
        ax.step(np.r_[0, x], np.r_[0, y], where="post", color=COLORS[k], lw=2, label=SHORT[k])
        ecdf[k] = {"x_sorted": x.tolist(), "cdf": y.tolist()}
        f_miss = float(np.mean(x < MISS))
        ax.annotate(f"{SHORT[k]}: {f_miss:.0%} below 0.1", xy=(MISS, f_miss), xytext=(MISS + 0.3, f_miss + ytext[k]),
                    fontsize=8.5, color=COLORS[k], arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    for v in (MISS, NAIL):
        ax.axvline(v, color=MUTED, lw=0.8)
    ax.set_xlim(0, min(hi, 2.0)); ax.set_ylim(0, 1)
    ax.set_xlabel("per-feature fidelity: best-of-4 target-feature activation / corpus max"); ax.set_ylabel("fraction of the 512 features at or below x")
    ax.set_title("The cumulative curve jumps at 0 then climbs slowly: the miss mass is a separate clump, not a tail\n"
                 f"(ECDF of per-feature fidelity, {n} held-out SAE features, best of 4 samples, three checkpoints)", fontsize=10.5, loc="left")
    ax.legend(frameon=False, loc="lower right"); style(ax)
    savefig(fig, out, "sae_normact_ecdf")
    json.dump({"thresholds": [MISS, NAIL], **ecdf}, open(os.path.join(out, "data", "sae_normact_ecdf.json"), "w"))

    # ---- (iii) fractions + BC / dip ----
    fig, ax = plt.subplots(figsize=(9.0, 3.6))
    segs = [("frac_missed_lt_0.1", "missed (<0.1)", "#c3c2b7"), ("frac_middle_0.1_to_0.9", "middle (0.1-0.9)", "#86b6ef"), ("frac_nailed_gt_0.9", "nailed (>0.9)", "#1c5cab")]
    ys = np.arange(len(ORDER))[::-1]
    for j, k in enumerate(ORDER):
        left = 0.0
        for key, lab, col in segs:
            v = S[k][key]
            ax.barh(ys[j], v, left=left, color=col, edgecolor="white", linewidth=2, height=0.62, label=lab if j == 0 else None)
            if v > 0.06:
                ax.text(left + v / 2, ys[j], f"{v:.0%}", ha="center", va="center", fontsize=9, color="white" if key != "frac_missed_lt_0.1" else INK)
            left += v
        dip = f"dip test p{fmt_p(S[k]['hartigan_dip_p'])}\n" if "hartigan_dip_p" in S[k] else ""
        ax.text(1.01, ys[j], f"{dip}BC {S[k]['bimodality_coefficient']:.2f} raw / {S[k]['bimodality_coefficient_clip1']:.2f} capped at 1",
                va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(ys); ax.set_yticklabels([SHORT[k] for k in ORDER]); ax.set_xlim(0, 1); ax.set_xticks([0, .25, .5, .75, 1]); ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("share of the 512 held-out SAE features")
    ax.set_title("Two thirds of the features sit in the two extreme bins and Hartigan's dip test rejects a single mode for every checkpoint\n"
                 "(missed / middle / nailed shares of per-feature fidelity; dip p < 0.05 = not unimodal; the tail-sensitive bimodality coefficient BC > 0.56 only once fidelity is capped at 1)",
                 fontsize=10, loc="left")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=9)
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(length=0); ax.spines["left"].set_visible(False)
    savefig(fig, out, "sae_normact_fractions")

    # ---- (iv) contrast: real-activation cosine ----
    rbins = np.round(np.arange(0, 1.0 + 1e-9, 0.025), 3)
    rd = hist_panels(D, lambda d: d["cos"]["realact"], "per-direction fidelity: best-of-4 max-token cosine between the generated text's layer-42 activation and the target activation",
                     "Contrast: for real held-out activations the same score is one hump -- this is what 'not bimodal' looks like\n"
                     f"({n} held-out real layer-42 activations, best of 4 samples, same three checkpoints)", "realact_cos_contrast", out, rbins,
                     annotate=lambda d, x: (f"dip test p{fmt_p(dip_test(x)['p'])}   " if dip_test(x) else "") + f"BC {bimodality_coefficient(x)[0]:.2f}",
                     label_left=True)
    json.dump(rd, open(os.path.join(out, "data", "realact_cos_contrast.json"), "w"))

    # ---- (v) same misses? Jaccard of bottom deciles + scatter ----
    dec = {k: set(np.argsort(D[k]["norm_act"])[: n // 10].tolist()) for k in ORDER}
    miss = {k: set(np.where(D[k]["norm_act"] < MISS)[0].tolist()) for k in ORDER}
    jac = lambda A, B: len(A & B) / max(1, len(A | B))
    pairs = [("sft", "rl"), ("sft", "fullft"), ("rl", "fullft")]
    exp_j = (n // 10) / (2 * n - n // 10) if n else 0   # expected Jaccard of two random 10% subsets: |A∩B|/|A∪B| ≈ 0.1n·0.1 / (0.19n)
    ov = {"bottom_decile_size": n // 10, "jaccard_bottom_decile": {f"{p}_vs_{q}": jac(dec[p], dec[q]) for p, q in pairs},
          "jaccard_bottom_decile_all3": len(dec["sft"] & dec["rl"] & dec["fullft"]) / max(1, len(dec["sft"] | dec["rl"] | dec["fullft"])),
          "jaccard_expected_if_independent": float(0.01 / 0.19),
          "missed_lt_0.1_size": {k: len(miss[k]) for k in ORDER},
          "jaccard_missed_lt_0.1": {f"{p}_vs_{q}": jac(miss[p], miss[q]) for p, q in pairs},
          "missed_all3": len(miss["sft"] & miss["rl"] & miss["fullft"]),
          "missed_all3_features": sorted(int(feats[i]) for i in (miss["sft"] & miss["rl"] & miss["fullft"])),
          "spearman_norm_act": {f"{p}_vs_{q}": float(stats.spearmanr(D[p]["norm_act"], D[q]["norm_act"]).statistic) for p, q in pairs},
          "pearson_norm_act": {f"{p}_vs_{q}": float(stats.pearsonr(D[p]["norm_act"], D[q]["norm_act"]).statistic) for p, q in pairs}}
    # RL vs SFT: does RL lift the misses or sharpen the hits?
    s_na, r_na = D["sft"]["norm_act"], D["rl"]["norm_act"]
    sm = s_na < MISS; sh = s_na > NAIL; smid = ~sm & ~sh
    ov["rl_effect"] = {"sft_missed_n": int(sm.sum()), "sft_missed_rl_still_missed": int((sm & (r_na < MISS)).sum()),
                       "sft_missed_rl_lifted_ge_0.1": int((sm & (r_na >= MISS)).sum()), "sft_missed_rl_nailed_gt_0.9": int((sm & (r_na > NAIL)).sum()),
                       "sft_missed_mean_rl": float(r_na[sm].mean()) if sm.any() else None,
                       "sft_nailed_n": int(sh.sum()), "sft_nailed_mean_sft": float(s_na[sh].mean()) if sh.any() else None,
                       "sft_nailed_mean_rl": float(r_na[sh].mean()) if sh.any() else None,
                       "sft_nailed_rl_dropped_lt_0.9": int((sh & (r_na <= NAIL)).sum()),
                       "sft_middle_n": int(smid.sum()), "sft_middle_mean_sft": float(s_na[smid].mean()) if smid.any() else None,
                       "sft_middle_mean_rl": float(r_na[smid].mean()) if smid.any() else None,
                       "rl_minus_sft_mean": float((r_na - s_na).mean()), "rl_gt_sft_frac": float(np.mean(r_na > s_na))}
    json.dump(ov, open(os.path.join(out, "data", "miss_overlap.json"), "w"), indent=1)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.9))
    lim = min(hi, 2.0)
    for ax, (p, q) in zip(axes, [("sft", "rl"), ("sft", "fullft")]):
        x, y = D[p]["norm_act"], D[q]["norm_act"]
        n_both = len(miss[p] & miss[q]); n_either = len(miss[p] | miss[q])
        ax.plot([0, lim], [0, lim], color=MUTED, lw=0.8)
        ax.axhline(MISS, color=GRID, lw=0.8); ax.axvline(MISS, color=GRID, lw=0.8)
        ax.add_patch(plt.Rectangle((0, 0), MISS, MISS, facecolor="#f0efec", edgecolor=INK, lw=0.9, zorder=1))
        ax.scatter(np.clip(x, 0, lim), np.clip(y, 0, lim), s=14, color=COLORS[q], alpha=0.55, edgecolor="white", linewidth=0.4, zorder=2,
                   label="one SAE feature (values > 2 drawn at 2)")
        ax.annotate(f"shaded corner = missed (<0.1) by BOTH: {n_both} features\n"
                    f"missed by {SHORT[p]} only: {int(np.sum((x < MISS) & (y >= MISS)))}   by {SHORT[q]} only: {int(np.sum((x >= MISS) & (y < MISS)))}\n"
                    f"Jaccard of the two miss sets {jac(miss[p], miss[q]):.2f} ({n_either} missed by at least one)",
                    xy=(MISS, MISS), xytext=(0.30, 0.30), fontsize=8.5, color=INK, va="bottom",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=GRID, alpha=0.9), arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))
        ax.set_xlim(-0.03, lim); ax.set_ylim(-0.03, lim)
        ax.set_xlabel(f"{SHORT[p]}: per-feature fidelity"); ax.set_ylabel(f"{SHORT[q]}: per-feature fidelity")
        rho = ov["spearman_norm_act"][f"{p}_vs_{q}"]
        ax.text(0.03, 0.98, f"Spearman rho = {rho:.2f}\nabove the diagonal: {np.mean(y > x):.0%} of features", transform=ax.transAxes, va="top", fontsize=9, color=INK2,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=GRID, alpha=0.9))
        ax.legend(frameon=True, framealpha=0.9, edgecolor=GRID, loc="lower right", fontsize=8); ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)
    re_ = ov["rl_effect"]
    fig.suptitle("Misses belong to the feature, not the run: an independent full fine-tune misses the same features as the SFT LoRA (right);\n"
                 f"RL (left) lifts {re_['sft_missed_rl_lifted_ge_0.1'] / max(1, re_['sft_missed_n']):.0%} of the SFT misses above 0.1 and pushes the already-inverted features past the corpus max\n"
                 f"(per-feature fidelity of the {n} held-out SAE features, one checkpoint against another)",
                 fontsize=10, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    savefig(fig, out, "sae_normact_scatter")

    # ---- best / worst features with their best generated text ----
    mean_na = np.mean([D[k]["norm_act"] for k in ORDER], axis=0)
    order = np.argsort(mean_na)
    def rowinfo(i):
        return {"feature": int(feats[i]), "corpus_peak": float(D["sft"]["corpus_peak"][i]), "mean_norm_act": float(mean_na[i]),
                **{f"norm_act_{k}": float(D[k]["norm_act"][i]) for k in ORDER},
                **{f"rank_{k}": (None if np.isnan(D[k]["rank"][i]) else int(D[k]["rank"][i])) for k in ORDER},
                **{f"best_text_{k}": D[k]["best_text"][i] for k in ORDER}}
    tb = {"ranking": "mean per-feature norm_act over the three checkpoints", "worst_20": [rowinfo(i) for i in order[:20]],
          "best_20": [rowinfo(i) for i in order[::-1][:20]]}
    json.dump(tb, open(os.path.join(out, "data", "best_worst_features.json"), "w"), indent=1)

    print(json.dumps({k: {m: round(v, 4) if isinstance(v, float) else v for m, v in S[k].items() if m != "quantiles"} for k in ORDER}, indent=1))
    print("overlap:", json.dumps({k: v for k, v in ov.items() if k != "missed_all3_features"}, indent=1))
    print("aggregates:", json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
