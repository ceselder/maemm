#!/usr/bin/env python3
"""Rank-based MLP-neuron fidelity metric for the Qwen3.6-27B activation->text inverter: figures + exact data for the report
~/shared/reports/maemm-mlp-rank-eval (built by that folder's build_html.py).

Inputs: the re-scored checkpoints of scripts/launchers/spawn_mlprank_rescore.py on the Modal volume
(/data/eval_ckpt/mlprank_<name>/{ckpt_<k>.json, perdir_ckpt_<k>.json}) and the ORIGINAL daemons' eval jsons of the same
checkpoints (/data/eval_ckpt/<tag>/ckpt_<k>.json) for the sanity reproduction of the pre-existing metrics. --fetch pulls
them with `modal volume get` into <report>/raw/.

    MODAL_PROFILE=safety-sahan python scripts/plot_mlp_rank_eval.py --fetch      # pull jsons, rebuild data/*.json + figures
    python scripts/plot_mlp_rank_eval.py                                         # reuse <report>/raw

Outputs -> <report>/{data/*.json, *.png + *.pdf}.
"""
import argparse
import json
import os
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(os.environ.get("MAEMM_MLPRANK_REPORT", "~/shared/reports/maemm-mlp-rank-eval")).expanduser()
RAW = OUT / "raw"
VOL = "maemm-data"
D_FF = 17408

# colour follows the entity GROUP (validated 3-slot categorical palette, all-pairs, light surface) + gray for chance
C_SFT, C_RL, C_PROD, C_CHANCE, C_INK = "#2a78d6", "#eb6834", "#1baf7a", "#898781", "#191919"
SURFACE = "#fcfcfb"

# name -> re-score tag / ckpt step / label / group / reference (original daemon) json
CKPTS = {
    "sft_realact23m":    dict(step=0, label="SFT: 23M real-act pretrain", short="SFT 23M", group="sft",
                              ref=("uplift_init_realact23m", 0), ckpt="/data/sft_mix/realact20m_prefix_lr1e-4/final"),
    "sft_midtrain":      dict(step=0, label="SFT: midtrain only", short="SFT midtrain", group="sft",
                              ref=("sft_mixeq_midtrain_only_from_base", 5665), ckpt="/data/sft_mix/mixeq_midtrain_only_from_base/final"),
    "rl_abl_initbase":   dict(step=300, label="RL 300 steps from no SFT", short="RL from base", group="rl",
                              ref=("rl_abl_initbase", 300), ckpt="/data/ckpts_rl_abl_initbase/final"),
    "rl_abl_initmid":    dict(step=300, label="RL 300 steps from midtrain", short="RL from midtrain", group="rl",
                              ref=("rl_abl_initmid", 300), ckpt="/data/ckpts_rl_abl_initmid/final"),
    "rl_abl_init23m":    dict(step=300, label="RL 300 steps from 23M pretrain", short="RL from 23M", group="rl",
                              ref=("rl_abl_init23m", 300), ckpt="/data/ckpts_rl_abl_init23m/final"),
    "rl_abl_initboth":   dict(step=300, label="RL 300 steps from pretrain+midtrain", short="RL from both", group="rl",
                              ref=("rl_abl_initboth", 300), ckpt="/data/ckpts_rl_abl_initboth/final"),
    "rl_abl_initnewfft": dict(step=300, label="RL 300 steps on full-FT midtrain base", short="RL on full-FT", group="rl",
                              ref=("rl_abl_initnewfft", 300), ckpt="/data/ckpts_rl_abl_initnewfft/final"),
    "rl_I_8x4096_nowarm": dict(step=150, label="production RL (step 150)", short="production RL", group="prod",
                               ref=("rl_I_8x4096_nowarm", 150), ckpt="/data/ckpts_rl_I_8x4096_nowarm/step_150"),
}
GROUP_COLOR = {"sft": C_SFT, "rl": C_RL, "prod": C_PROD}
GROUP_LABEL = {"sft": "SFT only (no RL)", "rl": "RL, 300 steps (SFT-init ablation arms)", "prod": "production RL run"}
ECDF_SERIES = [("sft_realact23m", C_SFT), ("rl_abl_init23m", C_RL), ("rl_I_8x4096_nowarm", C_PROD)]
SANITY_KEYS = ["eval/mean_all", "eval/sae/norm_act", "eval/sae/rank1_frac", "eval/realact/cos", "eval/bsf/cos", "eval/random/cos",
               "eval/mlp/cos", "eval/mlp/norm_act", "eval/mlp/fired10", "eval/mlp/fired25", "eval/mlp/fired50",
               "eval/mlp_pair/cos", "eval/mlp_pair/norm_act", "eval/mlp_pair/fired10", "eval/mlp_pair/any_norm_act"]
SANITY_TOL = 0.01


def vget(remote, local):
    local.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["modal", "volume", "get", VOL, remote, str(local), "--force"], capture_output=True, text=True)
    return r.returncode == 0 and local.exists()


def fetch(refetch=False):
    for name, c in CKPTS.items():
        for kind in ("ckpt", "perdir_ckpt"):
            loc = RAW / f"mlprank_{name}" / f"{kind}_{c['step']}.json"
            if loc.exists() and not refetch:
                continue
            ok = vget(f"/eval_ckpt/mlprank_{name}/{kind}_{c['step']}.json", loc)
            print(f"[fetch] {loc.relative_to(RAW)}: {'ok' if ok else 'MISSING (not scored yet?)'}", flush=True)
        tag, k = c["ref"]
        loc = RAW / "ref" / tag / f"ckpt_{k}.json"
        if not loc.exists() or refetch:
            ok = vget(f"/eval_ckpt/{tag}/ckpt_{k}.json", loc)
            print(f"[fetch] ref {tag}/ckpt_{k}.json: {'ok' if ok else 'MISSING'}", flush=True)


def load():
    res = {}
    for name, c in CKPTS.items():
        p = RAW / f"mlprank_{name}" / f"ckpt_{c['step']}.json"
        if not p.exists():
            print(f"[load] {name}: not scored (yet) -> skipped", flush=True)
            continue
        d = json.load(open(p))
        pp = RAW / f"mlprank_{name}" / f"perdir_ckpt_{c['step']}.json"
        pd = json.load(open(pp))["perdir"] if pp.exists() else None
        rp = RAW / "ref" / c["ref"][0] / f"ckpt_{c['ref'][1]}.json"
        ref = json.load(open(rp))["metrics"] if rp.exists() else None
        res[name] = {"metrics": d["metrics"], "protocol": d.get("protocol", {}), "perdir": pd, "ref": ref, "ckpt": d.get("ckpt")}
    return res


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(0.6); ax.spines[sp].set_color("#c9c7c1")
    ax.tick_params(width=0.6, colors="#52514e", labelsize=8.5)
    ax.yaxis.grid(True, lw=0.5, color="#e8e6e1"); ax.set_axisbelow(True)
    ax.set_facecolor(SURFACE)


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.png", dpi=170, facecolor=SURFACE)
    fig.savefig(OUT / f"{stem}.pdf", facecolor=SURFACE)
    plt.close(fig)
    print(f"[fig] {stem}.png/.pdf", flush=True)


def ecdf(x):
    x = np.sort(np.asarray(x, np.float64))
    return x, np.arange(1, len(x) + 1) / len(x)


# ---------------------------------------------------------------------------------------------------------------------
def fig_metric_bars(res, chance):
    names = [n for n in CKPTS if n in res]
    panels = [("eval/mlp/norm_act", "OLD  norm_act\n(mean of polarity·a / corpus max, best of 4)", "chance_norm_act"),
              ("eval/mlp/fired10", "OLD  fired10\n(share with norm_act ≥ 0.1)", "chance_fired10"),
              ("eval/mlp/corpus_pct_mean", "within-neuron corpus percentile\n(mean; 0.99 = beats 99% of its own corpus tokens)", "chance_corpus_pct_mean"),
              ("eval/mlp/rank1_frac", "NEW  rank-1 share\n(target is the most active of 17,408 neurons)", "chance_rank1_frac"),
              ("eval/mlp/rank_le10", "NEW  top-10 share\n(normalized cross-neuron rank ≤ 10)", "chance_rank_le10"),
              ("eval/mlp/mrr", "NEW  mean reciprocal rank", "chance_mrr")]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.4))
    x = np.arange(len(names))
    data = {"checkpoints": names, "labels": [CKPTS[n]["label"] for n in names], "panels": {}}
    for ax, (key, title, ck) in zip(axes.flat, panels):
        vals = [res[n]["metrics"].get(key, np.nan) for n in names]
        cols = [GROUP_COLOR[CKPTS[n]["group"]] for n in names]
        ax.bar(x, vals, width=0.62, color=cols, edgecolor=SURFACE, linewidth=1.0)
        ch = chance.get(f"eval/mlp/{ck}")
        if ch is not None:
            ax.axhline(ch, color=C_CHANCE, lw=1.2, ls=(0, (4, 3)))
            ax.text(len(names) - 0.4, ch, f" chance {ch:.3f}", color="#52514e", fontsize=7.5, va="bottom", ha="right")
        for xi, v in zip(x, vals):
            ax.text(xi, v, f"{v:.2f}" if key != "eval/mlp/norm_act" else f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color=C_INK)
        ax.set_title(title, fontsize=9.5, loc="left", color=C_INK)
        ax.set_xticks(x); ax.set_xticklabels([CKPTS[n]["short"] for n in names], rotation=28, ha="right", fontsize=8)
        style(ax)
        data["panels"][key] = {"values": [None if np.isnan(v) else float(v) for v in vals], "chance": ch, "title": title}
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLOR[g]) for g in ("sft", "rl", "prod")]
    fig.legend(handles + [plt.Line2D([0], [0], color=C_CHANCE, lw=1.2, ls=(0, (4, 3)))],
               [GROUP_LABEL[g] for g in ("sft", "rl", "prod")] + ["chance: random held-out corpus text, same protocol"],
               loc="lower center", ncol=4, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("The old fire-back metric saturates near or above 1.0 for every RL checkpoint; the cross-neuron rank still separates them\n"
                 "512 held-out layer-42 MLP neurons, Qwen3.6-27B activation→text inverter, best-of-4 generations, last-5-token window",
                 fontsize=11, x=0.01, ha="left", color=C_INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    save(fig, "metric_bars")
    json.dump(data, open(OUT / "data" / "metric_bars.json", "w"), indent=1)
    return data


def fig_rank_ecdf(res, chance_pd):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    data = {"series": {}, "x": "rank (log scale, 1 = most active of all 17,408 layer-42 neurons)"}
    specs = [("mlp", "rank", "single neurons (n = 512): normalized rank at the best token", lambda r: np.asarray(r)[:, 0]),
             ("mlp_pair", "rank", "co-firing pairs (n = 256): mean of the two members' ranks", lambda r: np.asarray(r).mean(1)),
             ("mlp", "raw_rank", "single neurons: RAW rank (polarity·a, no normalization)", lambda r: np.asarray(r)[:, 0])]
    for ax, (fam, key, title, fn) in zip(axes, specs):
        for name, col in ECDF_SERIES:
            if name not in res or not res[name]["perdir"]:
                continue
            r = fn(res[name]["perdir"]["extra"][fam][key])
            xs, ys = ecdf(r)
            ax.step(xs, ys, where="post", color=col, lw=2, label=CKPTS[name]["label"])
            data["series"][f"{fam}/{key}/{name}"] = {"rank_sorted": xs.tolist(), "ecdf": ys.tolist(), "le10": float(np.mean(r <= 10)), "rank1": float(np.mean(r == 1))}
        if chance_pd.get(fam):
            r = fn(chance_pd[fam][key])
            xs, ys = ecdf(r)
            ax.step(xs, ys, where="post", color=C_CHANCE, lw=1.6, ls=(0, (4, 3)), label="chance (random corpus text)")
            data["series"][f"{fam}/{key}/chance"] = {"rank_sorted": xs.tolist(), "ecdf": ys.tolist(), "le10": float(np.mean(r <= 10)), "rank1": float(np.mean(r == 1))}
        ax.axvline(10, color="#c9c7c1", lw=0.8)
        ax.text(10.6, 0.03, "top-10", fontsize=7.5, color="#52514e")
        ax.set_xscale("log"); ax.set_xlim(0.9, D_FF * 1.1); ax.set_ylim(0, 1)
        ax.set_xlabel("cross-neuron rank of the target neuron (log)", fontsize=9)
        ax.set_ylabel("share of held-out directions with rank ≤ x", fontsize=9)
        ax.set_title(title, fontsize=9.5, loc="left", color=C_INK)
        style(ax)
    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("RL moves most held-out neurons into the top-10 of all 17,408 neurons; SFT alone barely beats random corpus text\n"
                 "Cumulative distribution of the target neuron's rank on the best-of-4 generation (normalized rank; right: raw-magnitude rank for contrast)",
                 fontsize=11, x=0.01, ha="left", color=C_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "rank_ecdf")
    json.dump(data, open(OUT / "data" / "rank_ecdf.json", "w"), indent=1)
    return data


def fig_scatter(res, chance_pd):
    names = [n for n, _ in ECDF_SERIES if n in res and res[n]["perdir"]]
    fig, axes = plt.subplots(1, len(names) + 1, figsize=(4.6 * (len(names) + 1), 4.8), sharey=True, squeeze=False)
    axes = axes[0]
    data = {"quadrants": {}, "note": "x = OLD norm_act (best-of-4, floored at 1e-3 for the log axis); y = NEW normalized cross-neuron rank; "
                                     "quadrant shares over the 512 held-out single neurons"}
    for ax, name in zip(axes, names):
        pd = res[name]["perdir"]["extra"]["mlp"]
        na = np.asarray(pd["norm_act"], np.float64); rk = np.asarray(pd["rank"], np.float64)[:, 0]
        col = dict(ECDF_SERIES)[name]
        _scatter_panel(ax, na, rk, col, CKPTS[name]["label"], data, name)
    if chance_pd.get("mlp"):
        na = np.asarray(chance_pd["mlp"]["norm_act"], np.float64); rk = np.asarray(chance_pd["mlp"]["rank"], np.float64)[:, 0]
        _scatter_panel(axes[-1], na, rk, C_CHANCE, "chance: random held-out corpus text", data, "chance")
    axes[0].set_ylabel("NEW: cross-neuron rank at the best token (log; 1 = top)", fontsize=9)
    fig.suptitle("Where the old metric is uninformative: many neurons count as 'fired' (norm_act ≥ 0.1) while dozens of other neurons out-fire them,\n"
                 "and rank-1 inversions exist at every norm_act level — per-neuron old metric vs new rank, 512 held-out layer-42 MLP neurons",
                 fontsize=10.5, x=0.01, ha="left", color=C_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "normact_vs_rank")
    json.dump(data, open(OUT / "data" / "normact_vs_rank.json", "w"), indent=1)
    return data


def _scatter_panel(ax, na, rk, col, label, data, key):
    x = np.maximum(na, 1e-3)
    ax.scatter(x, rk, s=11, color=col, alpha=0.55, linewidths=0)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(8e-4, max(12, float(x.max()) * 1.3)); ax.set_ylim(0.8, D_FF * 1.3)
    ax.axvline(0.1, color="#c9c7c1", lw=0.8); ax.axhline(10, color="#c9c7c1", lw=0.8)
    q = {"fired_and_top10": float(np.mean((na >= 0.1) & (rk <= 10))), "fired_not_top10": float(np.mean((na >= 0.1) & (rk > 10))),
         "notfired_top10": float(np.mean((na < 0.1) & (rk <= 10))), "notfired_not_top10": float(np.mean((na < 0.1) & (rk > 10))),
         "rank1_frac": float(np.mean(rk == 1)), "fired10": float(np.mean(na >= 0.1)), "n": int(len(na)),
         "spearman_normact_rank": float(_spearman(na, rk))}
    data["quadrants"][key] = q
    kw = dict(fontsize=8, color="#52514e", transform=ax.transAxes)
    ax.text(0.98, 0.04, f"fired, top-10: {q['fired_and_top10']:.0%}", ha="right", **kw)
    ax.text(0.98, 0.96, f"fired, NOT top-10: {q['fired_not_top10']:.0%}", ha="right", va="top", **kw)
    ax.text(0.02, 0.04, f"not fired, top-10: {q['notfired_top10']:.0%}", ha="left", **kw)
    ax.text(0.02, 0.96, f"not fired, not top-10: {q['notfired_not_top10']:.0%}", ha="left", va="top", **kw)
    ax.set_title(f"{label}\nrank-1 share {q['rank1_frac']:.0%} · fired10 {q['fired10']:.0%} · Spearman(norm_act, rank) {q['spearman_normact_rank']:+.2f}",
                 fontsize=8.8, loc="left", color=C_INK)
    ax.set_xlabel("OLD: norm_act = polarity·a / corpus max (log)", fontsize=9)
    style(ax); ax.xaxis.grid(True, lw=0.5, color="#e8e6e1")


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0, 1]


# ---------------------------------------------------------------------------------------------------------------------
def summary(res):
    rows, sanity = {}, {}
    chance = {}
    for name, r in res.items():
        m = r["metrics"]
        rows[name] = {"label": CKPTS[name]["label"], "group": CKPTS[name]["group"], "ckpt": r["ckpt"], "ckpt_step": CKPTS[name]["step"],
                      "metrics": {k: v for k, v in m.items() if k.startswith("eval/mlp") or k in ("eval/mean_all", "eval/sae/norm_act", "eval/sae/rank1_frac",
                                                                                                    "eval/sae/mrr", "eval/sae/mean_rank", "time/inline_eval_s", "time/ckpt_eval_s")},
                      "protocol": r["protocol"]}
        for k, v in m.items():
            if "/chance_" in k:
                chance.setdefault(k, []).append(v)
        if r["ref"] is not None:
            cmp = {}
            for k in SANITY_KEYS:
                if k in m and k in r["ref"]:
                    cmp[k] = {"new": m[k], "ref": r["ref"][k], "delta": m[k] - r["ref"][k], "ok": abs(m[k] - r["ref"][k]) <= SANITY_TOL}
            sanity[name] = {"ref": f"/data/eval_ckpt/{CKPTS[name]['ref'][0]}/ckpt_{CKPTS[name]['ref'][1]}.json", "keys": cmp,
                            "all_ok": all(c["ok"] for c in cmp.values()), "max_abs_delta": max(abs(c["delta"]) for c in cmp.values()) if cmp else None}
    # the chance level depends on the base model only -> identical across checkpoints (assert, then keep one number)
    chance_one = {}
    for k, vs in chance.items():
        vs = np.asarray(vs)
        assert np.ptp(vs) < 1e-9, (k, vs)
        chance_one[k] = float(vs[0])
    return rows, sanity, chance_one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="pull the jsons from the Modal volume into <report>/raw first")
    ap.add_argument("--refetch", action="store_true")
    a = ap.parse_args()
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    if a.fetch or a.refetch:
        fetch(a.refetch)
    res = load()
    assert res, "no re-scored checkpoints found under raw/"
    rows, sanity, chance = summary(res)
    chance_pd = {}
    for n, r in res.items():   # any scored checkpoint carries the (identical) chance per-direction arrays
        if r["perdir"]:
            for fam in ("mlp", "mlp_pair"):
                if "chance" in r["perdir"]["extra"].get(fam, {}):
                    chance_pd.setdefault(fam, r["perdir"]["extra"][fam]["chance"])
    json.dump({"checkpoints": rows, "chance": chance, "sanity": sanity, "sanity_tolerance": SANITY_TOL,
               "d_ff": D_FF, "protocol_note": "512 held-out single neurons + 256 co-firing pairs (eval cache v2), best-of-4 generations at T=1, "
               "16-64 new tokens, scored on the clean base at layer 42 over the last 5 kept tokens; rank = 1 + #neurons whose polarity*a/corpus_max "
               "exceeds the target's at the target's best token; chance = the same protocol on 4 random 16-64-token windows of held-out FineFineWeb rows"},
              open(OUT / "data" / "summary.json", "w"), indent=1)
    for name, r in res.items():
        if r["perdir"]:
            per = {fam: {k: v for k, v in r["perdir"]["extra"][fam].items() if k != "chance"} for fam in r["perdir"]["extra"]}
            json.dump({"checkpoint": name, "ckpt": r["ckpt"], "perdir_extra": per}, open(OUT / "data" / f"perneuron_{name}.json", "w"))
    if chance_pd:
        json.dump(chance_pd, open(OUT / "data" / "perneuron_chance.json", "w"))
    fig_metric_bars(res, chance)
    fig_rank_ecdf(res, chance_pd)
    fig_scatter(res, chance_pd)
    print(json.dumps({n: {k.split("eval/mlp/")[-1]: round(v, 4) for k, v in rows[n]["metrics"].items() if k.startswith("eval/mlp/") and "chance" not in k and "best5" not in k and "raw" not in k}
                      for n in rows}, indent=1))
    print("[sanity]", json.dumps({n: {"all_ok": s["all_ok"], "max_abs_delta": round(s["max_abs_delta"], 4) if s["max_abs_delta"] is not None else None} for n, s in sanity.items()}, indent=1))
    print("[chance]", json.dumps({k.split("eval/")[-1]: round(v, 4) for k, v in chance.items() if "best5" not in k}, indent=1))


if __name__ == "__main__":
    main()
