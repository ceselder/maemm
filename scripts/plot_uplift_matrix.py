#!/usr/bin/env python3
"""Cross-uplift matrix plots + data for ~/shared/reports/maemm-uplift-matrix/ (see scripts/uplift_driver.py for the runs).

Pulls every evaluated checkpoint json of the experiment from the Modal volume (eval/eval_ckpt_daemon.py output,
/data/eval_ckpt/<tag>/ckpt_<k>.json), the arm banks' build_stats.json, and the RL training curves (wandb), writes
data/*.json (every plotted number) and the figures (PNG + PDF):

  uplift_sft_delta_heatmap      delta vs the common init after the 200k midtrain      (arms x eval families)
  uplift_rl_delta_heatmap       delta vs the common init after midtrain + RL@100      (arms x eval families)
  uplift_rl_vs_control_heatmap  delta vs the acts100 control after RL@100 = CROSS-UPLIFT (arms x eval families)
  uplift_family_bars_sft/rl     absolute per-family scores, every arm + the init line   (small multiples)
  uplift_rl_train               RL training reward per arm (100 steps)

    MODAL_PROFILE=safety-sahan python scripts/plot_uplift_matrix.py [--no-fetch] [--no-wandb]
"""
import argparse
import glob
import json
import os
import subprocess
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

OUT = os.path.expanduser("~/shared/reports/maemm-uplift-matrix")
RAW = f"{OUT}/raw"
MODAL = os.path.expanduser("~/modal_venv/bin/modal")
PROJ = "octahedral-systems/maxact-fast"

ARMS = ["acts100", "acts_sae", "acts_bsf", "acts_cluster", "acts_realact_long", "acts_mlp"]
ARM_LABEL = {"acts100": "100% real acts (control)", "acts_sae": "50% acts + 50% SAE features", "acts_bsf": "50% acts + 50% BSF blocks",
             "acts_cluster": "50% acts + 50% cluster probes", "acts_realact_long": "50% acts + 50% long-ctx acts",
             "acts_mlp": "50% acts + 50% MLP neurons"}
ARM_SHORT = {"acts100": "acts100", "acts_sae": "+SAE", "acts_bsf": "+BSF", "acts_cluster": "+probes", "acts_realact_long": "+long acts", "acts_mlp": "+MLP"}
# eval column: (metric key, short label, higher-is-better)
COLS = [("eval/mean_all", "mean_all\n(11 fams)", True),
        ("eval/realact/cos", "real acts", True), ("eval/realact_early/cos", "acts early", True), ("eval/realact_mid/cos", "acts mid", True),
        ("eval/realact_long/cos", "acts long", True), ("eval/indist_realact/cos", "acts in-dist", True), ("eval/indist_long/cos", "long in-dist", True),
        ("eval/bsf/cos", "BSF", True), ("eval/sae/norm_act", "SAE\nnorm act", True), ("eval/sae/rank1_frac", "SAE\nrank-1", True), ("eval/sae/verbalized_frac", "SAE\nverbalized", True),
        ("eval/cluster/cos", "probes", True), ("eval/indist_probe/cos", "probes\nin-dist", True), ("eval/jlens/cos", "J-lens", True),
        ("eval/mlp/cos", "MLP\nneurons cos", True), ("eval/mlp/norm_act", "MLP\nfire-back", True), ("eval/mlp_pair/cos", "MLP\npairs cos", True),
        ("eval/random/cos", "random\n(control, low)", False)]
# which columns are the arm's OWN training family (in-distribution for that arm; every arm also trains on real acts)
OWN = {"acts_sae": {"eval/sae/norm_act", "eval/sae/rank1_frac", "eval/sae/verbalized_frac"}, "acts_bsf": {"eval/bsf/cos"},
       "acts_cluster": {"eval/cluster/cos", "eval/indist_probe/cos"}, "acts_realact_long": {"eval/realact_long/cos", "eval/indist_long/cos"},
       "acts_mlp": {"eval/mlp/cos", "eval/mlp/norm_act", "eval/mlp_pair/cos"}, "acts100": set()}
REALACT_COLS = {"eval/realact/cos", "eval/realact_early/cos", "eval/realact_mid/cos", "eval/indist_realact/cos"}
RL_STEPS = [25, 50, 100]
# headline figure: the non-duplicate columns only, plain-language rows, one-line title
CORE_COLS = [("eval/mean_all", "mean of\n11 families"), ("eval/realact/cos", "real acts"), ("eval/realact_long/cos", "real acts\nlong ctx"),
             ("eval/sae/norm_act", "SAE\nnorm act"), ("eval/sae/rank1_frac", "SAE\nrank-1"), ("eval/sae/verbalized_frac", "SAE\nverbalized"), ("eval/bsf/cos", "BSF"), ("eval/cluster/cos", "cluster\nprobes"),
             ("eval/jlens/cos", "J-lens"), ("eval/mlp/cos", "MLP\nneurons"), ("eval/mlp/norm_act", "MLP\nfire-back"), ("eval/random/cos", "random\n(control)")]
ARM_ROW = {"acts100": "real acts only (control)", "acts_sae": "+ SAE features", "acts_bsf": "+ BSF blocks", "acts_cluster": "+ cluster probes",
           "acts_realact_long": "+ long-context acts", "acts_mlp": "+ MLP neurons"}

# palette (dataviz reference instance): diverging blue <-> red with a neutral gray midpoint; categorical slot 1 blue, slot 2 orange
BLUE, ORANGE, GRAY_MID, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#f0efec", "#0b0b0b", "#898781", "#e1e0d9"
DIVERGING = LinearSegmentedColormap.from_list("div_rb", ["#c0392b", "#e34948", "#f3b7a8", GRAY_MID, "#9ec5f4", "#2a78d6", "#0d366b"])


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def fetch(vol_path, local):
    if os.path.exists(local):
        return True
    os.makedirs(os.path.dirname(local), exist_ok=True)
    r = sh([MODAL, "volume", "get", "maemm-data", vol_path, local, "--force"])
    return r.returncode == 0 and os.path.exists(local)


def fetch_all(refresh_latest=True):
    """Pull every eval json + bank stats that exist on the volume (cheap; re-pulls nothing already local)."""
    env = dict(os.environ, MODAL_PROFILE=os.environ.get("MODAL_PROFILE", "safety-sahan"))
    os.environ.update(env)
    got = []
    if fetch("/eval_ckpt/uplift_init_realact23m/ckpt_0.json", f"{RAW}/init/ckpt_0.json"):
        got.append("init")
    for a in ARMS:
        if fetch(f"/banks/uplift_{a}/build_stats.json", f"{RAW}/banks/{a}.json"):
            got.append(f"bank:{a}")
        if fetch(f"/eval_ckpt/uplift_sft_{a}/ckpt_0.json", f"{RAW}/sft/{a}.json"):
            got.append(f"sft:{a}")
        for k in RL_STEPS:
            if fetch(f"/eval_ckpt/uplift_rl_{a}/ckpt_{k}.json", f"{RAW}/rl/{a}_{k}.json"):
                got.append(f"rl:{a}:{k}")
        if fetch(f"/sft_mix/uplift_sft_{a}/run_meta.json", f"{RAW}/sft_meta/{a}.json"):
            got.append(f"sftmeta:{a}")
    for name, path in REFS.items():
        if fetch(path, f"{RAW}/refs/{name}.json"):
            got.append(f"ref:{name}")
    return got


# reference checkpoints from earlier runs (v1 eval cache -> no mlp columns), for the "was the midtrain worth it" comparison
REFS = {"rl_A_from_init_no_midtrain_ckpt100": "/eval_ckpt/rl_A_randctx/ckpt_100.json",           # pristine init, realact random-ctx bank, 16x256, lr 7e-6, warmup 25
        "rl_B_rp500k_init_ckpt100": "/eval_ckpt/rl_B_randctx_probes/ckpt_100.json",                # 500k realact+probes init, acts+probes bank
        "rl_C_mix1m_midtrain_ckpt100": "/eval_ckpt/rl_C_mix1m/ckpt_100.json",                      # init + 1.1M all-families midtrain, all-families bank, lr 7e-6
        "sft_mix1m_midtrain_final": "/eval_ckpt/sft_mix1m_from_realact23m/ckpt_5000.json"}         # the 1.1M all-families midtrain final (before RL-C)
REF_LABEL = {"rl_A_from_init_no_midtrain_ckpt100": "RL-A @100: same init, NO midtrain, random-ctx real-acts bank (16x256, lr 7e-6, warmup 25)",
             "rl_B_rp500k_init_ckpt100": "RL-B @100: 500k realact+probes init, real-acts + probes bank (16x256, lr 7e-6)",
             "rl_C_mix1m_midtrain_ckpt100": "RL-C @100: same init + 1.1M all-families midtrain (lr 1e-4), all-families bank (16x256, lr 7e-6)",
             "sft_mix1m_midtrain_final": "1.1M all-families midtrain final (lr 1e-4, 4279 steps) = RL-C's init"}


def _derive(m):
    """add eval/sae/verbalized_frac = 1 - unverbalized_frac (share of held-out SAE features the inverter CAN make fire)."""
    if m and m.get("eval/sae/unverbalized_frac") is not None and m.get("eval/sae/verbalized_frac") is None:
        m["eval/sae/verbalized_frac"] = 1.0 - m["eval/sae/unverbalized_frac"]
    return m


def load_metrics(path):
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    m = d.get("metrics", d)
    return {"ckpt": d.get("ckpt"), "ckpt_step": d.get("ckpt_step"), "protocol": d.get("protocol"),
            "metrics": {k: v for k, v in m.items() if isinstance(v, (int, float))}}


def wandb_runs(no_wandb):
    """RL training curves + ids by run name (rl_uplift_<arm>) and the eval run ids."""
    out = {}
    if no_wandb:
        return out
    try:
        import wandb
        api = wandb.Api()
    except Exception as e:  # noqa
        print(f"[plot] wandb unavailable: {e}")
        return out
    for a in ARMS:
        rec = {}
        try:
            runs = list(api.runs(PROJ, filters={"display_name": f"rl_uplift_{a}"}, order="-created_at"))
            if runs:
                r = runs[0]
                h = r.history(keys=["reward/mean", "policy/entropy", "rollout/len_mean", "grad_norm"], pandas=False, samples=400)
                rec["rl_train"] = {"id": r.id, "state": r.state, "url": r.url,
                                   "points": [{"step": int(x["_step"]), "reward": x.get("reward/mean"), "entropy": x.get("policy/entropy"),
                                               "len": x.get("rollout/len_mean"), "grad_norm": x.get("grad_norm")}
                                              for x in h if x.get("reward/mean") is not None]}
            for kind, name in (("rl_eval", f"uplift_rl_{a}_eval"), ("sft_eval", f"uplift_sft_{a}_eval")):
                rr = list(api.runs(PROJ, filters={"display_name": name}, order="-created_at"))
                if rr:
                    rec[kind] = {"id": rr[0].id, "url": rr[0].url}
            meta = f"{RAW}/sft_meta/{a}.json"
            if os.path.exists(meta):
                wid = json.load(open(meta)).get("wandb_id")
                if wid:
                    try:
                        r = api.run(f"{PROJ}/{wid}")
                        h = r.history(keys=["train/loss"], pandas=False, samples=400)
                        pts = [{"step": int(x["_step"]), "loss": x.get("train/loss")} for x in h if x.get("train/loss") is not None]
                        if not pts:
                            h = r.history(keys=["loss"], pandas=False, samples=400)
                            pts = [{"step": int(x["_step"]), "loss": x.get("loss")} for x in h if x.get("loss") is not None]
                        rec["sft_train"] = {"id": wid, "state": r.state, "url": r.url, "points": pts}
                    except Exception as e:  # noqa
                        rec["sft_train"] = {"id": wid, "error": str(e)[:200]}
        except Exception as e:  # noqa
            rec["error"] = str(e)[:200]
        out[a] = rec
    return out


def assemble():
    init = load_metrics(f"{RAW}/init/ckpt_0.json")
    if init:
        _derive(init.get("metrics"))
    table = {"init": init, "arms": {}}
    for a in ARMS:
        rl = {str(k): load_metrics(f"{RAW}/rl/{a}_{k}.json") for k in RL_STEPS}
        bank = json.load(open(f"{RAW}/banks/{a}.json")) if os.path.exists(f"{RAW}/banks/{a}.json") else None
        table["arms"][a] = {"label": ARM_LABEL[a], "bank": bank, "sft": load_metrics(f"{RAW}/sft/{a}.json"), "rl": rl}
    return table


def matrix(table, stage, ref):
    """rows = arms, cols = COLS; value = metric(stage) - metric(ref) (ref: 'init' | 'acts100' | None). NaN where missing."""
    M = np.full((len(ARMS), len(COLS)), np.nan)
    for i, a in enumerate(ARMS):
        m = stage_metrics(table, a, stage)
        if m is None:
            continue
        for j, (k, _, _) in enumerate(COLS):
            v = m.get(k)
            if v is None:
                continue
            if ref == "init":
                r = (table["init"] or {}).get("metrics", {}).get(k)
            elif ref == "acts100":
                rm = stage_metrics(table, "acts100", stage)
                r = rm.get(k) if rm else None
            else:
                r = 0.0
            if r is not None:
                M[i, j] = v - r
    return M


def stage_metrics(table, arm, stage):
    s = table["arms"][arm]
    if stage == "sft":
        return _derive((s["sft"] or {}).get("metrics"))
    return _derive((s["rl"].get(str(stage)) or {}).get("metrics"))


def heatmap(M, title, subtitle, fname, vlim=0.10, ref_row=None):
    fig, ax = plt.subplots(figsize=(15.5, 5.2))
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)
    show = np.ma.masked_invalid(M)
    im = ax.imshow(show, cmap=DIVERGING, norm=norm, aspect="auto")
    ax.set_xticks(range(len(COLS))); ax.set_xticklabels([c[1] for c in COLS], fontsize=8.6)
    ax.set_yticks(range(len(ARMS))); ax.set_yticklabels([ARM_LABEL[a] for a in ARMS], fontsize=9.2)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    # 2px surface gaps between cells + own-family outline
    for i in range(len(ARMS) + 1):
        ax.axhline(i - 0.5, color="white", lw=2)
    for j in range(len(COLS) + 1):
        ax.axvline(j - 0.5, color="white", lw=2)
    for i, a in enumerate(ARMS):
        for j, (k, _, hib) in enumerate(COLS):
            v = M[i, j]
            if np.isnan(v):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=8, color=MUTED)
                continue
            strong = abs(v) > 0.6 * vlim
            ax.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=8.6, color="white" if strong else INK)
            if k in OWN[a]:
                ax.add_patch(plt.Rectangle((j - 0.45, i - 0.45), 0.9, 0.9, fill=False, ec=INK, lw=2.6, zorder=6))
    if ref_row is not None:
        ax.add_patch(plt.Rectangle((-0.5, ref_row - 0.5), len(COLS), 1, fill=False, ec=MUTED, lw=1.2, ls=(0, (2, 2))))
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cb.set_label("delta (clipped at ±%.2f for color; numbers exact)" % vlim, fontsize=8.5)
    cb.ax.tick_params(labelsize=8)
    import textwrap
    ax.set_title("\n".join(textwrap.wrap(title, 150)), fontsize=12, loc="left", color=INK, pad=30)
    ax.text(0, 1.012, "\n".join(textwrap.wrap(subtitle, 190)), transform=ax.transAxes, fontsize=8.6, color="#52514e", va="bottom")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/{fname}.{ext}", dpi=170)
    plt.close(fig)


def all_families_row(table, budget):
    """Reference row: the everything-mixed run (RL-E, 1.1M all-families midtrain -> RL on the 7-family bank) at the same RL step (100),
    as delta vs the same init. Fetched from wandb; None if unavailable."""
    ref = budget.get("reference_all_families")
    if not ref:
        return None
    try:
        import wandb
        rs = list(wandb.Api().runs(PROJ, filters={"display_name": ref["rl"]["eval_run"]}, order="-created_at"))
        row = _derive(dict(next(r for r in rs[0].history(pandas=False) if r.get("ckpt_step") == ref["rl"]["ckpt"] and r.get("eval/mean_all") is not None)))
    except Exception as e:
        print("[plot] all-families reference row unavailable:", e); return None
    init = (table["init"] or {}).get("metrics", {})
    return {"label": f"all 7 families mixed*  ·  {ref['midtrain']['n_examples'] / 1e6:.1f}M acts",
            "delta": {k: (row[k] - init[k]) if (row.get(k) is not None and init.get(k) is not None) else np.nan for k, _ in CORE_COLS},
            "abs": {k: row.get(k) for k, _ in CORE_COLS}, "source": ref}


TRAJ_COLS = [("eval/mean_all", "mean of 11 families"), ("eval/realact/cos", "real acts"), ("eval/realact_long/cos", "real acts, long ctx"), ("eval/sae/norm_act", "SAE norm act"),
             ("eval/sae/rank1_frac", "SAE rank-1"), ("eval/sae/verbalized_frac", "SAE verbalized (share that fires)"), ("eval/bsf/cos", "BSF"), ("eval/cluster/cos", "cluster probes"),
             ("eval/mlp/norm_act", "MLP fire-back")]
ARM_COLOR = {"acts100": "#7a7a7a", "acts_sae": "#b5542b", "acts_bsf": "#4a6fa5", "acts_cluster": "#2a7f62", "acts_realact_long": "#9a7fc4", "acts_mlp": "#c99a2e"}


def trajectories(table, fname, budget):
    """Every evaluated checkpoint of every arm as delta vs the shared init, against RL rollouts consumed (0 = right after the midtrain),
    with the everything-mixed run (RL-E) at ITS checkpoints as a reference — so arms can be compared at matched RL compute."""
    init = (table["init"] or {}).get("metrics", {})
    rps = budget["rl"]["directions_per_step"] * budget["rl"]["samples_per_direction"]      # 2048 rollouts / step for the arms
    out = {"x_is": "RL rollouts consumed (0 = after the 200k midtrain, before RL)", "arms": {}, "reference": None}
    for a in ARMS:
        pts = []
        m = stage_metrics(table, a, "sft")
        if m: pts.append({"rl_step": 0, "rollouts": 0, **{k: (m[k] - init[k]) if (m.get(k) is not None and init.get(k) is not None) else None for k, _ in TRAJ_COLS}})
        for st in RL_STEPS:
            m = stage_metrics(table, a, st)
            if m: pts.append({"rl_step": st, "rollouts": st * rps, **{k: (m[k] - init[k]) if (m.get(k) is not None and init.get(k) is not None) else None for k, _ in TRAJ_COLS}})
        out["arms"][a] = {"label": ARM_ROW[a], "points": pts}
    ref = budget.get("reference_all_families"); ref_pts = []
    if ref:
        try:
            import wandb
            rs = list(wandb.Api().runs(PROJ, filters={"display_name": ref["rl"]["eval_run"]}, order="-created_at"))
            for r in sorted([r for r in rs[0].history(pandas=False) if r.get("ckpt_step") is not None and r.get("eval/mean_all") is not None], key=lambda r: r["ckpt_step"]):
                if r["ckpt_step"] <= 100:
                    r = _derive(dict(r))
                    ref_pts.append({"rl_step": int(r["ckpt_step"]), "rollouts": int(r["ckpt_step"]) * ref["rl"]["rollouts_per_step"],
                                    **{k: (r[k] - init[k]) if (r.get(k) is not None and init.get(k) is not None) else None for k, _ in TRAJ_COLS}})
            out["reference"] = {"label": "all 7 families mixed (RL-E; 1.1M midtrain, 4096 rollouts/step)", "points": ref_pts}
        except Exception as e:
            print("[plot] trajectory reference unavailable:", e)
    json.dump(out, open(f"{OUT}/data/trajectories.json", "w"), indent=1)
    fig, axes = plt.subplots(3, 3, figsize=(15, 10.5), sharex=True)
    for ax, (k, title) in zip(axes.flat, TRAJ_COLS):
        for a in ARMS:
            pts = [p for p in out["arms"][a]["points"] if p.get(k) is not None]
            ax.plot([p["rollouts"] / 1e3 for p in pts], [p[k] for p in pts], "o-", color=ARM_COLOR[a], ms=4, lw=1.6, label=ARM_ROW[a])
        pts = [p for p in ref_pts if p.get(k) is not None]
        if pts:
            ax.plot([p["rollouts"] / 1e3 for p in pts], [p[k] for p in pts], "s--", color="#111111", ms=4, lw=1.4, label=out["reference"]["label"])
        ax.axhline(0, color="#999999", lw=0.8); ax.set_title(title, fontsize=10); ax.grid(alpha=0.25); ax.tick_params(labelsize=8.5)
        ax.set_xticks([0, 51, 102, 205, 410]); ax.set_xticklabels(["0\n(after\nmidtrain)", "51k", "102k", "205k", "410k"], fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("RL rollouts consumed", fontsize=9.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("Δ vs shared init", fontsize=9.5)
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=8.8, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Every checkpoint: the 200k midtrain alone lowers every family (x = 0), RL recovers and then lifts them within 50k–100k rollouts;\n"
                 "at matched rollouts the everything-mixed run (dashed, 5x more midtrain tokens) tracks the best single-family arm rather than beating it", fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/{fname}.{ext}", dpi=170)
    plt.close(fig)


def heatmap_clean(table, fname, budget):
    """Headline: delta vs the shared init after midtrain + RL@100, core columns only, per-row training budget in the label,
    plus (below a gap) the everything-mixed reference run at the same RL step."""
    import textwrap
    cols = [k for k, _ in CORE_COLS]; full = [k for k, _, _ in COLS]
    M_full = matrix(table, 100, "init"); M = M_full[:, [full.index(k) for k in cols]]
    extra = all_families_row(table, budget)
    mt = budget["midtrain"]
    def acts_label(a):
        fams = mt[a]["per_family"]; n_real = fams.get("realact", {}).get("n", 0); n_x = sum(v["n"] for f, v in fams.items() if f != "realact")
        return f"{ARM_ROW[a]}  ·  {n_real / 1e3:.0f}k real + {n_x / 1e3:.0f}k acts" if n_x else f"{ARM_ROW[a]}  ·  {n_real / 1e3:.0f}k acts"
    rows = [acts_label(a) for a in ARMS]
    if extra:
        M = np.vstack([M, np.full((1, len(cols)), np.nan), np.array([[extra["delta"][k] for k in cols]])])   # blank spacer row, then the reference
        rows += ["", extra["label"]]
        json.dump({"row": extra["label"], "delta_vs_init": {k: (None if np.isnan(v) else float(v)) for k, v in extra["delta"].items()}, "absolute": extra["abs"],
                   "source": extra["source"]}, open(f"{OUT}/data/all_families_reference_row.json", "w"), indent=1)
    n_rows = M.shape[0]
    fig = plt.figure(figsize=(14.5, 6.0 + (0.9 if extra else 0)))
    ax = fig.add_axes([0.215, 0.15, 0.66, 0.62])   # [left, bottom, width, height] — fixed so the cells stay wide
    vlim = 0.10
    im = ax.imshow(np.ma.masked_invalid(M), cmap=DIVERGING, norm=TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim), aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([l for _, l in CORE_COLS], fontsize=9.5)
    ax.set_yticks(range(n_rows)); ax.set_yticklabels(rows, fontsize=10.5)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for i in range(n_rows + 1):
        ax.axhline(i - 0.5, color="white", lw=2)
    for j in range(len(cols) + 1):
        ax.axvline(j - 0.5, color="white", lw=2)
    if extra:
        ax.add_patch(plt.Rectangle((-0.5, len(ARMS) - 0.5), len(cols), 1, color="white", zorder=5))   # blank the spacer row
    for i in range(n_rows):
        a = ARMS[i] if i < len(ARMS) else None
        for j, k in enumerate(cols):
            v = M[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=10.5, color="white" if abs(v) > 0.6 * vlim else INK, zorder=7)
            if a and k in OWN[a]:
                ax.add_patch(plt.Rectangle((j - 0.45, i - 0.45), 0.9, 0.9, fill=False, ec=INK, lw=2.4, zorder=6))
    cax = fig.add_axes([0.89, 0.15, 0.012, 0.62])
    cb = fig.colorbar(im, cax=cax); cb.set_label("Δ vs init", fontsize=9); cb.ax.tick_params(labelsize=8)
    gen = budget["rl"]["generated_tokens_M"]; lo, hi = min(gen.values()), max(gen.values())
    fig.text(0.02, 0.93 if not extra else 0.94, "Midtrain + 100 RL steps: structured direction families lift SAE and MLP fidelity, real acts improve in every mixed arm, J-lens never moves",
             fontsize=12.5, color=INK, va="center")
    mt_lo, mt_hi = min(v["target_tokens"] for v in mt.values()) / 1e6, max(v["target_tokens"] for v in mt.values()) / 1e6
    sub = (f"Δ held-out score vs the shared init (23M real-act SFT). Each row: midtrain on 200k (direction, text) pairs — 100k real activations + 100k of the named family "
           f"({mt_lo:.0f}–{mt_hi:.0f}M target tokens) → 100 RL steps (CISPO loss, group-relative advantages, 16 samples/direction) × 2048 rollouts ({lo:.0f}–{hi:.0f}M generated tokens). Black box = the arm's own family.")
    if extra:
        r = extra["source"]
        sub += (f"  *Reference, not part of the matrix: midtrain on the full 1.1M all-families mix (≈{r['midtrain']['target_tokens_est'] / 1e6:.0f}M target tokens) → 100 RL steps at 2x the batch "
                f"(4096 rollouts/step, ≈{r['rl']['generated_tokens_M']:.0f}M generated tokens) on the 7-family bank incl. MLP neurons; same init lineage.")
    fig.text(0.02, 0.855 if not extra else 0.875, "\n".join(textwrap.wrap(sub, 165)), fontsize=9, color="#52514e", va="top")
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/{fname}.{ext}", dpi=170)
    plt.close(fig)


def family_bars(table, stage, title, fname):
    init = (table["init"] or {}).get("metrics", {})
    cols = [c for c in COLS if c[0] not in ("eval/sae/rank1_frac",)]
    n = len(cols); ncol = 4; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(15.5, 3.1 * nrow))
    for ax, (k, lab, hib) in zip(axes.flat, cols):
        vals = []
        for a in ARMS:
            m = stage_metrics(table, a, stage)
            vals.append(np.nan if not m or k not in m else m[k])
        x = np.arange(len(ARMS))
        colors = [ORANGE if a == "acts100" else BLUE for a in ARMS]
        ax.bar(x, [0 if np.isnan(v) else v for v in vals], width=0.62, color=colors, edgecolor="none")
        for xi, v in zip(x, vals):
            if np.isnan(v):
                ax.text(xi, 0.002, "n/a", ha="center", va="bottom", fontsize=7.5, color=MUTED)
        if k in init:
            ax.axhline(init[k], color=INK, lw=1.2, ls=(0, (3, 2)))
        ax.set_title(lab.replace("\n", " "), fontsize=9.6, loc="left")
        ax.set_xticks(x); ax.set_xticklabels([ARM_SHORT[a] for a in ARMS], fontsize=8, rotation=0)
        ax.tick_params(axis="y", labelsize=8, length=0); ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color("#c3c2b7")
        finite = [v for v in vals if not np.isnan(v)] + ([init[k]] if k in init else [])
        if finite:
            lo, hi = min(finite), max(finite)
            pad = max(0.02, 0.25 * (hi - lo))
            ax.set_ylim(max(0, lo - pad) if k != "eval/random/cos" else 0, hi + pad)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    import textwrap
    fig.suptitle("\n".join(textwrap.wrap(title, 160)), fontsize=12, x=0.01, ha="left", color=INK)
    fig.text(0.01, 0.945, "\n".join(textwrap.wrap("orange = 100%-real-activations control arm; blue = 50/50 arms; dashed line = the common init before any midtraining. "
             "Bars: 512 held-out directions per family, best-of-4 samples, cosine of the clean model's layer-42 activation (SAE: normalized feature activation; "
             "MLP fire-back: normalized neuron activation; random: lower is better).", 200)), fontsize=8.3, color="#52514e", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/{fname}.{ext}", dpi=170)
    plt.close(fig)


def rl_train_plot(runs, fname):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 3.8))
    series = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]   # categorical slots 1-6, fixed order by arm
    for (k, ttl) in zip(("reward", "entropy", "len"), ("training reward (max cos over the last-5 window, minus length penalty)",
                                                     "policy entropy (nats / token)", "mean rollout length (tokens)")):
        ax = axes[["reward", "entropy", "len"].index(k)]
        for a, c in zip(ARMS, series):
            pts = (runs.get(a, {}).get("rl_train") or {}).get("points", [])
            xs = [p["step"] for p in pts if p.get(k) is not None]; ys = [p[k] for p in pts if p.get(k) is not None]
            if xs:
                ax.plot(xs, ys, color=c, lw=1.8, label=ARM_SHORT[a])
                ax.text(xs[-1] + 1, ys[-1], ARM_SHORT[a], fontsize=7.5, color=c, va="center")
        ax.set_title(ttl, fontsize=9.5, loc="left"); ax.set_xlabel("RL step (128 directions x 16 samples each)", fontsize=8.5)
        ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(labelsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8.5, ncol=6, loc="lower center", bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("RL raises every arm's training reward within 100 steps; the SAE / probe / MLP banks are the hardest (lowest reward)\n"
                 "yet transfer the most, so a bank's reward level does not predict its cross-uplift", fontsize=11.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.90))
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/{fname}.{ext}", dpi=170)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--refresh", action="store_true", help="delete the local raw copies first (re-pull everything)")
    a = ap.parse_args()
    os.makedirs(f"{OUT}/data", exist_ok=True); os.makedirs(RAW, exist_ok=True)
    if a.refresh:
        for p in glob.glob(f"{RAW}/**/*.json", recursive=True):
            os.remove(p)
    if not a.no_fetch:
        got = fetch_all()
        print(f"[plot] fetched/present: {got}")
    table = assemble()
    runs = wandb_runs(a.no_wandb)
    have_sft = [x for x in ARMS if table["arms"][x]["sft"]]
    have_rl = [x for x in ARMS if table["arms"][x]["rl"].get("100")]
    print(f"[plot] init {'OK' if table['init'] else 'MISSING'} | sft evals {have_sft} | rl@100 evals {have_rl}")

    M_sft = matrix(table, "sft", "init"); M_rl = matrix(table, 100, "init"); M_x = matrix(table, 100, "acts100")
    M_sft_x = matrix(table, "sft", "acts100")
    json.dump({"description": "cross-uplift matrix: every evaluated checkpoint (init, per-arm SFT final, per-arm RL ckpts 25/50/100); "
                              "metrics = eval/eval_ckpt_daemon.py output (512 dirs/family, bo4, T=1, 16-64 tokens, v2 cache with mlp/mlp_pair)",
               "columns": [{"key": k, "label": l.replace("\n", " "), "higher_is_better": h} for k, l, h in COLS],
               "own_family_columns": {a: sorted(v) for a, v in OWN.items()}, "realact_columns_in_every_arm": sorted(REALACT_COLS),
               "table": table, "generated": time.time()}, open(f"{OUT}/data/eval_table.json", "w"), indent=1)
    json.dump({"columns": [k for k, _, _ in COLS], "arms": ARMS,
               "delta_sft_vs_init": {a: {k: (None if np.isnan(M_sft[i, j]) else float(M_sft[i, j])) for j, (k, _, _) in enumerate(COLS)} for i, a in enumerate(ARMS)},
               "delta_rl100_vs_init": {a: {k: (None if np.isnan(M_rl[i, j]) else float(M_rl[i, j])) for j, (k, _, _) in enumerate(COLS)} for i, a in enumerate(ARMS)},
               "delta_rl100_vs_acts100": {a: {k: (None if np.isnan(M_x[i, j]) else float(M_x[i, j])) for j, (k, _, _) in enumerate(COLS)} for i, a in enumerate(ARMS)},
               "delta_sft_vs_acts100": {a: {k: (None if np.isnan(M_sft_x[i, j]) else float(M_sft_x[i, j])) for j, (k, _, _) in enumerate(COLS)} for i, a in enumerate(ARMS)}},
              open(f"{OUT}/data/deltas.json", "w"), indent=1)
    json.dump({"arms": {a: table["arms"][a]["bank"] for a in ARMS}}, open(f"{OUT}/data/banks.json", "w"), indent=1)
    # CSVs: long (arm, stage, metric, absolute, delta_vs_init, delta_vs_control) + wide heatmap (RL@100 delta vs init, core columns)
    import csv
    init_m = _derive((table["init"] or {}).get("metrics", {})) or {}
    with open(f"{OUT}/data/uplift_matrix_long.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["arm", "arm_label", "stage", "rl_step", "metric", "value", "init_value", "delta_vs_init", "control_value_same_stage", "delta_vs_control"])
        for a in ARMS:
            for stage in ["sft"] + RL_STEPS:
                m = stage_metrics(table, a, stage); c = stage_metrics(table, "acts100", stage)
                if not m:
                    continue
                for k, _, _ in COLS:
                    v = m.get(k); iv = init_m.get(k); cv = (c or {}).get(k)
                    w.writerow([a, ARM_LABEL[a], "after_midtrain" if stage == "sft" else "rl", 0 if stage == "sft" else stage, k,
                                v, iv, None if (v is None or iv is None) else v - iv, cv, None if (v is None or cv is None) else v - cv])
    with open(f"{OUT}/data/uplift_heatmap_wide.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["arm", "arm_label"] + [k for k, _ in CORE_COLS])
        for i, a in enumerate(ARMS):
            w.writerow([a, ARM_LABEL[a]] + [(None if np.isnan(M_rl[i, [k for k, _, _ in COLS].index(k2)]) else round(float(M_rl[i, [k for k, _, _ in COLS].index(k2)]), 4)) for k2, _ in CORE_COLS])
    json.dump({n: {"label": REF_LABEL[n], "source": p, **(load_metrics(f"{RAW}/refs/{n}.json") or {})} for n, p in REFS.items()},
              open(f"{OUT}/data/references.json", "w"), indent=1)
    json.dump(runs, open(f"{OUT}/data/runs.json", "w"), indent=1)

    heatmap(M_sft, "The 200k midtrain (lr 1e-4, one epoch) LOWERS every held-out family in every arm — even the 100%-real-activations control — "
                   "except the arm whose second half is long-context activation windows; SAE and probe rows keep only their own family",
            "delta vs the common init (23M realact-only SFT final) after 1 epoch of SFT on 100k real acts + 100k of family X (control: 200k real acts). "
            "Rows = training arm, columns = held-out eval family; outlined cells = the arm's own X family. The shared real-acts half is a short-context "
            "prefix harvest (exact re-encodable pairs), a different target style from the init's activation windows.",
            "uplift_sft_delta_heatmap")
    heatmap(M_rl, "100 RL steps on the same bank undo the midtrain damage in every arm and lift real-activation fidelity above the init; "
                  "every structured-direction arm (SAE, BSF, probes, MLP) also lifts SAE-feature fidelity and MLP fire-back far above the init",
            "delta vs the common init after SFT + RL@100 (CISPO/ScaleRL cosine reward over the last 5 tokens, 128 directions x 16 samples per step, lr 1e-5, "
            "10 warmup steps). Outlined = the arm's own X family.",
            "uplift_rl_delta_heatmap")
    heatmap(M_x, "Cross-uplift at RL@100 vs the 100%-real-activations control: every structured-direction family (SAE, BSF, probes, MLP) transfers to SAE-feature "
                 "fidelity (+0.12 to +0.38) and MLP fire-back (+0.21 to +0.40) and lifts real activations (+0.02 to +0.04); long-context windows lift only activations; "
                 "nothing moves J-lens",
            "delta vs the acts100 control arm at RL@100 (same init, same step count, same recipe; only the 100k X half of the bank differs). "
            "Outlined = the arm's own X family; real-activation columns are in-distribution for every arm.",
            "uplift_rl_vs_control_heatmap", ref_row=0)
    if os.path.exists(f"{OUT}/data/budget.json"):
        heatmap_clean(table, "uplift_rl_delta_clean", json.load(open(f"{OUT}/data/budget.json")))
        trajectories(table, "uplift_trajectories", json.load(open(f"{OUT}/data/budget.json")))
    family_bars(table, "sft", "Absolute held-out scores after the 200k midtrain, per eval family: the control (orange) vs each 50/50 arm (blue) vs the init (dashed)",
                "uplift_family_bars_sft")
    family_bars(table, 100, "Absolute held-out scores after midtrain + RL@100, per eval family: the control (orange) vs each 50/50 arm (blue) vs the init (dashed)",
                "uplift_family_bars_rl")
    if any((runs.get(x, {}).get("rl_train") or {}).get("points") for x in ARMS):
        rl_train_plot(runs, "uplift_rl_train")
    print(f"[plot] wrote {OUT}/data/*.json + figures")


if __name__ == "__main__":
    main()
