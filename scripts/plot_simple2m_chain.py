#!/usr/bin/env python3
"""SIMPLE2M chain report: ONE full-FT SFT from the base model (exact 50/50: 8-64-token-context activations with FULL-context targets +
2M-SAE encoder/decoder max-act rows) -> full-parameter RL (whole-span cosine reward, even activations/SAE pool, contexts 64-2048).

Idempotent: re-mirrors the evaluator JSONs from the Modal volume (/data/eval_ckpt/<run>/ckpt_N.json -> ~/shared/overnight/eval_ckpt_json/),
re-pulls the SFT loss and RL dynamics from wandb, rewrites data/*.json + every figure (PNG + PDF), then runs the report folder's build_html.py.

    python scripts/plot_simple2m_chain.py            # refresh everything
    python scripts/plot_simple2m_chain.py --no-html --no-mirror
"""
import argparse
import glob
import json
import os
import subprocess
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(os.environ.get("MAEMM_SIMPLE2M_REPORT", "~/shared/reports/maemm-simple2m-chain")).expanduser()
MIRROR = Path("~/shared/overnight/eval_ckpt_json").expanduser()
IDS = Path("~/shared/overnight/simple2m/ids.json").expanduser()
REGISTRY = Path("~/shared/overnight/simple2m/doc_registry.json").expanduser()
PROJ = "celestedeschamphelaere-personal/maxact-fast"
SFT_RUN, RL_RUN = "simple2m_sft", "rl_simple2m_8x2048_anywin"
SFT_WANDB, RL_WANDB = "e6w71sth", "5ud9qiuh"
EFF_BATCH = 4096
# reference arms (same evaluator families; v2 cache = the same 11 cosine families + 131k sae; they lack the v3 slice families)
REFS = {
    "a":    {"run": "rl_fullparam_fft104m_mix5m_8x2048", "label": "arm A: midtrain init, 8x2048, last-5 reward", "init_run": "sft_mix5msft_midtrain_fft_from_fft104m", "init_step": 671, "color": "#7b3fb8", "ls": "-"},
    "d":    {"run": "rl_fullparam_fft104m_mix5m_8x2048_paperlr_anywin", "label": "arm D: midtrain init, 8x2048, paper optimizer (lr 5e-7), whole-span reward", "init_run": "sft_mix5msft_midtrain_fft_from_fft104m", "init_step": 671, "color": "#8a6d1f", "ls": (0, (6, 2))},
    "s":    {"run": "rl_fullparam_fft104m_sae2m_8x2048_anywin", "label": "arm S: 2M-SAE midtrain init, 8x2048, whole-span reward", "init_run": "sft_mix5m_sae2m_midtrain_fft_from_fft104m", "init_step": 1737, "color": "#1f8a70", "ls": "-."},
    "base": {"run": "rl_fullparam_fft104m_mix5m_8x512", "label": "the .425 arm: midtrain init, 8x512, last-5 reward", "init_run": "sft_mix5msft_midtrain_fft_from_fft104m", "init_step": 671, "color": "#2b6cb0", "ls": ":"},
}
THIS_C, SFT_C = "#c0392b", "#c0392b"
INK, INK2, GRID = "#2b2b2b", "#6b6b6b", "#e6e2dc"
COS_FAMS = ["realact", "realact_early", "realact_mid", "realact_long", "indist_realact", "indist_long", "bsf", "cluster", "indist_probe", "jlens"]
FIRE = [("eval/sae/fired", "131k SAE features fired (gate)"), ("eval/sae2m_enc/fired", "held-out 2M-SAE ENCODER dirs fired"),
        ("eval/sae2m_dec/fired", "held-out 2M-SAE DECODER dirs fired"), ("eval/mlp/norm_act", "L42 MLP neurons norm_act")]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": GRID, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.titlecolor": INK, "figure.facecolor": "#fbfaf7", "axes.facecolor": "#fbfaf7", "savefig.facecolor": "#fbfaf7"})


def mirror(runs):
    for run in runs:
        (MIRROR / run).mkdir(parents=True, exist_ok=True)
        r = subprocess.run(["modal", "volume", "get", "--force", "maemm-data", f"/eval_ckpt/{run}/", str(MIRROR) + "/"], capture_output=True, text=True,
                           env={**os.environ, "MODAL_PROFILE": "safety-sahan"})
        print(f"[mirror] {run}: rc={r.returncode}")


def evals(run):
    rows = {}
    for f in glob.glob(str(MIRROR / run / "**" / "ckpt_*.json"), recursive=True):
        d = json.load(open(f)); m = d["metrics"]
        rows[int(d["ckpt_step"])] = {k: float(v) for k, v in m.items() if isinstance(v, (int, float)) and (k.startswith("eval/") or k.startswith("time/"))}
    return [dict(ckpt_step=s, **rows[s]) for s in sorted(rows)]


def wandb_series(run_id, keys):
    try:
        import wandb
        api = wandb.Api(timeout=120); r = api.run(f"{PROJ}/{run_id}")
        by = {}
        for src in (r.scan_history(keys=["_step"] + keys), r.history(samples=20000, pandas=False)):
            for x in src:
                s = x.get("_step")
                if s is None: continue
                d = by.setdefault(int(s), {})
                for k in keys:
                    v = x.get(k)
                    if isinstance(v, (int, float)) and v == v: d[k] = float(v)
        steps = sorted(by)
        return {"run_id": run_id, "state": r.state, "url": r.url, "step": steps, **{k: [by[s].get(k) for s in steps] for k in keys}}
    except Exception as e:  # noqa
        print(f"[wandb] {run_id}: {type(e).__name__}: {e}")
        return None


def wrap(s, n=118):
    import textwrap
    return "\n".join(textwrap.wrap(s, n))


def savefig(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=170, bbox_inches="tight"); fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight"); plt.close(fig)
    print("wrote", OUT / f"{name}.png")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--no-html", action="store_true"); ap.add_argument("--no-mirror", action="store_true"); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "data").mkdir(exist_ok=True)
    if not a.no_mirror:
        mirror([f"sft_{SFT_RUN}", RL_RUN])
    ids = json.load(open(IDS)); registry = json.load(open(REGISTRY)) if REGISTRY.exists() else None
    sft = evals(f"sft_{SFT_RUN}"); rl = evals(RL_RUN)
    refs = {}
    for k, R in REFS.items():
        ev = evals(R["run"]); init = [r for r in evals(R["init_run"]) if r["ckpt_step"] == R["init_step"]]
        refs[k] = {**R, "evals": ev, "init_mean_all": init[0]["eval/mean_all"] if init else None}
    sft_final = sft[-1] if sft else None
    fetched = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    sft_dyn = wandb_series(SFT_WANDB, ["loss", "lr", "ex_per_s"])
    rl_dyn = wandb_series(RL_WANDB, ["reward/mean", "policy/entropy", "grad_norm", "policy/sampler_abs_dlogp", "ratio/clipfrac", "rollout/len_mean", "time/step_s",
                                    "reward/peak_last_frac", "reward/peak_dist_mean", "policy/offpolicy_lag_steps"])
    rl_last_step = max(rl_dyn["step"]) if rl_dyn and rl_dyn["step"] else None
    best_rl = max(rl, key=lambda r: r["eval/mean_all"]) if rl else None

    # ---------------- headline: held-out fidelity through the chain ----------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.6), sharey=True, gridspec_kw={"width_ratios": [1, 1.25]})
    xs = [r["ckpt_step"] * EFF_BATCH / 1e6 for r in sft]; ys = [r["eval/mean_all"] for r in sft]
    ax1.plot(xs, ys, "-o", color=SFT_C, lw=2.2, ms=6, label="this chain: SFT from base (50/50 acts 8-64 ctx full-context targets + 2M-SAE enc/dec)")
    for x, y in zip(xs, ys): ax1.annotate(f"{y:.3f}", (x, y), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8, color=INK2)
    for k in ("a", "s"):
        if refs[k]["init_mean_all"] is not None:
            ax1.axhline(refs[k]["init_mean_all"], color=refs[k]["color"], ls=":", lw=1.1)
            ax1.annotate(f"{refs[k]['init_mean_all']:.3f} = {'5M-mix midtrain init (arms A/D/.425)' if k == 'a' else '2M-SAE midtrain init (arm S)'}", (xs[0], refs[k]["init_mean_all"]),
                         xytext=(2, 3), textcoords="offset points", fontsize=7.8, color=refs[k]["color"])
    ax1.set_xlabel("SFT: training rows seen (millions; 4,096 per step, exact 50/50 activations : SAE rows)"); ax1.set_ylabel("held-out fidelity: mean cosine over 10 direction families (eval cache v3 == v2 families)")
    ax1.set_title("SFT alone: held-out fidelity FALLS while train loss falls", fontsize=10, loc="left"); ax1.grid(color=GRID, lw=0.6); ax1.set_xlim(0, xs[-1] * 1.08 if xs else 8.5)
    # RL panel
    rx = [0] + [r["ckpt_step"] for r in rl]; ry = ([sft_final["eval/mean_all"]] if sft_final else []) + [r["eval/mean_all"] for r in rl]
    ax2.plot(rx, ry, "-o", color=THIS_C, lw=2.4, ms=6.5, label=f"this chain: RL from the SFT final (init {ry[0]:.3f}), 8x2048, lr 1e-6, WHOLE-span reward, even acts/SAE pool", zorder=5)
    for x, y in zip(rx[1:], ry[1:]): ax2.annotate(f"{y:.3f}", (x, y), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8.2, color=THIS_C, fontweight="bold")
    for k, R in refs.items():
        if not R["evals"]: continue
        xx = [0] + [r["ckpt_step"] for r in R["evals"]]; yy = [R["init_mean_all"]] + [r["eval/mean_all"] for r in R["evals"]]
        ax2.plot(xx, yy, ls=R["ls"], marker="s", ms=3.6, lw=1.4, color=R["color"], label=R["label"], alpha=0.95)
    ax2.set_xlabel("RL step (16,384 rollouts per step for the 8x2048 arms, 4,096 for 8x512; step 0 = the SFT init)"); ax2.grid(color=GRID, lw=0.6)
    ax2.set_xlim(-8, max(320, (max(rx) if rx else 300) + 20))
    ax2.set_title("RL repairs it and passes the midtrain-init arms at matched steps", fontsize=10, loc="left")
    ax2.legend(loc="lower right", fontsize=7.6, frameon=False)
    ax1.legend(loc="upper right", fontsize=7.6, frameon=False)
    ymin = min([min(ys)] + [min(r["eval/mean_all"] for r in R["evals"]) for R in refs.values() if R["evals"]] + [R["init_mean_all"] for R in refs.values() if R["init_mean_all"]]) - 0.02
    ax1.set_ylim(ymin, 0.45)
    claim = (f"A single 8M-row SFT from the base model (50/50 activations of 8-64 tokens of context with full-context targets + 2M-SAE encoder/decoder rows) DECLINES on held-out "
             f"fidelity from {ys[0]:.3f} ({xs[0]:.0f}M rows) to {ys[-1]:.3f} ({xs[-1]:.0f}M) while its train loss keeps falling; full-parameter RL from that final "
             + (f"reaches {best_rl['eval/mean_all']:.3f} at step {best_rl['ckpt_step']} (arm A {next((r['eval/mean_all'] for r in refs['a']['evals'] if r['ckpt_step'] == best_rl['ckpt_step']), float('nan')):.3f} at the same step from a {refs['a']['init_mean_all']:.3f} init)"
                if best_rl else "is running") + f" — Qwen3.6-27B activation-to-text inverter, 512 held-out directions per family, best-of-4 at T=1" + (f" (RL at step {rl_last_step} of 300)" if rl_last_step and rl_last_step < 300 else ""))
    fig.suptitle(wrap(claim, 150), fontsize=10.2, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9)); savefig(fig, "fidelity_chain")

    # ---------------- per-family cosine through the chain ----------------
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8), sharey=True, gridspec_kw={"width_ratios": [1, 1.25]})
    cols = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, fam in enumerate(COS_FAMS):
        k = f"eval/{fam}/cos"
        axes[0].plot(xs, [r.get(k, np.nan) for r in sft], "-o", ms=3.5, lw=1.5, color=cols[i], label=fam)
        axes[1].plot(rx, ([sft_final.get(k, np.nan)] if sft_final else []) + [r.get(k, np.nan) for r in rl], "-o", ms=3.5, lw=1.5, color=cols[i], label=fam)
    axes[0].set_xlabel("SFT rows seen (millions)"); axes[1].set_xlabel("RL step"); axes[0].set_ylabel("best-of-4 max-token cosine to the injected direction (512 held-out dirs per family)")
    axes[0].set_title("SFT: every real-activation family declines; long context most", fontsize=10, loc="left"); axes[1].set_title("RL: every family recovers past its SFT peak", fontsize=10, loc="left")
    for ax in axes: ax.grid(color=GRID, lw=0.6)
    axes[1].legend(fontsize=7.6, frameon=False, ncol=2, loc="lower right")
    fig.suptitle(wrap("Per-family held-out cosine through the chain: SFT (left, vs rows seen) then RL (right, vs step; step 0 = the SFT final) — real activations by context bucket "
                      "(early <=512 / mid 512-2048 / long 2048-8192 document position), BSF, cluster probes, J-lens, in-distribution families", 150), fontsize=10.2, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92)); savefig(fig, "families_chain")

    # ---------------- SAE / MLP firing through the chain ----------------
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4), sharey=True, gridspec_kw={"width_ratios": [1, 1.25]})
    fc = ["#2b6cb0", "#c0392b", "#8a6d1f", "#1f8a70"]
    for i, (k, lab) in enumerate(FIRE):
        axes[0].plot(xs, [r.get(k, np.nan) for r in sft], "-o", ms=4, lw=1.8, color=fc[i], label=lab)
        axes[1].plot(rx, ([sft_final.get(k, np.nan)] if sft_final else []) + [r.get(k, np.nan) for r in rl], "-o", ms=4, lw=1.8, color=fc[i], label=lab)
        for x, y in zip(rx[1:], [r.get(k, np.nan) for r in rl]): axes[1].annotate(f"{y:.2f}", (x, y), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7.5, color=fc[i])
    axes[0].set_xlabel("SFT rows seen (millions)"); axes[1].set_xlabel("RL step"); axes[0].set_ylabel("fraction of held-out features fired above the SAE's learned gate (MLP: norm_act)")
    axes[0].set_title("SFT: held-out 2M-SAE features barely fire (<10%)", fontsize=10, loc="left"); axes[1].set_title("RL: they fire at 38-44% by step 100", fontsize=10, loc="left")
    for ax in axes: ax.grid(color=GRID, lw=0.6); ax.set_ylim(0, 1)
    axes[1].legend(fontsize=7.8, frameon=False, loc="upper left")
    fig.suptitle(wrap("Feature-firing metrics through the chain — held-out 2M-SAE features (100k eval split, 512 scored; encoder columns and decoder rows injected, firing judged by the encoder "
                      "at gate 1.683), the 131k-SAE eval features (never trained on in this chain; gate 1.654) and layer-42 MLP neurons", 150), fontsize=10.2, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92)); savefig(fig, "heldout_firing")

    # ---------------- RL dynamics ----------------
    if rl_dyn:
        fig, axes = plt.subplots(2, 3, figsize=(14.5, 6.6)); axes = axes.ravel()
        panels = [("reward/mean", "reward (mean over 16,384 rollouts)"), ("policy/entropy", "policy entropy"), ("grad_norm", "grad norm (clip at 1)"),
                  ("policy/sampler_abs_dlogp", "sampler |Δ log p| (off-policy drift)"), ("ratio/clipfrac", "CISPO clip fraction"), ("rollout/len_mean", "rollout length (tokens)")]
        st = np.array(rl_dyn["step"])
        for ax, (k, lab) in zip(axes, panels):
            v = np.array([np.nan if x is None else x for x in rl_dyn[k]], float)
            ax.plot(st, v, color="#b8b2a8", lw=0.8)
            if len(v) >= 10:
                ker = np.ones(10) / 10; sm = np.convolve(np.nan_to_num(v, nan=np.nanmean(v)), ker, mode="valid"); ax.plot(st[9:], sm, color=THIS_C, lw=1.8)
            ax.set_title(lab, fontsize=9.5, loc="left"); ax.grid(color=GRID, lw=0.6); ax.set_xlabel("RL step")
            if k == "policy/sampler_abs_dlogp": ax.axhline(0.10, color="#c0392b", ls=":", lw=1); ax.annotate("watch level .10", (st[0], 0.10), xytext=(3, 3), textcoords="offset points", fontsize=7.5, color="#c0392b")
        fig.suptitle(wrap(f"RL training dynamics (run {RL_RUN}, thin = per step, thick = 10-step mean): reward climbs steadily; the one off-policy burst (|Δ log p| up to .26 at steps 30-33) "
                          "coincided with rollouts lengthening to 90 tokens and the sampler lag reaching 2 steps, and decayed within 10 steps as the length penalty pulled rollouts back to ~55 tokens", 150),
                     fontsize=10.2, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.9)); savefig(fig, "rl_dynamics")

    # ---------------- SFT loss ----------------
    if sft_dyn:
        fig, ax = plt.subplots(figsize=(9.5, 4.4)); st = np.array(sft_dyn["step"]); v = np.array([np.nan if x is None else x for x in sft_dyn["loss"]], float)
        ax.plot(st, v, color="#b8b2a8", lw=0.7)
        if len(v) >= 25: ker = np.ones(25) / 25; ax.plot(st[24:], np.convolve(np.nan_to_num(v, nan=np.nanmean(v)), ker, mode="valid"), color=SFT_C, lw=1.8, label="train NLL (25-step mean)")
        ax2 = ax.twinx(); ax2.plot([r["ckpt_step"] for r in sft], ys, "o-", color="#2b6cb0", lw=1.4, ms=5, label="held-out mean_all at each checkpoint")
        ax.set_xlabel("SFT optimizer step (4,096 rows each)"); ax.set_ylabel("train NLL of the target text"); ax2.set_ylabel("held-out mean_all", color="#2b6cb0")
        ax.grid(color=GRID, lw=0.6); h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, fontsize=8, frameon=False, loc="upper right")
        fig.suptitle(wrap(f"SFT overfits its own distribution: train NLL falls {np.nanmean(v[:20]):.2f} -> {np.nanmean(v[-20:]):.2f} while held-out fidelity falls {ys[0]:.3f} -> {ys[-1]:.3f} (8M rows, one epoch, lr 1e-5, batch 4,096, full fine-tune of Qwen3.6-27B)", 130), fontsize=10, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.9)); savefig(fig, "sft_loss")

    # ---------------- data ----------------
    dd = OUT / "data"
    json.dump({"generated_at": fetched, "run": SFT_RUN, "wandb": SFT_WANDB, "eff_batch": EFF_BATCH, "rows_per_ckpt": {r["ckpt_step"]: r["ckpt_step"] * EFF_BATCH for r in sft}, "evals": sft}, open(dd / "sft_evals.json", "w"), indent=1)
    json.dump({"generated_at": fetched, "run": RL_RUN, "wandb": RL_WANDB, "init": sft_final, "evals": rl, "best": best_rl, "last_train_step": rl_last_step}, open(dd / "rl_evals.json", "w"), indent=1)
    json.dump({"generated_at": fetched, "arms": {k: {kk: vv for kk, vv in R.items() if kk not in ("ls",)} for k, R in refs.items()}}, open(dd / "reference_arms.json", "w"), indent=1)
    if rl_dyn: json.dump({"generated_at": fetched, **rl_dyn}, open(dd / "rl_dynamics.json", "w"), indent=1)
    if sft_dyn: json.dump({"generated_at": fetched, **sft_dyn}, open(dd / "sft_loss.json", "w"), indent=1)
    comp = {"sft_mix": ids.get("compose_sft", {}), "rl_pool": ids.get("compose_rl", {}), "feature_split": ids.get("feature_split"), "doc_ranges": ids.get("doc_ranges"),
            "sft": {k: v for k, v in ids.get("sft", {}).items()}, "rl": {k: v for k, v in ids.get("rl", {}).items()}, "recipe_v2": ids.get("recipe_v2"),
            "banks": {k: {kk: vv for kk, vv in ids[k].items() if kk in ("out", "args", "spawned", "note")} for k in ("bank_sae2m_sft", "bank_sae2m_rl", "coll_sft_ctx8_64", "coll_rl_ctx64_2048") if k in ids}}
    json.dump({"generated_at": fetched, **comp}, open(dd / "data_composition.json", "w"), indent=1, default=str)
    if registry: json.dump(registry, open(dd / "doc_registry.json", "w"), indent=1)
    matched = {}
    for r in rl:
        s = r["ckpt_step"]; matched[s] = {"this": r["eval/mean_all"], **{k: next((x["eval/mean_all"] for x in R["evals"] if x["ckpt_step"] == s), None) for k, R in refs.items()}}
    summ = {"generated_at": fetched, "sft_curve": [(r["ckpt_step"], round(r["eval/mean_all"], 4)) for r in sft], "sft_final_mean_all": sft_final["eval/mean_all"] if sft_final else None,
            "sft_best": max(sft, key=lambda r: r["eval/mean_all"])["ckpt_step"] if sft else None, "rl_curve": [(r["ckpt_step"], round(r["eval/mean_all"], 4)) for r in rl],
            "rl_best": best_rl, "rl_last_train_step": rl_last_step, "rl_state": rl_dyn["state"] if rl_dyn else None, "matched_step_mean_all": matched,
            "ref_inits": {k: R["init_mean_all"] for k, R in refs.items()},
            "heldout_2m": {"sft_final": {k: sft_final.get(k) for k in ("eval/sae2m_enc/fired", "eval/sae2m_dec/fired", "eval/sae2m_enc/norm_act", "eval/sae2m_dec/norm_act")} if sft_final else None,
                           "rl": {r["ckpt_step"]: {k: r.get(k) for k in ("eval/sae2m_enc/fired", "eval/sae2m_dec/fired", "eval/sae2m_enc/norm_act", "eval/sae2m_dec/norm_act")} for r in rl}}}
    json.dump(summ, open(dd / "summary.json", "w"), indent=1, default=str)
    print(json.dumps({k: summ[k] for k in ("sft_curve", "rl_curve", "matched_step_mean_all")}))
    if not a.no_html and (OUT / "build_html.py").exists():
        subprocess.run(["python3", str(OUT / "build_html.py")], check=False)


if __name__ == "__main__":
    main()
