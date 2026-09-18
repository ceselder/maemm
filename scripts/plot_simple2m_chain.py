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
# Scorer (fix 2026-09-18): every cosine in the suite used to be cos(unit(h), d) with a RAW layer-42 activation against mean-free targets,
# compressing every number by ||h-mu||/||h|| (source text ~.5). The re-score with the centered scorer cos(unit(h-mu), d) lives under tags
# <run>_c. MAEMM_SCORER=legacy rebuilds the old report from the old tags (kept for the record; data_legacy_scorer/ is a frozen copy).
SCORER = os.environ.get("MAEMM_SCORER", "centered")
SUF = "_c" if SCORER == "centered" else ""
SCORER_NOTE = ("centered scorer cos(unit(h − μ), d), μ = layer-42 corpus mean" if SCORER == "centered"
               else "LEGACY raw-activation scorer cos(unit(h), d): every value compressed by ‖h−μ‖/‖h‖ (source text ≈ .5)")


SFT20_RUN, RL20_RUN = "sft_simple2m20m_sft", "rl_simple2m20m_8x2048_anywin"   # the SFT-scaling retest chain (2026-09-18): 20M rows, 80/20 acts/SAE
NATIVE_CENTERED = {"rl_simple2m_8x2048_anywin_centered", SFT20_RUN, RL20_RUN}
# distillation chain (2026-09-18): 2M fresh activations x best-of-16 rollouts of the step-250 policy (raw-reward selection) -> SFT from base -> RL (raw reward)
DIST_SFT_RAW, DIST_SFT_CEN = "sft_distill_fresh2m_bo16_sft_raw", "sft_distill_fresh2m_bo16_sft"      # same checkpoints, two evaluators
DIST_RL = "rl_distill_fresh2m_bo16_8x2048_anywin"                                                   # raw-reward RL from the distilled SFT final (raw evaluator)   # runs evaluated with the centered scorer from the start (no _c re-score tag)


def tagc(run):
    return run if run in NATIVE_CENTERED else run + SUF
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
# the three "yolo" RL variants launched 2026-09-17 18:57Z from the SAME SFT final (scripts/launchers/spawn_simple2m_rl_variants.py); equal 16,384 rollouts per step
VARIANTS = {
    "v16x1024":              {"run": "rl_simple2m_16x1024_anywin",         "short": "16x1024 whole-span", "label": "variant: 16 prompts x 1,024 rollouts, lr 1e-6, WHOLE-span reward",          "color": "#d98a2b", "ls": (0, (4, 2)), "reward_note": "whole-span reward (comparable to the main arm)"},
    "v8x2048_last16_lr5e-7": {"run": "rl_simple2m_8x2048_last16_lr5e-7",   "short": "last-16 reward, lr 5e-7", "label": "variant: 8x2048, lr 5e-7, reward on the LAST 16 tokens only",              "color": "#5c8fd6", "ls": (0, (1, 1.5)), "reward_note": "reward = max activation over the last 16 tokens (NOT comparable to whole-span reward)"},
    "centered_reward":       {"run": "rl_simple2m_8x2048_anywin_centered",    "short": "CENTERED reward (8x2048)", "label": "main recipe again with the FIXED reward cos(unit(h − μ), d); every other flag identical", "color": "#111111", "ls": "-", "reward_note": "centered whole-span reward (the fix); comparable to the main arm's centered re-score"},
    "v8x2048_last16_lr1e-6": {"run": "rl_simple2m_8x2048_last16_lr1e-6",   "short": "last-16 reward, lr 1e-6", "label": "variant: 8x2048, lr 1e-6, reward on the LAST 16 tokens only",              "color": "#3f9a4e", "ls": (0, (4, 1.5, 1, 1.5)), "reward_note": "reward = max activation over the last 16 tokens (NOT comparable to whole-span reward)"},
}
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
        mirror([tagc(x) for x in [f"sft_{SFT_RUN}", RL_RUN, SFT20_RUN, RL20_RUN] + [V["run"] for V in VARIANTS.values()] + [R["run"] for R in REFS.values()] + sorted({R["init_run"] for R in REFS.values()})])
        mirror([DIST_SFT_RAW, DIST_SFT_CEN, DIST_RL, f"sft_{SFT_RUN}", f"sft_{SFT_RUN}_c", RL_RUN])   # distillation chain + raw/centered source-text baselines
    ids = json.load(open(IDS)); registry = json.load(open(REGISTRY)) if REGISTRY.exists() else None
    sft = evals(tagc(f"sft_{SFT_RUN}")); rl = evals(tagc(RL_RUN))
    if not sft or not rl:
        raise SystemExit(f"[{SCORER}] no evals yet for {tagc(f'sft_{SFT_RUN}')} / {tagc(RL_RUN)} — nothing to plot")
    refs = {}
    for k, R in REFS.items():
        ev = evals(tagc(R["run"])); init = [r for r in evals(tagc(R["init_run"])) if r["ckpt_step"] == R["init_step"]]
        refs[k] = {**R, "evals": ev, "init_mean_all": init[0]["eval/mean_all"] if init else None}
    sft_final = sft[-1] if sft else None
    var_ids = ids.get("rl_variants", {})
    variants = {}
    for k, V in VARIANTS.items():
        wid = var_ids.get(k, {}).get("wandb")
        dyn = wandb_series(wid, ["reward/mean", "policy/entropy", "grad_norm", "policy/sampler_abs_dlogp", "ratio/clipfrac", "rollout/len_mean", "time/step_s"]) if wid else None
        if dyn is None:  # wandb id not recorded: look the run up by display name
            try:
                import wandb as _w
                rs = list(_w.Api().runs(PROJ, filters={"display_name": V["run"]}))
                if rs:
                    dyn = wandb_series(rs[-1].id, ["reward/mean", "policy/entropy", "grad_norm", "policy/sampler_abs_dlogp", "ratio/clipfrac", "rollout/len_mean", "time/step_s"])
            except Exception as e:  # noqa
                print(f"[variants] wandb lookup failed for {V['run']}: {e}")
        variants[k] = {**V, "evals": evals(tagc(V["run"])), "dyn": dyn, "ids": var_ids.get(k, {}), "last_train_step": max(dyn["step"]) if dyn and dyn["step"] else None}
    fetched = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    sft_dyn = wandb_series(SFT_WANDB, ["loss", "lr", "ex_per_s"])
    rl_dyn = wandb_series(RL_WANDB, ["reward/mean", "policy/entropy", "grad_norm", "policy/sampler_abs_dlogp", "ratio/clipfrac", "rollout/len_mean", "time/step_s",
                                    "reward/peak_last_frac", "reward/peak_dist_mean", "policy/offpolicy_lag_steps"])
    rl_last_step = max(rl_dyn["step"]) if rl_dyn and rl_dyn["step"] else None
    best_rl = max(rl, key=lambda r: r["eval/mean_all"]) if rl else None

    sft20 = evals(tagc(SFT20_RUN)); rl20 = evals(tagc(RL20_RUN))

    # ---------------- headline: held-out fidelity through the chain ----------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.6), sharey=True, gridspec_kw={"width_ratios": [1, 1.25]})
    xs = [r["ckpt_step"] * EFF_BATCH / 1e6 for r in sft]; ys = [r["eval/mean_all"] for r in sft]
    ax1.plot(xs, ys, "-o", color=SFT_C, lw=2.2, ms=6, label="this chain: SFT from base (50/50 acts 8-64 ctx full-context targets + 2M-SAE enc/dec)")
    for x, y in zip(xs, ys): ax1.annotate(f"{y:.3f}", (x, y), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8, color=INK2)
    for k in ("a", "s"):
        if refs[k]["init_mean_all"] is not None:
            ax1.axhline(refs[k]["init_mean_all"], color=refs[k]["color"], ls=":", lw=1.1)
            ax1.annotate(f"{refs[k]['init_mean_all']:.3f} = {'5M-mix midtrain init (arms A/D/.425)' if k == 'a' else '2M-SAE midtrain init (arm S)'}", (xs[-1] if xs else 8, refs[k]["init_mean_all"]),
                         xytext=(-2, 4 if k == 'a' else -11), textcoords="offset points", fontsize=7.8, color=refs[k]["color"], ha="right")
    ax1.set_xlabel("SFT: training rows seen (millions; 4,096 per step, exact 50/50 activations : SAE rows)"); ax1.set_ylabel(f"held-out fidelity: mean cosine over 10 direction families\n({SCORER_NOTE})")
    ax1.set_title("SFT alone: held-out fidelity FALLS while train loss falls", fontsize=10, loc="left"); ax1.grid(color=GRID, lw=0.6); ax1.set_xlim(0, xs[-1] * 1.08 if xs else 8.5)
    # RL panel
    rx = [0] + [r["ckpt_step"] for r in rl]; ry = ([sft_final["eval/mean_all"]] if sft_final else []) + [r["eval/mean_all"] for r in rl]
    ax2.plot(rx, ry, "-o", color=THIS_C, lw=2.4, ms=6.5, label=f"this chain: RL from the SFT final (init {ry[0]:.3f}), 8x2048, lr 1e-6, WHOLE-span reward, even acts/SAE pool", zorder=5)
    for x, y in zip(rx[1:], ry[1:]): ax2.annotate(f"{y:.3f}", (x, y), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8.2, color=THIS_C, fontweight="bold")
    for k, R in refs.items():
        if not R["evals"]: continue
        xx = [0] + [r["ckpt_step"] for r in R["evals"]]; yy = [R["init_mean_all"]] + [r["eval/mean_all"] for r in R["evals"]]
        ax2.plot(xx, yy, ls=R["ls"], marker="s", ms=3.6, lw=1.4, color=R["color"], label=R["label"], alpha=0.95)
    for k, V in variants.items():
        if not V["evals"] or not sft_final: continue
        xx = [0] + [r["ckpt_step"] for r in V["evals"]]; yy = [sft_final["eval/mean_all"]] + [r["eval/mean_all"] for r in V["evals"]]
        ax2.plot(xx, yy, ls=V["ls"], marker="D", ms=3.8, lw=1.5, color=V["color"], label=V["label"], zorder=4)
    ax2.set_xlabel("RL step (16,384 rollouts per step for the 8x2048 arms, 4,096 for 8x512; step 0 = the SFT init)"); ax2.grid(color=GRID, lw=0.6)
    ax2.set_xlim(-8, max(320, (max(rx) if rx else 300) + 20))
    late = [(r["ckpt_step"], r["eval/mean_all"], next((x["eval/mean_all"] for x in refs["a"]["evals"] if x["ckpt_step"] == r["ckpt_step"]), None)) for r in rl if r["ckpt_step"] >= 150]
    behind = [t for t in late if t[2] is not None and t[1] < t[2] - 0.003]
    ax2.set_title("RL repairs it: ahead of the midtrain-init arms through step 100, level at 150" + (f", behind arm A from step {behind[0][0]} on" if behind else ""), fontsize=10, loc="left")
    ax2.legend(loc="lower right", fontsize=7.0, frameon=False)
    ax1.legend(loc="upper right", fontsize=7.6, frameon=False)
    ymin = min([min(ys)] + [min(r["eval/mean_all"] for r in R["evals"]) for R in refs.values() if R["evals"]] + [R["init_mean_all"] for R in refs.values() if R["init_mean_all"]]) - 0.02
    ymax = max([max(ys)] + [max(r["eval/mean_all"] for r in rl)] + [max((r["eval/mean_all"] for r in R["evals"]), default=0) for R in refs.values()]
               + [max((r["eval/mean_all"] for r in V["evals"]), default=0) for V in variants.values()]) + 0.02
    ax1.set_ylim(ymin, ymax)
    claim = (f"A single 8M-row SFT from the base model (50/50 activations of 8-64 tokens of context with full-context targets + 2M-SAE encoder/decoder rows) DECLINES on held-out "
             f"fidelity from {ys[0]:.3f} ({xs[0]:.0f}M rows) to {ys[-1]:.3f} ({xs[-1]:.0f}M) while its train loss keeps falling; full-parameter RL from that final "
             + (f"reaches {best_rl['eval/mean_all']:.3f} at step {best_rl['ckpt_step']} (arm A {next((r['eval/mean_all'] for r in refs['a']['evals'] if r['ckpt_step'] == best_rl['ckpt_step']), float('nan')):.3f} at the same step from a {(refs['a']['init_mean_all'] if refs['a']['init_mean_all'] is not None else float('nan')):.3f} init)"
                if best_rl else "is running") + f" — Qwen3.6-27B activation-to-text inverter, 512 held-out directions per family, best-of-4 at T=1; {SCORER_NOTE}" + (f" (RL at step {rl_last_step} of 300)" if rl_last_step and rl_last_step < 300 and rl_dyn.get("state") == "running" else ""))
    fig.suptitle(wrap(claim, 165), fontsize=10.2, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.87)); savefig(fig, "fidelity_chain")

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
    axes[0].set_title("SFT: held-out 2M-SAE features barely fire (<10%)", fontsize=10, loc="left")
    pk = max(rl, key=lambda r: r.get("eval/sae2m_enc/fired", 0)) if rl else None
    axes[1].set_title(f"RL: they fire at {pk['eval/sae2m_enc/fired']*100:.0f}% / {pk['eval/sae2m_dec/fired']*100:.0f}% (enc / dec) by step {pk['ckpt_step']}, then DECAY to {rl[-1]['eval/sae2m_enc/fired']*100:.0f}% / {rl[-1]['eval/sae2m_dec/fired']*100:.0f}% at step {rl[-1]['ckpt_step']}" if pk and rl[-1]["ckpt_step"] > pk["ckpt_step"] else f"RL: they fire at {pk['eval/sae2m_enc/fired']*100:.0f}% / {pk['eval/sae2m_dec/fired']*100:.0f}% by step {pk['ckpt_step']}", fontsize=10, loc="left")
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
                          "coincided with rollouts lengthening to 90 tokens and the sampler lag reaching 2 steps, and decayed within 10 steps as the length penalty pulled rollouts back to ~55 tokens; "
                          "a second, milder excursion (|Δ log p| .10-.12 at steps 270-290, clip fraction .15-.17) came with entropy falling .65 -> .55 and coincides with the held-out decline from step 250 to 300", 150),
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

    # ---------------- RL variants: same SFT init, different batch shape / reward window / lr ----------------
    if any(V["evals"] for V in variants.values()):
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2))
        arms = [("main", {"run": RL_RUN, "short": "main arm (8x2048 whole-span)", "label": "main arm: 8x2048, lr 1e-6, whole-span reward", "color": THIS_C, "ls": "-", "evals": rl, "dyn": rl_dyn})] + list(variants.items())
        for k, V in arms:
            if not V["evals"]: continue
            xx = [0] + [r["ckpt_step"] for r in V["evals"]]
            axes[0].plot(xx, [sft_final["eval/mean_all"]] + [r["eval/mean_all"] for r in V["evals"]], ls=V["ls"], marker="o", ms=4.5, lw=2 if k == "main" else 1.6, color=V["color"], label=V["short"])
            if k == "main":
                for x, y in zip(xx[1:], [r["eval/mean_all"] for r in V["evals"]]): axes[0].annotate(f"{y:.3f}", (x, y), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7, color=V["color"])
            axes[1].plot(xx, [sft_final.get("eval/sae2m_enc/fired", np.nan)] + [r.get("eval/sae2m_enc/fired", np.nan) for r in V["evals"]], ls=V["ls"], marker="o", ms=4, lw=1.6, color=V["color"], label=f"{V['short']}: encoder")
            axes[1].plot(xx, [sft_final.get("eval/sae2m_dec/fired", np.nan)] + [r.get("eval/sae2m_dec/fired", np.nan) for r in V["evals"]], ls=V["ls"], marker="s", ms=3.5, lw=1.0, color=V["color"], alpha=0.6, label=f"{V['short']}: decoder")
            if V["dyn"] and V["dyn"]["step"]:
                st = np.array(V["dyn"]["step"]); ev = np.array([np.nan if x is None else x for x in V["dyn"]["policy/entropy"]], float)
                if len(ev) >= 10: ker = np.ones(10) / 10; axes[2].plot(st[9:], np.convolve(np.nan_to_num(ev, nan=np.nanmean(ev)), ker, mode="valid"), ls=V["ls"], lw=1.7, color=V["color"], label=V["short"])
        a_ev = refs["a"]["evals"]
        if a_ev: axes[0].plot([r["ckpt_step"] for r in a_ev], [r["eval/mean_all"] for r in a_ev], ls="-", marker="s", ms=3, lw=1.1, color=refs["a"]["color"], alpha=0.8, label="arm A (midtrain init, last-5)")
        axes[0].set_title("held-out fidelity (mean_all) per checkpoint", fontsize=10, loc="left"); axes[0].set_ylabel("mean cosine over 10 held-out direction families"); axes[0].legend(fontsize=6.8, frameon=False, loc="lower right"); axes[0].set_ylim(bottom=sft_final["eval/mean_all"] - 0.06)
        axes[1].set_title("held-out 2M-SAE features fired (encoder = circles, decoder = squares)", fontsize=10, loc="left"); axes[1].set_ylabel("fraction of 512 held-out features above the gate"); axes[1].legend(fontsize=6.8, frameon=False, loc="upper left", ncol=2); axes[1].set_ylim(0, 0.62)
        axes[2].set_title("policy entropy (10-step mean)", fontsize=10, loc="left"); axes[2].set_ylabel("entropy (nats / token)"); axes[2].legend(fontsize=7, frameon=False)
        for ax in axes: ax.grid(color=GRID, lw=0.6); ax.set_xlabel("RL step (16,384 rollouts per step in every arm; step 0 = the shared SFT init)")
        prog = ", ".join(f"{V['short']} at step {V['last_train_step']}" for k, V in variants.items() if V["last_train_step"])
        n_arms = 1 + sum(1 for V in variants.values() if V["evals"])
        fig.suptitle(wrap(f"{n_arms} RL runs from the SAME SFT init: the main arm (8 prompts x 2,048 rollouts, whole-span reward, lr 1e-6) vs 16x1,024 prompts/rollouts, and vs rewarding only the last 16 generated tokens at lr 5e-7 / 1e-6 "
                          "— the three legacy lr 1e-6 arms are indistinguishable on mean_all (~.60 from step 100) and in EVERY one of them held-out 2M-SAE firing peaks (at step 150 for 8x2048, at step 250 for 16x1024) and then decays; the lr 5e-7 arm reaches the same .600 by step 300 with held-out 2M firing still RISING (encoder .26 -> .34 from step 150 to 300) and the best MLP norm_act (.69 vs .58): the learning rate, not the batch shape or the reward window, controls the post-150 decay"
                          + (f" (in progress: {prog})" if prog else ""), 165), fontsize=10.2, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.88)); savefig(fig, "rl_variants")

    # ---------------- SFT-scaling retest: 20M rows at 80/20 acts/SAE vs the 8M 50/50 chain, vs rows seen ----------------
    if sft20:
        fams20 = [("eval/mean_all", "mean_all (10 held-out families)"), ("eval/realact/cos", "real activations (cos)"), ("eval/realact_long/cos", "long-context activations (cos)"),
                  ("eval/mlp/norm_act", "L42 MLP neurons (norm_act)"), ("eval/sae/norm_act", "131k-SAE features (norm_act)"), ("eval/sae2m_enc/fired", "held-out 2M-SAE encoder dirs fired")]
        fig, axes = plt.subplots(2, 3, figsize=(15, 7.6)); axes = axes.ravel()
        x8 = [r["ckpt_step"] * EFF_BATCH / 1e6 for r in sft]; x20 = [r["ckpt_step"] * EFF_BATCH / 1e6 for r in sft20]
        for ax, (k, lab) in zip(axes, fams20):
            ax.plot(x8, [r.get(k, np.nan) for r in sft], "-o", color=SFT_C, ms=4.5, lw=1.8, label="8M-row SFT: 50/50 activations : 2M-SAE rows (4M + 2M enc + 2M dec)")
            ax.plot(x20, [r.get(k, np.nan) for r in sft20], "-s", color="#111111", ms=4.5, lw=1.8, label="20M-row SFT: 80/20 (16M activations + 2M enc + 2M dec), i.i.d. step windows")
            for xx, r in zip(x20, sft20): ax.annotate(f"{r.get(k, np.nan):.3f}", (xx, r.get(k, np.nan)), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7, color="#111111")
            ax.set_title(lab, fontsize=9.5, loc="left"); ax.grid(color=GRID, lw=0.6); ax.set_xlabel("SFT rows seen (millions; 4,096 per step)"); ax.set_xlim(0, 21)
        axes[0].legend(fontsize=7.2, frameon=False, loc="upper right")
        last20 = sft20[-1]; at8 = next((r for r in sft if r["ckpt_step"] == last20["ckpt_step"]), None)
        cmp = (f"at {x20[-1]:.0f}M rows the 80/20 run scores {last20['eval/mean_all']:.3f} vs the 50/50 run's {at8['eval/mean_all']:.3f} at the same step" if at8 else f"latest 80/20 point {last20['eval/mean_all']:.3f} at {x20[-1]:.0f}M rows")
        fig.suptitle(wrap(f"SFT-scaling retest: does more activation data cap higher? Held-out fidelity vs rows seen for the 8M 50/50 chain (declined .491 -> .420) and the 20M 80/20 chain "
                          f"(2M+2M SAE rows held fixed, short-context full-target activations scaled 4M -> 16M; same lr 1e-5 one-cycle, batch 4,096, full FT from base) — {cmp}; "
                          f"both curves on the centered scorer; the 20M run is at step {sft20[-1]['ckpt_step']} of 4,883" + (f"; its RL has {len(rl20)} evals" if rl20 else ""), 165), fontsize=10.2, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.9)); savefig(fig, "sft_scaling")

    # ---------------- distillation: SFT on best-of-16 rollouts of the policy vs SFT on the source text ----------------
    d_sft_raw, d_sft_cen, d_rl = evals(DIST_SFT_RAW), evals(DIST_SFT_CEN), evals(DIST_RL)
    o_sft_raw, o_sft_cen = evals(f"sft_{SFT_RUN}"), evals(f"sft_{SFT_RUN}_c")
    o_rl_raw = evals(RL_RUN)
    if d_sft_raw or d_sft_cen:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), gridspec_kw={"width_ratios": [1.15, 1.25, 1]})
        DC, OC = "#c0392b", "#6b6b6b"
        ax = axes[0]
        for ev, col, ls, lab in ((o_sft_raw, OC, "-", "source-text SFT (8M rows), RAW scorer"), (d_sft_raw, DC, "-", "distilled SFT (best-of-16 rollouts, 2M rows), RAW scorer"),
                                 (o_sft_cen, OC, "--", "source-text SFT, centered scorer"), (d_sft_cen, DC, "--", "distilled SFT, centered scorer")):
            if ev:
                ax.plot([r["ckpt_step"] * EFF_BATCH / 1e6 for r in ev], [r["eval/mean_all"] for r in ev], ls=ls, marker="o", ms=4.5, lw=2 if col == DC else 1.4, color=col, label=lab)
        if o_rl_raw:
            b = max(o_rl_raw, key=lambda r: r["eval/mean_all"]); ax.axhline(b["eval/mean_all"], color="#2b6cb0", ls=":", lw=1.1)
            ax.annotate(f"{b['eval/mean_all']:.3f} = source-text chain AFTER RL (best ckpt, raw)", (8.3, b["eval/mean_all"]), xytext=(0, -10), textcoords="offset points", fontsize=7.5, color="#2b6cb0", ha="right")
        ax.set_xlabel("SFT rows seen (millions; both runs: full FT from base, batch 4,096, lr 1e-5)"); ax.set_ylabel("held-out fidelity: mean cosine over 10 direction families")
        ax.set_title("Distilled targets vs source-text targets, held-out fidelity vs rows seen", fontsize=10, loc="left"); ax.grid(color=GRID, lw=0.6); ax.legend(fontsize=7.2, frameon=False, loc="upper right"); ax.set_xlim(0, 8.6)
        ax = axes[1]
        if d_sft_raw and o_sft_raw:
            dl = d_sft_raw[-1]; ob = max(o_sft_raw, key=lambda r: r["eval/mean_all"])
            fams = [("realact", "eval/realact/cos"), ("early", "eval/realact_early/cos"), ("long", "eval/realact_long/cos"), ("bsf", "eval/bsf/cos"), ("cluster", "eval/cluster/cos"), ("jlens", "eval/jlens/cos"),
                    ("131k SAE\nnorm_act", "eval/sae/norm_act"), ("2M enc\nfired", "eval/sae2m_enc/fired"), ("2M dec\nfired", "eval/sae2m_dec/fired"), ("MLP\nnorm_act", "eval/mlp/norm_act")]
            x = np.arange(len(fams)); w = 0.38
            ax.bar(x - w / 2, [ob.get(k, np.nan) for _, k in fams], w, color=OC, label=f"source-text SFT, best ckpt (step {ob['ckpt_step']}, {ob['ckpt_step'] * EFF_BATCH / 1e6:.0f}M rows)")
            ax.bar(x + w / 2, [dl.get(k, np.nan) for _, k in fams], w, color=DC, label=f"distilled SFT, step {dl['ckpt_step']} ({dl['ckpt_step'] * EFF_BATCH / 1e6:.2f}M rows)")
            for xi, (_, k) in zip(x, fams):
                ax.annotate(f"{dl.get(k, np.nan):.2f}", (xi + w / 2, dl.get(k, np.nan)), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7, color=DC)
            ax.set_xticks(x); ax.set_xticklabels([n for n, _ in fams], fontsize=7.5); ax.set_ylabel("held-out score (raw scorer)"); ax.grid(color=GRID, lw=0.6, axis="y")
            ax.set_title("Every family up, incl. families NOT in the bank (SAE, BSF, cluster, MLP)", fontsize=9.5, loc="left"); ax.legend(fontsize=7.2, frameon=False, loc="upper left")
        ax = axes[2]
        if o_rl_raw:
            ax.plot([0] + [r["ckpt_step"] for r in o_rl_raw], [o_sft_raw[-1]["eval/mean_all"] if o_sft_raw else np.nan] + [r["eval/mean_all"] for r in o_rl_raw], "-s", ms=3.5, lw=1.4, color=OC, label="RL from the source-text SFT final (main arm, raw reward)")
        if d_rl:
            ax.plot([0] + [r["ckpt_step"] for r in d_rl], [d_sft_raw[-1]["eval/mean_all"] if d_sft_raw else np.nan] + [r["eval/mean_all"] for r in d_rl], "-o", ms=5, lw=2.2, color=DC, label="RL from the distilled SFT final (raw reward)")
            for r in d_rl: ax.annotate(f"{r['eval/mean_all']:.3f}", (r["ckpt_step"], r["eval/mean_all"]), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7.5, color=DC)
        elif d_sft_raw:
            ax.plot([0], [d_sft_raw[-1]["eval/mean_all"]], "o", ms=6, color=DC, label="distilled SFT final (RL pending)")
        ax.set_xlabel("RL step (8x2048, lr 1e-6, whole-span RAW reward; step 0 = the SFT init)"); ax.set_title("RL on top of each SFT final", fontsize=9.5, loc="left"); ax.grid(color=GRID, lw=0.6); ax.legend(fontsize=7.2, frameon=False, loc="lower right")
        ax.set_xlim(-8, 320)
        peak = max(d_sft_raw, key=lambda r: r["eval/mean_all"]) if d_sft_raw else None
        tail = (f", vs {max(r['eval/mean_all'] for r in o_sft_raw):.3f} for the best source-text SFT checkpoint and {max(r['eval/mean_all'] for r in o_rl_raw):.3f} for that chain after RL" if o_sft_raw and o_rl_raw else "")
        fig.suptitle(wrap("Rejection-sampling distillation: 2,000,000 FRESH Ultra-FineWeb activations (1M with 8-64-token contexts, 1M full-document 64-2048), 16 rollouts each from the RL'd inverter (step 250), "
                          "the best-cosine rollout kept as the SFT target -> a fresh full fine-tune of the base model" + (f" reaches {peak['eval/mean_all']:.3f} held-out (raw scorer) at {peak['ckpt_step'] * EFF_BATCH / 1e6:.2f}M rows" if peak else "") + tail, 170), fontsize=10.2, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.9)); savefig(fig, "distill_chain")
        json.dump({"generated_at": fetched, "distilled_sft_raw": d_sft_raw, "distilled_sft_centered": d_sft_cen, "distilled_rl_raw": d_rl, "source_text_sft_raw": o_sft_raw,
                   "source_text_sft_centered": o_sft_cen, "source_text_rl_raw": o_rl_raw, "eff_batch": EFF_BATCH,
                   "harvest": {k: v for k, v in ids.get("harvest", {}).get("harvest_full_v1", {}).items() if k in ("plan", "spec", "n_samples", "top_k", "configs", "banks")},
                   "distill": ids.get("distill")}, open(OUT / "data" / "distill.json", "w"), indent=1, default=str)
    dd = OUT / "data"
    json.dump({"generated_at": fetched, "scorer": SCORER, "run": SFT_RUN, "wandb": SFT_WANDB, "eff_batch": EFF_BATCH, "rows_per_ckpt": {r["ckpt_step"]: r["ckpt_step"] * EFF_BATCH for r in sft}, "evals": sft}, open(dd / "sft_evals.json", "w"), indent=1)
    json.dump({"generated_at": fetched, "scorer": SCORER, "sft20": {"run": SFT20_RUN, "rows_per_step": EFF_BATCH, "steps_total": 4883, "mix": "16M acts (ctx 8-64, full-context targets; docs [5.5M,5.625M) + [6.1M,~6.48M)) + 2M enc + 2M dec", "evals": sft20},
               "rl20": {"run": RL20_RUN, "evals": rl20}, "sft8_evals": sft}, open(dd / "sft_scaling.json", "w"), indent=1, default=str)
    json.dump({"generated_at": fetched, "scorer": SCORER, "run": RL_RUN, "wandb": RL_WANDB, "init": sft_final, "evals": rl, "best": best_rl, "last_train_step": rl_last_step}, open(dd / "rl_evals.json", "w"), indent=1)
    json.dump({"generated_at": fetched, "arms": {k: {kk: vv for kk, vv in R.items() if kk not in ("ls",)} for k, R in refs.items()}}, open(dd / "reference_arms.json", "w"), indent=1)
    json.dump({"generated_at": fetched, "shared_init": sft_final, "variants": {k: {kk: vv for kk, vv in V.items() if kk not in ("ls", "dyn")} for k, V in variants.items()},
               "dynamics": {k: V["dyn"] for k, V in variants.items() if V["dyn"]}}, open(dd / "rl_variants.json", "w"), indent=1, default=str)
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
    summ = {"generated_at": fetched, "scorer": SCORER, "scorer_note": SCORER_NOTE, "sft_curve": [(r["ckpt_step"], round(r["eval/mean_all"], 4)) for r in sft], "sft_final_mean_all": sft_final["eval/mean_all"] if sft_final else None,
            "sft_best": max(sft, key=lambda r: r["eval/mean_all"])["ckpt_step"] if sft else None, "rl_curve": [(r["ckpt_step"], round(r["eval/mean_all"], 4)) for r in rl],
            "rl_best": best_rl, "rl_last_train_step": rl_last_step, "rl_state": rl_dyn["state"] if rl_dyn else None, "matched_step_mean_all": matched,
            "ref_inits": {k: R["init_mean_all"] for k, R in refs.items()},
            "variants": {k: {"run": V["run"], "curve": [(r["ckpt_step"], round(r["eval/mean_all"], 4)) for r in V["evals"]], "last_train_step": V["last_train_step"],
                             "heldout_2m": {r["ckpt_step"]: {kk: r.get(kk) for kk in ("eval/sae2m_enc/fired", "eval/sae2m_dec/fired")} for r in V["evals"]}} for k, V in variants.items()},
            "heldout_2m": {"sft_final": {k: sft_final.get(k) for k in ("eval/sae2m_enc/fired", "eval/sae2m_dec/fired", "eval/sae2m_enc/norm_act", "eval/sae2m_dec/norm_act")} if sft_final else None,
                           "rl": {r["ckpt_step"]: {k: r.get(k) for k in ("eval/sae2m_enc/fired", "eval/sae2m_dec/fired", "eval/sae2m_enc/norm_act", "eval/sae2m_dec/norm_act")} for r in rl}}}
    json.dump(summ, open(dd / "summary.json", "w"), indent=1, default=str)
    print(json.dumps({k: summ[k] for k in ("sft_curve", "rl_curve", "matched_step_mean_all")}))
    if not a.no_html and (OUT / "build_html.py").exists():
        subprocess.run(["python3", str(OUT / "build_html.py")], check=False)


if __name__ == "__main__":
    main()
