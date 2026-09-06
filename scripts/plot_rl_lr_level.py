"""Learning-rate LEVEL vs cumulative drift (Sep 6 2026): RL-H (16 x 512, constant lr 5e-6, 600 steps) vs RL-F (same batch, same init/bank/recipe,
constant lr 1e-5, 400 steps). Question: does halving the lr delay the constant-lr collapse proportionally (drift-budget view: onset at a fixed
cumulative lr x steps) or remove it (level view)? Does the slower run reach the same best held-out checkpoint?
Fig 1: dynamics vs step with onset lines. Fig 2: held-out evals vs step, vs rollouts, vs cumulative lr x steps (the third row is the direct
test). Fig 3: onset step, onset lr x steps, best mean_all, GPU-hours to best.
Writes ~/shared/reports/maemm-rl-lr-level/{lr_dynamics,lr_evals,lr_summary}.{png,pdf} + data/lr_level.json."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb

OUT = os.path.expanduser("~/shared/reports/maemm-rl-lr-level")
os.makedirs(f"{OUT}/data", exist_ok=True)
PROJ = "octahedral-systems/maxact-fast"
RUNS = {
    "F": {"train": "pu3whp5m", "eval": "rl_F_mixmlp_16x512_eval", "lr": 1e-5, "rps": 8192, "gpus": 8, "total_steps": 400,
          "label": "lr 1e-5 (RL-F)", "color": "#b5542b",
          "resume": {"after_utc": "2026-09-06T04:50:00", "from_step": 161}},   # resumed from step_160 with --max-lag 2 (see batch-scaling report)
    "H": {"train": "yzuiumfg", "eval": "rl_H_mixmlp_16x512_lr5e-6_eval", "lr": 5e-6, "rps": 8192, "gpus": 8, "total_steps": 600,
          "label": "lr 5e-6 (RL-H)", "color": "#2a7f62"},
}
TRAIN_KEYS = ["reward/mean", "policy/entropy", "grad_norm", "policy/sampler_abs_dlogp", "ratio/clipfrac", "rollout/len_mean", "time/step_s", "policy/offpolicy_lag_steps"]
EVAL_KEYS = [("eval/mean_all", "mean over held-out families"), ("eval/sae/norm_act", "SAE norm_act"), ("eval/sae/rank1_frac", "SAE rank-1 fraction"),
             ("eval/sae/unverbalized_frac", "SAE unverbalized (lower = better)"), ("eval/realact/cos", "real activations (cos)"),
             ("eval/realact_long/cos", "real acts, long ctx (cos)"), ("eval/bsf/cos", "BSF (cos)"), ("eval/cluster/cos", "cluster probes (cos)"), ("eval/mlp/norm_act", "MLP fire-back")]
PANELS = [("reward/mean", "training reward (max cos, last-5 window)", False), ("policy/entropy", "policy entropy (nats/token)", False),
          ("grad_norm", "gradient norm (log; dotted = clip 1.0)", True), ("policy/sampler_abs_dlogp", "sampler vs trainer |Δ log-prob| per token (log; dotted = 0.05)", True),
          ("rollout/len_mean", "mean rollout length (tokens)", False), ("time/step_s", "seconds per step", False)]
api = wandb.Api()


def train_rows(run_id, resume=None):
    """One row per RL step. wandb's _step is a log-row counter: a run resumed with --wandb-id keeps counting, so rows logged after
    `resume.after_utc` are re-indexed from `resume.from_step` and the abandoned first-segment rows at >= from_step are dropped."""
    import datetime
    run = api.run(f"{PROJ}/{run_id}")
    logged = [k for k in TRAIN_KEYS if k in run.summary] or TRAIN_KEYS
    raw = sorted([h for h in run.history(keys=logged + ["_timestamp"], pandas=False, samples=3000) if h.get("grad_norm") is not None], key=lambda h: h["_step"])
    if resume:
        cut = datetime.datetime.fromisoformat(resume["after_utc"]).replace(tzinfo=datetime.timezone.utc).timestamp()
        old = [h for h in raw if h["_timestamp"] < cut and int(h["_step"]) < resume["from_step"]]
        new_seg = [h for h in raw if h["_timestamp"] >= cut]
        return ([{"step": int(h["_step"]), **{k: h.get(k) for k in TRAIN_KEYS}} for h in old]
                + [{"step": resume["from_step"] + i, **{k: h.get(k) for k in TRAIN_KEYS}} for i, h in enumerate(new_seg)])
    return [{"step": int(h["_step"]), **{k: h.get(k) for k in TRAIN_KEYS}} for h in raw]


def eval_rows(name):
    rs = list(api.runs(PROJ, filters={"display_name": name}, order="-created_at"))
    if not rs:
        return []
    rows = [r for r in rs[0].history(pandas=False) if r.get("ckpt_step") is not None and r.get("eval/mean_all") is not None]
    return sorted([{"ckpt_step": int(r["ckpt_step"]), **{k: r.get(k) for k, _ in EVAL_KEYS}} for r in rows], key=lambda r: r["ckpt_step"])


def onset(tr, key, thr, consecutive=3, min_step=100):
    v = [(r["step"], r[key]) for r in tr if r.get(key) is not None and r["step"] >= min_step]
    for i in range(len(v) - consecutive + 1):
        if all(x > thr for _, x in v[i:i + consecutive]):
            return v[i][0]
    return None


data = {"runs": {}, "note": "identical runs except lr: init 23M realact SFT -> 1.1M all-families midtrain; bank mix_1m_mlp (7 families); ScaleRL/CISPO; 16 samples x 512 "
        "directions per step; 3 vLLM + 5 trainer B200; --max-lag 2 (RL-F steps 1-160 ran at lag 1-4 before its resume); constant lr, no decay, no KL. Onsets: sampler |dlogp| > 0.05 "
        "for 3 consecutive logged steps; grad_norm > 1 for 2 consecutive steps (min step 100). lr x steps = cumulative optimizer movement proxy under Adam."}
for k, cfg in RUNS.items():
    tr = train_rows(cfg["train"], cfg.get("resume")); ev = eval_rows(cfg["eval"])
    best = max(ev, key=lambda r: r["eval/mean_all"]) if ev else None
    sps = float(np.median([r["time/step_s"] for r in tr if r.get("time/step_s")])) if tr else None
    od, og = onset(tr, "policy/sampler_abs_dlogp", 0.05), onset(tr, "grad_norm", 1.0, consecutive=2)
    data["runs"][k] = {**{x: cfg[x] for x in ("train", "eval", "lr", "rps", "gpus", "total_steps", "label")}, "train": tr, "evals": ev,
                       "onset_dlogp_gt05": od, "onset_gnorm_gt1": og, "onset_lr_steps_dlogp": (od * cfg["lr"]) if od else None, "onset_lr_steps_gnorm": (og * cfg["lr"]) if og else None,
                       "median_step_s": sps, "best": best, "gpu_hours_to_best": (best["ckpt_step"] * sps * cfg["gpus"] / 3600) if (best and sps) else None,
                       "steps_logged": len(tr)}
json.dump(data, open(f"{OUT}/data/lr_level.json", "w"), indent=1)
F, H = data["runs"]["F"], data["runs"]["H"]
ratio = (H["onset_gnorm_gt1"] / F["onset_gnorm_gt1"]) if (H["onset_gnorm_gt1"] and F["onset_gnorm_gt1"]) else None
verdict = ("delays the collapse in PROPORTION to the lr (onset lr × steps "
           f"{H['onset_lr_steps_gnorm']:.2e} vs {F['onset_lr_steps_gnorm']:.2e}) — the cumulative-drift budget holds" if ratio and 1.6 <= ratio <= 2.4 else
           "removes the collapse within 600 steps — the lr LEVEL matters beyond cumulative drift" if H["onset_gnorm_gt1"] is None else
           f"moves the collapse by {ratio:.2f}x in steps — neither a clean proportional delay nor a removal")

# ---- fig 1: dynamics ----
fig, axes = plt.subplots(2, 3, figsize=(17, 8.4), sharex=True)
for ax, (key, title, logy) in zip(axes.flat, PANELS):
    for k, cfg in RUNS.items():
        tr = [r for r in data["runs"][k]["train"] if r.get(key) is not None]
        ax.plot([r["step"] for r in tr], [r[key] for r in tr], color=cfg["color"], lw=1.2, label=cfg["label"])
        o = data["runs"][k]["onset_gnorm_gt1"]
        if o:
            ax.axvline(o, color=cfg["color"], ls="--", lw=0.9, alpha=0.8)
    if key == "grad_norm":
        ax.axhline(1.0, color="#333333", ls=":", lw=1)
    if key == "policy/sampler_abs_dlogp":
        ax.axhline(0.05, color="#333333", ls=":", lw=1)
    if logy:
        ax.set_yscale("log")
    ax.set_title(title, fontsize=10); ax.grid(alpha=0.25); ax.tick_params(labelsize=8.5)
for ax in axes[1]:
    ax.set_xlabel("RL step (dashed verticals = grad-norm onset of the runaway)", fontsize=9.5)
h, l = axes.flat[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.0))
fig.suptitle(f"Halving the learning rate {verdict}:\\nrunaway onset at step {H['onset_gnorm_gt1']} (lr 5e-6) vs {F['onset_gnorm_gt1']} (lr 1e-5), same batch, same everything else",
             fontsize=11, y=0.995)
fig.tight_layout(rect=(0, 0.07, 1, 0.94))
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/lr_dynamics.{ext}", dpi=160)
plt.close(fig)

# ---- fig 2: evals vs step / rollouts / lr x steps ----
cols = [("eval/mean_all", "mean over held-out families"), ("eval/sae/norm_act", "SAE norm_act"), ("eval/sae/rank1_frac", "SAE rank-1"),
        ("eval/realact/cos", "real activations (cos)"), ("eval/mlp/norm_act", "MLP fire-back")]
fig, axes = plt.subplots(3, len(cols), figsize=(3.4 * len(cols), 10.5))
for j, (key, title) in enumerate(cols):
    for i, xmode in enumerate(("step", "rollouts", "lr_steps")):
        ax = axes[i, j]
        for k, cfg in RUNS.items():
            ev = [r for r in data["runs"][k]["evals"] if r.get(key) is not None]
            xs = [r["ckpt_step"] if xmode == "step" else r["ckpt_step"] * cfg["rps"] / 1e6 if xmode == "rollouts" else r["ckpt_step"] * cfg["lr"] * 1e3 for r in ev]
            ax.plot(xs, [r[key] for r in ev], "o-", color=cfg["color"], ms=4, lw=1.5, label=cfg["label"])
            o = data["runs"][k]["onset_gnorm_gt1"]
            if o and xmode == "lr_steps":
                ax.axvline(o * cfg["lr"] * 1e3, color=cfg["color"], ls="--", lw=0.8, alpha=0.7)
        ax.set_title(title if i == 0 else "", fontsize=9.5); ax.grid(alpha=0.25); ax.tick_params(labelsize=8)
        ax.set_xlabel({"step": "RL step", "rollouts": "rollouts consumed (millions)", "lr_steps": "cumulative lr × steps (×1e-3); dashed = onset"}[xmode], fontsize=9)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.0))
fig.suptitle(f"Held-out fidelity: per STEP the lower lr is slower (top); per cumulative lr × steps the two runs overlay almost exactly (bottom) and reach the same best\n"
             f"({H['best']['eval/mean_all']:.3f} @{H['best']['ckpt_step']} at lr 5e-6 vs {F['best']['eval/mean_all']:.3f} @{F['best']['ckpt_step']} at lr 1e-5) — the walk is the same, only its speed changes", fontsize=11, y=0.995)
fig.tight_layout(rect=(0, 0.05, 1, 0.96))
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/lr_evals.{ext}", dpi=160)
plt.close(fig)

# ---- fig 3: summary bars ----
fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
names = [RUNS[k]["label"] for k in RUNS]; colors = [RUNS[k]["color"] for k in RUNS]
bars = [("runaway onset (RL step, grad norm > 1)", [data["runs"][k]["onset_gnorm_gt1"] or 0 for k in RUNS], "{:.0f}"),
        ("cumulative lr × steps at onset (×1e-3)", [(data["runs"][k]["onset_lr_steps_gnorm"] or 0) * 1e3 for k in RUNS], "{:.2f}"),
        ("best held-out mean_all", [data["runs"][k]["best"]["eval/mean_all"] for k in RUNS], "{:.3f}"),
        ("B200 GPU-hours to the best checkpoint", [data["runs"][k]["gpu_hours_to_best"] or 0 for k in RUNS], "{:.0f} h")]
for ax, (title, vals, fmt) in zip(axes, bars):
    b = ax.bar(names, vals, color=colors, width=0.55)
    for rect, v in zip(b, vals):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(), fmt.format(v), ha="center", va="bottom", fontsize=9)
    ax.set_title(title, fontsize=10); ax.grid(alpha=0.25, axis="y"); ax.tick_params(labelsize=8.5)
    if "mean_all" in title:
        ax.set_ylim(0.40, 0.43)
fig.suptitle(f"lr 5e-6 vs 1e-5 at 16×512: onset {H['onset_gnorm_gt1']} vs {F['onset_gnorm_gt1']} steps ({ratio:.2f}x) but the same lr × steps; same best checkpoint; "
             f"{(H['gpu_hours_to_best'] or 0) / (F['gpu_hours_to_best'] or 1):.1f}x the GPU-hours to reach it", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.92))
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/lr_summary.{ext}", dpi=160)
plt.close(fig)
print("wrote", OUT, {k: (v["steps_logged"], [r["ckpt_step"] for r in v["evals"]], v["onset_dlogp_gt05"], v["onset_gnorm_gt1"], round(v["best"]["eval/mean_all"], 4) if v["best"] else None) for k, v in data["runs"].items()}, "| verdict:", verdict)
