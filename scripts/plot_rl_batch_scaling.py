"""Batch scaling for the inverter's RL (Sep 6 2026): RL-F (16 samples x 512 directions = 8,192 rollouts/step) vs RL-E (16 x 256 = 4,096),
otherwise identical (init = 23M realact SFT + 1.1M all-families midtrain; 7-family bank incl. layer-42 MLP neurons; ScaleRL/CISPO; constant
lr 1e-5; 400 steps). Question: does a 2x batch move the constant-lr collapse wall (RL-E onset step 261) and/or raise the best held-out
checkpoint (RL-E best .4210 @250)? A 16x1024 run (RL-G) was launched alongside and cancelled at step 2 on cost grounds.
Fig 1: training dynamics vs step. Fig 2: held-out evals per checkpoint vs step AND vs rollouts. Fig 3: onset step / best eval / GPU-hours.
Writes ~/shared/reports/maemm-rl-batch-scaling/{batch_dynamics,batch_evals,batch_summary}.{png,pdf} + data/batch_scaling.json."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb

OUT = os.path.expanduser("~/shared/reports/maemm-rl-batch-scaling")
os.makedirs(f"{OUT}/data", exist_ok=True)
PROJ = "octahedral-systems/maxact-fast"
RUNS = {
    "E": {"train": "p57zffg6", "eval": "rl_E_mixmlp_from_mixsft_eval", "rps": 4096, "gpus": 8, "split": "2 rollout + 6 trainer",
          "label": "16 × 256 = 4,096 rollouts/step (RL-E)", "color": "#4a6fa5"},
    "F": {"train": "pu3whp5m", "eval": "rl_F_mixmlp_16x512_eval", "rps": 8192, "gpus": 8, "split": "3 rollout + 5 trainer",
          "label": "16 × 512 = 8,192 rollouts/step (RL-F)", "color": "#b5542b",
          # resumed from step_160 at 04:46Z Sep 6 with --max-lag 2: wandb's _step is a ROW COUNTER, so resumed rows continue at _step 178+ while
          # meaning RL step 161+; rows logged after the resume are re-indexed and the abandoned rows (RL steps 161-177, lag 4-5) are dropped
          "resume": {"after_utc": "2026-09-06T04:50:00", "from_step": 161}},
}
TRAIN_KEYS = ["reward/mean", "policy/entropy", "grad_norm", "policy/sampler_abs_dlogp", "ratio/clipfrac", "rollout/len_mean", "time/step_s", "policy/offpolicy_lag_steps"]
EVAL_KEYS = [("eval/mean_all", "mean over held-out families"), ("eval/sae/norm_act", "SAE norm_act"), ("eval/sae/rank1_frac", "SAE rank-1 fraction"),
             ("eval/sae/unverbalized_frac", "SAE unverbalized (lower = better)"), ("eval/realact/cos", "real activations (cos)"),
             ("eval/realact_long/cos", "real acts, long ctx (cos)"), ("eval/bsf/cos", "BSF (cos)"), ("eval/cluster/cos", "cluster probes (cos)"), ("eval/mlp/norm_act", "MLP fire-back")]
PANELS = [("reward/mean", "training reward (max cos, last-5 window)", False), ("policy/entropy", "policy entropy (nats/token)", False),
          ("grad_norm", "gradient norm (log; dotted = clip 1.0)", True), ("policy/sampler_abs_dlogp", "sampler vs trainer |Δ log-prob| per token (log)", True),
          ("rollout/len_mean", "mean rollout length (tokens)", False), ("time/step_s", "seconds per step", False)]
api = wandb.Api()


def train_rows(run_id, resume=None):
    """One row per RL step. wandb's _step is a log-row counter, not the RL step: a resumed run keeps counting, so for a run with
    `resume` = {after_utc, from_step} the rows logged after that time are re-indexed from_step, from_step+1, ... and the abandoned
    rows of the first segment at >= from_step are dropped."""
    import datetime
    run = api.run(f"{PROJ}/{run_id}")
    logged = [k for k in TRAIN_KEYS if k in run.summary] or TRAIN_KEYS
    raw = [h for h in run.history(keys=logged + ["_timestamp"], pandas=False, samples=3000) if h.get("grad_norm") is not None]
    raw.sort(key=lambda h: h["_step"])
    if resume:
        cut = datetime.datetime.fromisoformat(resume["after_utc"]).replace(tzinfo=datetime.timezone.utc).timestamp()
        old = [h for h in raw if h["_timestamp"] < cut and int(h["_step"]) < resume["from_step"]]
        new_seg = [h for h in raw if h["_timestamp"] >= cut]
        rows = [{"step": int(h["_step"]), **{k: h.get(k) for k in TRAIN_KEYS}} for h in old]
        rows += [{"step": resume["from_step"] + i, **{k: h.get(k) for k in TRAIN_KEYS}} for i, h in enumerate(new_seg)]
        return rows
    return [{"step": int(h["_step"]), **{k: h.get(k) for k in TRAIN_KEYS}} for h in raw]


def eval_rows(name):
    rs = list(api.runs(PROJ, filters={"display_name": name}, order="-created_at"))
    if not rs:
        return []
    rows = [r for r in rs[0].history(pandas=False) if r.get("ckpt_step") is not None and r.get("eval/mean_all") is not None]
    return sorted([{"ckpt_step": int(r["ckpt_step"]), **{k: r.get(k) for k, _ in EVAL_KEYS}} for r in rows], key=lambda r: r["ckpt_step"])


def onset(tr, key, thr, consecutive=3, min_step=100):
    """first step from which `key` exceeds thr for `consecutive` consecutive logged steps (robust to single spikes)."""
    v = [(r["step"], r[key]) for r in tr if r.get(key) is not None and r["step"] >= min_step]
    for i in range(len(v) - consecutive + 1):
        if all(x > thr for _, x in v[i:i + consecutive]):
            return v[i][0]
    return None


data = {"runs": {}, "note": "same init (23M realact SFT -> 1.1M all-families midtrain), same 7-family bank (mix_1m_mlp), ScaleRL/CISPO, constant lr 1e-5, 400 steps, "
        "held-out eval 512 dirs/family best-of-4 T=1 with the v2 cache (mlp families). RL-F ran steps 1-160 with off-policy lag drifting 1->4 (3+5 split, "
        "queue cap 8), was resumed from its step-160 checkpoint (adapter + optimizer state) with --max-lag 2 for steps 161-400. RL-G (16x1024) cancelled at step 2."}
for k, cfg in RUNS.items():
    tr = train_rows(cfg["train"], cfg.get("resume")); ev = eval_rows(cfg["eval"])
    best = max(ev, key=lambda r: r["eval/mean_all"]) if ev else None
    sps = float(np.median([r["time/step_s"] for r in tr if r.get("time/step_s")])) if tr else None
    data["runs"][k] = {**{x: cfg[x] for x in ("train", "eval", "rps", "gpus", "split", "label")}, "resume": cfg.get("resume"), "train": tr, "evals": ev,
                       "onset_dlogp_gt05": onset(tr, "policy/sampler_abs_dlogp", 0.05), "onset_gnorm_gt1": onset(tr, "grad_norm", 1.0, consecutive=2),
                       "median_step_s": sps, "best": best,
                       "gpu_hours_to_best": (best["ckpt_step"] * sps * cfg["gpus"] / 3600) if (best and sps) else None,
                       "rollouts_to_best": best["ckpt_step"] * cfg["rps"] if best else None,
                       "gpu_hours_400": (400 * sps * cfg["gpus"] / 3600) if sps else None}
json.dump(data, open(f"{OUT}/data/batch_scaling.json", "w"), indent=1)
E, F = data["runs"]["E"], data["runs"]["F"]

# ---- fig 1: dynamics ----
fig, axes = plt.subplots(2, 3, figsize=(17, 8.2), sharex=True)
for ax, (key, title, logy) in zip(axes.flat, PANELS):
    for k, cfg in RUNS.items():
        tr = [r for r in data["runs"][k]["train"] if r.get(key) is not None]
        ax.plot([r["step"] for r in tr], [r[key] for r in tr], color=cfg["color"], lw=1.2, label=cfg["label"])
    if key == "grad_norm":
        ax.axhline(1.0, color="#333333", ls=":", lw=1)
    if key == "policy/sampler_abs_dlogp":
        ax.axhline(0.05, color="#333333", ls=":", lw=1)
    if logy:
        ax.set_yscale("log")
    for k, cfg in RUNS.items():
        o = data["runs"][k]["onset_dlogp_gt05"]
        if o:
            ax.axvline(o, color=cfg["color"], ls="--", lw=0.8, alpha=0.7)
    ax.set_title(title, fontsize=10); ax.grid(alpha=0.25); ax.tick_params(labelsize=8.5)
for ax in axes[1]:
    ax.set_xlabel("RL step (dashed verticals = sampler-drift onset)", fontsize=9.5)
h, l = axes.flat[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.0))
dF, dE = F["onset_dlogp_gt05"], E["onset_dlogp_gt05"]; gF, gE = F["onset_gnorm_gt1"], E["onset_gnorm_gt1"]
verdict = ("arrives EARLIER" if (gF or 10**9) < (gE or 10**9) else "is not delayed")
fig.suptitle(f"Doubling the RL batch does not delay the constant-lr collapse — it {verdict}: sampler-drift onset step {dF} vs {dE}, grad-norm explosion {gF} vs {gE} (16×512 vs 16×256);\n"
             f"the 2x batch runs a sharper policy (lower entropy at the same step) with a lower gradient norm until it breaks", fontsize=11, y=0.995)
fig.tight_layout(rect=(0, 0.07, 1, 0.94))
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/batch_dynamics.{ext}", dpi=160)
plt.close(fig)

# ---- fig 2: evals vs step (row 1) and vs rollouts (row 2) ----
cols = [("eval/mean_all", "mean over held-out families"), ("eval/sae/norm_act", "SAE norm_act"), ("eval/sae/rank1_frac", "SAE rank-1"),
        ("eval/realact/cos", "real activations (cos)"), ("eval/mlp/norm_act", "MLP fire-back")]
fig, axes = plt.subplots(2, len(cols), figsize=(3.4 * len(cols), 7.6))
for j, (key, title) in enumerate(cols):
    for i, xmode in enumerate(("step", "rollouts")):
        ax = axes[i, j]
        for k, cfg in RUNS.items():
            ev = [r for r in data["runs"][k]["evals"] if r.get(key) is not None]
            xs = [r["ckpt_step"] if xmode == "step" else r["ckpt_step"] * cfg["rps"] / 1e6 for r in ev]
            ax.plot(xs, [r[key] for r in ev], "o-", color=cfg["color"], ms=4, lw=1.5, label=cfg["label"])
        ax.set_title(title if i == 0 else "", fontsize=9.5); ax.grid(alpha=0.25); ax.tick_params(labelsize=8)
        ax.set_xlabel("RL step" if xmode == "step" else "rollouts consumed (millions)", fontsize=9)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.0))
fig.suptitle(f"Held-out fidelity per checkpoint: at the same STEP the 2x batch is slightly ahead (best {F['best']['eval/mean_all']:.3f} @{F['best']['ckpt_step']} vs "
             f"{E['best']['eval/mean_all']:.3f} @{E['best']['ckpt_step']}); at the same ROLLOUTS it is behind — rollouts buy less than step count", fontsize=11, y=0.995)
fig.tight_layout(rect=(0, 0.07, 1, 0.95))
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/batch_evals.{ext}", dpi=160)
plt.close(fig)

# ---- fig 3: summary bars ----
fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
names = [RUNS[k]["label"].split(" (")[0] for k in RUNS]; colors = [RUNS[k]["color"] for k in RUNS]
bars = [("sampler-drift onset (RL step)", [data["runs"][k]["onset_dlogp_gt05"] or 0 for k in RUNS], "{:.0f}"),
        ("best held-out mean_all", [data["runs"][k]["best"]["eval/mean_all"] for k in RUNS], "{:.3f}"),
        ("best-checkpoint MLP fire-back", [data["runs"][k]["best"]["eval/mlp/norm_act"] or 0 for k in RUNS], "{:.3f}"),
        ("B200 GPU-hours to the best checkpoint", [data["runs"][k]["gpu_hours_to_best"] or 0 for k in RUNS], "{:.0f} h")]
for ax, (title, vals, fmt) in zip(axes, bars):
    b = ax.bar(names, vals, color=colors, width=0.55)
    for rect, v in zip(b, vals):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(), fmt.format(v), ha="center", va="bottom", fontsize=9)
    ax.set_title(title, fontsize=10); ax.grid(alpha=0.25, axis="y"); ax.tick_params(labelsize=8.5)
    if "mean_all" in title:
        ax.set_ylim(0.40, 0.43)
    if "fire-back" in title:
        ax.set_ylim(0.70, 0.85)
fig.suptitle(f"2x batch: collapse {'earlier' if (gF or 10**9) < (gE or 10**9) else 'not later'} ({gF} vs {gE}), best checkpoint {F['best']['eval/mean_all']:.3f} vs {E['best']['eval/mean_all']:.3f}, "
             f"{F['gpu_hours_to_best'] / E['gpu_hours_to_best']:.1f}x the GPU-hours to reach it", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.92))
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/batch_summary.{ext}", dpi=160)
plt.close(fig)
print("wrote", OUT, {k: (len(v["train"]), [r["ckpt_step"] for r in v["evals"]], v["onset_dlogp_gt05"], v["onset_gnorm_gt1"], v["best"]["eval/mean_all"] if v["best"] else None) for k, v in data["runs"].items()})
