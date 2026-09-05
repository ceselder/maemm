"""Collapse ablation (Sep 5 2026): RL-D (Qwen3.6-27B inverter, ScaleRL/CISPO, lr 1e-5 constant) blew up after step ~273.
Seven arms resume RL-D from its last healthy checkpoint (step 250) for 60 steps, each with ONE change, to find what prevents
the runaway. Figure 1: training dynamics per arm vs global step with RL-D's own trajectory (steps 200-310) as the grey reference.
Figure 2: held-out eval at the 280 and 310 checkpoints per arm vs RL-D at 250 and 300. Writes
~/shared/reports/maemm-collapse-ablation/{ablation_dynamics,ablation_evals}.{png,pdf} + data/*.json (every plotted number)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb

OUT = os.path.expanduser("~/shared/reports/maemm-collapse-ablation")
os.makedirs(f"{OUT}/data", exist_ok=True)
PROJ = "octahedral-systems/maxact-fast"
ARMS = {  # arm key -> (label written for a reader who never saw the code, colour)
    "control": ("unchanged (lr 1e-5, IS cap 5, AdamW β2 0.999 / eps 1e-8, grad clip 1.0, length penalty)", "#444444"),
    "lr3e-6": ("learning rate 3e-6 instead of 1e-5", "#2a7f62"),
    "lrdecay": ("linear lr decay (1e-5 → 0 at step 400; 4e-6 → 2.4e-6 in this window)", "#4a6fa5"),
    "adam95": ("AdamW β2 0.95 / eps 1e-15 (MiniMax-M1 settings)", "#9a3b8f"),
    "iscap2": ("IS-weight cap 2 instead of 5", "#c99a2e"),
    "nolenpen": ("no length penalty", "#b5542b"),
    "gclip03": ("gradient-norm clip 0.3 instead of 1.0", "#111111"),
}
REF_TRAIN, REF_EVAL = "bc3nzllu", "rl_D_mix1m_lr1e-5_from_realact23m_mixsft_eval"   # RL-D
TRAIN_KEYS = ["reward/mean", "policy/entropy", "grad_norm", "grad_norm_did_clip", "policy/sampler_abs_dlogp", "ratio/clipfrac", "rollout/len_mean", "lr"]
EVAL_KEYS = [("eval/mean_all", "mean over held-out families"), ("eval/sae/norm_act", "SAE norm_act"), ("eval/sae/rank1_frac", "SAE rank-1 fraction"),
             ("eval/sae/unverbalized_frac", "SAE unverbalized (lower = better)"), ("eval/realact/cos", "real activations (cos)"),
             ("eval/bsf/cos", "BSF (cos)"), ("eval/cluster/cos", "cluster probes (cos)")]

api = wandb.Api()


def train_rows(run, keys=TRAIN_KEYS):
    # wandb's keys= filter drops rows missing ANY key: the RL-D reference predates the "lr" metric, so fetch it without "lr"
    return [{"step": int(h["_step"]), **{k: h.get(k) for k in keys}} for h in run.history(keys=keys, pandas=False, samples=1000) if h.get("grad_norm") is not None]


def eval_rows(name):
    rs = list(api.runs(PROJ, filters={"display_name": name}, order="-created_at"))
    if not rs:
        return []
    rows = [r for r in rs[0].history(pandas=False) if "ckpt_step" in r and r.get("eval/mean_all") is not None]
    return sorted([{"ckpt_step": int(r["ckpt_step"]), **{k: r.get(k) for k, _ in EVAL_KEYS}} for r in rows], key=lambda r: r["ckpt_step"])


data = {"reference": {"train_id": REF_TRAIN, "eval_run": REF_EVAL, "train": [r for r in train_rows(api.run(f"{PROJ}/{REF_TRAIN}"), [k for k in TRAIN_KEYS if k != "lr"]) if 225 <= r["step"] <= 310],
                      "evals": [r for r in eval_rows(REF_EVAL) if r["ckpt_step"] in (250, 300)]},
        "arms": {}, "note": "all arms resume RL-D step_250 (adapter + optimizer state) for steps 251..310; 16 samples x 256 directions per step; bank mix_1m_v2; "
        "1 vLLM rollout + 3 HF trainer B200. Onsets: first step with grad_norm > 1.0 and first step with sampler |dlogp| > 0.05."}
for arm, (label, _) in ARMS.items():
    rs = list(api.runs(PROJ, filters={"display_name": f"rl_ablate_{arm}"}, order="-created_at"))
    tr = train_rows(rs[0]) if rs else []
    g = np.array([r["grad_norm"] for r in tr]); dl = np.array([r["policy/sampler_abs_dlogp"] for r in tr]); st = np.array([r["step"] for r in tr])
    data["arms"][arm] = {"label": label, "train_id": rs[0].id if rs else None, "state": rs[0].state if rs else None, "train": tr, "evals": eval_rows(f"rl_ablate_{arm}_eval"),
                         "onset_gnorm_gt1": int(st[g > 1][0]) if (g > 1).any() else None, "onset_dlogp_gt05": int(st[dl > 0.05][0]) if (dl > 0.05).any() else None,
                         "steps_gnorm_gt1": int((g > 1).sum()), "n_steps": len(tr),
                         "last10": {k: float(np.mean([r[k] for r in tr[-10:]])) for k in ("reward/mean", "grad_norm", "policy/sampler_abs_dlogp", "rollout/len_mean", "policy/entropy")} if tr else {}}
json.dump(data, open(f"{OUT}/data/collapse_ablation.json", "w"), indent=1)

# ---- figure 1: dynamics ----
PANELS = [("grad_norm", "gradient norm (log; dotted = clip 1.0)", True), ("policy/sampler_abs_dlogp", "sampler vs trainer |Δ log-prob| per token", True),
          ("reward/mean", "training reward (max cos, last-5 window)", False), ("policy/entropy", "policy entropy (nats/token)", False),
          ("rollout/len_mean", "mean rollout length (tokens)", False), ("ratio/clipfrac", "fraction of tokens with clipped IS ratio", True)]
fig, axes = plt.subplots(2, 3, figsize=(17, 8.6), sharex=True)
ref = data["reference"]["train"]
for ax, (k, title, logy) in zip(axes.flat, PANELS):
    ax.plot([r["step"] for r in ref], [r[k] for r in ref], color="#b0b0b0", lw=2.6, ls="--", label="RL-D itself (the run being resumed), steps 225-310")
    for arm, (label, col) in ARMS.items():
        tr = data["arms"][arm]["train"]
        if tr:
            ax.plot([r["step"] for r in tr], [r[k] for r in tr], color=col, lw=1.3, label=label)
    if k == "grad_norm":
        ax.axhline(1.0, color="#333333", ls=":", lw=1)
    if logy:
        ax.set_yscale("log")
    ax.axvline(250, color="#999999", ls="--", lw=0.8); ax.set_xlim(224, 312); ax.set_title(title, fontsize=10); ax.grid(alpha=0.25); ax.tick_params(labelsize=8.5)
for ax in axes[1]:
    ax.set_xlabel("global RL step (all arms resume RL-D at 250)", fontsize=9.5)
h, l = axes.flat[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=8.8, bbox_to_anchor=(0.5, 0.0))
surv = [a for a in ARMS if data["arms"][a]["onset_dlogp_gt05"] is None]
fig.suptitle("Resuming the collapsing RL run from its last healthy checkpoint with one change each: shrinking the effective step (lower lr, lr decay, grad clip 0.3)\n"
             f"delays the runaway; the IS-weight cap and the length penalty do not — arms with no sampler-drift onset by step 310: {', '.join(surv) if surv else 'none'}",
             fontsize=11, y=0.995)
fig.tight_layout(rect=(0, 0.17, 1, 0.95))
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/ablation_dynamics.{ext}", dpi=160)
plt.close(fig)

# ---- figure 2: evals at 280 / 310 vs D@250 / D@300 ----
fig, axes = plt.subplots(1, len(EVAL_KEYS), figsize=(3.0 * len(EVAL_KEYS), 5.2))
names = list(ARMS); x = np.arange(len(names))
refe = {r["ckpt_step"]: r for r in data["reference"]["evals"]}
for ax, (k, title) in zip(axes, EVAL_KEYS):
    v280 = [next((r[k] for r in data["arms"][a]["evals"] if r["ckpt_step"] == 280), np.nan) for a in names]
    v310 = [next((r[k] for r in data["arms"][a]["evals"] if r["ckpt_step"] == 310), np.nan) for a in names]
    ax.bar(x - 0.2, v280, 0.38, color=[ARMS[a][1] for a in names], alpha=0.45, label="checkpoint 280")
    ax.bar(x + 0.2, v310, 0.38, color=[ARMS[a][1] for a in names], label="checkpoint 310")
    if 250 in refe:
        ax.axhline(refe[250][k], color="#555555", ls="--", lw=1.1, label="RL-D at 250 (start of every arm)")
    if 300 in refe:
        ax.axhline(refe[300][k], color="#b5542b", ls=":", lw=1.3, label="RL-D at 300 (collapsed)")
    lo = np.nanmin(v280 + v310 + [refe[s][k] for s in refe]); hi = np.nanmax(v280 + v310 + [refe[s][k] for s in refe])
    ax.set_ylim(lo - 0.15 * (hi - lo + 1e-3), hi + 0.15 * (hi - lo + 1e-3))
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8); ax.set_title(title, fontsize=9.5); ax.grid(alpha=0.25, axis="y"); ax.tick_params(labelsize=8)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.0))
fig.suptitle("Held-out fidelity of each arm's 280 and 310 checkpoints vs the checkpoint they started from (RL-D 250) and RL-D's own collapsed 300",
             fontsize=11, y=0.995)
fig.tight_layout(rect=(0, 0.1, 1, 0.94))
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/ablation_evals.{ext}", dpi=160)
plt.close(fig)
print("wrote", OUT, {a: (data["arms"][a]["n_steps"], len(data["arms"][a]["evals"]), data["arms"][a]["onset_gnorm_gt1"], data["arms"][a]["onset_dlogp_gt05"]) for a in ARMS})
