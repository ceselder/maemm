"""Collapse ablation (Sep 5 2026): RL-D (Qwen3.6-27B inverter, ScaleRL/CISPO, lr 1e-5 constant) blew up after step ~273.
Arms resume RL-D from its last healthy checkpoint (step 250) for 60 steps, each with ONE change, to find what prevents the
runaway. Round 1 = step-size / clipping / penalty knobs; round 2 = mechanism-level knobs (advantage normalization, batch,
trust region, KL, entropy floor, fresh optimizer state).
Per round: figure 1 = training dynamics per arm vs global step with RL-D (and, for round 2, the round-1 control) as grey
references; figure 2 = held-out eval at the 280 and 310 checkpoints per arm vs RL-D at 250 and 300.
Writes ~/shared/reports/maemm-collapse-ablation/{ablation,ablation2}_{dynamics,evals}.{png,pdf} + data/collapse_ablation{,2}.json."""
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
ROUND1 = {  # arm key -> (label for a reader who never saw the code, colour)
    "control": ("unchanged (lr 1e-5, IS cap 5, AdamW β2 0.999 / eps 1e-8, grad clip 1.0, length penalty)", "#444444"),
    "lr3e-6": ("learning rate 3e-6 instead of 1e-5", "#2a7f62"),
    "lrdecay": ("linear lr decay (1e-5 → 0 at step 400; 4e-6 → 2.4e-6 in this window)", "#4a6fa5"),
    "adam95": ("AdamW β2 0.95 / eps 1e-15 (MiniMax-M1 settings)", "#9a3b8f"),
    "iscap2": ("IS-weight cap 2 instead of 5", "#c99a2e"),
    "nolenpen": ("no length penalty", "#b5542b"),
    "gclip03": ("gradient-norm clip 0.3 instead of 1.0", "#111111"),
}
ROUND2 = {
    "control_fresh": ("unchanged, fresh AdamW moments (optimizer state not loaded)", "#444444"),
    "rawadv_fresh": ("raw advantages (no batch std normalization), fresh AdamW moments", "#2a7f62"),
    "rawadv": ("raw advantages, LOADED AdamW moments (≈ lr/25 for ~100s of steps — confounded)", "#8fd1b5"),
    "rawadv_gclip03": ("raw advantages + grad clip 0.3, loaded moments (confounded the same way)", "#c7e9d6"),
    "groups512": ("512 directions per step (2x batch, 8192 rollouts)", "#4a6fa5"),
    "ppoclip": ("PPO clipped surrogate eps 0.2 instead of CISPO", "#c99a2e"),
    "kl002": ("KL 0.02 to the SFT init", "#9a3b8f"),
    "enttarget": ("adaptive entropy floor 1.2", "#b5542b"),
}
REF_TRAIN, REF_EVAL = "bc3nzllu", "rl_D_mix1m_lr1e-5_from_realact23m_mixsft_eval"   # RL-D
TRAIN_KEYS = ["reward/mean", "policy/entropy", "grad_norm", "grad_norm_did_clip", "policy/sampler_abs_dlogp", "ratio/clipfrac", "rollout/len_mean", "lr", "policy/kl_to_init"]
EVAL_KEYS = [("eval/mean_all", "mean over held-out families"), ("eval/sae/norm_act", "SAE norm_act"), ("eval/sae/rank1_frac", "SAE rank-1 fraction"),
             ("eval/sae/unverbalized_frac", "SAE unverbalized (lower = better)"), ("eval/realact/cos", "real activations (cos)"),
             ("eval/bsf/cos", "BSF (cos)"), ("eval/cluster/cos", "cluster probes (cos)")]
PANELS = [("grad_norm", "gradient norm (log; dotted = clip 1.0)", True), ("policy/sampler_abs_dlogp", "sampler vs trainer |Δ log-prob| per token", True),
          ("reward/mean", "training reward (max cos, last-5 window)", False), ("policy/entropy", "policy entropy (nats/token)", False),
          ("rollout/len_mean", "mean rollout length (tokens)", False), ("ratio/clipfrac", "fraction of tokens with clipped IS ratio", True)]

api = wandb.Api()


def train_rows(run, keys=TRAIN_KEYS):
    """wandb's keys= filter drops rows missing ANY requested key, so only request keys the run actually logged."""
    logged = [k for k in keys if k in run.summary]
    for must in ("reward/mean", "grad_norm"):
        if must not in logged:
            logged.append(must)
    rows = run.history(keys=logged, pandas=False, samples=1000)
    return [{"step": int(h["_step"]), **{k: h.get(k) for k in keys}} for h in rows if h.get("grad_norm") is not None]


def eval_rows(name):
    rs = list(api.runs(PROJ, filters={"display_name": name}, order="-created_at"))
    if not rs:
        return []
    rows = [r for r in rs[0].history(pandas=False) if "ckpt_step" in r and r.get("eval/mean_all") is not None]
    return sorted([{"ckpt_step": int(r["ckpt_step"]), **{k: r.get(k) for k, _ in EVAL_KEYS}} for r in rows], key=lambda r: r["ckpt_step"])


def arm_record(label, run_name, eval_name):
    rs = list(api.runs(PROJ, filters={"display_name": run_name}, order="-created_at"))
    tr = train_rows(rs[0]) if rs else []
    g = np.array([r["grad_norm"] for r in tr]); dl = np.array([r["policy/sampler_abs_dlogp"] for r in tr]); st = np.array([r["step"] for r in tr])
    return {"label": label, "train_id": rs[0].id if rs else None, "state": rs[0].state if rs else None, "train": tr, "evals": eval_rows(eval_name),
            "onset_gnorm_gt1": int(st[g > 1][0]) if (g > 1).any() else None, "onset_dlogp_gt05": int(st[dl > 0.05][0]) if (dl > 0.05).any() else None,
            "steps_gnorm_gt1": int((g > 1).sum()), "n_steps": len(tr),
            "last10": {k: float(np.mean([r[k] for r in tr[-10:]])) for k in ("reward/mean", "grad_norm", "policy/sampler_abs_dlogp", "rollout/len_mean", "policy/entropy")} if tr else {}}


def render(arms, run_prefix, prefix, title_dyn, extra_refs):
    """arms: key -> (label, colour); run_prefix: wandb run-name prefix; extra_refs: list of (label, train rows) plotted as extra grey references."""
    ref_run = api.run(f"{PROJ}/{REF_TRAIN}")
    data = {"reference": {"train_id": REF_TRAIN, "eval_run": REF_EVAL, "train": [r for r in train_rows(ref_run) if 225 <= r["step"] <= 310],
                          "evals": [r for r in eval_rows(REF_EVAL) if r["ckpt_step"] in (250, 300)]},
            "arms": {a: arm_record(lab, f"{run_prefix}{a}", f"{run_prefix}{a}_eval") for a, (lab, _) in arms.items()},
            "note": "all arms resume RL-D step_250 (adapter; optimizer state unless 'fresh') for steps 251..310; 16 samples x 256 directions per step unless stated; "
                    "bank mix_1m_v2; 1 vLLM rollout + 3 HF trainer B200. Onsets: first step with grad_norm > 1.0 and first step with sampler |dlogp| > 0.05."}
    json.dump(data, open(f"{OUT}/data/collapse_{prefix}.json", "w"), indent=1)

    fig, axes = plt.subplots(2, 3, figsize=(17, 8.8), sharex=True)
    ref = data["reference"]["train"]
    for ax, (k, title, logy) in zip(axes.flat, PANELS):
        ax.plot([r["step"] for r in ref], [r[k] for r in ref], color="#b0b0b0", lw=2.6, ls="--", label="RL-D itself (the run being resumed), steps 225-310")
        for rl, rrows in extra_refs:
            ax.plot([r["step"] for r in rrows], [r[k] for r in rrows], color="#d8c8b0", lw=2.2, ls=":", label=rl)
        for a, (label, col) in arms.items():
            tr = data["arms"][a]["train"]
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
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=8.6, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(title_dyn(data), fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0.19, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/{prefix}_dynamics.{ext}", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(EVAL_KEYS), figsize=(3.0 * len(EVAL_KEYS), 5.6))
    names = list(arms); x = np.arange(len(names)); refe = {r["ckpt_step"]: r for r in data["reference"]["evals"]}
    for ax, (k, title) in zip(axes, EVAL_KEYS):
        v280 = [next((r[k] for r in data["arms"][a]["evals"] if r["ckpt_step"] == 280), np.nan) for a in names]
        v310 = [next((r[k] for r in data["arms"][a]["evals"] if r["ckpt_step"] == 310), np.nan) for a in names]
        ax.bar(x - 0.2, v280, 0.38, color=[arms[a][1] for a in names], alpha=0.45, label="checkpoint 280")
        ax.bar(x + 0.2, v310, 0.38, color=[arms[a][1] for a in names], label="checkpoint 310")
        if 250 in refe:
            ax.axhline(refe[250][k], color="#555555", ls="--", lw=1.1, label="RL-D at 250 (start of every arm)")
        if 300 in refe:
            ax.axhline(refe[300][k], color="#b5542b", ls=":", lw=1.3, label="RL-D at 300 (collapsed)")
        vals = [v for v in v280 + v310 + [refe[s][k] for s in refe] if v is not None and not np.isnan(v)]
        lo, hi = min(vals), max(vals); ax.set_ylim(lo - 0.15 * (hi - lo + 1e-3), hi + 0.15 * (hi - lo + 1e-3))
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8); ax.set_title(title, fontsize=9.5); ax.grid(alpha=0.25, axis="y"); ax.tick_params(labelsize=8)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Held-out fidelity of each arm's 280 and 310 checkpoints vs the checkpoint they started from (RL-D 250) and RL-D's own collapsed 300", fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/{prefix}_evals.{ext}", dpi=160)
    plt.close(fig)
    print(f"wrote {prefix}:", {a: (data["arms"][a]["n_steps"], len(data["arms"][a]["evals"]), data["arms"][a]["onset_gnorm_gt1"], data["arms"][a]["onset_dlogp_gt05"]) for a in arms})
    return data


def title1(d):
    surv = [a for a in ROUND1 if d["arms"][a]["onset_dlogp_gt05"] is None]
    return ("Resuming the collapsing RL run from its last healthy checkpoint with one change each: shrinking the effective step (lower lr, lr decay, grad clip 0.3)\n"
            f"delays the runaway; the IS-weight cap and the length penalty do not — arms with no sampler-drift onset by step 310: {', '.join(surv) if surv else 'none'}")


def title2(d):
    on = {a: d["arms"][a]["onset_dlogp_gt05"] for a in ROUND2}
    fmt = lambda a: str(on[a]) if on[a] else "none"
    return ("Round 2, mechanism-level changes at the SAME lr 1e-5: none prevents the runaway. Sampler-drift onset — fresh Adam moments "
            f"{fmt('control_fresh')} (control 277), raw advantages {fmt('rawadv_fresh')}, 2x batch {fmt('groups512')}, PPO clip {fmt('ppoclip')}, KL 0.02 {fmt('kl002')}, entropy floor {fmt('enttarget')};\n"
            "the raw-advantage arms with loaded Adam moments look stable only because the stale second moment cuts their effective lr ~25x")


d1 = render(ROUND1, "rl_ablate_", "ablation", title1, [])
render(ROUND2, "rl_ablate2_", "ablation2", title2, [("round-1 control (same recipe, loaded moments)", d1["arms"]["control"]["train"])])
