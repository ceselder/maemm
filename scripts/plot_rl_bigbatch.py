"""RL batch scaling incl. the 8 x 4096 run (RL-I, no warmup): held-out evals vs RL step and vs rollouts consumed for
RL-E (16x256 = 4,096 rollouts/step), RL-F (16x512 = 8,192), RL-I (8x4096 = 32,768). Same init (23M realact SFT + 1.1M mix
midtrain), same 7-family bank, ScaleRL/CISPO, constant lr 1e-5. E/F evals live in octahedral-systems/maxact-fast, I's in the
personal entity. Writes ~/shared/reports/maemm-rl-batch-scaling/bigbatch_evals.{png,pdf} + data/bigbatch.json. Re-run as RL-I grows."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb

OUT = os.path.expanduser("~/shared/reports/maemm-rl-batch-scaling")
os.makedirs(f"{OUT}/data", exist_ok=True)
RUNS = {
    "E": {"proj": "octahedral-systems/maxact-fast", "eval": "rl_E_mixmlp_from_mixsft_eval", "rps": 4096, "label": "16 x 256 = 4,096 rollouts/step (RL-E)", "color": "#4a6fa5"},
    "F": {"proj": "octahedral-systems/maxact-fast", "eval": "rl_F_mixmlp_16x512_eval", "rps": 8192, "label": "16 x 512 = 8,192 rollouts/step (RL-F)", "color": "#b5542b"},
    "I": {"proj": "celestedeschamphelaere-personal/maxact-fast", "eval": "rl_I_mixmlp_8x4096_lr1e-5_nowarm_eval", "rps": 32768, "label": "8 x 4096 = 32,768 rollouts/step (RL-I, no warmup)", "color": "#2a7f62"},
}
KEYS = [("eval/mean_all", "mean over held-out families"), ("eval/sae/norm_act", "SAE fire-rate (norm_act)"), ("eval/realact/cos", "real activations (cos)"),
        ("eval/cluster/cos", "cluster probes (cos)"), ("eval/mlp/norm_act", "MLP fire-back")]
api = wandb.Api()


def eval_rows(proj, name):
    rs = list(api.runs(proj, filters={"display_name": name}, order="-created_at"))
    rows = [r for r in rs[0].history(pandas=False) if r.get("ckpt_step") is not None and r.get("eval/mean_all") is not None] if rs else []
    return sorted([{"ckpt_step": int(r["ckpt_step"]), **{k: r.get(k) for k, _ in KEYS}} for r in rows], key=lambda r: r["ckpt_step"])


data = {"runs": {}, "note": "rollouts = ckpt_step x rollouts/step. RL-I resumed at step 25 onto the prefix-cache trainer (same math)."}
for k, cfg in RUNS.items():
    ev = eval_rows(cfg["proj"], cfg["eval"])
    data["runs"][k] = {**{x: cfg[x] for x in ("eval", "rps", "label")}, "evals": ev, "best": max(ev, key=lambda r: r["eval/mean_all"]) if ev else None}
json.dump(data, open(f"{OUT}/data/bigbatch.json", "w"), indent=1)

fig, axes = plt.subplots(2, len(KEYS), figsize=(3.4 * len(KEYS), 7.6))
for row, (xkey, xlabel) in enumerate((("step", "RL step"), ("rollouts", "rollouts consumed (millions)"))):
    for ax, (k, title) in zip(axes[row], KEYS):
        for r, cfg in RUNS.items():
            ev = [e for e in data["runs"][r]["evals"] if e.get(k) is not None]
            xs = [e["ckpt_step"] if xkey == "step" else e["ckpt_step"] * cfg["rps"] / 1e6 for e in ev]
            ax.plot(xs, [e[k] for e in ev], "o-", color=cfg["color"], lw=1.4, ms=4 if len(ev) > 1 else 8, label=cfg["label"])
        ax.set_title(title, fontsize=9.5); ax.set_xlabel(xlabel, fontsize=9); ax.grid(alpha=0.25); ax.tick_params(labelsize=8)
I = data["runs"]["I"]; E = data["runs"]["E"]
i_last = I["evals"][-1] if I["evals"] else None
same_roll = None
if i_last:
    target = i_last["ckpt_step"] * 32768 / 4096
    same_roll = min(E["evals"], key=lambda e: abs(e["ckpt_step"] - target))
head = (f"8 x 4096 at step {i_last['ckpt_step']}: mean {i_last['eval/mean_all']:.3f} — ahead of 16 x 256 at the same STEP ({next((e['eval/mean_all'] for e in E['evals'] if e['ckpt_step'] == i_last['ckpt_step']), float('nan')):.3f}), "
        f"behind it at the same ROLLOUTS (16 x 256 step {same_roll['ckpt_step']}: {same_roll['eval/mean_all']:.3f})" if i_last and same_roll else "8 x 4096 run: no eval yet")
fig.suptitle("Held-out fidelity vs RL step (top) and vs rollouts consumed (bottom) for three rollout batch sizes at constant lr 1e-5\n" + head, fontsize=10.5, y=0.995)
h, l = axes[0][0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.07, 1, 0.93))
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/bigbatch_evals.{ext}", dpi=160)
print("wrote", {r: [(e["ckpt_step"], round(e["eval/mean_all"], 3)) for e in data["runs"][r]["evals"]] for r in RUNS})


# ---- reward dynamics: E / F / I vs step and vs rollouts ----
import datetime as _dt
TRAIN = {"E": ("octahedral-systems/maxact-fast/p57zffg6", None), "F": ("octahedral-systems/maxact-fast/pu3whp5m", ("2026-09-06T04:50:00", 161)),
         "I": ("celestedeschamphetaere-personal/maxact-fast/t8brhw8t".replace("phetaere", "phelaere"), None)}
def train_rows(path, resume):
    r = api.run(path); rows = sorted([h for h in r.history(keys=["reward/mean", "policy/entropy", "grad_norm", "_timestamp"], pandas=False, samples=3000) if h.get("reward/mean") is not None], key=lambda h: h["_step"])
    if resume:
        cut = _dt.datetime.fromisoformat(resume[0]).replace(tzinfo=_dt.timezone.utc).timestamp()
        old = [h for h in rows if h["_timestamp"] < cut and int(h["_step"]) < resume[1]]; new = [h for h in rows if h["_timestamp"] >= cut]
        return [(int(h["_step"]), h["reward/mean"], h["policy/entropy"], h["grad_norm"]) for h in old] + [(resume[1] + i, h["reward/mean"], h["policy/entropy"], h["grad_norm"]) for i, h in enumerate(new)]
    return [(int(h["_step"]), h["reward/mean"], h["policy/entropy"], h["grad_norm"]) for h in rows]
dyn = {k: train_rows(*TRAIN[k]) for k in TRAIN}
data["dynamics"] = {k: [{"step": s, "reward": r, "entropy": e, "grad_norm": g} for s, r, e, g in v] for k, v in dyn.items()}
json.dump(data, open(f"{OUT}/data/bigbatch.json", "w"), indent=1)
def sm(y, w=9):
    y = np.asarray(y, float); return np.convolve(y, np.ones(w) / w, mode="same") if len(y) >= w else y
fig, axes = plt.subplots(2, 3, figsize=(16, 8.2))
for col, (key, title, logy) in enumerate((("reward", "training reward (mean cos, last-5 window)", False), ("entropy", "policy entropy (nats/token)", False), ("grad_norm", "gradient norm (log; dotted = clip 1.0)", True))):
    for row, xmode in enumerate(("step", "rollouts")):
        ax = axes[row][col]
        for k, cfg in RUNS.items():
            v = dyn[k]; xs = np.array([s for s, *_ in v], float); ys = np.array([{"reward": r, "entropy": e, "grad_norm": g}[key] for _, r, e, g in v], float)
            if xmode == "rollouts": xs = xs * cfg["rps"] / 1e6
            ax.plot(xs, ys, color=cfg["color"], lw=0.6, alpha=0.3); ax.plot(xs, sm(ys), color=cfg["color"], lw=1.8, label=cfg["label"])
        if logy: ax.set_yscale("log"); ax.axhline(1.0, color="#333", ls=":", lw=1)
        ax.set_title(title, fontsize=9.5); ax.set_xlabel("RL step" if xmode == "step" else "rollouts consumed (millions)", fontsize=9); ax.grid(alpha=0.25); ax.tick_params(labelsize=8)
h_, l_ = axes[0][0].get_legend_handles_labels(); fig.legend(h_, l_, loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.0))
I_on = 178
fig.suptitle("Reward dynamics at three rollout batch sizes, constant lr 1e-5: the 8x batch (8 x 4096) climbs the same reward curve per step, breaks EARLIER in steps (grad-norm onset 178 vs 246 / 268)\n"
             "and far later in rollouts (5.8M vs 1-2M); per rollout it is the least sample-efficient. Thin = raw, thick = 9-step moving average; RL-I resumed at steps 25 and 50 (trainer changes, same math)", fontsize=10, y=0.995)
fig.tight_layout(rect=(0, 0.06, 1, 0.93))
for ext in ("png", "pdf"): fig.savefig(f"{OUT}/bigbatch_reward.{ext}", dpi=160)
print("wrote reward dynamics:", {k: (len(v), v[-1][0], round(v[-1][1], 3)) for k, v in dyn.items()})
