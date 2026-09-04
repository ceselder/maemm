"""RL-A vs RL-B (Sep 4): training reward/entropy per step and every held-out eval metric per checkpoint, for the two
parallel RL runs (A: init = 23M realact-only SFT final, bank = random-ctx real activations; B: init = 500k realact+probes
SFT, bank = random-ctx real activations + cluster probes). Writes ~/shared/reports/maemm-rl-ab/{rl_ab_train,rl_ab_evals}.{png,pdf}
+ data/*.json + a minimal report.html. Every plotted number is in data/."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

OUT = os.path.expanduser("~/shared/reports/maemm-rl-ab")
os.makedirs(f"{OUT}/data", exist_ok=True)
PROJ = "octahedral-systems/maxact-fast"
RUNS = {
    "A: init 23M realact-only SFT, bank random-ctx acts": {"train": "a8o4ea1i", "eval": "rl_A_randctx_from_realact23m_eval", "color": "#b5542b"},
    "B: init 500k realact+probes SFT, bank random-ctx acts + probes": {"train": "vrdgg54v", "eval": "rl_B_randctx_probes_from_rp500k_eval", "color": "#4a6fa5"},
}
REF = {"previous best RL (16x128 ScaleRL @300, same init as B)": {"eval/mean_all": 0.4104, "eval/sae/norm_act": 0.795, "eval/sae/rank1_frac": 0.320, "eval/realact/cos": 0.525},
       "init B: 500k realact+probes SFT": {"eval/mean_all": 0.359, "eval/sae/norm_act": 0.61},
       "init A: 23M realact-only SFT final": {"eval/mean_all": 0.369, "eval/sae/norm_act": 0.418, "eval/realact/cos": 0.477}}
TRAIN_KEYS = ["reward/mean", "policy/entropy", "grad_norm", "rollout/len_mean"]
EVAL = [("eval/mean_all", "mean over held-out families"), ("eval/realact/cos", "real activations (cos)"),
        ("eval/realact_long/cos", "long-context real activations (cos)"), ("eval/sae/norm_act", "SAE features (norm. activation)"),
        ("eval/sae/rank1_frac", "SAE rank-1 fraction"), ("eval/bsf/cos", "BSF subspace (cos)"), ("eval/cluster/cos", "cluster probes (cos)"),
        ("extra/locality/win5_share", "activation mass in last-5 window")]

api = wandb.Api()
data = {"runs": {}, "references": REF}
for label, cfg in RUNS.items():
    tr = api.run(f"{PROJ}/{cfg['train']}")
    hist = tr.history(keys=TRAIN_KEYS, pandas=False, samples=2000)
    train = [{"step": int(h["_step"]), **{k: h.get(k) for k in TRAIN_KEYS}} for h in hist if h.get("reward/mean") is not None]
    ev_runs = list(api.runs(PROJ, filters={"display_name": cfg["eval"]}, order="-created_at"))
    rows = []
    if ev_runs:
        rows = sorted([r for r in ev_runs[0].history(pandas=False) if "ckpt_step" in r and r.get("eval/mean_all") is not None], key=lambda r: r["ckpt_step"])
        rows = [{"ckpt_step": int(r["ckpt_step"]), **{m: r.get(m) for m, _ in EVAL}} for r in rows]
    data["runs"][label] = {"train_id": cfg["train"], "eval_run": cfg["eval"], "state": tr.state, "train": train, "evals": rows}
json.dump(data, open(f"{OUT}/data/rl_ab.json", "w"), indent=1)

# ---- figure 1: training dynamics ----
fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
for ax, (k, title) in zip(axes, [("reward/mean", "training reward (max cos, last-5 window)"), ("policy/entropy", "policy entropy (nats/token)"),
                                 ("grad_norm", "grad norm"), ("rollout/len_mean", "mean rollout length (tokens)")]):
    for label, cfg in RUNS.items():
        pts = [(p["step"], p[k]) for p in data["runs"][label]["train"] if p.get(k) is not None]
        if pts:
            ax.plot([s for s, _ in pts], [v for _, v in pts], color=cfg["color"], lw=1.2, alpha=0.9, label=label.split(":")[0])
    ax.set_title(title, fontsize=9.5); ax.set_xlabel("RL step (4096 rollouts each)", fontsize=8.5); ax.grid(alpha=0.25); ax.tick_params(labelsize=8)
axes[0].legend(frameon=False, fontsize=8.5)
fig.suptitle("RL from two inits on random-context activation banks: A starts with higher reward on its activation-only bank, B carries the probe families",
             fontsize=10.5)
fig.tight_layout(rect=(0, 0, 1, 0.93))
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/rl_ab_train.{ext}", dpi=160)
plt.close(fig)

# ---- figure 2: evals per checkpoint ----
fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True)
for ax, (m, title) in zip(axes.flat, EVAL):
    for label, cfg in RUNS.items():
        rows = [r for r in data["runs"][label]["evals"] if r.get(m) is not None]
        if rows:
            ax.plot([r["ckpt_step"] for r in rows], [r[m] for r in rows], "o-", color=cfg["color"], ms=3.5, lw=1.5, label=label)
    for i, (rname, vals) in enumerate(REF.items()):
        if m in vals:
            ax.axhline(vals[m], color=["#2a7f62", "#7a7a7a", "#c99a2e"][i], ls=":", lw=1.2, label=rname)
    ax.set_title(title, fontsize=9.5); ax.grid(alpha=0.25); ax.tick_params(labelsize=8)
for ax in axes[1]:
    ax.set_xlabel("RL checkpoint step", fontsize=9)
seen = {}
for ax in axes.flat:
    for h, l in zip(*ax.get_legend_handles_labels()):
        seen.setdefault(l, h)
fig.legend(seen.values(), seen.keys(), loc="lower center", ncol=2, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Held-out fidelity per RL checkpoint: both inits reach mean_all ≈0.41; the probe-trained init B keeps SAE fidelity (~0.75) that "
             "the realact-only init A never recovers (~0.46)", fontsize=10.5, y=0.995)
fig.tight_layout(rect=(0, 0.08, 1, 0.965))
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/rl_ab_evals.{ext}", dpi=160)
plt.close(fig)

# ---- minimal report.html ----
def best(label, m="eval/mean_all"):
    rows = [r for r in data["runs"][label]["evals"] if r.get(m) is not None]
    return max(rows, key=lambda r: r[m]) if rows else None
A, B = list(RUNS)
bA, bB = best(A), best(B)
html = f"""<!doctype html><html><head><meta charset="utf-8"><title>RL-A vs RL-B: init and bank composition</title>
<link rel="stylesheet" href="/reports/static/claude.css"></head><body><main>
<h1>RL-A vs RL-B: does the 23M realact SFT make a better RL start?</h1>
<div class="tldr"><b>TL;DR</b> Both runs reach mean_all ≈ 0.41 (A best {bA['eval/mean_all']:.3f} @ {bA['ckpt_step']}, B best {bB['eval/mean_all']:.3f} @ {bB['ckpt_step']}).
A wins on real activations ({bA['eval/realact/cos']:.3f} vs {bB['eval/realact/cos']:.3f}, the best realact cosine of any run) but its SAE-feature fidelity stays at
{bA['eval/sae/norm_act']:.2f} vs {bB['eval/sae/norm_act']:.2f} for B: the realact-only SFT init lost SAE competence and a realact-only RL bank does not teach it back.
Both runs complete (400 steps each). Regenerated by scripts/plot_rl_ab.py.</div>
<figure><img src="rl_ab_evals.png" alt="evals per checkpoint"><figcaption>Every held-out metric per RL checkpoint (512 per family, best-of-4, T=1). Dotted: references. Data: data/rl_ab.json.</figcaption></figure>
<figure><img src="rl_ab_train.png" alt="training dynamics"><figcaption>Training reward, entropy, grad norm, rollout length per RL step. Data: data/rl_ab.json.</figcaption></figure>
<details><summary>Setup</summary><ul>
<li>Both: 8xB200 (2 vLLM rollout + 6 HF trainer), 16 rollouts x 256 prompts/step, ScaleRL recipe (CISPO eps 5, prompt-level agg, batch-level advantage norm, zero-var filter, NPR 0.9 @ cos 0.7), lr 7e-6 (25-step warmup), no KL/entropy term, length penalty 0.00025/token past 8, max 192 new tokens, reward = max cosine over the last 5 tokens of the clean base L42 activation, 400 steps, autocast bf16, fp32 head.</li>
<li>A: init /data/sft_mix/realact20m_prefix_lr1e-4/final (23M realact-only SFT), bank /data/banks/rl_randctx (200k realact directions, ctx ~U[9,512]). wandb train a8o4ea1i.</li>
<li>B: init /data/sft_mix/last5_rp/final (500k realact+probes SFT), bank /data/banks/rl_randctx_probes (same 200k + 100k eval-excluded cluster probes). wandb train vrdgg54v.</li>
</ul></details></main></body></html>"""
open(f"{OUT}/report.html", "w").write(html)
print(f"wrote {OUT}: A evals {len(data['runs'][A]['evals'])} rows ({data['runs'][A]['state']}), B evals {len(data['runs'][B]['evals'])} rows ({data['runs'][B]['state']})")
