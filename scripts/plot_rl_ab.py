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
    "C: init 23M-realact SFT + 1.1M all-families SFT, bank all-families mix, lr 7e-6": {"train": "oo7dvzz7", "eval": "rl_C_mix1m_from_realact23m_mixsft_eval", "color": "#2a7f62"},
    "D: same as C but lr 1e-5": {"train": "bc3nzllu", "eval": "rl_D_mix1m_lr1e-5_from_realact23m_mixsft_eval", "color": "#9a3b8f"},
    "E: same as D, bank + MLP-neuron family (7 families), lr 1e-5": {"train": "p57zffg6", "eval": "rl_E_mixmlp_from_mixsft_eval", "color": "#c99a2e"},
}
REF = {"previous best RL (16x128 ScaleRL @300, same init as B)": {"eval/mean_all": 0.4104, "eval/sae/norm_act": 0.795, "eval/sae/rank1_frac": 0.320, "eval/realact/cos": 0.525},
       "init B: 500k realact+probes SFT": {"eval/mean_all": 0.359, "eval/sae/norm_act": 0.61},
       "init A: 23M realact-only SFT final": {"eval/mean_all": 0.369, "eval/sae/norm_act": 0.418, "eval/realact/cos": 0.477},
       "init C: + 1.1M all-families SFT final": {"eval/mean_all": 0.340, "eval/sae/norm_act": 0.450, "eval/realact/cos": 0.444}}
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
    if k == "grad_norm":
        ax.set_yscale("log"); ax.axhline(1.0, color="#333333", ls=":", lw=1, label="clip (max-grad-norm 1.0)")
    ax.set_title(title, fontsize=9.5); ax.set_xlabel("RL step (4096 rollouts each)", fontsize=8.5); ax.grid(alpha=0.25); ax.tick_params(labelsize=8)
axes[0].legend(frameon=False, fontsize=8.5)
fig.suptitle("Training dynamics of the five RL runs: C, D and E all collapse under a constant learning rate (grad norm explodes once entropy falls to ~1.1: "
             "C at step ~300, D at ~273, E at ~261), A and B on the smaller banks do not", fontsize=10.5)
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
            ax.axhline(vals[m], color=["#111111", "#7a7a7a", "#c99a2e", "#9a3b8f"][i], ls=":", lw=1.2, label=rname)
    ax.set_title(title, fontsize=9.5); ax.grid(alpha=0.25); ax.tick_params(labelsize=8)
for ax in axes[1]:
    ax.set_xlabel("RL checkpoint step", fontsize=9)
seen = {}
for ax in axes.flat:
    for h, l in zip(*ax.get_legend_handles_labels()):
        seen.setdefault(l, h)
fig.legend(seen.values(), seen.keys(), loc="lower center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.03))
fig.suptitle("Held-out fidelity per RL checkpoint: from the realact+all-families SFT init, lr 1e-5 (D, E) beats lr 7e-6 (C) at every matched checkpoint and all three reach "
             "mean ~0.42 / SAE ~0.90 by step 200-250, then the constant-lr collapse drags step 300+ back down; adding the MLP family (E) costs nothing on the other families", fontsize=10, y=0.995)
fig.tight_layout(rect=(0, 0.13, 1, 0.965))
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/rl_ab_evals.{ext}", dpi=160)
plt.close(fig)

# ---- minimal report.html ----
def best(label, m="eval/mean_all"):
    rows = [r for r in data["runs"][label]["evals"] if r.get(m) is not None]
    return max(rows, key=lambda r: r[m]) if rows else None
A, B, C, D, E = list(RUNS)
bA, bB, bC, bD, bE = best(A), best(B), best(C), best(D), best(E)
def fmt(b, m="eval/mean_all"):
    return "–" if b is None else f"{b[m]:.3f} @ {b['ckpt_step']}"
html = f"""<!doctype html><html><head><meta charset="utf-8"><title>RL-A vs RL-B: init and bank composition</title>
<link rel="stylesheet" href="/reports/static/claude.css"></head><body><main>
<h1>RL runs A–E: init, bank, learning rate, and the constant-lr collapse</h1>
<div class="tldr"><b>TL;DR</b> Best mean_all per run: A {fmt(bA)}, B {fmt(bB)}, C {fmt(bC)} (SAE {bC['eval/sae/norm_act']:.3f}), D {fmt(bD)} (SAE {bD['eval/sae/norm_act']:.3f}), E {fmt(bE)} (SAE {bE['eval/sae/norm_act']:.3f}, MLP fire-back {bE.get('eval/mlp/norm_act', float('nan')):.3f}).
<b>Learning rate:</b> lr 1e-5 (D) beats lr 7e-6 (C) at every matched checkpoint (e.g. step 100: {[r for r in data['runs'][D]['evals'] if r['ckpt_step']==100][0]['eval/mean_all']:.3f} vs {[r for r in data['runs'][C]['evals'] if r['ckpt_step']==100][0]['eval/mean_all']:.3f} mean, SAE {[r for r in data['runs'][D]['evals'] if r['ckpt_step']==100][0]['eval/sae/norm_act']:.3f} vs {[r for r in data['runs'][C]['evals'] if r['ckpt_step']==100][0]['eval/sae/norm_act']:.3f}).
<b>Collapse:</b> C, D and E all blow up under the constant learning rate (no decay, no KL, no entropy term): once entropy falls to ~1.1 the gradient norm crosses the clip (C step ~300, D ~273, E ~261) and every later step is clipped, the sampler/trainer log-prob gap grows 5-10x, reward falls .35→.25 and rollouts shorten 42→28 tokens; step-300/400 checkpoints score far below step 200-250 (C@400 .315, D@300 .346). The best checkpoints are the pre-collapse ones (C step 300, D step 200/250, E step 250).
<b>MLP family:</b> E equals D on every shared family within noise while lifting held-out MLP-neuron fire-back from .475 (C@100) to {bE.get('eval/mlp/norm_act', float('nan')):.3f}.
A wins on real activations ({bA['eval/realact/cos']:.3f}) but never learns SAE/probes (its RL bank has none); B keeps SAE ~.75. Fix for the next runs: <code>--lr-decay linear</code> (added to rl_disagg.py, not yet used). Regenerated by scripts/plot_rl_ab.py.</div>
<figure><img src="rl_ab_evals.png" alt="evals per checkpoint"><figcaption>Every held-out metric per RL checkpoint (512 per family, best-of-4, T=1). Dotted: references. Data: data/rl_ab.json.</figcaption></figure>
<figure><img src="rl_ab_train.png" alt="training dynamics"><figcaption>Training reward, entropy, grad norm, rollout length per RL step. Data: data/rl_ab.json.</figcaption></figure>
<h2>Does the SFT init matter? Same RL recipe from three starting points, every logged eval metric</h2>
<figure><img src="rl_pretrain_effect.png" alt="pretrain effect on RL"><figcaption>Same ScaleRL recipe (lr 7e-6, last-5 cosine reward), x = rollouts seen (log). Inits: SFT on real activations only
(23M examples; RL-A, whose bank is real activations only), SFT on the 500k realact+probes datamix (the earlier 8x256 run on the all-families bank, 2048 rollouts/step), and SFT on real activations (23M)
continued on the 1.1M all-families datamix (RL-C, all-families bank, 4096 rollouts/step). Twelve non-duplicate metrics are shown (mean, real acts short/long, SAE norm_act / rank-1 / unverbalized / cosine, BSF, cluster probes, J-lens, random control,
last-5 activation share); every logged metric is still in the JSON. Dotted lines: each init before RL. Data: data/rl_pretrain_effect.json; script scripts/plot_rl_pretrain_effect.py.</figcaption></figure>
<h2>Every held-out eval on one axis, per arm</h2>
<figure><img src="rl_all_evals_one_axis.png" alt="all evals on one axis per arm"><figcaption>One panel per RL arm, every named held-out eval as a line on a shared axis: SAE norm_act
(feature activation on generated text ÷ corpus max), SAE rank-1 fraction, SAE unverbalized fraction (lower = better), SAE/BSF/cluster-probe/real-activation (short and long context)
cosines, J-lens cosine (unembed row pulled back to layer 42), and random directions as the control floor. Same data as the figure above (data/rl_pretrain_effect.json).</figcaption></figure>
<details><summary>Setup</summary><ul>
<li>Both: 8xB200 (2 vLLM rollout + 6 HF trainer), 16 rollouts x 256 prompts/step, ScaleRL recipe (CISPO eps 5, prompt-level agg, batch-level advantage norm, zero-var filter, NPR 0.9 @ cos 0.7), lr 7e-6 (25-step warmup), no KL/entropy term, length penalty 0.00025/token past 8, max 192 new tokens, reward = max cosine over the last 5 tokens of the clean base L42 activation, 400 steps, autocast bf16, fp32 head.</li>
<li>A: init /data/sft_mix/realact20m_prefix_lr1e-4/final (23M realact-only SFT), bank /data/banks/rl_randctx (200k realact directions, ctx ~U[9,512]). wandb train a8o4ea1i.</li>
<li>B: init /data/sft_mix/last5_rp/final (500k realact+probes SFT), bank /data/banks/rl_randctx_probes (same 200k + 100k eval-excluded cluster probes). wandb train vrdgg54v.</li>
<li>C: init /data/sft_mix/mix1m_from_realact23m/final (the 23M realact SFT continued for one epoch on the 1.1M all-families mix, lr 1e-4), bank /data/banks/mix_1m_v2 (250k realact p14-91 + 250k realact_long p256-511 + 236k SAE-feature dirs + 118k BSF dirs + 242k cluster probes, eval hold-outs excluded), lr 7e-6. wandb train oo7dvzz7. Collapsed after step 300 (grad norm 4-50, clipped every step).</li>
<li>D: identical to C with lr 1e-5. wandb train bc3nzllu, save /data/ckpts_rl_D_mix1m_lr1e-5. Collapsed after step ~273.</li>
<li>E: identical to D but bank /data/banks/mix_1m_mlp (mix_1m_v2 + 113,814 layer-42 MLP-neuron / co-firing-pair directions = 1,209,088 rows, 7 families); evaluated with eval cache v2 (adds held-out mlp / mlp_pair families: cosine + fire-back). wandb train p57zffg6, save /data/ckpts_rl_E_mixmlp. Collapse onset step ~261.</li>
<li>Collapse diagnosis (wandb oo7dvzz7 / bc3nzllu / p57zffg6): grad_norm_did_clip flips to 1 and stays; policy/sampler_abs_dlogp .03→.1-.3; rollout/mean_logp -1.3→-3.8; no NaN, no truncation, NPR dropped 0.1% of the bank, 256/256 groups kept, off-policy lag ≤2. Trainer only had --warmup-steps (constant lr after warmup); --lr-decay {{linear,cosine}} + --lr-min-frac added on 2026-09-05.</li>
</ul></details></main></body></html>"""
open(f"{OUT}/report.html", "w").write(html)
print(f"wrote {OUT}: A evals {len(data['runs'][A]['evals'])} rows ({data['runs'][A]['state']}), B evals {len(data['runs'][B]['evals'])} rows ({data['runs'][B]['state']})")
