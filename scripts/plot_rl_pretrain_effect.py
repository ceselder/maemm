"""Does the 23M-activation SFT pretrain change what RL reaches? Same all-families RL recipe (ScaleRL, lr 7e-6, last-5 cosine
reward) from two starting points: (a) the 500k realact+probes SFT (two earlier runs on the 'everything' bank, 2048 rollouts/step),
(b) the 23M realact pretrain + 1.1M all-families midtrain (RL-C, 4096 rollouts/step). Held-out evals vs rollouts seen (log-x).
Writes ~/shared/reports/maemm-rl-ab/rl_pretrain_effect.{png,pdf} + data/rl_pretrain_effect.json."""
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
    "500k SFT init → RL 16x128 (2048 rollouts/step)": {"eval": "rl_everything_16x128_disagg_scalerl_lr7e-6_4gpu_eval", "rps": 2048, "color": "#7a7a7a", "ls": "-"},
    "500k SFT init → RL 8x256 (2048 rollouts/step)": {"eval": "rl_everything_8x256_disagg_scalerl_lr7e-6_eval", "rps": 2048, "color": "#4a6fa5", "ls": "--"},
    "23M-act pretrain + mix midtrain → RL-C 16x256 (4096 rollouts/step)": {"eval": "rl_C_mix1m_from_realact23m_mixsft_eval", "rps": 4096, "color": "#b5542b", "ls": "-"},
}
INIT = {"500k SFT init": {"eval/mean_all": 0.359, "eval/sae/norm_act": 0.61, "color": "#4a6fa5"},
        "23M pretrain + midtrain init": {"eval/mean_all": 0.340, "eval/sae/norm_act": 0.450, "eval/realact/cos": 0.444, "color": "#b5542b"}}
SKIP = {"extra/locality/n_features", "extra/locality/n_texts", "extra/adversarial/n_pending_scored"}
LABELS = {"eval/mean_all": "mean over held-out families", "eval/realact/cos": "real activations (cos)", "eval/realact_early/cos": "real acts, early positions (cos)",
          "eval/realact_mid/cos": "real acts, mid positions (cos)", "eval/realact_long/cos": "real acts, long context (cos)",
          "eval/indist_realact/cos": "in-distribution real acts (cos)", "eval/indist_long/cos": "in-distribution long acts (cos)",
          "eval/indist_probe/cos": "in-distribution probes (cos)", "eval/sae/norm_act": "SAE norm. activation", "eval/sae/fired": "SAE fired fraction",
          "eval/sae/rank1_frac": "SAE rank-1 fraction", "eval/sae/rank_le5": "SAE rank <= 5 fraction", "eval/sae/mrr": "SAE mean reciprocal rank",
          "eval/sae/mean_rank": "SAE mean rank (lower = better)", "eval/sae/unverbalized_frac": "SAE unverbalized fraction (lower = better)",
          "eval/sae/unverbalized_p10": "SAE unverbalized p10 (lower = better)", "eval/sae/beat_corpus": "SAE beats corpus max (fraction)", "eval/sae/cos": "SAE direction cos",
          "eval/bsf/cos": "BSF subspace (cos)", "eval/cluster/cos": "cluster probes (cos)", "eval/jlens/cos": "J-lens (cos, fully held-out)", "eval/random/cos": "random directions (cos, floor)",
          "extra/locality/win5_share": "activation mass in last-5 window", "extra/locality/win3_share": "activation mass in last-3 window",
          "extra/locality/win5_share_crop32": "last-5 share (32-token crop)", "extra/locality/peak_in_last5_frac": "peak in last 5 tokens (fraction)",
          "extra/locality/fire_frac": "fire fraction (locality set)", "extra/locality/feat_fire_frac": "feature fire fraction", "extra/locality/gini": "activation Gini (concentration)",
          "extra/locality/peak_share": "peak token share of activation", "extra/locality/spread_half": "tokens holding half the activation",
          "extra/locality/peak_pos_mean": "peak position (mean, relative)", "extra/locality/peak_pos_median": "peak position (median, relative)",
          "extra/locality/peak_act_fired_mean": "peak activation when fired (mean)", "extra/locality/peak_norm_best_mean": "peak norm. activation, best-of-4 (mean)",
          "extra/locality/n_tokens_mean": "generated tokens (mean)"}


def _metric_list(rows):
    ks = sorted(k for k in rows[-1] if (k.startswith("eval/") or k.startswith("extra/")) and isinstance(rows[-1][k], (int, float))
                and not k.startswith("eval/all/") and k not in SKIP)
    return [(k, LABELS.get(k, k)) for k in ks]


api = wandb.Api()
data = {"runs": {}, "inits": INIT, "note": "same ScaleRL recipe (CISPO, batch adv norm, NPR, lr 7e-6, no KL, last-5 cosine reward); "
        "banks: 'everything' (5 families x 100k) for the 500k-init runs, mix_1m_v2 (same 5 families, 1.1M) for RL-C"}
raw = {}
for label, cfg in RUNS.items():
    rs = list(api.runs(PROJ, filters={"display_name": cfg["eval"]}, order="-created_at"))
    raw[label] = sorted([r for r in rs[0].history(pandas=False) if "ckpt_step" in r and r.get("eval/mean_all") is not None], key=lambda r: r["ckpt_step"]) if rs else []
METRICS = _metric_list(raw["23M-act pretrain + mix midtrain → RL-C 16x256 (4096 rollouts/step)"])
for label, cfg in RUNS.items():
    rows = [{"ckpt_step": int(r["ckpt_step"]), "rollouts": int(r["ckpt_step"]) * cfg["rps"], **{m: r.get(m) for m, _ in METRICS}} for r in raw[label]]
    data["runs"][label] = {"eval_run": cfg["eval"], "rollouts_per_step": cfg["rps"], "evals": rows}
json.dump(data, open(f"{OUT}/data/rl_pretrain_effect.json", "w"), indent=1)

import math
ncol = 6
nrow = math.ceil(len(METRICS) / ncol)
fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 2.9 * nrow), sharex=True)
for ax in axes.flat[len(METRICS):]:
    ax.axis("off")
for ax, (m, title) in zip(axes.flat, METRICS):
    for label, cfg in RUNS.items():
        rows = [r for r in data["runs"][label]["evals"] if r.get(m) is not None]
        if rows:
            ax.plot([r["rollouts"] / 1e6 for r in rows], [r[m] for r in rows], "o", ls=cfg["ls"], color=cfg["color"], ms=3.5, lw=1.5, label=label)
    for name, vals in INIT.items():
        if m in vals:
            ax.axhline(vals[m], color=vals["color"], ls=":", lw=1.1, label=f"{name} (before RL)")
    ax.set_xscale("log"); ax.set_title(title, fontsize=8.5); ax.grid(alpha=0.25); ax.tick_params(labelsize=8)
    ax.set_xticks([0.05, 0.1, 0.2, 0.4, 0.8]); ax.set_xticklabels(["0.05", "0.1", "0.2", "0.4", "0.8"]); ax.minorticks_off()
for ax in axes[-1]:
    ax.set_xlabel("rollouts seen (millions, log)", fontsize=9)
seen = {}
for ax in axes.flat:
    for h, l in zip(*ax.get_legend_handles_labels()):
        seen.setdefault(l, h)
fig.legend(seen.values(), seen.keys(), loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.005))
fig.suptitle("Same all-families RL recipe, different starting points: the 23M-activation pretrain (+ mix midtrain) starts lower but overtakes the 500k-SFT start "
             "on the mean, SAE features, real activations and J-lens by ~0.4M rollouts; still behind on BSF and cluster probes (all logged eval metrics)", fontsize=11, y=0.995)
fig.tight_layout(rect=(0, 0.04, 1, 0.975))
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/rl_pretrain_effect.{ext}", dpi=160)
plt.close(fig)
print("wrote rl_pretrain_effect:", {k: len(v["evals"]) for k, v in data["runs"].items()})
