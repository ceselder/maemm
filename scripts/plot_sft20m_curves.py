"""Scaled realact-only SFT (23M examples, Qwen3.6-27B inverter): training loss vs step and every held-out eval metric vs
examples seen, with the RL bests and the older SFT eval curve (2.5M realact+probes) as references.

Writes ~/shared/reports/maemm-sft-20m/{sft20m_train_loss,sft20m_evals_vs_examples}.{png,pdf} + data/*.json + a minimal
report.html (extended later by build_html.py). Every number plotted is in data/.
"""
import json
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

OUT = os.path.expanduser("~/shared/reports/maemm-sft-20m")
os.makedirs(f"{OUT}/data", exist_ok=True)
PROJ = "octahedral-systems/maxact-fast"
TRAIN_ID, EVAL_NAME, EFF_BATCH = "da7cxuz3", "realact20m_prefix_lr1e-4_eval", 512
OLD = {"name": "sft_big_rp_eval", "eff_batch": 128, "label": "2.5M realact+probes SFT (Sep 1)"}
REF = {"old 500k SFT init (realact+probes)": {"eval/mean_all": 0.359, "eval/sae/norm_act": 0.61},
       "best RL (16x128 ScaleRL ckpt 300)": {"eval/mean_all": 0.4104, "eval/sae/norm_act": 0.795, "eval/sae/rank1_frac": 0.320, "eval/realact/cos": 0.525},
       "8x256 ScaleRL ckpt 160": {"eval/sae/norm_act": 0.820, "eval/sae/rank1_frac": 0.363}}
METRICS = [("eval/mean_all", "mean over held-out families"), ("eval/realact/cos", "real activations (cos)"),
           ("eval/sae/norm_act", "SAE features (norm. activation)"), ("eval/sae/rank1_frac", "SAE rank-1 fraction"),
           ("eval/bsf/cos", "BSF subspace (cos)"), ("eval/cluster/cos", "cluster probes (cos)"), ("eval/jlens/cos", "J-lens (cos)"),
           ("extra/locality/win5_share", "activation mass in last-5 window")]

api = wandb.Api()

# ---- training loss ----
tr = api.run(f"{PROJ}/{TRAIN_ID}")
hist = [h for h in tr.history(keys=["loss", "lr"], pandas=False, samples=5000) if h.get("loss") is not None]
loss = {"run": tr.name, "id": TRAIN_ID, "eff_batch": EFF_BATCH, "steps_total": 44922,
        "points": [{"step": int(h["_step"]), "examples": int(h["_step"]) * EFF_BATCH, "loss": float(h["loss"]), "lr": h.get("lr")} for h in hist]}
json.dump(loss, open(f"{OUT}/data/train_loss.json", "w"), indent=1)


def eval_rows(name, eff_batch):
    runs = list(api.runs(PROJ, filters={"display_name": name}, order="-created_at"))
    if not runs:
        return None
    rows = [r for r in runs[0].history(pandas=False) if "ckpt_step" in r and r.get("eval/mean_all") is not None]
    rows.sort(key=lambda r: r["ckpt_step"])
    keys = [m for m, _ in METRICS]
    return {"run": name, "id": runs[0].id, "eff_batch": eff_batch,
            "rows": [{"ckpt_step": int(r["ckpt_step"]), "examples": int(r["ckpt_step"]) * eff_batch, **{k: r.get(k) for k in keys}} for r in rows]}


ev = eval_rows(EVAL_NAME, EFF_BATCH)
old = eval_rows(OLD["name"], OLD["eff_batch"])
json.dump({"new": ev, "old": old, "references": REF}, open(f"{OUT}/data/evals_vs_examples.json", "w"), indent=1)

# ---- figure 1: training loss ----
steps = [p["step"] for p in loss["points"]]; ls = [p["loss"] for p in loss["points"]]
win = 20
smooth = [statistics.mean(ls[max(0, i - win + 1): i + 1]) for i in range(len(ls))]
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.plot([s * EFF_BATCH / 1e6 for s in steps], ls, color="#c9c2b8", lw=0.8, label="per-log-step loss (every 50 steps)")
ax.plot([s * EFF_BATCH / 1e6 for s in steps], smooth, color="#b5542b", lw=2, label=f"{win}-point running mean")
ax.set_xlabel("examples seen (millions)"); ax.set_ylabel("next-token cross-entropy (target tokens)")
ax.set_title("Training loss keeps falling across the 23M-example realact-only SFT (Qwen3.6-27B inverter, lr 1e-4, eff. batch 512)", fontsize=9.5)
ax.grid(alpha=0.25); ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/sft20m_train_loss.{ext}", dpi=160)
plt.close(fig)

# ---- figure 2: evals vs examples ----
fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True)
for ax, (m, label) in zip(axes.flat, METRICS):
    if ev:
        xs = [r["examples"] / 1e6 for r in ev["rows"] if r.get(m) is not None]
        ys = [r[m] for r in ev["rows"] if r.get(m) is not None]
        ax.plot(xs, ys, "o-", color="#b5542b", ms=3.5, lw=1.5, label="this run: 23M realact-only, prefix-cache")
    if old:
        xs = [r["examples"] / 1e6 for r in old["rows"] if r.get(m) is not None]
        ys = [r[m] for r in old["rows"] if r.get(m) is not None]
        if xs:
            ax.plot(xs, ys, "s--", color="#4a6fa5", ms=3, lw=1.2, label=OLD["label"])
    for i, (rname, vals) in enumerate(REF.items()):
        if m in vals:
            ax.axhline(vals[m], color=["#777777", "#2a7f62", "#6b4c9a"][i], ls=":", lw=1.2, label=rname)
    ax.set_title(label, fontsize=9.5); ax.grid(alpha=0.25); ax.set_xscale("log")
    ax.tick_params(labelsize=8)
for ax in axes[1]:
    ax.set_xlabel("examples seen (millions, log)", fontsize=9)
handles, labels = axes[0, 0].get_legend_handles_labels()
seen = {}
for h, l in zip(handles, labels):
    seen.setdefault(l, h)
fig.legend(seen.values(), seen.keys(), loc="lower center", ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.01))
fig.suptitle("Held-out fidelity is flat from 0.6M to 11.5M examples of realact-only SFT while RL (dotted) is the only thing that moved it",
             fontsize=11, y=0.995)
fig.tight_layout(rect=(0, 0.06, 1, 0.97))
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/sft20m_evals_vs_examples.{ext}", dpi=160)
plt.close(fig)

# ---- minimal report.html (claude.css) ----
last = ev["rows"][-1] if ev else {}
html = f"""<!doctype html><html><head><meta charset="utf-8"><title>23M realact-only SFT: does scale move fidelity?</title>
<link rel="stylesheet" href="/reports/static/claude.css"></head><body><main>
<h1>23M-example realact-only SFT of the Qwen3.6-27B inverter</h1>
<div class="tldr"><b>TL;DR</b> Training loss keeps falling (2.44 → 1.95) but every held-out fidelity metric is flat from 0.6M to
{last.get('examples', 0) / 1e6:.1f}M examples: mean_all {min(r['eval/mean_all'] for r in ev['rows']):.3f}–{max(r['eval/mean_all'] for r in ev['rows']):.3f},
realact 0.470–0.483, SAE norm_act drifting down 0.478 → 0.45. The only thing that has moved these metrics is RL with the cosine reward (0.359 → 0.410).
Live run; this page is regenerated by scripts/plot_sft20m_curves.py.</div>
<figure><img src="sft20m_evals_vs_examples.png" alt="evals vs examples"><figcaption>Every eval metric vs examples seen (log-x), with the older
2.5M realact+probes SFT curve (if available) and RL bests as dotted references. Data: data/evals_vs_examples.json.</figcaption></figure>
<figure><img src="sft20m_train_loss.png" alt="train loss"><figcaption>Training loss vs examples seen. Data: data/train_loss.json.</figcaption></figure>
<details><summary>Run details</summary><ul>
<li>Bank: /data/banks/realact_short_20m_all — 23,000,000 (direction, target) pairs from 860,824 FineFineWeb docs; ctx 8–256 tokens, targets 8–32 tokens ending at the firing token; direction = unit(act_L42 − μ).</li>
<li>Trainer: sft/pretrain.py --prefix-cache (exact shared prompt prefix, transformers fork ceselder/transformers@e52940e), --autocast-bf16, no grad-ckpt, micro-batch 64/GPU × 8 B200 = eff. batch 512, lr 1e-4 OneCycle, max_seq 160, LoRA r64/α16 rsLoRA. wandb train {TRAIN_ID}, eval {ev['id'] if ev else '-'}.</li>
<li>Eval: 512 per family, best-of-4 at T=1, same protocol as the RL runs (eval/eval_ckpt_daemon.py).</li>
</ul></details></main></body></html>"""
open(f"{OUT}/report.html", "w").write(html)
print(f"wrote {OUT}: train points {len(loss['points'])}, eval rows {len(ev['rows']) if ev else 0}, old rows {len(old['rows']) if old else 'none'}")
