"""Training-loss curve of the 104M FULL fine-tune (eff batch 4096, lr 1e-5) vs the previous best pretrain (23M LoRA r64 @ eff 512, lr 1e-4)
and the 2M full-FT sweep run, all on a matched x-axis of EXAMPLES SEEN (= optimizer step x effective batch). Second panel: held-out mean_all
per checkpoint vs examples for the same runs. Loss = mean CE over target tokens (same definition in every run). Re-run as the FFT grows.
Writes ~/shared/reports/maemm-sft-fullft-104m/loss_vs_prev.{png,pdf} + data/loss_vs_prev.json."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb

OUT = os.path.expanduser("~/shared/reports/maemm-sft-fullft-104m"); os.makedirs(f"{OUT}/data", exist_ok=True)
RUNS = {
    "fft104m": {"path": "celestedeschamphelaere-personal/maxact-fast/jp82mr9a", "eff": 4096, "eval": ("celestedeschamphelaere-personal/maxact-fast", "realact104m_fullft_b4096_lr1e-5_eval"),
                "label": "full fine-tune, 104M corpus, eff batch 4,096, lr 1e-5 (running)", "color": "#2a7f62"},
    "lora23m": {"path": "octahedral-systems/maxact-fast/da7cxuz3", "eff": 512, "eval": ("octahedral-systems/maxact-fast", "realact20m_prefix_lr1e-4_eval"),
                "label": "previous best: LoRA r64, 23M corpus, eff batch 512, lr 1e-4", "color": "#b5542b"},
    "fft2m": {"path": "octahedral-systems/maxact-fast/9ce7q1n2", "eff": 512, "eval": None,
              "label": "full fine-tune, 2M subset, eff batch 512, lr 1e-5 (sweep)", "color": "#4a6fa5"},
}
api = wandb.Api()
data = {"runs": {}}
for k, cfg in RUNS.items():
    r = api.run(cfg["path"])
    rows = [x for x in r.history(keys=["loss", "_step"], pandas=False, samples=20000) if x.get("loss") is not None]
    rows.sort(key=lambda x: x["_step"])
    steps = np.array([x["_step"] for x in rows], float); loss = np.array([x["loss"] for x in rows], float)
    ev = []
    if cfg["eval"]:
        rs = list(api.runs(cfg["eval"][0], filters={"display_name": cfg["eval"][1]}, order="-created_at"))
        if rs:
            ev = sorted([(int(x["ckpt_step"]), float(x["eval/mean_all"])) for x in rs[0].history(pandas=False) if x.get("ckpt_step") is not None and x.get("eval/mean_all") is not None])
    data["runs"][k] = {"wandb": cfg["path"], "eff_batch": cfg["eff"], "label": cfg["label"], "n_rows": len(rows), "last_step": int(steps[-1]) if len(steps) else None,
                       "loss_steps": steps.tolist(), "loss": loss.tolist(), "evals": [{"ckpt_step": s, "examples": s * cfg["eff"], "mean_all": m} for s, m in ev]}
json.dump(data, open(f"{OUT}/data/loss_vs_prev.json", "w"))

def smooth(x, y, n_ex_window=500_000, eff=1):
    stride = float(np.median(np.diff(x))) * 1e6 if len(x) > 1 else eff   # examples between logged points
    w = max(1, int(round(n_ex_window / max(stride, 1)))); 
    if len(y) < w: return x, y
    c = np.convolve(y, np.ones(w) / w, mode="valid"); return x[w - 1:], c

fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
ax = axes[0]
for k, cfg in RUNS.items():
    d = data["runs"][k]; st = np.array(d["loss_steps"]); lo = np.array(d["loss"]); ex = st * cfg["eff"] / 1e6
    ax.plot(ex, lo, color=cfg["color"], lw=0.5, alpha=0.25)
    xs, ys = smooth(ex, lo, eff=cfg["eff"]); ax.plot(xs, ys, color=cfg["color"], lw=2, label=cfg["label"] + f" — smoothed over 0.5M examples")
ax.set_xscale("log"); ax.set_xlabel("examples seen (millions, log)"); ax.set_ylabel("training loss (mean CE over target tokens)"); ax.set_ylim(1.7, 3.6); ax.grid(alpha=0.25)
ax.set_title("Training loss vs examples seen", fontsize=10.5); ax.legend(fontsize=8, loc="upper right")
ax = axes[1]
for k, cfg in RUNS.items():
    ev = data["runs"][k]["evals"]
    if ev: ax.plot([e["examples"] / 1e6 for e in ev], [e["mean_all"] for e in ev], "o-", color=cfg["color"], lw=1.6, ms=5, label=cfg["label"])
ax.axhline(0.375, color="#b5542b", ls=":", lw=1, label="LoRA 23M best checkpoint (.375)")
ax.set_xscale("log"); ax.set_xlabel("examples seen (millions, log)"); ax.set_ylabel("held-out mean fidelity (all families)"); ax.grid(alpha=0.25)
ax.set_title("Held-out fidelity per checkpoint vs examples seen", fontsize=10.5); ax.legend(fontsize=8, loc="lower right")
f = data["runs"]["fft104m"]; l = data["runs"]["lora23m"]
fl = np.array(f["loss"]); ll = np.array(l["loss"]); f_ex = f["last_step"] * 4096
# loss at matched examples: LoRA loss around the FFT's current example count
li = np.argmin(np.abs(np.array(l["loss_steps"]) * 512 - f_ex)); lo_match = float(np.mean(ll[max(0, li - 20): li + 20])); fo = float(np.mean(fl[-50:]))
fig.suptitle(f"Full fine-tune at 8x the batch tracks the LoRA pretrain's training loss at matched examples ({fo:.3f} vs {lo_match:.3f} at {f_ex / 1e6:.1f}M)\n"
             f"and its held-out fidelity (.367-.370 so far vs the LoRA run's flat .364-.375): the eval is not moving with the loss yet", fontsize=10.5, y=1.0)
fig.tight_layout(rect=(0, 0, 1, 0.94))
for ext in ("png", "pdf"): fig.savefig(f"{OUT}/loss_vs_prev.{ext}", dpi=160)
print("wrote", {k: (data["runs"][k]["n_rows"], data["runs"][k]["last_step"], [(e["ckpt_step"], round(e["mean_all"], 3)) for e in data["runs"][k]["evals"]][:8]) for k in RUNS}, "| fft last50 loss", round(fo, 3), "lora at matched examples", round(lo_match, 3))
