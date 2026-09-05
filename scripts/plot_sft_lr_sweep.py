"""SFT (pretrain objective) learning-rate sweep at a fixed 2M-example budget (Qwen3.6-27B inverter, LoRA r64 rsLoRA):
five arms lr in {3e-5, 1e-4, 3e-4, 1e-3, 3e-3}, identical data (seeded 2M subset of the 23M realact bank), schedule
(OneCycle, 2% warmup, linear anneal over 3907 steps at eff. batch 512) and trainer (--prefix-cache). Pulls every arm's
training loss + held-out eval rows from wandb and writes ~/shared/reports/maemm-sft-lr-sweep/
{lr_sweep_final_vs_lr,lr_sweep_curves,lr_sweep_train_loss}.{png,pdf} + data/*.json. Every plotted number is in data/.
report.html is rendered by ~/shared/reports/maemm-sft-lr-sweep/build_html.py from the same json."""
import json
import math
import os
import statistics
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import wandb

OUT = os.path.expanduser("~/shared/reports/maemm-sft-lr-sweep")
os.makedirs(f"{OUT}/data", exist_ok=True)
PROJ = "octahedral-systems/maxact-fast"
LRS = [3e-5, 1e-4, 3e-4, 5e-4, 1e-3, 3e-3]
# arms whose loss blew up right after the 78-step warmup (8-10, then >6.6 for 1100+ steps) and were cancelled at ~step 1250;
# their only evaluated checkpoint is step_976 (0.5M examples), plotted as a hollow "x" and never used for the claim
DIVERGED = {1e-3: "loss 9.9 at step 100, 6.7 at step 1200; cancelled at ~step 1250",
            3e-3: "loss 8.7 at step 100, 7.4 at step 1200; cancelled at ~step 1250"}
EFF_BATCH, FINAL_STEP, N_EXAMPLES = 512, 3907, 2_000_000
RUN = lambda lr: f"lrsweep2m_lr{lr:g}"
# ordinal lrs -> one hue, light -> dark (plus distinct markers so identity is never color-alone)
COLORS = [plt.cm.copper(x) for x in (0.88, 0.74, 0.6, 0.46, 0.3, 0.12)]
MARKERS = ["o", "s", "^", "D", "v", "P"]
# FULL fine-tuning arms (sft/fullft.py: every weight trainable, FSDP2, fp32 masters, bf16 compute) on the SAME 2M subset,
# schedule and effective batch as the LoRA arms; lower lrs because all 27B weights move
FT_LRS = [1e-5, 3e-5]
RUN_FT = lambda lr: f"fullft2m_lr{lr:g}"
FT_COLORS = ["#7fa7d6", "#2f5597"]
FT_MARKER = "*"
REF = {"23M-example run at lr 1e-4 (OneCycle over 23M; range over its checkpoints)":
           {"eval/mean_all": (0.364, 0.375), "eval/sae/norm_act": (0.418, 0.478), "eval/realact/cos": (0.470, 0.483)},
       "best RL so far (RL-C ckpt 160)": {"eval/mean_all": 0.4131, "eval/sae/norm_act": 0.8467, "eval/realact/cos": 0.536}}
EVAL = [("eval/mean_all", "mean over held-out families"), ("eval/realact/cos", "real activations (cos)"),
        ("eval/realact_long/cos", "long-context real activations (cos)"), ("eval/sae/norm_act", "SAE features (norm. activation)"),
        ("eval/sae/rank1_frac", "SAE rank-1 fraction"), ("eval/bsf/cos", "BSF subspace (cos)"), ("eval/cluster/cos", "cluster probes (cos)"),
        ("eval/mlp/cos", "layer-42 MLP neuron writes (cos)"), ("eval/jlens/cos", "J-lens (cos)"), ("extra/locality/win5_share", "activation mass in last-5 window")]
FINAL_PANELS = EVAL[:8]

api = wandb.Api()


def runs_named(name):
    return list(api.runs(PROJ, filters={"display_name": name}, order="-created_at"))


data = {"arms": {}, "fullft": {}, "references": REF, "eff_batch": EFF_BATCH, "final_step": FINAL_STEP, "n_examples": N_EXAMPLES,
        "diverged": {RUN(lr): why for lr, why in DIVERGED.items()}}
STAT_KEYS = ["mfu", "tflops", "ex_per_s", "tok_per_s", "peak_mem_gb"]


def load_arm(name, lr, kind):
    arm = {"lr": lr, "run": name, "kind": kind, "train": None, "evals": [], "eval_run_id": None, "train_state": None, "train_stats": {}}
    tr = runs_named(name)
    if tr:
        hist = [h for h in tr[0].history(keys=["loss", "lr"] + STAT_KEYS, pandas=False, samples=5000) if h.get("loss") is not None]
        arm["train"] = {"id": tr[0].id, "points": [{"step": int(h["_step"]), "loss": float(h["loss"]), "lr": h.get("lr")} for h in hist]}
        arm["train_state"] = tr[0].state
        for k in STAT_KEYS:   # medians over the run (skip the first logged step: compile / warm-up)
            vals = [float(h[k]) for h in hist[1:] if h.get(k) is not None]
            if vals:
                arm["train_stats"][k] = statistics.median(vals)
        if "ex_per_s" in arm["train_stats"]:   # per-rank ex/s -> seconds per optimizer step (eff. batch 512 over 8 ranks)
            arm["train_stats"]["s_per_step"] = (EFF_BATCH / 8) / arm["train_stats"]["ex_per_s"]
    ev = runs_named(f"{name}_eval")
    if ev:
        rows = [r for r in ev[0].history(pandas=False) if "ckpt_step" in r and r.get("eval/mean_all") is not None]
        rows.sort(key=lambda r: r["ckpt_step"])
        arm["eval_run_id"] = ev[0].id
        arm["evals"] = [{"ckpt_step": int(r["ckpt_step"]), "examples": int(r["ckpt_step"]) * EFF_BATCH, **{m: r.get(m) for m, _ in EVAL}} for r in rows]
    return arm


for lr in LRS:
    data["arms"][RUN(lr)] = load_arm(RUN(lr), lr, "lora_r64")
for lr in FT_LRS:
    data["fullft"][RUN_FT(lr)] = load_arm(RUN_FT(lr), lr, "full_ft")


def final_row(arm):
    rows = arm["evals"]
    if not rows:
        return None
    fin = [r for r in rows if r["ckpt_step"] >= FINAL_STEP]
    return fin[0] if fin else rows[-1]           # fall back to the latest available checkpoint while the run is live


finals = {lr: final_row(data["arms"][RUN(lr)]) for lr in LRS}
have = [lr for lr in LRS if finals[lr] and lr not in DIVERGED]     # stable arms with an evaluated checkpoint
shown = [lr for lr in LRS if finals[lr]]                            # incl. diverged arms (their 0.5M row)
data["final_rows"] = {RUN(lr): finals[lr] for lr in LRS}
ft_finals = {lr: final_row(data["fullft"][RUN_FT(lr)]) for lr in FT_LRS}
ft_have = [lr for lr in FT_LRS if ft_finals[lr]]
data["final_rows_fullft"] = {RUN_FT(lr): ft_finals[lr] for lr in FT_LRS}
ALL_X = sorted(set(LRS) | set(FT_LRS))
json.dump(data, open(f"{OUT}/data/lr_sweep.json", "w"), indent=1)

# ---- headline: final checkpoint metric vs lr ----
if have:
    means = {lr: finals[lr]["eval/mean_all"] for lr in have}
    best_lr = max(means, key=means.get)
    spread = max(means.values()) - min(means.values())
    base = means.get(1e-4)
    core = [lr for lr in have if lr <= 3e-4] or have            # the arms whose CE ties; 5e-4 is the edge-of-stability probe
    core_vals = [means[lr] for lr in core]
    spread_core = max(core_vals) - min(core_vals)
    if spread_core < 0.01 and base is not None and means[best_lr] - base <= 0.005:
        claim = (f"SFT learning rate does not move held-out fidelity between lr {min(core):g} and {max(core):g} "
                 f"(mean_all {min(core_vals):.3f}-{max(core_vals):.3f}) at a fixed 2M-example budget")
        edge = [lr for lr in have if lr > max(core)]
        if edge and means[edge[0]] < min(core_vals) - 0.005:
            claim += f"; {edge[0]:g} is slightly worse ({means[edge[0]]:.3f})"
    elif base is not None and means[best_lr] - base > 0.005:
        claim = f"lr {best_lr:g} beats the production lr 1e-4 on held-out fidelity (mean_all {means[best_lr]:.3f} vs {base:.3f}) at a fixed 2M-example SFT budget"
    else:
        claim = f"lr {best_lr:g} is the best SFT learning rate at a fixed 2M-example budget (mean_all {means[best_lr]:.3f}); higher lrs do not help"
    if DIVERGED:
        claim += f"; lr >= {min(DIVERGED):g} diverges (train loss 7-10 from the first 100 steps)"
    ft_done = [lr for lr in ft_have if ft_finals[lr]["ckpt_step"] >= FINAL_STEP]    # only ANNEALED finals earn a verdict
    if ft_done:
        ft_means = {lr: ft_finals[lr]["eval/mean_all"] for lr in ft_done}
        ft_best = max(ft_means, key=ft_means.get)
        d = ft_means[ft_best] - means[best_lr]
        verdict = ("beats" if d > 0.005 else "ties" if abs(d) <= 0.005 else "trails")
        claim += f". Full fine-tuning (lr {ft_best:g}) {verdict} the best LoRA arm: mean_all {ft_means[ft_best]:.3f} vs {means[best_lr]:.3f}"
        data["claim_fullft"] = {"best_ft_lr": ft_best, "ft_mean_all": ft_means[ft_best], "lora_best_lr": best_lr, "lora_mean_all": means[best_lr], "verdict": verdict,
                                "n_final": len(ft_done), "n_arms": len(FT_LRS)}
    elif ft_have:
        claim += ". Full fine-tuning arms still running (latest evaluated checkpoint: " + ", ".join(
            f"lr {lr:g} step {ft_finals[lr]['ckpt_step']} mean_all {ft_finals[lr]['eval/mean_all']:.3f}" for lr in ft_have) + ")"
    data["fullft_in_progress"] = [RUN_FT(lr) for lr in ft_have if lr not in ft_done]
    data["claim"] = claim
    json.dump(data, open(f"{OUT}/data/lr_sweep.json", "w"), indent=1)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.2))
    for ax, (m, title) in zip(axes.flat, FINAL_PANELS):
        xs = [lr for lr in have if finals[lr].get(m) is not None]
        ys = [finals[lr][m] for lr in xs]
        ax.plot(xs, ys, "-", color="#b5542b", lw=1.5, zorder=2, label="LoRA r64 (rsLoRA, all-linear)")
        # y-range from the stable arms + full-FT arms + references only: the diverged arms sit at ~0 and would flatten every difference
        fx = [lr for lr in ft_have if ft_finals[lr].get(m) is not None]
        span = ys + [ft_finals[lr][m] for lr in fx] + [v for vals in REF.values() for v in ([vals[m]] if m in vals and not isinstance(vals[m], tuple) else list(vals[m]) if m in vals else [])]
        if span:
            lo, hi = min(span), max(span); pad = max(0.02, 0.18 * (hi - lo))
            lo, hi = lo - pad, hi + pad
            ax.set_ylim(lo, hi)
        else:
            lo, hi = ax.get_ylim()
        for i, (lr, y, c, mk) in enumerate(zip(LRS, [finals[lr].get(m) if finals[lr] else None for lr in LRS], COLORS, MARKERS)):
            if y is None:
                continue
            if lr in DIVERGED:
                y_draw = lo + 0.06 * (hi - lo)
                ax.plot([lr], [y_draw], "x", color="#9a3b8f", ms=8, mew=1.6, zorder=3, clip_on=False,
                        label="diverged arm (train loss 7-10): its 0.5M-example checkpoint, drawn OFF SCALE at the axis bottom")
                ax.annotate(f"{y:.3f}\n(off scale)", (lr, y_draw), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=6.5, color="#9a3b8f")
            else:
                ax.plot([lr], [y], mk, color=c, ms=7, mec="#333", mew=0.5, zorder=3)
                ax.annotate(f"{y:.3f}", (lr, y), textcoords="offset points", xytext=(0, 7 if i % 2 == 0 else -13), ha="center", fontsize=7.5, color="#333")
        if fx:   # full fine-tuning arms: a second series (star markers, blue dashed)
            ax.plot(fx, [ft_finals[lr][m] for lr in fx], "--", color="#2f5597", lw=1.3, zorder=2)
            for lr, c in zip(FT_LRS, FT_COLORS):
                if lr in fx:
                    y = ft_finals[lr][m]
                    ax.plot([lr], [y], FT_MARKER, color=c, ms=11, mec="#1b365d", mew=0.6, zorder=4, label="full fine-tune (all 27B weights, FSDP2)")
                    ax.annotate(f"{y:.3f}", (lr, y), textcoords="offset points", xytext=(0, -13), ha="center", fontsize=7.5, color="#1b365d")
        for i, (rname, vals) in enumerate(REF.items()):
            if m in vals:
                v = vals[m]
                if isinstance(v, tuple):
                    ax.axhspan(v[0], v[1], color="#7a7a7a", alpha=0.18, lw=0, label=rname)
                else:
                    ax.axhline(v, color="#2a7f62", ls=":", lw=1.2, label=rname)
        ax.set_xscale("log"); ax.set_xticks(ALL_X); ax.set_xticklabels([f"{lr:g}" for lr in ALL_X], fontsize=7.5, rotation=30)
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_title(title, fontsize=9.5); ax.grid(alpha=0.25); ax.tick_params(labelsize=8)
    for ax in axes[1]:
        ax.set_xlabel("peak learning rate (OneCycle, log)", fontsize=9)
    seen = {}
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(l, h)
    fig.legend(seen.values(), seen.keys(), loc="lower center", ncol=2, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(textwrap.fill(claim, 150) + "\n(Qwen3.6-27B inverter, LoRA r64 rsLoRA, one epoch over the same 2M real-activation examples per arm; final annealed checkpoint; 512 dirs/family, best-of-4)",
                 fontsize=9.5, y=0.995)
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/lr_sweep_final_vs_lr.{ext}", dpi=160)
    plt.close(fig)

# ---- curves: every metric vs examples seen, one line per lr ----
fig, axes = plt.subplots(2, 5, figsize=(17, 6.8), sharex=True)
for ax, (m, title) in zip(axes.flat, EVAL):
    for lr, c, mk in zip(LRS, COLORS, MARKERS):
        if lr in DIVERGED:      # one ~0 point each; shown off-scale in the headline figure instead
            continue
        rows = [r for r in data["arms"][RUN(lr)]["evals"] if r.get(m) is not None]
        if rows:
            ax.plot([r["examples"] / 1e6 for r in rows], [r[m] for r in rows], "-", marker=mk, color=c, ms=4, lw=1.4, mec="#333", mew=0.4, label=f"LoRA lr {lr:g}")
    for lr, c in zip(FT_LRS, FT_COLORS):
        rows = [r for r in data["fullft"][RUN_FT(lr)]["evals"] if r.get(m) is not None]
        if rows:
            ax.plot([r["examples"] / 1e6 for r in rows], [r[m] for r in rows], "--", marker=FT_MARKER, color=c, ms=8, lw=1.4,
                    mec="#1b365d", mew=0.5, label=f"full FT lr {lr:g}")
    for i, (rname, vals) in enumerate(REF.items()):
        if m in vals:
            v = vals[m]
            if isinstance(v, tuple):
                ax.axhspan(v[0], v[1], color="#7a7a7a", alpha=0.18, lw=0, label=rname)
            else:
                ax.axhline(v, color="#2a7f62", ls=":", lw=1.2, label=rname)
    ax.set_title(title, fontsize=9.5); ax.grid(alpha=0.25); ax.tick_params(labelsize=8)
for ax in axes[1]:
    ax.set_xlabel("examples seen (millions)", fontsize=9)
seen = {}
for ax in axes.flat:
    for h, l in zip(*ax.get_legend_handles_labels()):
        seen.setdefault(l, h)
fig.legend(seen.values(), seen.keys(), loc="lower center", ncol=4, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.01))
fig.suptitle("Held-out fidelity along training for each stable SFT learning rate, LoRA r64 (solid) vs full fine-tuning (dashed stars)\n"
             "(same 2M real-activation examples, OneCycle schedule; intermediate checkpoints are not annealed; "
             + ", ".join(f"{lr:g}" for lr in sorted(DIVERGED)) + " diverged and are omitted)", fontsize=10, y=0.995)
fig.tight_layout(rect=(0, 0.07, 1, 0.95))
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/lr_sweep_curves.{ext}", dpi=160)
plt.close(fig)

# ---- training loss per lr ----
fig, ax = plt.subplots(figsize=(9, 4.4))
win = 10
loss_summary = {}
for lr, c, arm_d, is_ft in ([(lr, c, data["arms"][RUN(lr)], False) for lr, c in zip(LRS, COLORS)]
                            + [(lr, c, data["fullft"][RUN_FT(lr)], True) for lr, c in zip(FT_LRS, FT_COLORS)]):
    tr = arm_d["train"]
    if not tr or not tr["points"]:
        continue
    ls = [p["loss"] for p in tr["points"]]; st = [p["step"] for p in tr["points"]]
    sm = [statistics.mean(ls[max(0, i - win + 1): i + 1]) for i in range(len(ls))]
    fin = [x for x in ls if math.isfinite(x)]
    loss_summary[arm_d["run"]] = {"first10_mean": statistics.mean(fin[:10]) if fin else None, "last10_mean": statistics.mean(fin[-10:]) if fin else None,
                                  "min": min(fin) if fin else None, "n_nonfinite": len(ls) - len(fin), "last_step": st[-1], "kind": arm_d["kind"],
                                  **{f"median_{k}": v for k, v in arm_d["train_stats"].items()}}
    if is_ft:
        ax.plot([s * EFF_BATCH / 1e6 for s in st], sm, color=c, lw=1.8, ls="--", label=f"full FT lr {lr:g}")
        continue
    ax.plot([s * EFF_BATCH / 1e6 for s in st], sm, color=c if lr not in DIVERGED else "#9a3b8f", lw=1.6,
            ls="-" if lr not in DIVERGED else ["--", ":", "-."][sorted(DIVERGED).index(lr) % 3],   # distinct dashes per diverged arm
            label=f"lr {lr:g}" + (" (diverged, cancelled)" if lr in DIVERGED else ""))
ax.set_xlabel("examples seen (millions)"); ax.set_ylabel(f"next-token CE on target tokens ({win}-point running mean)")
if loss_summary:
    lo = min(v["last10_mean"] for v in loss_summary.values() if v["last10_mean"] is not None)
    ax.set_ylim(top=min(10.5, max(v["first10_mean"] for v in loss_summary.values() if v["first10_mean"]) + 0.3), bottom=max(0.0, lo - 0.3))
ax.set_title("Training loss per learning rate on the same 2M real-activation examples: LoRA r64 (solid) vs full fine-tuning (dashed) "
             "(eff. batch 512, OneCycle 2% warmup + linear anneal)", fontsize=9.5)
ax.grid(alpha=0.25); ax.legend(frameon=False, fontsize=8.5)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/lr_sweep_train_loss.{ext}", dpi=160)
plt.close(fig)
json.dump(loss_summary, open(f"{OUT}/data/train_loss_summary.json", "w"), indent=1)

print(f"wrote {OUT}: arms with evals {[f'{lr:g}:{len(data['arms'][RUN(lr)]['evals'])}' for lr in LRS]}, stable finals {len(have)}/{len(LRS) - len(DIVERGED)}, "
      f"train runs {sum(1 for lr in LRS if data['arms'][RUN(lr)]['train'])} | full-FT arms with evals {[f'{lr:g}:{len(data['fullft'][RUN_FT(lr)]['evals'])}' for lr in FT_LRS]}, finals {len(ft_have)}/{len(FT_LRS)}")
