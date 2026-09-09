"""Full-parameter fine-tune of the Qwen3.6-27B activation->text inverter on 104M real-activation examples:
held-out fidelity per checkpoint, per-family fidelity, and training loss, all on a shared x-axis of EXAMPLES SEEN
(= optimizer step x effective batch), against (i) the LoRA-23M baseline, (ii) the batch-8,192 / lr-3e-5 comparison
arm (stopped by the user at 27M examples) and (iii) the 2M-subset full-FT sweep.

Everything is pulled from wandb, so the script is idempotent: re-run it after the final checkpoint's eval lands and
the figures / JSON / headline sentence update themselves (the headline is conditional on a programmatic band check).

Writes to ~/shared/reports/maemm-sft-fullft-104m/:
  fidelity_vs_examples.{png,pdf}   eval/mean_all vs examples (log-x), LoRA-baseline range shaded
  per_family_vs_examples.{png,pdf} small multiples, one eval family per panel
  loss_curves.{png,pdf}            training loss vs examples, dashed verticals at the optimizer-state resets
  data/fidelity_vs_examples.json, data/per_family.json, data/loss_curves.json   (every plotted number)
Then render the report:  python ~/shared/reports/maemm-sft-fullft-104m/build_html.py
"""
import datetime as dt
import json
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import numpy as np
import wandb

OUT = os.path.expanduser("~/shared/reports/maemm-sft-fullft-104m")
os.makedirs(f"{OUT}/data", exist_ok=True)
IDS = os.path.expanduser("~/shared/overnight/sft_fullft_100m_ids.json")
TOTAL_STEPS = 25391          # 104M examples / 4,096 per step
N_GPUS = 8                   # B200s per run
BAND_EPS = 0.004             # |mean_all - LoRA midpoint| <= eps  ->  "noise" (table colouring, per the brief)
FAM_EPS = 0.002              # per-family: outside the LoRA run's own min..max by more than this -> good / bad
SMOOTH_M = 0.5               # loss smoothing window, millions of examples
PERS = "celestedeschamphelaere-personal/maxact-fast"
OCTA = "octahedral-systems/maxact-fast"
# palette validated with the dataviz validator (all-pairs, light surface): blue / orange / violet; gray = neutral reference
COL = {"blue": "#2a78d6", "orange": "#eb6834", "violet": "#4a3aa7", "gray": "#52514e", "band": "#c3c2b7",
       "ink": "#0b0b0b", "muted": "#898781", "grid": "#e1e0d9"}

RUNS = {
    "fft104m": dict(train=f"{PERS}/jp82mr9a", eff=4096, eval=(PERS, "realact104m_fullft_b4096_lr1e-5_eval"),
                    ids="realact104m_fullft_b4096_lr1e-5", color=COL["blue"], marker="o", plot=True,
                    label="full-parameter fine-tune, 104M examples, batch 4,096, lr 1e-5"),
    "fft104m_b8k": dict(train=f"{PERS}/6o76ie7z", eff=8192, eval=(PERS, "realact104m_fullft_b8192_lr3e-5_eval"),
                        ids="realact104m_fullft_b8192_lr3e-5", color=COL["orange"], marker="s", plot=True,
                        label="full-parameter fine-tune, same corpus, batch 8,192, lr 3e-5 (stopped at 27M)"),
    "lora23m": dict(train=f"{OCTA}/da7cxuz3", eff=512, eval=(OCTA, "realact20m_prefix_lr1e-4_eval"),
                    ids=None, color=COL["gray"], marker="^", plot=True,
                    label="LoRA baseline (rsLoRA r64/α16), 23M examples, batch 512, lr 1e-4"),
    "fft2m": dict(train=f"{OCTA}/9ce7q1n2", eff=512, eval=(OCTA, "fullft2m_lr1e-05_eval"),
                  ids=None, color=COL["violet"], marker="D", plot=True,
                  label="full-parameter fine-tune, 2M subset, batch 512, lr 1e-5"),
    "fft2m_lr3e5": dict(train=f"{OCTA}/c6pxmq4k", eff=512, eval=(OCTA, "fullft2m_lr3e-05_eval"),
                        ids=None, color=None, marker=None, plot=False,          # appendix table only
                        label="full-parameter fine-tune, 2M subset, batch 512, lr 3e-5"),
}
PLOT_ORDER = ["lora23m", "fft2m", "fft104m_b8k", "fft104m"]      # draw the hero series last (on top)

# eval families shown as small multiples: (wandb key, plain-word label, part of mean_all?)
FAMILIES = [  # (wandb key, panel title, part of mean_all?, short name used in the claim sentence)
    ("eval/sae/norm_act", "SAE features: normalised peak activation", False, "SAE features"),
    ("eval/realact/cos", "real activations, short context (cosine)", True, "short-context real activations"),
    ("eval/realact_long/cos", "real activations, long context (cosine)", True, "long-context real activations"),
    ("eval/bsf/cos", "subspace-featurizer (BSF) directions (cosine)", True, "BSF directions"),
    ("eval/cluster/cos", "linear-probe / cluster directions (cosine)", True, "probe directions"),
    ("eval/jlens/cos", "J-lens directions (cosine)", True, "J-lens directions"),
]
# mean_all = mean over the higher-is-better cosine families (control `random` and the SAE family excluded);
# verified numerically below (max abs error stored in per_family.json)
MEAN_ALL_KEYS = ["eval/realact/cos", "eval/realact_early/cos", "eval/realact_mid/cos", "eval/realact_long/cos",
                 "eval/indist_realact/cos", "eval/indist_long/cos", "eval/indist_probe/cos",
                 "eval/bsf/cos", "eval/cluster/cos", "eval/jlens/cos"]


def wrap(text, width):
    return textwrap.fill(text, width)


def utc(t):
    return dt.datetime.fromtimestamp(float(t), dt.UTC).strftime("%Y-%m-%d %H:%M:%SZ")


api = wandb.Api()


def fetch_loss(path):
    """Union of the sampled history (covers the whole run, <=10k rows) and scan_history (every row it returns;
    for the 104M run it silently skips steps 1,863-13,999 of the resumed leg). Keyed by the trainer's global step."""
    r = api.run(path)
    rows = {}
    for x in r.history(keys=["loss", "_step", "_timestamp"], pandas=False, samples=10000):
        if x.get("loss") is not None:
            rows[int(x["_step"])] = (float(x["_timestamp"]), float(x["loss"]))
    for x in r.scan_history(keys=["loss", "_step", "_timestamp"]):
        if x.get("loss") is not None:
            rows[int(x["_step"])] = (float(x["_timestamp"]), float(x["loss"]))
    steps = np.array(sorted(rows), dtype=int)
    return dict(steps=steps, ts=np.array([rows[s][0] for s in steps]), loss=np.array([rows[s][1] for s in steps]),
                state=r.state, created=r.created_at, name=r.name, id=r.id, summary_step=r.summary.get("_step"))


def fetch_evals(project, display_name, eff):
    """All rows from every wandb run with that display name (evaluator respawns keep the name); newest run wins on a
    repeated ckpt_step. Returns rows sorted by ckpt_step with every numeric eval/* key + examples."""
    runs = list(api.runs(project, filters={"display_name": display_name}, order="-created_at"))
    rows = {}
    for r in reversed(runs):
        for x in r.history(pandas=False, samples=5000):
            if x.get("ckpt_step") is None or x.get("eval/mean_all") is None:
                continue
            rows[int(x["ckpt_step"])] = {k: float(v) for k, v in x.items()
                                         if k.startswith("eval/") and isinstance(v, (int, float))}
    out = [dict(ckpt_step=s, examples=s * eff, **rows[s]) for s in sorted(rows)]
    return out, [r.id for r in runs]


ids = json.load(open(IDS)) if os.path.exists(IDS) else {}
train, evals, eval_ids = {}, {}, {}
for k, cfg in RUNS.items():
    train[k] = fetch_loss(cfg["train"])
    evals[k], eval_ids[k] = fetch_evals(cfg["eval"][0], cfg["eval"][1], cfg["eff"])
    print(f"{k:12s} loss rows {len(train[k]['steps']):6d} last step {int(train[k]['steps'][-1]):6d} ({train[k]['state']}) | "
          f"evals {len(evals[k])}: {[(e['ckpt_step'], round(e['eval/mean_all'], 4)) for e in evals[k]][-4:]}")

# ------------------------------------------------------------------ band check + headline ---------------------------
lora_vals = np.array([e["eval/mean_all"] for e in evals["lora23m"]])
lo, hi = float(lora_vals.min()), float(lora_vals.max())
mid = (lo + hi) / 2
a1 = evals["fft104m"]
a1_vals = np.array([e["eval/mean_all"] for e in a1]); a1_ex = np.array([e["examples"] for e in a1]) / 1e6
in_band = (a1_vals >= lo) & (a1_vals <= hi)
in_band_tol = (a1_vals >= lo - FAM_EPS) & (a1_vals <= hi + FAM_EPS)
best_i = int(np.argmax(a1_vals))
lora_ex_max = max(e["examples"] for e in evals["lora23m"]) / 1e6
ratio = a1_ex.max() / lora_ex_max
final_done = int(train["fft104m"]["steps"][-1]) >= TOTAL_STEPS - 1 or any(e["ckpt_step"] >= TOTAL_STEPS - 1 for e in a1)
span = f"every checkpoint from {a1_ex.min():.0f}M to {a1_ex.max():.0f}M examples"
if in_band.all():
    verdict = "all_in_band"
    headline = (f"{ratio:.1f}× the LoRA baseline's data and full-parameter training do not move held-out fidelity: "
                f"{span} lands inside the LoRA band {lo:.3f}–{hi:.3f}")
elif in_band_tol.all():
    verdict = "all_within_tol"
    headline = (f"{ratio:.1f}× the LoRA baseline's data and full-parameter training do not move held-out fidelity: "
                f"{span} lands within {FAM_EPS:.3f} of the LoRA band {lo:.3f}–{hi:.3f}")
elif (a1_vals > hi + BAND_EPS).any():
    j = int(np.argmax(a1_vals)); verdict = "some_above"
    headline = (f"Full-parameter training on {ratio:.1f}× the data moves held-out fidelity above the LoRA band "
                f"{lo:.3f}–{hi:.3f}: best {a1_vals[j]:.3f} at {a1_ex[j]:.0f}M examples "
                f"({int(in_band.sum())}/{len(a1_vals)} checkpoints still inside the band)")
else:
    j = int(np.argmin(a1_vals)); verdict = "some_below"
    headline = (f"Full-parameter training on {ratio:.1f}× the data does not lift held-out fidelity above the LoRA band "
                f"{lo:.3f}–{hi:.3f}; {int((~in_band).sum())} of {len(a1_vals)} checkpoints fall below it "
                f"(lowest {a1_vals[j]:.3f} at {a1_ex[j]:.0f}M examples)")
print("HEADLINE:", headline)

subtitle_exp = ("Qwen3.6-27B activation-to-text inverter; held-out mean cosine over 10 direction families "
                "(512 held-out directions each, best-of-4 samples, last-5-token window); x = training examples seen (log)")

# ------------------------------------------------------------------ figure 1: fidelity vs examples ------------------
plt.rcParams.update({"font.size": 10, "axes.edgecolor": COL["muted"], "axes.labelcolor": COL["ink"],
                     "xtick.color": COL["ink"], "ytick.color": COL["ink"]})
fig, ax = plt.subplots(figsize=(12, 6.6))
ax.axhspan(lo, hi, color=COL["band"], alpha=0.32, lw=0,
           label=f"LoRA baseline: range of its {len(lora_vals)} checkpoints ({lo:.3f}–{hi:.3f})")
for k in PLOT_ORDER:
    cfg = RUNS[k]; ev = evals[k]
    if not ev:
        continue
    x = [e["examples"] / 1e6 for e in ev]; y = [e["eval/mean_all"] for e in ev]
    hero = k == "fft104m"
    ax.plot(x, y, marker=cfg["marker"], color=cfg["color"], lw=2 if hero else 1.3, ms=7 if hero else 6,
            mec="white", mew=0.9, alpha=1 if hero else 0.9, label=cfg["label"], zorder=4 if hero else 3)
ax.annotate(f"{a1_vals[-1]:.3f} @ {a1_ex[-1]:.0f}M" + ("" if final_done else "  (run still training)"),
            xy=(a1_ex[-1], a1_vals[-1]), xytext=(8, 14), textcoords="offset points", fontsize=9, color=COL["ink"],
            ha="left", arrowprops=dict(arrowstyle="-", color=COL["muted"], lw=0.8))
ax.annotate(f"best {a1_vals[best_i]:.3f} @ {a1_ex[best_i]:.1f}M", xy=(a1_ex[best_i], a1_vals[best_i]),
            xytext=(0, 18), textcoords="offset points", fontsize=9, color=COL["ink"], ha="center",
            arrowprops=dict(arrowstyle="-", color=COL["muted"], lw=0.8))
ax.set_xscale("log"); ax.set_xlim(0.4, 140)
ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}M"))
ax.xaxis.set_major_locator(mt.LogLocator(base=10, subs=(1, 2, 5), numticks=12))
ax.set_ylim(min(0.352, a1_vals.min() - 0.006), max(0.386, a1_vals.max() + 0.008))
ax.yaxis.set_major_locator(mt.MultipleLocator(0.005)); ax.yaxis.set_major_formatter(mt.FormatStrFormatter("%.3f"))
ax.set_xlabel("training examples seen (log scale)")
ax.set_ylabel("held-out fidelity (mean cosine over 10 direction families)")
ax.grid(axis="y", color=COL["grid"], lw=0.8); ax.spines[["top", "right"]].set_visible(False)
ax.set_title(wrap(headline, 128) + "\n" + wrap(subtitle_exp, 128), fontsize=10.5, loc="left")
ax.legend(loc="lower right", fontsize=8.6, frameon=False)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/fidelity_vs_examples.{ext}", dpi=170)
plt.close(fig)

json.dump({
    "generated_utc": utc(dt.datetime.now(dt.UTC).timestamp()),
    "headline": headline, "verdict": verdict, "subtitle": subtitle_exp,
    "lora_band": {"min": lo, "max": hi, "mid": mid, "n_ckpts": int(len(lora_vals)), "sd": float(lora_vals.std(ddof=1)),
                  "max_examples": lora_ex_max * 1e6, "wandb_train": RUNS["lora23m"]["train"], "wandb_eval": eval_ids["lora23m"]},
    "band_check": {"n_in_band": int(in_band.sum()), "n_in_band_tol": int(in_band_tol.sum()), "n_total": int(len(a1_vals)),
                   "tol": FAM_EPS, "noise_eps": BAND_EPS, "best": {"mean_all": float(a1_vals[best_i]), "ckpt_step": a1[best_i]["ckpt_step"],
                   "examples": a1[best_i]["examples"]}, "min": float(a1_vals.min()), "max": float(a1_vals.max()),
                   "sd_ckpt_to_ckpt": float(a1_vals.std(ddof=1)), "data_ratio_vs_lora": float(ratio), "final_evaluated": bool(final_done)},
    "series": {k: {"label": RUNS[k]["label"], "eff_batch": RUNS[k]["eff"], "wandb_train": RUNS[k]["train"], "wandb_eval_runs": eval_ids[k],
                   "plotted": RUNS[k]["plot"],
                   "points": [{"ckpt_step": e["ckpt_step"], "examples": e["examples"], "mean_all": e["eval/mean_all"]} for e in evals[k]]}
               for k in RUNS},
}, open(f"{OUT}/data/fidelity_vs_examples.json", "w"), indent=1)

# ------------------------------------------------------------------ figure 2: per-family small multiples -----------
def first_last(vals, n=3):
    vals = np.asarray(vals)
    if len(vals) < 2 * n:
        return float(vals[-1] - vals[0])
    return float(vals[-n:].mean() - vals[:n].mean())

fam_stats = {}
for key, lab, in_mean, short in FAMILIES:
    lv = np.array([e[key] for e in evals["lora23m"] if key in e])
    av = [e[key] for e in a1 if key in e]
    fam_stats[key] = {"label": lab, "short": short, "in_mean_all": in_mean,
                      "lora_min": float(lv.min()), "lora_max": float(lv.max()), "lora_mean": float(lv.mean()),
                      "fft104m_first3_to_last3": first_last(av), "fft104m_first": float(av[0]), "fft104m_last": float(av[-1]),
                      "fft104m_last_vs_lora_max": float(av[-1] - lv.max()), "fft104m_last_vs_lora_min": float(av[-1] - lv.min())}
mean_all_err = max(abs(np.mean([e[k] for k in MEAN_ALL_KEYS]) - e["eval/mean_all"]) for e in a1)
rising = [fam_stats[k]["short"] for k, *_ in FAMILIES if fam_stats[k]["fft104m_first3_to_last3"] > 0.005]
falling = [fam_stats[k]["short"] for k, *_ in FAMILIES if fam_stats[k]["fft104m_first3_to_last3"] < -0.005]
d_mean = first_last(a1_vals)
if rising and falling:
    fam_claim = (f"The flat mean hides families moving in opposite directions as the full fine-tune sees more data: "
                 f"{' and '.join(rising)} rise, while {', '.join(falling)} fall (first 3 vs last 3 checkpoints; the mean itself {d_mean:+.3f})")
elif falling:
    fam_claim = f"More full-fine-tune data lowers {', '.join(falling)} while the other families stay flat (mean_all {d_mean:+.3f})"
elif rising:
    fam_claim = f"More full-fine-tune data raises {', '.join(rising)} while the other families stay flat (mean_all {d_mean:+.3f})"
else:
    fam_claim = f"No eval family moves by more than 0.005 across the full fine-tune (mean_all {d_mean:+.3f})"
print("FAMILY CLAIM:", fam_claim)

fig, axes = plt.subplots(2, 3, figsize=(15, 8.2), sharex=True)
for ax, (key, lab, in_mean, short) in zip(axes.flat, FAMILIES):
    fs = fam_stats[key]
    ax.axhspan(fs["lora_min"], fs["lora_max"], color=COL["band"], alpha=0.32, lw=0)
    for k in PLOT_ORDER:
        cfg = RUNS[k]; ev = [e for e in evals[k] if key in e]
        if not ev:
            continue
        hero = k == "fft104m"
        ax.plot([e["examples"] / 1e6 for e in ev], [e[key] for e in ev], marker=cfg["marker"], color=cfg["color"],
                lw=1.8 if hero else 1.1, ms=5.5 if hero else 4.5, mec="white", mew=0.7, zorder=4 if hero else 3)
    ax.set_xscale("log"); ax.set_xlim(0.4, 140)
    ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}M"))
    ax.xaxis.set_major_locator(mt.LogLocator(base=10, subs=(1, 3), numticks=10))
    ax.set_title(lab + ("" if in_mean else "  [not in mean_all]"), fontsize=9.6, loc="left")
    ax.text(0.02, 0.05, f"full FT 104M, first 3 vs last 3 ckpts: {fs['fft104m_first3_to_last3']:+.3f}",
            transform=ax.transAxes, fontsize=8.4, color=COL["ink"])
    ax.grid(axis="y", color=COL["grid"], lw=0.8); ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_formatter(mt.FormatStrFormatter("%.3f")); ax.tick_params(labelsize=8.5)
for ax in axes[1]:
    ax.set_xlabel("training examples seen (log)", fontsize=9)
handles = [plt.Line2D([], [], color=RUNS[k]["color"], marker=RUNS[k]["marker"], lw=1.5, ms=6, mec="white", label=RUNS[k]["label"])
           for k in ["fft104m", "fft104m_b8k", "lora23m", "fft2m"]]
handles.append(plt.Rectangle((0, 0), 1, 1, color=COL["band"], alpha=0.32, label="LoRA baseline: range over its checkpoints"))
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.6, frameon=False, bbox_to_anchor=(0.5, -0.005))
fig.suptitle(wrap(fam_claim, 165) + "\n" + wrap("Qwen3.6-27B activation-to-text inverter, held-out directions (512 per family, best-of-4, "
             "last-5-token window); shaded = the LoRA baseline's own min-max over its checkpoints", 165), fontsize=10.5, x=0.01, y=0.992, ha="left", va="top")
fig.subplots_adjust(left=0.05, right=0.99, top=0.86, bottom=0.13, wspace=0.22, hspace=0.3)
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/per_family_vs_examples.{ext}", dpi=170)
plt.close(fig)

json.dump({
    "generated_utc": utc(dt.datetime.now(dt.UTC).timestamp()), "claim": fam_claim,
    "families_plotted": [{"key": k, "label": l, "in_mean_all": m, "short": sh} for k, l, m, sh in FAMILIES],
    "mean_all_keys": MEAN_ALL_KEYS, "mean_all_recomputation_max_abs_err": float(mean_all_err),
    "family_stats": fam_stats, "fam_eps": FAM_EPS,
    "series": {k: {"label": RUNS[k]["label"], "eff_batch": RUNS[k]["eff"], "points": evals[k]} for k in RUNS},
}, open(f"{OUT}/data/per_family.json", "w"), indent=1)

# ------------------------------------------------------------------ figure 3: training loss --------------------------
def smooth_by_x(x, y, w):
    """Centred moving average over a window of w units of x (handles uneven row density)."""
    x = np.asarray(x, float); y = np.asarray(y, float); c = np.concatenate([[0.0], np.cumsum(y)])
    lo_i = np.searchsorted(x, x - w / 2, "left"); hi_i = np.searchsorted(x, x + w / 2, "right")
    return x, (c[hi_i] - c[lo_i]) / np.maximum(hi_i - lo_i, 1)


def segments(d, gap_min=20):
    """Split a run's rows at wall-clock gaps > gap_min minutes; return list of (i0, i1) index ranges and the gaps."""
    ts, st = d["ts"], d["steps"]
    gaps = [i for i in range(len(ts) - 1) if ts[i + 1] - ts[i] > gap_min * 60]
    cuts = [0] + [g + 1 for g in gaps] + [len(ts)]
    segs = [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]
    gap_info = [{"step_before": int(st[g]), "step_after": int(st[g + 1]), "minutes": float((ts[g + 1] - ts[g]) / 60),
                 "utc_before": utc(ts[g]), "utc_after": utc(ts[g + 1])} for g in gaps]
    return segs, gap_info


def median_s_per_step(d, i0, i1):
    st, ts = d["steps"][i0:i1], d["ts"][i0:i1]
    if len(st) < 3:
        return None
    ds, dts = np.diff(st), np.diff(ts)
    ok = (dts < 120) & (ds > 0)
    return float(np.median(dts[ok] / ds[ok])) if ok.any() else None


d1 = train["fft104m"]
segs, gaps = segments(d1)
legs_json = ids.get(RUNS["fft104m"]["ids"], {}).get("legs", [])
resume_steps = sorted({int(l["from_step"]) for l in legs_json}) or [1813, 14504]
boundaries = []
for rs in resume_steps:
    g = next((g for g in gaps if g["step_before"] >= rs), None)
    before = after = None
    if g:
        m_b = (d1["steps"] > g["step_before"] - 100) & (d1["steps"] <= g["step_before"])
        m_a = (d1["steps"] >= g["step_after"]) & (d1["steps"] < g["step_after"] + 100)
        before, after = float(d1["loss"][m_b].mean()), float(d1["loss"][m_a].mean())
    boundaries.append({"resume_step": rs, "examples": rs * RUNS["fft104m"]["eff"], "wandb_gap": g,
                       "redone_steps_visible_in_wandb": (g["step_before"] - rs) if g else None,
                       "loss_mean_100_before_gap": before, "loss_mean_100_after_gap": after,
                       "loss_delta_after_resume": (after - before) if (before is not None) else None})
leg_sps = [{"steps": [int(d1["steps"][i0]), int(d1["steps"][i1 - 1])], "median_s_per_step": median_s_per_step(d1, i0, i1),
            "utc": [utc(d1["ts"][i0]), utc(d1["ts"][i1 - 1])]} for i0, i1 in segs]
last_step = int(d1["steps"][-1])


def gpu_hours(upto_step):
    """steps x measured s/step x GPUs, per leg (slow leg to the 1st resume, fast legs after) + the re-done steps."""
    sps = [l["median_s_per_step"] or 6.6 for l in leg_sps]
    cut = resume_steps[0] if resume_steps else 0
    fast = float(np.median(sps[1:])) if len(sps) > 1 else sps[0]
    redone = sum(b["redone_steps_visible_in_wandb"] or 0 for b in boundaries)
    sec = cut * sps[0] + max(0, upto_step - cut) * fast + redone * fast
    return sec * N_GPUS / 3600


def tail_mean(d, n_examples, eff):
    m = d["steps"] > d["steps"][-1] - n_examples / eff
    return float(d["loss"][m].mean()), int(m.sum())


fft_last50 = float(d1["loss"][d1["steps"] > last_step - 50].mean())
fft_last_half_m, _ = tail_mean(d1, 0.5e6, 4096)
lora = train["lora23m"]; lora_last_half_m, lora_n = tail_mean(lora, 0.5e6, 512)
lora_end_ex = int(lora["steps"][-1]) * 512
m23 = np.abs(d1["steps"] * 4096 - lora_end_ex) <= 0.25e6
fft_at_lora_end = float(d1["loss"][m23].mean()) if m23.any() else None
arm2 = train["fft104m_b8k"]; arm2_last50 = float(arm2["loss"][arm2["steps"] > arm2["steps"][-1] - 50].mean())
arm2_end_ex = int(arm2["steps"][-1]) * 8192
gh_now, gh_final = gpu_hours(last_step), gpu_hours(TOTAL_STEPS)
print(f"loss: FFT last50 {fft_last50:.4f} | FFT last 0.5M ex {fft_last_half_m:.4f} | LoRA last 0.5M ex {lora_last_half_m:.4f} "
      f"| FFT at LoRA's end ({lora_end_ex/1e6:.1f}M) {fft_at_lora_end} | arm2 last50 {arm2_last50:.4f} | GPU-h now {gh_now:.0f} final {gh_final:.0f}")
print("boundaries:", json.dumps(boundaries, indent=None)[:600]); print("legs s/step:", leg_sps)

fig, ax = plt.subplots(figsize=(13, 7))
for k in PLOT_ORDER:
    cfg, d = RUNS[k], train[k]
    ex = d["steps"] * cfg["eff"] / 1e6
    hero = k == "fft104m"
    ax.plot(ex, d["loss"], color=cfg["color"], lw=0.4, alpha=0.11, zorder=2)
    xs, ys = smooth_by_x(ex, d["loss"], SMOOTH_M)
    ax.plot(xs, ys, color=cfg["color"], lw=2.2 if hero else 1.7, zorder=4 if hero else 3,
            label=cfg["label"] + (" (still training)" if hero and not final_done else ""))
ymax, ymin = 3.05, 1.5
for i, b in enumerate(boundaries):
    xb = b["examples"] / 1e6
    ax.axvline(xb, ls="--", color=COL["ink"], lw=1, alpha=0.7, zorder=5)
    g = b["wandb_gap"] or {}
    what = ("switched to the fast FSDP config (prefix-share + prefetch)" if i == 0 else "24 h Modal cap hit; resumed by hand")
    lines = [f"optimizer-state reset {i + 1}: resumed from step {b['resume_step']:,} ({xb:.1f}M examples)", what,
             f"AdamW moments re-initialised, data re-tokenised, {b['redone_steps_visible_in_wandb'] or 0}+ steps re-done"]
    if b["loss_delta_after_resume"] is not None:
        lines.append(f"loss over the 100 steps before vs after: {b['loss_mean_100_before_gap']:.3f} vs "
                     f"{b['loss_mean_100_after_gap']:.3f} ({b['loss_delta_after_resume']:+.3f})")
    ax.annotate("\n".join(wrap(l, 50) for l in lines), xy=(xb, ymax - 0.02), xytext=(-8, 0), textcoords="offset points",
                fontsize=7.8, ha="right", va="top", color=COL["ink"], zorder=6,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COL["grid"], lw=0.8))
LEAD = dict(arrowstyle="-", color=COL["muted"], lw=0.8, shrinkB=2)
ax.annotate(f"full FT: {fft_last50:.3f} (last 50 steps, {last_step * 4096 / 1e6:.0f}M)", xy=(last_step * 4096 / 1e6, fft_last50),
            xytext=(34, ymin + 0.03), textcoords="data", fontsize=8.6, color=COL["ink"], ha="left", va="bottom", zorder=7, arrowprops=LEAD)
ax.annotate(f"LoRA baseline: {lora_last_half_m:.3f} at its end ({lora_end_ex / 1e6:.0f}M)", xy=(lora_end_ex / 1e6, lora_last_half_m),
            xytext=(30, 2.52), textcoords="data", fontsize=8.6, color=COL["ink"], ha="left", va="center", zorder=7, arrowprops=LEAD)
if fft_at_lora_end is not None:
    ax.annotate(f"full FT at the same {lora_end_ex / 1e6:.0f}M: {fft_at_lora_end:.3f}", xy=(lora_end_ex / 1e6, fft_at_lora_end),
                xytext=(30, 2.38), textcoords="data", fontsize=8.6, color=COL["ink"], ha="left", va="center", zorder=7, arrowprops=LEAD)
ax.annotate(f"batch-8,192 arm, stopped by the user at {arm2_end_ex / 1e6:.0f}M: {arm2_last50:.3f}", xy=(arm2_end_ex / 1e6, arm2_last50),
            xytext=(30, 2.24), textcoords="data", fontsize=8.6, color=COL["ink"], ha="left", va="center", zorder=7, arrowprops=LEAD)
ax.set_xscale("log"); ax.set_xlim(0.2, 160); ax.set_ylim(ymin, ymax)
ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}M"))
ax.xaxis.set_major_locator(mt.LogLocator(base=10, subs=(1, 2, 5), numticks=12))
ax.set_xlabel("training examples seen (log scale)"); ax.set_ylabel("training loss (mean cross-entropy over target tokens)")
ax.grid(axis="y", color=COL["grid"], lw=0.8); ax.spines[["top", "right"]].set_visible(False)
loss_claim = (f"Training loss keeps falling to {fft_last50:.2f} (LoRA baseline ended at {lora_last_half_m:.2f}; the full fine-tune "
              f"sat at {fft_at_lora_end:.2f} at that same {lora_end_ex / 1e6:.0f}M) while held-out fidelity does not move: the objective and the eval disagree"
              if fft_at_lora_end is not None else f"Training loss keeps falling to {fft_last50:.2f} while held-out fidelity does not move")
ax.set_title(wrap(loss_claim, 135) + "\n" + wrap(f"Qwen3.6-27B activation-to-text inverter; bold = {SMOOTH_M:g}M-example moving average, faint = every logged step; "
             "dashed = optimizer-state resets at the leg boundaries of the 104M run", 135), fontsize=10.5, loc="left")
ax.legend(loc="lower left", fontsize=8.6, frameon=False)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/loss_curves.{ext}", dpi=170)
plt.close(fig)

json.dump({
    "generated_utc": utc(dt.datetime.now(dt.UTC).timestamp()), "claim": loss_claim, "smooth_window_examples": SMOOTH_M * 1e6,
    "stats": {"fft104m_last50_steps": fft_last50, "fft104m_last_0p5M_examples": fft_last_half_m, "fft104m_last_step": last_step,
              "fft104m_total_steps": TOTAL_STEPS, "fft104m_state": d1["state"], "fft104m_final_evaluated": bool(final_done),
              "lora23m_last_0p5M_examples": lora_last_half_m, "lora23m_end_examples": lora_end_ex,
              "fft104m_at_lora_end_examples": fft_at_lora_end, "fft104m_b8k_last50_steps": arm2_last50, "fft104m_b8k_end_examples": arm2_end_ex,
              "fft104m_b8k_state": arm2["state"], "fft104m_b8k_median_s_per_step": median_s_per_step(arm2, 0, len(arm2["steps"])),
              "gpu_hours_so_far": gh_now, "gpu_hours_projected_final": gh_final, "n_gpus": N_GPUS,
              "gpu_hours_note": "steps x measured median s/step per leg x 8 GPUs / 3600, plus re-done steps; excludes staging/tokenisation between legs"},
    "fft104m_legs": leg_sps, "fft104m_boundaries": boundaries, "fft104m_wandb_gaps": gaps,
    "series": {k: {"label": RUNS[k]["label"], "eff_batch": RUNS[k]["eff"], "wandb": RUNS[k]["train"], "state": train[k]["state"],
                   "created": str(train[k]["created"]), "n_rows": int(len(train[k]["steps"])),
                   "steps": train[k]["steps"].tolist(), "timestamps": [round(float(t), 1) for t in train[k]["ts"]],
                   "loss": [round(float(v), 5) for v in train[k]["loss"]]} for k in RUNS if RUNS[k]["plot"]},
}, open(f"{OUT}/data/loss_curves.json", "w"))
print("wrote", OUT)
