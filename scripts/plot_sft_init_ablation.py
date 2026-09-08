#!/usr/bin/env python3
"""SFT-init ablation of the Qwen3.6-27B activation->text inverter: does the SFT starting point matter once you RL on the same bank?

Three LoRA policies (r64 / alpha16 rsLoRA) get the IDENTICAL RL recipe (ScaleRL/CISPO GRPO, group-relative advantages, 8 samples x 512
directions = 4,096 rollouts/step, constant lr 7e-6 after 25 warmup steps, no KL, length penalty, 300 steps, ckpt every 25) on ONE shared bank
(/data/banks/mix_eq_1p45m: six equal families x 241,741) and differ only in the init:
  init23m  = 23M real-activation LoRA pretrain              (/data/sft_mix/realact20m_prefix_lr1e-4/final)
  initmid  = midtrain-only LoRA on the same equal bank       (/data/sft_mix/mixeq_midtrain_only_from_base/final, 1 epoch from the raw base)
  initbase = fresh LoRA on the raw base model, no SFT at all (--init-adapter none)

Idempotent: every run re-pulls the eval + training histories from wandb (and, unless --no-fetch-transcripts, the rollout transcripts from the
Modal volume), rewrites data/*.json, every figure as PNG + PDF, then (unless --no-html) runs the report folder's build_html.py.

    python scripts/plot_sft_init_ablation.py                       # refresh everything
    python scripts/plot_sft_init_ablation.py --no-fetch-transcripts   # reuse /tmp/abl_tx_<arm>.jsonl

Outputs -> ~/shared/reports/maemm-sft-init-ablation/{data/*.json, sft_init_*.png|pdf}
"""
import argparse
import datetime as dt
import json
import math
import os
import re
import statistics
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb

OUT = Path(os.environ.get("MAEMM_SFT_INIT_REPORT", "~/shared/reports/maemm-sft-init-ablation")).expanduser()
PROJ = "celestedeschamphelaere-personal/maxact-fast"
REF_PROJ = "octahedral-systems/maxact-fast"
UPLIFT_DATA = Path("~/shared/reports/maemm-uplift-matrix/data").expanduser()   # the 23M init's held-out eval (same v2 protocol) lives here

# ---- arms: colour follows the entity (validated 3-slot categorical palette, all-pairs, on the ivory surface) ----
ARMS = {
    "init23m": dict(label="23M real-activation pretrain", short="23M pretrain", color="#2a78d6", marker="o",
                    train="rl_abl_init23m_8x512_lr7e-6", eval="rl_abl_init23m_8x512_lr7e-6_eval",
                    init_adapter="/data/sft_mix/realact20m_prefix_lr1e-4/final", ckpt_dir="/data/ckpts_rl_abl_init23m",
                    init_desc="LoRA SFT on 23M real L42 activations (eff batch 512, lr 1e-4); NOT trained on the RL bank"),
    "initmid": dict(label="midtrain only (1.45M mixed, from base)", short="midtrain only", color="#eb6834", marker="s",
                    train="rl_abl_initmid_8x512_lr7e-6", eval="rl_abl_initmid_8x512_lr7e-6_eval",
                    init_adapter="/data/sft_mix/mixeq_midtrain_only_from_base/final", ckpt_dir="/data/ckpts_rl_abl_initmid",
                    init_desc="LoRA SFT from the raw base on the SAME equal six-family bank RL uses (1 epoch of 1.45M, eff batch 256, lr 1e-4); no real-activation pretrain"),
    "initbase": dict(label="no SFT (base model + fresh LoRA)", short="no SFT", color="#1baf7a", marker="^",
                     train="rl_abl_initbase_8x512_lr7e-6", eval="rl_abl_initbase_8x512_lr7e-6_eval",
                     init_adapter=None, ckpt_dir="/data/ckpts_rl_abl_initbase",
                     init_desc="fresh LoRA on Qwen/Qwen3.6-27B, --init-adapter none; no SFT of any kind"),
}
REF = dict(label="reference: earlier run RL-C (23M pretrain + 1.1M midtrain; 16x256 rollouts, different 1.1M bank)", short="RL-C reference",
           color="#898781", eval="rl_C_mix1m_from_realact23m_mixsft_eval")
MIDTRAIN_SFT_EVAL = "mixeq_midtrain_only_from_base_eval"      # the midtrain-only init's own SFT eval; final ckpt rows = 5664 / 5665
MIDTRAIN_FINAL_STEPS = (5664, 5665)
RANDOM_FLOOR = 0.03                                            # cosine to a random direction; what an untrained inverter scores

# chart chrome (text never wears a series colour)
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.titlecolor": INK, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "grid.color": GRID, "grid.linewidth": 0.8, "axes.grid": True, "axes.axisbelow": True, "legend.frameon": False,
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})

# eval metrics: (key, panel title, one-line meaning). "eval/sae/verbalized_frac" is derived = 1 - unverbalized_frac.
FAMILY_PANELS = [
    ("eval/sae/norm_act", "SAE features: normalized activation", "held-out SAE feature activation on the generated text / that feature's corpus max"),
    ("eval/sae/verbalized_frac", "SAE features: verbalized fraction", "1 - unverbalized_frac: held-out features the inverter can make fire at all"),
    ("eval/realact/cos", "real activations (cos)", "cosine of the layer-42 activation to a held-out real activation"),
    ("eval/realact_long/cos", "long-context real activations (cos)", "same, activations taken at positions 256-511"),
    ("eval/bsf/cos", "BSF subspace blocks (cos)", "cosine to held-out block-sparse-featurizer directions"),
    ("eval/cluster/cos", "cluster probes (cos)", "cosine to held-out linear cluster-probe directions"),
    ("eval/mlp/norm_act", "MLP neurons: fire-back (normalized act.)", "held-out layer-42 MLP neuron activation on the text / neuron's corpus max"),
    ("eval/random/cos", "random directions (control floor, cos)", "should stay ~0.03; lower is better"),
]
TABLE_COLS = [("eval/mean_all", "mean fidelity"), ("eval/sae/norm_act", "SAE norm act"), ("eval/sae/verbalized_frac", "SAE verbalized"),
              ("eval/sae/rank1_frac", "SAE rank-1"), ("eval/realact/cos", "real acts"), ("eval/realact_long/cos", "long-ctx acts"),
              ("eval/bsf/cos", "BSF"), ("eval/cluster/cos", "probes"), ("eval/mlp/norm_act", "MLP fire-back"), ("eval/mlp/fired10", "MLP fired>=10%"),
              ("eval/jlens/cos", "J-lens"), ("eval/random/cos", "random (floor)")]
TRAIN_PANELS = [("reward/mean", "reward: mean cosine (last-5-token window)"), ("policy/entropy", "policy entropy (nats / token)"),
                ("grad_norm", "gradient norm before clipping at 1.0 (log scale)"), ("policy/sampler_abs_dlogp", "sampler vs trainer |delta log p| per token (log scale)"),
                ("rollout/len_mean", "response length (tokens)"), ("ratio/clipfrac", "CISPO clipped-token fraction")]
LOG_PANELS = {"grad_norm", "policy/sampler_abs_dlogp"}
TRAIN_EXTRA = ["reward/shaped_mean", "reward/std", "reward/trunc_frac", "policy/kl_to_init", "time/step_s", "lr", "trainer/real_tokens_per_rollout",
               "scalerl/is_trunc_frac", "var/zero_var_group_frac", "grad_norm_did_clip"]
DEGENERATE_MAX_TOK, DEGENERATE_TTR = 10, 0.6   # <=10 tokens (generation floor is 8) OR word type/token ratio < 0.6 (repetition)


# ------------------------------------------------------------------ fetch ------------------------------------------------------------------
def _num(x):
    return isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))


def _eval_rows(run):
    rows = {}
    for r in run.scan_history():
        if _num(r.get("ckpt_step")) and _num(r.get("eval/mean_all")):
            row = {k: v for k, v in r.items() if (k.startswith("eval/") or k.startswith("extra/") or k.startswith("time/")) and _num(v)}
            row["ckpt_step"] = int(r["ckpt_step"])
            if _num(row.get("eval/sae/unverbalized_frac")):
                row["eval/sae/verbalized_frac"] = 1.0 - row["eval/sae/unverbalized_frac"]
            rows[row["ckpt_step"]] = row          # keep the latest row per checkpoint (re-evals overwrite)
    return [rows[k] for k in sorted(rows)]


def _run(api, proj, name):
    rs = list(api.runs(proj, filters={"display_name": name}, order="-created_at"))
    if not rs:
        raise SystemExit(f"wandb run not found: {proj}/{name}")
    return rs[0]


def _mean_rows(rows):
    keys = set().union(*[set(r) for r in rows])
    return {k: float(np.mean([r[k] for r in rows if _num(r.get(k))])) for k in keys if any(_num(r.get(k)) for r in rows)}


def fetch_wandb():
    api = wandb.Api(timeout=180)
    out = {"arms": {}, "ref": {}, "midtrain_sft": {}, "fetched_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
    # --- the midtrain-only init's own SFT eval -> its before-RL row + full SFT trajectory ---
    sft = _run(api, PROJ, MIDTRAIN_SFT_EVAL)
    sft_rows = _eval_rows(sft)
    final = [r for r in sft_rows if r["ckpt_step"] in MIDTRAIN_FINAL_STEPS] or sft_rows[-1:]
    out["midtrain_sft"] = {"run": MIDTRAIN_SFT_EVAL, "id": sft.id, "state": sft.state, "rows": sft_rows,
                           "final_steps_used": [r["ckpt_step"] for r in final], "final": _mean_rows(final)}
    # --- the 23M init: eval from the uplift report (same v2 protocol); fallback = the numbers quoted in that report ---
    init23 = None
    try:
        init23 = json.load(open(UPLIFT_DATA / "eval_table.json"))["table"]["init"]["metrics"]
        init23 = {k: v for k, v in init23.items() if _num(v) and (k.startswith("eval/") or k.startswith("extra/"))}
        init23_src = f"{UPLIFT_DATA}/eval_table.json  table.init.metrics (ckpt {ARMS['init23m']['init_adapter']}, v2 held-out eval)"
    except Exception as e:  # noqa: BLE001
        init23 = {"eval/mean_all": 0.3684, "eval/sae/norm_act": 0.4158, "eval/sae/unverbalized_frac": 0.4648, "eval/sae/rank1_frac": 0.1895,
                  "eval/realact/cos": 0.4775, "eval/realact_long/cos": 0.4066, "eval/bsf/cos": 0.2959, "eval/cluster/cos": 0.2260,
                  "eval/mlp/norm_act": 0.1212, "eval/mlp/fired10": 0.2363, "eval/jlens/cos": 0.1033, "eval/random/cos": 0.0325}
        init23_src = f"hard-coded fallback (uplift eval_table.json unreadable: {e})"
    init23["eval/sae/verbalized_frac"] = 1.0 - init23["eval/sae/unverbalized_frac"]
    # --- RL-C reference: before-RL row from the uplift report's references.json ---
    try:
        refinit = json.load(open(UPLIFT_DATA / "references.json"))["sft_mix1m_midtrain_final"]["metrics"]
        refinit = {k: v for k, v in refinit.items() if _num(v) and k.startswith("eval/")}
        refinit_src = f"{UPLIFT_DATA}/references.json  sft_mix1m_midtrain_final (RL-C's init = 23M pretrain + 1.1M midtrain final)"
    except Exception as e:  # noqa: BLE001
        refinit = {"eval/mean_all": 0.3403, "eval/sae/norm_act": 0.4501, "eval/realact/cos": 0.4445, "eval/sae/unverbalized_frac": 0.4473}
        refinit_src = f"hard-coded fallback ({e})"
    refinit["eval/sae/verbalized_frac"] = 1.0 - refinit["eval/sae/unverbalized_frac"]
    before = {"init23m": (init23, init23_src),
              "initmid": (out["midtrain_sft"]["final"], f"wandb {PROJ}/{MIDTRAIN_SFT_EVAL} mean of ckpt_step {out['midtrain_sft']['final_steps_used']}"),
              "initbase": ({"eval/mean_all": RANDOM_FLOOR, "eval/random/cos": RANDOM_FLOOR},
                           "NOT measured: an untrained LoRA scores the random-direction floor (~0.03 cosine) by construction; per-family values unknown")}
    # --- the three arms ---
    for arm, cfg in ARMS.items():
        ev = _run(api, PROJ, cfg["eval"])
        tr = _run(api, PROJ, cfg["train"])
        hist = [r for r in tr.scan_history() if _num(r.get("_step"))]
        keys = [k for k, _ in TRAIN_PANELS] + TRAIN_EXTRA
        train = {"step": [int(r["_step"]) for r in hist]}
        for k in keys:
            train[k] = [r.get(k) if _num(r.get(k)) else None for r in hist]
        conf = {k: v for k, v in tr.config.items() if not isinstance(v, (dict, list))}
        out["arms"][arm] = {"eval_run": cfg["eval"], "eval_id": ev.id, "eval_state": ev.state, "train_run": cfg["train"], "train_id": tr.id,
                            "train_state": tr.state, "train_last_step": int(max(train["step"])) if train["step"] else None,
                            "evals": _eval_rows(ev), "before_rl": before[arm][0], "before_rl_source": before[arm][1], "train": train, "config": conf}
    # --- RL-C ---
    rc = _run(api, REF_PROJ, REF["eval"])
    out["ref"] = {"eval_run": REF["eval"], "project": REF_PROJ, "id": rc.id, "state": rc.state, "evals": _eval_rows(rc), "before_rl": refinit,
                  "before_rl_source": refinit_src, "config": {k: v for k, v in rc.config.items() if not isinstance(v, (dict, list))},
                  "note": "RL-C: same lr 7e-6 / ScaleRL recipe but 16x256 rollouts per step, init = 23M pretrain + 1.1M all-families midtrain, bank mix_1m_v2 "
                          "(1.1M, five families, no MLP neurons). Its ckpt 400 row (mean_all 0.315) is a late collapse and is NOT plotted (arms stop at 300)."}
    return out


def fetch_transcripts(arms, fetch=True):
    tx = {}
    for arm in arms:
        local = Path(f"/tmp/abl_tx_{arm}.jsonl")
        remote = f"ckpts_rl_abl_{arm}/transcripts.jsonl"
        if fetch:
            cmd = f"source ~/modal_venv/bin/activate && export MODAL_PROFILE=safety-sahan && modal volume get maemm-data {remote} {local} --force"
            try:
                subprocess.run(["bash", "-lc", cmd], check=True, timeout=900, capture_output=True)
            except Exception as e:  # noqa: BLE001
                print(f"[warn] transcript fetch failed for {arm}: {e}; using cached {local} if present")
        rows = []
        if local.exists():
            for line in open(local):
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        tx[arm] = rows
    return tx


# --------------------------------------------------------------- derived numbers ---------------------------------------------------------------
def _ttr(text):
    w = re.findall(r"[a-z0-9]+", text.lower())
    return (len(set(w)) / len(w)) if len(w) >= 4 else 1.0


def is_degenerate(r):
    return bool(r["n_tok"] <= DEGENERATE_MAX_TOK or _ttr(r["text"]) < DEGENERATE_TTR)


def transcript_stats(rows):
    by_step = {}
    for r in rows:
        by_step.setdefault(int(r["step"]), []).append(r)
    stats = []
    for s in sorted(by_step):
        rs = by_step[s]
        stats.append({"step": s, "n": len(rs), "cos_mean": statistics.mean(r["cos"] for r in rs), "reward_mean": statistics.mean(r["reward"] for r in rs),
                      "n_tok_mean": statistics.mean(r["n_tok"] for r in rs), "n_tok_median": statistics.median(r["n_tok"] for r in rs),
                      "degenerate_share": sum(is_degenerate(r) for r in rs) / len(rs), "ttr_mean": statistics.mean(_ttr(r["text"]) for r in rs)})
    return stats


def transcript_examples(rows, steps):
    ex = []
    by_step = {}
    for r in rows:
        by_step.setdefault(int(r["step"]), []).append(r)
    for s in steps:
        rs = sorted(by_step.get(s, []), key=lambda r: -r["cos"])
        if not rs:
            continue
        picks = [("best of 16", rs[0]), ("median of 16", rs[len(rs) // 2])]
        for tag, r in picks:
            ex.append({"step": s, "pick": tag, "family": r["family"], "vec_idx": r["vec_idx"], "target": r["target"], "text": r["text"], "cos": r["cos"],
                       "reward": r["reward"], "n_tok": r["n_tok"], "degenerate": is_degenerate(r)})
    return ex


def rolling(y, w):
    y = np.array([np.nan if v is None else v for v in y], dtype=float)
    ok = ~np.isnan(y)
    num = np.convolve(np.where(ok, y, 0.0), np.ones(w), "same")
    den = np.convolve(ok.astype(float), np.ones(w), "same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def series(arm_data, key, with_before=True):
    xs, ys = [], []
    if with_before and _num(arm_data["before_rl"].get(key)):
        xs.append(0); ys.append(arm_data["before_rl"][key])
    for r in arm_data["evals"]:
        if _num(r.get(key)):
            xs.append(r["ckpt_step"]); ys.append(r[key])
    return xs, ys


def at_step(arm_data, step, key="eval/mean_all"):
    for r in arm_data["evals"]:
        if r["ckpt_step"] == step and _num(r.get(key)):
            return r[key]
    return None


def last_eval(arm_data, key="eval/mean_all"):
    rows = [r for r in arm_data["evals"] if _num(r.get(key))]
    return (rows[-1]["ckpt_step"], rows[-1][key]) if rows else (None, None)


def fmt(v, nd=3):
    return "n/a" if v is None else f"{v:.{nd}f}"


def signed(v, nd=3):
    return "n/a" if v is None else f"{v:+.{nd}f}"


# -------------------------------------------------------------------- plots --------------------------------------------------------------------
def style_ax(ax):
    ax.grid(True, axis="y"); ax.grid(False, axis="x")
    ax.tick_params(length=0)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS)


def save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / f"{stem}.png")


def add_ref(ax, ref, key, label=None):
    xs, ys = [], []
    if _num(ref["before_rl"].get(key)):
        xs.append(0); ys.append(ref["before_rl"][key])
    for r in ref["evals"]:
        if r["ckpt_step"] <= 300 and _num(r.get(key)):
            xs.append(r["ckpt_step"]); ys.append(r[key])
    if xs:
        ax.plot(xs, ys, ls="--", lw=1.4, color=REF["color"], label=label, zorder=1)


def add_arm(ax, arm, d, key, lw=2.0, ms=6.5, label=None):
    xs, ys = series(d, key)
    cfg = ARMS[arm]
    if not xs:
        return
    ax.plot(xs, ys, color=cfg["color"], lw=lw, marker=cfg["marker"], ms=ms, mec="white", mew=1.2, label=label, zorder=3, solid_capstyle="round")
    if xs[0] == 0 and arm == "initbase":   # before-RL value is the floor by construction, not a measurement -> hollow marker
        ax.plot([0], [ys[0]], marker=cfg["marker"], ms=ms + 1, mfc="white", mec=cfg["color"], mew=1.6, ls="none", zorder=4)


def claim_headline(D, S):
    a, b, c = (at_step(D["arms"][k], S) for k in ("init23m", "initmid", "initbase"))
    if None in (a, b):
        return f"Held-out fidelity vs RL step for three SFT starting points (matched step {S})"
    gap = b - a
    if abs(gap) < 0.01:
        cmp_ = f"Both SFT inits land within {abs(gap):.3f} of each other after {S} RL steps (23M pretrain {a:.3f}, midtrain-only {b:.3f})"
    elif gap > 0:
        cmp_ = f"The midtrain-only init leads the 23M pretrain by {gap:.3f} after {S} RL steps ({b:.3f} vs {a:.3f}) despite starting lower"
    else:
        cmp_ = f"The 23M pretrain leads the midtrain-only init by {-gap:.3f} after {S} RL steps ({a:.3f} vs {b:.3f})"
    base = f"skipping SFT entirely reaches only {c:.3f}" if c is not None else "skipping SFT entirely lags far behind"
    return f"{cmp_}; {base}"


def plot_headline(D, S):
    fig, ax = plt.subplots(figsize=(10, 5.6))
    add_ref(ax, D["ref"], "eval/mean_all", REF["label"])
    ends = []
    for arm, d in D["arms"].items():
        cfg = ARMS[arm]
        state = "" if d["train_state"] == "finished" else f"  [still running, RL step {d['train_last_step']}]"
        add_arm(ax, arm, d, "eval/mean_all", label=cfg["label"] + state)
        s, v = last_eval(d)
        if s is not None:
            ends.append((v, s, cfg))
    # direct end labels (text in ink, identity from the marker colour); push apart only when two ends sit at nearby x
    ends.sort(key=lambda t: t[0])
    ys = [e[0] for e in ends]
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < 0.014 and abs(ends[i][1] - ends[i - 1][1]) < 40:
            ys[i] = ys[i - 1] + 0.014
    for (v, s, cfg), y in zip(ends, ys):
        ax.annotate(f"{cfg['short']}: {v:.3f}", xy=(s, v), xytext=(s + 6, y), fontsize=9, color=INK2, va="center",
                    arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.8) if abs(y - v) > 0.004 else None)
    if _num(D["arms"]["init23m"]["before_rl"].get("eval/mean_all")):
        ax.annotate("before RL\n(SFT init alone)", xy=(0, D["arms"]["init23m"]["before_rl"]["eval/mean_all"]), xytext=(9, 0.30), fontsize=8.5, color=MUTED,
                    arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.8))
    ax.annotate("no SFT: random-direction\nfloor by construction (~0.03)", xy=(0, RANDOM_FLOOR), xytext=(9, 0.07), fontsize=8.5, color=MUTED,
                arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.8))
    ax.axvline(S, color=GRID, lw=1.0, zorder=0)
    ax.text(S, 0.005, f"step {S}: last checkpoint\nall three arms have", fontsize=8, color=MUTED, ha="center", va="bottom")
    ax.set_xlim(-6, 392); ax.set_ylim(0, 0.47)
    ax.set_xticks([0, 25, 50, 100, 150, 200, 250, 300])
    ax.set_xlabel("RL step (4,096 rollouts per step; step 0 = the SFT init before any RL)")
    ax.set_ylabel("held-out fidelity: mean cosine over 10 direction families")
    ax.set_title(claim_headline(D, S) + "\nSame RL recipe (CISPO GRPO, 8 x 512 rollouts/step, lr 7e-6, 300 steps) on the same equal six-family bank; "
                 "Qwen3.6-27B activation-to-text inverter, 512 held-out directions per family, best-of-4", fontsize=10.5, loc="left", pad=10)
    ax.legend(loc="lower right", fontsize=8.5)
    style_ax(ax)
    save(fig, "sft_init_mean_fidelity")


def plot_families(D, S):
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.6), sharex=True)
    for ax, (key, title, _) in zip(axes.flat, FAMILY_PANELS):
        add_ref(ax, D["ref"], key, REF["label"])
        for arm, d in D["arms"].items():
            add_arm(ax, arm, d, key, lw=1.8, ms=5.5, label=ARMS[arm]["label"])
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xticks([0, 50, 100, 150, 200, 250, 300]); ax.set_xlim(-6, 312)
        if key == "eval/random/cos":
            ax.set_ylim(0, 0.08)
        else:
            ax.set_ylim(0, None)
        style_ax(ax)
    for ax in axes[-1]:
        ax.set_xlabel("RL step (0 = before RL)", fontsize=9)
    # which family separates the two SFT inits most / least at the matched step (computed, not asserted)
    gaps = []
    for key, title, _ in FAMILY_PANELS:
        if key == "eval/random/cos":
            continue
        v23, vmid = at_step(D["arms"]["init23m"], S, key), at_step(D["arms"]["initmid"], S, key)
        if v23 is not None and vmid is not None:
            gaps.append((abs(vmid - v23), title.split(":")[0].split(" (")[0], vmid, v23))
    gaps.sort()
    hi, lo = gaps[-1], gaps[0]
    sae_txt = f"{hi[1]} ({hi[2]:.2f} vs {hi[3]:.2f} at step {S}, midtrain-only vs 23M pretrain)"
    ra_txt = f"{lo[1]} ({lo[2]:.3f} vs {lo[3]:.3f})"
    fig.suptitle(f"The two SFT inits differ most on {sae_txt} and least on {ra_txt}; the no-SFT policy stays near the floor in every family\n"
                 "Per-family held-out fidelity vs RL step - same CISPO GRPO recipe (8 x 512 rollouts/step, lr 7e-6) on the same equal six-family bank, "
                 "Qwen3.6-27B activation-to-text inverter; hollow marker = floor by construction, not measured", fontsize=11, x=0.01, ha="left", y=0.995)
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    save(fig, "sft_init_per_family")


def _tail_mean(d, key, n=50):
    ys = [v for v in d["train"][key][-n:] if _num(v)]
    return float(np.mean(ys)) if ys else None


def plot_train(D):
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 7.8), sharex=True)
    W = 9
    for ax, (key, title) in zip(axes.flat, TRAIN_PANELS):
        for arm, d in D["arms"].items():
            x = np.array(d["train"]["step"]); y = d["train"][key]
            if not any(_num(v) for v in y):
                continue
            yy = np.array([np.nan if v is None else v for v in y], dtype=float)
            ax.plot(x, yy, color=ARMS[arm]["color"], lw=0.8, alpha=0.28, zorder=2)
            ax.plot(x, rolling(y, W), color=ARMS[arm]["color"], lw=2.0, label=ARMS[arm]["label"], zorder=3, solid_capstyle="round")
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlim(0, 305); ax.set_xticks([0, 50, 100, 150, 200, 250, 300])
        if key in LOG_PANELS:
            ax.set_yscale("log")
        elif key in ("ratio/clipfrac", "reward/mean"):
            ax.set_ylim(0, None)
        style_ax(ax)
    for ax in axes[-1]:
        ax.set_xlabel("RL step", fontsize=9)
    b, a = D["arms"]["initbase"], D["arms"]["init23m"]
    fig.suptitle(f"Same recipe, very different optimisation: the no-SFT policy collapses to ~{fmt(_tail_mean(b, 'rollout/len_mean'), 0)}-token, "
                 f"entropy {fmt(_tail_mean(b, 'policy/entropy'), 2)} outputs with {100 * (_tail_mean(b, 'ratio/clipfrac') or 0):.0f}% clipped tokens, "
                 f"while the 23M pretrain trains smoothly ({fmt(_tail_mean(a, 'rollout/len_mean'), 0)} tokens, entropy {fmt(_tail_mean(a, 'policy/entropy'), 2)}, "
                 f"{100 * (_tail_mean(a, 'ratio/clipfrac') or 0):.1f}% clipped; last-50-step means)\n"
                 "RL training dynamics vs step for the three SFT inits - CISPO GRPO, 8 x 512 rollouts/step, lr 7e-6 (25 warmup), no KL; "
                 f"thin line = per-step value, thick = {W}-step centred moving average", fontsize=11, x=0.01, ha="left", y=0.995)
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    save(fig, "sft_init_train_dynamics")


def matched_numbers(D, S):
    res = {"matched_step": S, "arms": {}, "ref": {}}
    for arm, d in D["arms"].items():
        before = d["before_rl"].get("eval/mean_all")
        m, f = at_step(d, S), at_step(d, 300)
        ls, lv = last_eval(d)
        res["arms"][arm] = {"label": ARMS[arm]["label"], "before_rl": before, "before_rl_measured": arm != "initbase", f"at_{S}": m, "at_300": f,
                            "last_step": ls, "last_value": lv, f"delta_{S}": None if m is None or before is None else m - before,
                            "delta_300": None if f is None or before is None else f - before}
    r = D["ref"]
    res["ref"] = {"label": REF["label"], "before_rl": r["before_rl"].get("eval/mean_all"), f"at_{S}": at_step(r, S), "at_300": at_step(r, 300)}
    return res


def plot_bars(D, S, M):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw={"width_ratios": [3, 2]})
    arms = list(ARMS)
    w = 0.24
    # left: absolute level at before / matched / 300
    cats = [("before RL", "before_rl"), (f"RL step {S}", f"at_{S}"), ("RL step 300", "at_300")]
    ax = axes[0]
    for i, arm in enumerate(arms):
        vals = [M["arms"][arm][k] for _, k in cats]
        xs = np.arange(len(cats)) + (i - 1) * (w + 0.02)
        for x, v, (_, k) in zip(xs, vals, cats):
            if v is None:
                ax.text(x, 0.01, "pending" if k == "at_300" else "n/a", rotation=90, fontsize=8, color=MUTED, ha="center", va="bottom")
                continue
            hollow = (arm == "initbase" and k == "before_rl")
            ax.bar(x, v, width=w, color="white" if hollow else ARMS[arm]["color"], edgecolor=ARMS[arm]["color"] if hollow else "none", lw=1.5,
                   label=ARMS[arm]["label"] if k == "before_rl" or (arm == "initbase" and k == f"at_{S}") else None, zorder=3)
            ax.text(x, v + 0.006, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5, color=INK2)
    for _, k in cats[1:]:
        pass
    rv = M["ref"].get(f"at_{S}")
    if rv is not None:
        ax.axhline(rv, color=REF["color"], ls="--", lw=1.2, zorder=1)
        ax.text(2.42, rv + 0.004, f"RL-C reference @ {S}: {rv:.3f}", fontsize=8, color=MUTED, ha="right", va="bottom")
    ax.set_xticks(np.arange(len(cats))); ax.set_xticklabels([c for c, _ in cats])
    ax.set_ylim(0, 0.5); ax.set_ylabel("held-out fidelity (mean cosine, 10 families)")
    ax.set_title("absolute level (hollow bar = floor by construction, not measured)", fontsize=10, loc="left")
    hnd = {l: h for h, l in zip(*ax.get_legend_handles_labels())}
    ax.legend(hnd.values(), hnd.keys(), loc="upper left", fontsize=8.5)
    style_ax(ax)
    # right: gain from RL
    ax = axes[1]
    dcats = [(f"gain by step {S}", f"delta_{S}"), ("gain by step 300", "delta_300")]
    for i, arm in enumerate(arms):
        xs = np.arange(len(dcats)) + (i - 1) * (w + 0.02)
        for x, (_, k) in zip(xs, dcats):
            v = M["arms"][arm][k]
            if v is None:
                ax.text(x, 0.004, "pending", rotation=90, fontsize=8, color=MUTED, ha="center", va="bottom")
                continue
            ax.bar(x, v, width=w, color=ARMS[arm]["color"], zorder=3)
            ax.text(x, v + 0.004, f"{v:+.3f}", ha="center", va="bottom", fontsize=8.5, color=INK2)
    ax.set_xticks(np.arange(len(dcats))); ax.set_xticklabels([c for c, _ in dcats])
    ax.set_ylim(0, 0.26); ax.set_ylabel("fidelity gained from RL (after - before)")
    ax.set_title("what RL added on top of each init", fontsize=10, loc="left")
    style_ax(ax)
    a = M["arms"]
    fig.suptitle(f"RL adds {signed(a['init23m'][f'delta_{S}'])} to the 23M pretrain, {signed(a['initmid'][f'delta_{S}'])} to the midtrain-only init and "
                 f"{signed(a['initbase'][f'delta_{S}'])} to the no-SFT policy by step {S} - the SFT inits land at {fmt(a['init23m'][f'at_{S}'])} / "
                 f"{fmt(a['initmid'][f'at_{S}'])}, no SFT at {fmt(a['initbase'][f'at_{S}'])}\n"
                 "Held-out mean fidelity before RL vs after a matched number of RL steps, three SFT starting points, same CISPO GRPO recipe and bank "
                 "(Qwen3.6-27B activation-to-text inverter)", fontsize=11, x=0.01, ha="left", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "sft_init_matched_step_bars")


def plot_transcripts(TX):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), sharex=True)
    panels = [("cos_mean", "mean cosine of the 16 sampled rollouts"), ("n_tok_mean", "mean response length (tokens)"), ("degenerate_share", "degenerate share of sampled rollouts")]
    for ax, (key, title) in zip(axes, panels):
        for arm, st in TX["stats"].items():
            if not st:
                continue
            x = [s["step"] for s in st]; y = [s[key] for s in st]
            ax.plot(x, y, color=ARMS[arm]["color"], lw=0.9, alpha=0.3, zorder=2)
            ax.plot(x, rolling(y, 3), color=ARMS[arm]["color"], lw=2.0, label=ARMS[arm]["label"], zorder=3, solid_capstyle="round")
        ax.set_title(title, fontsize=10, loc="left"); ax.set_xlim(0, 305); ax.set_xticks([0, 50, 100, 150, 200, 250, 300]); ax.set_xlabel("RL step", fontsize=9)
        if key == "degenerate_share":
            ax.set_ylim(0, 1.02)
        else:
            ax.set_ylim(0, None)
        style_ax(ax)
    tails = {arm: (np.mean([s["n_tok_mean"] for s in st[-10:]]) if st else None, np.mean([s["degenerate_share"] for s in st[-10:]]) if st else None)
             for arm, st in TX["stats"].items()}
    tb, ta = tails["initbase"], tails["init23m"]
    fig.suptitle(f"The no-SFT arm's rollouts shrink to ~{fmt(tb[0], 0)} tokens and {100 * (tb[1] or 0):.0f}% are degenerate (repetition or at the 8-token floor); "
                 f"the 23M-pretrain arm holds ~{fmt(ta[0], 0)} tokens with {100 * (ta[1] or 0):.0f}% degenerate (last 10 logged steps)\n"
                 f"Per-step statistics of the logged rollout transcripts (16 rollouts = 4 groups x 4 samples every 5 RL steps); degenerate = <= {DEGENERATE_MAX_TOK} tokens "
                 f"or word type/token ratio < {DEGENERATE_TTR}; thick line = 3-point moving average", fontsize=11, x=0.01, ha="left", y=1.0)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.02, 1, 0.88))
    save(fig, "sft_init_transcript_stats")


# ---------------------------------------------------------------- data files ----------------------------------------------------------------
def write_data(D, TX, S, M):
    dd = OUT / "data"; dd.mkdir(parents=True, exist_ok=True)
    meta = {"generated_at": D["fetched_at"], "matched_step": S, "wandb_project": PROJ, "ref_project": REF_PROJ, "random_floor": RANDOM_FLOOR,
            "arms": {a: {k: d[k] for k in ("eval_run", "eval_id", "eval_state", "train_run", "train_id", "train_state", "train_last_step", "before_rl_source")}
                     | {"label": ARMS[a]["label"], "init_adapter": ARMS[a]["init_adapter"], "ckpt_dir": ARMS[a]["ckpt_dir"], "init_desc": ARMS[a]["init_desc"],
                        "eval_steps": [r["ckpt_step"] for r in d["evals"]]} for a, d in D["arms"].items()},
            "ref": {k: D["ref"][k] for k in ("eval_run", "project", "id", "state", "before_rl_source", "note")} | {"label": REF["label"]},
            "midtrain_sft": {k: D["midtrain_sft"][k] for k in ("run", "id", "state", "final_steps_used")},
            "family_panels": FAMILY_PANELS, "train_panels": TRAIN_PANELS, "degenerate_rule": {"max_tok": DEGENERATE_MAX_TOK, "ttr_lt": DEGENERATE_TTR}}
    json.dump(meta, open(dd / "meta.json", "w"), indent=1)
    json.dump({"arms": {a: {"label": ARMS[a]["label"], "before_rl": d["before_rl"], "before_rl_source": d["before_rl_source"], "evals": d["evals"]}
                        for a, d in D["arms"].items()},
               "ref": {"label": REF["label"], "before_rl": D["ref"]["before_rl"], "evals": D["ref"]["evals"], "note": D["ref"]["note"]},
               "midtrain_sft_trajectory": D["midtrain_sft"]["rows"]}, open(dd / "eval_curves.json", "w"), indent=1)
    train = {}
    for a, d in D["arms"].items():
        t = dict(d["train"])
        for k, _ in TRAIN_PANELS:
            t[k + "__ma9"] = [None if np.isnan(v) else float(v) for v in rolling(d["train"][k], 9)]
        train[a] = {"label": ARMS[a]["label"], "run": d["train_run"], "id": d["train_id"], "state": d["train_state"], "series": t}
    json.dump(train, open(dd / "train_dynamics.json", "w"), indent=1)
    json.dump(M, open(dd / "matched_step.json", "w"), indent=1)
    json.dump({"rule": meta["degenerate_rule"], "stats": TX["stats"], "examples": TX["examples"]}, open(dd / "transcripts.json", "w"), indent=1)
    # checkpoint x family table (before-RL row first), coloured in build_html relative to the 23M init's before-RL row
    tbl = {"columns": TABLE_COLS, "reference_row": D["arms"]["init23m"]["before_rl"], "arms": {}}
    for a, d in D["arms"].items():
        rows = [{"ckpt_step": 0, "kind": "before RL", **{k: d["before_rl"].get(k) for k, _ in TABLE_COLS}}]
        rows += [{"ckpt_step": r["ckpt_step"], "kind": "RL", **{k: r.get(k) for k, _ in TABLE_COLS}} for r in d["evals"]]
        tbl["arms"][a] = {"label": ARMS[a]["label"], "rows": rows}
    tbl["ref"] = {"label": REF["label"], "rows": [{"ckpt_step": 0, "kind": "before RL", **{k: D["ref"]["before_rl"].get(k) for k, _ in TABLE_COLS}}]
                  + [{"ckpt_step": r["ckpt_step"], "kind": "RL", **{k: r.get(k) for k, _ in TABLE_COLS}} for r in D["ref"]["evals"]]}
    json.dump(tbl, open(dd / "checkpoint_table.json", "w"), indent=1)
    cfgs = {a: d["config"] for a, d in D["arms"].items()}
    allk = sorted(set().union(*[set(c) for c in cfgs.values()]))
    diff = {k: {a: cfgs[a].get(k) for a in cfgs} for k in allk if len({json.dumps(cfgs[a].get(k), default=str) for a in cfgs}) > 1}
    shared = {k: cfgs["init23m"].get(k) for k in allk if k not in diff}
    json.dump({"shared": shared, "differs": diff, "ref_config": D["ref"]["config"]}, open(dd / "run_config.json", "w"), indent=1, default=str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch-transcripts", action="store_true", help="reuse /tmp/abl_tx_<arm>.jsonl instead of pulling from the Modal volume")
    ap.add_argument("--no-html", action="store_true", help="skip running the report folder's build_html.py")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    D = fetch_wandb()
    for a, d in D["arms"].items():
        print(f"{a}: train {d['train_state']} @ step {d['train_last_step']}; eval ckpts {[r['ckpt_step'] for r in d['evals']]}; "
              f"before-RL mean_all {fmt(d['before_rl'].get('eval/mean_all'))}")
    common = set.intersection(*[{r["ckpt_step"] for r in d["evals"]} for d in D["arms"].values()])
    S = max(common) if common else 25
    tx_rows = fetch_transcripts(list(ARMS), fetch=not args.no_fetch_transcripts)
    TX = {"stats": {a: transcript_stats(r) for a, r in tx_rows.items()}, "examples": {}}
    for a, r in tx_rows.items():
        steps = sorted({s for s in (100, max((x["step"] for x in r), default=None)) if s is not None})
        TX["examples"][a] = transcript_examples(r, steps)
    M = matched_numbers(D, S)
    write_data(D, TX, S, M)
    plot_headline(D, S); plot_families(D, S); plot_train(D); plot_bars(D, S, M)
    if any(TX["stats"].values()):
        plot_transcripts(TX)
    print(json.dumps(M, indent=1))
    bh = OUT / "build_html.py"
    if not args.no_html and bh.exists():
        subprocess.run(["python3", str(bh)], check=True)


if __name__ == "__main__":
    main()
