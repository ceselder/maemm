#!/usr/bin/env python3
"""SFT-init ablation of the Qwen3.6-27B activation->text inverter: does the SFT starting point matter once you RL on the same bank?

Six arms. Five policies get the IDENTICAL LoRA RL recipe (ScaleRL/CISPO GRPO, group-relative advantages, 8 samples x 512 directions = 4,096 rollouts/step,
constant lr 7e-6 after 25 warmup steps, no KL, length penalty, 300 steps, ckpt every 25) on ONE shared bank (/data/banks/mix_eq_1p45m: six
equal families x 241,741) and differ only in the init (all RL policies are LoRA r64 / alpha16 rsLoRA):
  init23m    = 23M real-activation LoRA pretrain only                       (/data/sft_mix/realact20m_prefix_lr1e-4/final)
  initmid    = midtrain-only LoRA on the equal bank, from the raw base        (/data/sft_mix/mixeq_midtrain_only_from_base/final)
  initboth   = 23M pretrain LoRA continued with a LoRA midtrain on the bank   (/data/sft_mix/mixeq_midtrain_from_realact23m/final)
  initnewfft = NEW 104M full-fine-tune pretrain + full-FT midtrain; RL LoRA starts fresh on those weights (policy base
               /data/sft_mix/mixeq_midtrain_fft_from_fft23m_v2/final)
  initbase   = fresh LoRA on the raw base model, no SFT at all (--init-adapter none)
plus one RL-mode arm on the initnewfft init:
  initnewfft_fullparam = FULL-PARAMETER RL (every weight trainable, lr 1e-6, same 8x512 recipe / bank / 300 steps; full-model checkpoints)

Idempotent: every run re-pulls the eval + training histories from wandb, the inits' own SFT evals, the MLP-rank re-score summary written by
scripts/plot_mlp_rank_eval.py (if present) and, unless --no-fetch-transcripts, the rollout transcripts from the Modal volume; rewrites
data/*.json, every figure as PNG + PDF, then (unless --no-html) runs the report folder's build_html.py.

    python scripts/plot_sft_init_ablation.py                          # refresh everything
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
UPLIFT_DATA = Path("~/shared/reports/maemm-uplift-matrix/data").expanduser()      # the 23M init's held-out eval (same v2 protocol) lives here
MLPRANK_DATA = Path("~/shared/reports/maemm-mlp-rank-eval/data").expanduser()     # rank-based MLP re-score of the finals (plot_mlp_rank_eval.py)

# ---- arms (legend/table order). Colour follows the entity: 5-slot palette validated all-pairs on the ivory surface (magenta needs the
#      relief rule -> every chart has direct labels or a table twin). before = how the before-RL row is obtained.
ARMS = {
    "init23m": dict(label="23M pretrain only", short="23M pretrain", color="#2a78d6", marker="o",
                    train="rl_abl_init23m_8x512_lr7e-6", eval="rl_abl_init23m_8x512_lr7e-6_eval",
                    init_adapter="/data/sft_mix/realact20m_prefix_lr1e-4/final", policy_base="Qwen/Qwen3.6-27B", ckpt_dir="/data/ckpts_rl_abl_init23m",
                    before=("uplift",), mlprank=("sft_realact23m", "rl_abl_init23m"), pretrain=True, midtrain=False,
                    init_desc="LoRA SFT on 23M real layer-42 activations (eff batch 512, lr 1e-4); never saw the RL bank"),
    "initmid": dict(label="midtrain only", short="midtrain only", color="#d95926", marker="s",
                    train="rl_abl_initmid_8x512_lr7e-6", eval="rl_abl_initmid_8x512_lr7e-6_eval",
                    init_adapter="/data/sft_mix/mixeq_midtrain_only_from_base/final", policy_base="Qwen/Qwen3.6-27B", ckpt_dir="/data/ckpts_rl_abl_initmid",
                    before=("sft_eval", "mixeq_midtrain_only_from_base_eval", (5664, 5665)), mlprank=("sft_midtrain", "rl_abl_initmid"), pretrain=False, midtrain=True,
                    init_desc="LoRA SFT from the raw base on the SAME equal six-family bank RL uses (1 epoch = 1.45M examples, eff batch 256, lr 1e-4); no real-activation pretrain"),
    "initboth": dict(label="23M pretrain + midtrain", short="pretrain + midtrain", color="#4a3aa7", marker="D",
                     train="rl_abl_initboth_8x512_lr7e-6", eval="rl_abl_initboth_8x512_lr7e-6_eval",
                     init_adapter="/data/sft_mix/mixeq_midtrain_from_realact23m/final", policy_base="Qwen/Qwen3.6-27B", ckpt_dir="/data/ckpts_rl_abl_initboth",
                     before=("sft_eval", "mixeq_midtrain_from_realact23m_eval", (5665,)), mlprank=(None, "rl_abl_initboth"), pretrain=True, midtrain=True,
                     init_desc="the 23M real-activation LoRA pretrain continued with a LoRA midtrain on the equal bank (1 epoch, same recipe as midtrain-only)"),
    "initnewfft": dict(label="new full-FT pretrain + full-FT midtrain (LoRA RL)", short="new full-FT + midtrain", color="#e87ba4", marker="P",
                       train="rl_abl_initnewfft_8x512_lr7e-6", eval="rl_abl_initnewfft_8x512_lr7e-6_eval",
                       init_adapter=None, policy_base="/data/sft_mix/mixeq_midtrain_fft_from_fft23m_v2/final", ckpt_dir="/data/ckpts_rl_abl_initnewfft",
                       before=("sft_eval", "mixeq_midtrain_fft_from_fft23m_v2_eval", (354, 355)), mlprank=(None, "rl_abl_initnewfft"), pretrain=True, midtrain=True,
                       init_desc="NEW pretrain: 104M-example FULL fine-tune (evaluated at 23M) + full-FT midtrain on the equal bank; the RL LoRA starts FRESH on "
                                 "those full-model weights (policy base = that dir) instead of continuing a midtrain adapter"),
    "initnewfft_fullparam": dict(label="new full-FT pretrain + full-FT midtrain, FULL-PARAMETER RL (lr 1e-6)", short="full-FT init, full-param RL",
                                 color="#006300", marker="X", ls="--", full_param=True,
                                 train="rl_abl_initnewfft_fullparam_8x512", eval="rl_abl_initnewfft_fullparam_8x512_eval",
                                 init_adapter=None, policy_base="/data/sft_mix/mixeq_midtrain_fft_from_fft23m_v2/final", ckpt_dir="/data/ckpts_rl_abl_initnewfft_fullparam",
                                 before=("sft_eval", "mixeq_midtrain_fft_from_fft23m_v2_eval", (354, 355)), mlprank=(None, "rl_abl_initnewfft_fullparam"), pretrain=True, midtrain=True,
                                 init_desc="the SAME init as the LoRA-RL arm above (new 104M full-FT pretrain + full-FT midtrain), but RL updates EVERY weight (no LoRA) "
                                           "at lr 1e-6 instead of 7e-6; checkpoints are full-model dirs"),
    "initbase": dict(label="no SFT (base model)", short="no SFT", color="#199e70", marker="^",
                     train="rl_abl_initbase_8x512_lr7e-6", eval="rl_abl_initbase_8x512_lr7e-6_eval",
                     init_adapter=None, policy_base="Qwen/Qwen3.6-27B", ckpt_dir="/data/ckpts_rl_abl_initbase",
                     before=("floor",), mlprank=(None, "rl_abl_initbase"), pretrain=False, midtrain=False,
                     init_desc="fresh LoRA on Qwen/Qwen3.6-27B, --init-adapter none; no SFT of any kind"),
}
SFT_ARMS = [a for a in ARMS if a != "initbase"]                            # every arm that had some SFT
LORA_SFT_ARMS = [a for a in SFT_ARMS if not ARMS[a].get("full_param")]     # the init ablation proper: identical LoRA-RL recipe
FULLPARAM_PAIR = ("initnewfft_fullparam", "initnewfft")                    # same init, full-parameter RL vs LoRA RL
REF = dict(label="reference: earlier run RL-C (23M pretrain + 1.1M midtrain; 16x256 rollouts, different 1.1M bank)", short="RL-C reference",
           color="#898781", eval="rl_C_mix1m_from_realact23m_mixsft_eval")
RANDOM_FLOOR = 0.03      # cosine to a random direction; what an untrained inverter scores
STEPS = [100, 200, 300]  # the matched-step table

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
RANK_COLS = [("eval/mlp/rank1_frac", "MLP rank-1 (re-score)"), ("eval/mlp/rank_le10", "MLP top-10 (re-score)")]
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


def _uplift_init23m():
    try:
        m = json.load(open(UPLIFT_DATA / "eval_table.json"))["table"]["init"]["metrics"]
        m = {k: v for k, v in m.items() if _num(v) and (k.startswith("eval/") or k.startswith("extra/"))}
        src = f"{UPLIFT_DATA}/eval_table.json  table.init.metrics (ckpt {ARMS['init23m']['init_adapter']}, v2 held-out eval)"
    except Exception as e:  # noqa: BLE001
        m = {"eval/mean_all": 0.3684, "eval/sae/norm_act": 0.4158, "eval/sae/unverbalized_frac": 0.4648, "eval/sae/rank1_frac": 0.1895,
             "eval/realact/cos": 0.4775, "eval/realact_long/cos": 0.4066, "eval/bsf/cos": 0.2959, "eval/cluster/cos": 0.2260,
             "eval/mlp/norm_act": 0.1212, "eval/mlp/fired10": 0.2363, "eval/jlens/cos": 0.1033, "eval/random/cos": 0.0325}
        src = f"hard-coded fallback (uplift eval_table.json unreadable: {e})"
    m["eval/sae/verbalized_frac"] = 1.0 - m["eval/sae/unverbalized_frac"]
    return m, src


def fetch_wandb():
    api = wandb.Api(timeout=180)
    out = {"arms": {}, "ref": {}, "sft_evals": {}, "fetched_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
    for arm, cfg in ARMS.items():
        kind = cfg["before"][0]
        if kind == "uplift":
            before, src = _uplift_init23m()
        elif kind == "sft_eval":
            _, run_name, steps = cfg["before"]
            sft = _run(api, PROJ, run_name)
            rows = _eval_rows(sft)
            final = [r for r in rows if r["ckpt_step"] in steps] or rows[-1:]
            before = _mean_rows(final)
            used = [r["ckpt_step"] for r in final]
            src = f"wandb {PROJ}/{run_name} ckpt_step {used}" + (" (mean of the two rows)" if len(used) > 1 else "") + \
                  (f"; eval cache {sft.config.get('cache')}" if sft.config.get("cache") else "")
            out["sft_evals"][arm] = {"run": run_name, "id": sft.id, "state": sft.state, "rows": rows, "final_steps_used": used, "cache": sft.config.get("cache")}
        else:
            before, src = ({"eval/mean_all": RANDOM_FLOOR, "eval/random/cos": RANDOM_FLOOR},
                           "NOT measured: an untrained LoRA scores the random-direction floor (~0.03 cosine) by construction; per-family values unknown")
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
                            "evals": _eval_rows(ev), "before_rl": before, "before_rl_source": src, "train": train, "config": conf}
    # --- RL-C reference (before-RL row from the uplift report's references.json) ---
    try:
        refinit = json.load(open(UPLIFT_DATA / "references.json"))["sft_mix1m_midtrain_final"]["metrics"]
        refinit = {k: v for k, v in refinit.items() if _num(v) and k.startswith("eval/")}
        refinit_src = f"{UPLIFT_DATA}/references.json  sft_mix1m_midtrain_final (RL-C's init = 23M pretrain + 1.1M midtrain final)"
    except Exception as e:  # noqa: BLE001
        refinit = {"eval/mean_all": 0.3403, "eval/sae/norm_act": 0.4501, "eval/realact/cos": 0.4445, "eval/sae/unverbalized_frac": 0.4473}
        refinit_src = f"hard-coded fallback ({e})"
    refinit["eval/sae/verbalized_frac"] = 1.0 - refinit["eval/sae/unverbalized_frac"]
    rc = _run(api, REF_PROJ, REF["eval"])
    out["ref"] = {"eval_run": REF["eval"], "project": REF_PROJ, "id": rc.id, "state": rc.state, "evals": _eval_rows(rc), "before_rl": refinit,
                  "before_rl_source": refinit_src, "config": {k: v for k, v in rc.config.items() if not isinstance(v, (dict, list))},
                  "note": "RL-C: same lr 7e-6 / ScaleRL recipe but 16x256 rollouts per step, init = 23M pretrain + 1.1M all-families midtrain, bank mix_1m_v2 "
                          "(1.1M, five families, no MLP neurons). Its ckpt 400 row (mean_all 0.315) is a late collapse and is NOT plotted (arms stop at 300)."}
    return out


def load_mlprank():
    """Rank-based MLP metric (re-score of the finals + two SFT inits by scripts/plot_mlp_rank_eval.py). None if that report has not been built."""
    p = MLPRANK_DATA / "summary.json"
    if not p.exists():
        return None
    s = json.load(open(p))
    ck = {n: {"ckpt": c.get("ckpt"), "ckpt_step": c.get("ckpt_step"), "label": c.get("label"),
              "metrics": {k: v for k, v in c["metrics"].items() if k.startswith("eval/mlp") and _num(v)}} for n, c in s["checkpoints"].items()}
    chance = {k: v for k, v in s.get("chance", {}).items() if k in ("eval/mlp/chance_rank1_frac", "eval/mlp/chance_rank_le10", "eval/mlp/chance_mrr")}
    return {"source": str(p), "checkpoints": ck, "chance": chance, "protocol_note": s.get("protocol_note")}


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
        for tag, r in [("best of 16", rs[0]), ("median of 16", rs[len(rs) // 2])]:
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


def matched_numbers(D, S):
    res = {"matched_step": S, "steps": STEPS, "arms": {}, "ref": {}, "reading": {}}
    allsteps = sorted(set(STEPS + [S]))
    for arm, d in D["arms"].items():
        before = d["before_rl"].get("eval/mean_all")
        ls, lv = last_eval(d)
        e = {"label": ARMS[arm]["label"], "short": ARMS[arm]["short"], "before_rl": before, "before_rl_measured": arm != "initbase",
             "last_step": ls, "last_value": lv, "at": {}, "delta": {}}
        for s in allsteps:
            v = at_step(d, s)
            e["at"][str(s)] = v
            e["delta"][str(s)] = None if v is None or before is None else v - before
        res["arms"][arm] = e
    r = D["ref"]
    res["ref"] = {"label": REF["label"], "before_rl": r["before_rl"].get("eval/mean_all"), "at": {str(s): at_step(r, s) for s in allsteps}}

    def diff(a, b, s):
        va, vb = res["arms"][a]["at"][str(s)], res["arms"][b]["at"][str(s)]
        return None if va is None or vb is None else va - vb
    for s in allsteps:
        sft = {a: res["arms"][a]["at"][str(s)] for a in LORA_SFT_ARMS if res["arms"][a]["at"][str(s)] is not None}   # same-recipe LoRA arms only
        base = res["arms"]["initbase"]["at"][str(s)]
        res["reading"][str(s)] = {
            "midtrain_on_top_of_pretrain": diff("initboth", "init23m", s),            # (pretrain + midtrain) - pretrain only
            "pretrain_on_top_of_midtrain": diff("initboth", "initmid", s),            # (pretrain + midtrain) - midtrain only
            "midtrain_only_vs_pretrain_only": diff("initmid", "init23m", s),
            "new_vs_old_pretrain": diff("initnewfft", "initboth", s),                # new full-FT pretrain+midtrain - LoRA pretrain+midtrain
            "fullparam_vs_lora_same_init": diff(*FULLPARAM_PAIR, s),                 # full-parameter RL (lr 1e-6) - LoRA RL (lr 7e-6), identical init
            "sft_spread": (max(sft.values()) - min(sft.values())) if len(sft) >= 2 else None,
            "sft_best": max(sft, key=sft.get) if sft else None, "sft_worst": min(sft, key=sft.get) if sft else None,
            "any_sft_vs_none": (min(sft.values()) - base) if (sft and base is not None) else None,
            "n_sft_arms": len(sft)}
    # LoRA RL vs full-parameter RL on the identical init: per-checkpoint deltas on the mean and every family panel
    fp, lo = FULLPARAM_PAIR
    keys = ["eval/mean_all"] + [k for k, _, _ in FAMILY_PANELS]
    common = sorted({r["ckpt_step"] for r in D["arms"][fp]["evals"]} & {r["ckpt_step"] for r in D["arms"][lo]["evals"]})
    res["lora_vs_fullparam"] = {"full_param_arm": fp, "lora_arm": lo, "keys": keys, "rows": [
        {"ckpt_step": s, **{k: {"full_param": at_step(D["arms"][fp], s, k), "lora": at_step(D["arms"][lo], s, k),
                                "delta": (None if at_step(D["arms"][fp], s, k) is None or at_step(D["arms"][lo], s, k) is None
                                          else at_step(D["arms"][fp], s, k) - at_step(D["arms"][lo], s, k))} for k in keys}} for s in common]}
    return res


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
        ax.plot(xs, ys, ls="--", lw=1.3, color=REF["color"], label=label, zorder=1)


def add_arm(ax, arm, d, key, lw=2.0, ms=6.5, label=None):
    xs, ys = series(d, key)
    cfg = ARMS[arm]
    if not xs:
        return
    ax.plot(xs, ys, color=cfg["color"], lw=lw, ls=cfg.get("ls", "-"), marker=cfg["marker"], ms=ms, mec="white", mew=1.2, label=label, zorder=3,
            solid_capstyle="round")
    if xs[0] == 0 and arm == "initbase":   # before-RL value is the floor by construction, not a measurement -> hollow marker
        ax.plot([0], [ys[0]], marker=cfg["marker"], ms=ms + 1, mfc="white", mec=cfg["color"], mew=1.6, ls="none", zorder=4)


def arm_label(D, arm):
    d = D["arms"][arm]
    return ARMS[arm]["label"] + ("" if d["train_state"] == "finished" else f"  [running, RL step {d['train_last_step']}]")


def claim_headline(M, S):
    rd = M["reading"][str(S)]
    vals = {a: M["arms"][a]["at"][str(S)] for a in LORA_SFT_ARMS if M["arms"][a]["at"][str(S)] is not None}
    if len(vals) < 2:
        return f"Held-out fidelity vs RL step for six starting points / RL modes (matched step {S})"
    hi, lo = rd["sft_best"], rd["sft_worst"]
    if rd["sft_spread"] < 0.02:
        p1 = (f"All {len(vals)} SFT inits given the same LoRA RL land within {rd['sft_spread']:.3f} of each other after {S} steps "
              f"({ARMS[lo]['short']} {vals[lo]:.3f} to {ARMS[hi]['short']} {vals[hi]:.3f})")
    else:
        p1 = f"SFT inits spread {rd['sft_spread']:.3f} after {S} LoRA-RL steps ({ARMS[hi]['short']} {vals[hi]:.3f} best, {ARMS[lo]['short']} {vals[lo]:.3f} worst)"
    fpd = rd.get("fullparam_vs_lora_same_init")
    p2 = f"; full-parameter RL on the new init adds {fpd:+.3f} over LoRA RL" if fpd is not None else ""
    c = M["arms"]["initbase"]["at"][str(S)]
    p2 += f"; skipping SFT entirely reaches only {c:.3f}" if c is not None else ""
    v300 = {a: M["arms"][a]["at"]["300"] for a in SFT_ARMS if M["arms"][a]["at"]["300"] is not None}
    if len(v300) >= 2 and S != 300:
        b = max(v300, key=v300.get)
        p2 += f"; at step 300 {ARMS[b]['short']} leads ({v300[b]:.3f})"
    return p1 + p2


def plot_headline(D, S, M):
    fig, ax = plt.subplots(figsize=(10.5, 6.6))
    add_ref(ax, D["ref"], "eval/mean_all", REF["label"])
    ends = []
    for arm, d in D["arms"].items():
        add_arm(ax, arm, d, "eval/mean_all", label=arm_label(D, arm))
        s, v = last_eval(d)
        if s is not None:
            ends.append((v, s, ARMS[arm]))
    # direct end labels (text in ink, identity from the marker colour); push apart only when two ends sit at nearby x
    ends.sort(key=lambda t: t[0])
    ys = [e[0] for e in ends]
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < 0.015 and abs(ends[i][1] - ends[i - 1][1]) < 40:
            ys[i] = ys[i - 1] + 0.015
    for (v, s, cfg), y in zip(ends, ys):
        ax.annotate(f"{cfg['short']}: {v:.3f}", xy=(s, v), xytext=(s + 7, y), fontsize=8.8, color=INK2, va="center",
                    arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.8) if abs(y - v) > 0.004 else None)
    if _num(D["arms"]["initmid"]["before_rl"].get("eval/mean_all")):
        ax.annotate("before RL: each SFT init\nevaluated on its own", xy=(0, D["arms"]["initmid"]["before_rl"]["eval/mean_all"]), xytext=(12, 0.27),
                    fontsize=8.5, color=MUTED, arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.8))
    ax.annotate("no SFT: random-direction\nfloor by construction (~0.03)", xy=(0, RANDOM_FLOOR), xytext=(12, 0.06), fontsize=8.5, color=MUTED,
                arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.8))
    ax.axvline(S, color=GRID, lw=1.0, zorder=0)
    ax.text(S, 0.005, f"step {S}: last checkpoint\nall five arms have", fontsize=8, color=MUTED, ha="center", va="bottom")
    ax.set_xlim(-6, 400); ax.set_ylim(0, 0.5)
    ax.set_xticks([0, 25, 50, 100, 150, 200, 250, 300])
    ax.set_xlabel("RL step (4,096 rollouts per step; step 0 = the SFT init before any RL)")
    ax.set_ylabel("held-out fidelity: mean cosine over 10 direction families")
    ax.set_title(claim_headline(M, S) + "\nSame RL recipe (CISPO GRPO, 8 x 512 rollouts/step, lr 7e-6, 300 steps) on the same equal six-family bank; "
                 "Qwen3.6-27B activation-to-text inverter, 512 held-out directions per family, best-of-4", fontsize=10.5, loc="left", pad=10)
    style_ax(ax)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8.6, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    save(fig, "sft_init_mean_fidelity")


def plot_families(D, S):
    fig, axes = plt.subplots(2, 4, figsize=(17.5, 8), sharex=True)
    for ax, (key, title, _) in zip(axes.flat, FAMILY_PANELS):
        for arm, d in D["arms"].items():
            add_arm(ax, arm, d, key, lw=1.8, ms=5.5, label=ARMS[arm]["label"])
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xticks([0, 50, 100, 150, 200, 250, 300]); ax.set_xlim(-6, 312)
        ax.set_ylim(0, 0.08 if key == "eval/random/cos" else None)
        style_ax(ax)
    for ax in axes[-1]:
        ax.set_xlabel("RL step (0 = before RL)", fontsize=9)
    # which family separates the SFT inits most / least at the matched step (computed, not asserted)
    gaps = []
    for key, title, _ in FAMILY_PANELS:
        if key == "eval/random/cos":
            continue
        vals = {a: at_step(D["arms"][a], S, key) for a in LORA_SFT_ARMS}
        vals = {a: v for a, v in vals.items() if v is not None}
        if len(vals) >= 2:
            hi, lo = max(vals, key=vals.get), min(vals, key=vals.get)
            gaps.append((vals[hi] - vals[lo], title.split(":")[0].split(" (")[0], hi, vals[hi], lo, vals[lo]))
    gaps.sort()
    top, bot = gaps[-1], gaps[0]
    fp, lo_ = FULLPARAM_PAIR
    fp_sae, lo_sae = at_step(D["arms"][fp], S, "eval/sae/norm_act"), at_step(D["arms"][lo_], S, "eval/sae/norm_act")
    fp_txt = (f"; full-parameter RL on the new init tracks its LoRA-RL twin family by family (SAE {fmt(fp_sae, 2)} vs {fmt(lo_sae, 2)})"
              if fp_sae is not None and lo_sae is not None else "")
    fig.suptitle(f"The LoRA-RL SFT inits differ most on {top[1]} (spread {top[0]:.2f} at step {S}: {ARMS[top[2]]['short']} {top[3]:.2f} vs {ARMS[top[4]]['short']} {top[5]:.2f}) "
                 f"and least on {bot[1]} (spread {bot[0]:.3f}){fp_txt}; the no-SFT policy stays near the floor in every family\n"
                 "Per-family held-out fidelity vs RL step - same CISPO GRPO recipe (8 x 512 rollouts/step) on the same equal six-family bank; LoRA arms lr 7e-6, "
                 "full-parameter arm lr 1e-6 (dashed); Qwen3.6-27B activation-to-text inverter; hollow marker = floor by construction, not measured",
                 fontsize=10.5, x=0.01, ha="left", y=0.995)
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    save(fig, "sft_init_per_family")


def _tail_mean(d, key, n=50):
    ys = [v for v in d["train"][key][-n:] if _num(v)]
    return float(np.mean(ys)) if ys else None


def plot_train(D):
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.2), sharex=True)
    W = 9
    for ax, (key, title) in zip(axes.flat, TRAIN_PANELS):
        for arm, d in D["arms"].items():
            x = np.array(d["train"]["step"]); y = d["train"][key]
            if not any(_num(v) for v in y):
                continue
            yy = np.array([np.nan if v is None else v for v in y], dtype=float)
            ax.plot(x, yy, color=ARMS[arm]["color"], lw=0.8, alpha=0.25, zorder=2)
            ax.plot(x, rolling(y, W), color=ARMS[arm]["color"], lw=2.0, label=arm_label(D, arm), zorder=3, solid_capstyle="round")
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlim(0, 305); ax.set_xticks([0, 50, 100, 150, 200, 250, 300])
        if key in LOG_PANELS:
            ax.set_yscale("log")
        elif key in ("ratio/clipfrac", "reward/mean"):
            ax.set_ylim(0, None)
        style_ax(ax)
    for ax in axes[-1]:
        ax.set_xlabel("RL step", fontsize=9)
    b, n, f = D["arms"]["initbase"], D["arms"]["initnewfft"], D["arms"]["initnewfft_fullparam"]
    lora_clip = np.mean([_tail_mean(D["arms"][k], "ratio/clipfrac") or 0 for k in ("init23m", "initmid", "initboth")])
    lora_dlp = np.mean([_tail_mean(D["arms"][k], "policy/sampler_abs_dlogp") or 0 for k in ("init23m", "initmid", "initboth")])
    lora_ent = np.mean([_tail_mean(D["arms"][k], "policy/entropy") or 0 for k in ("init23m", "initmid", "initboth")])
    fig.suptitle(f"Same recipe, different optimisation: no SFT collapses to ~{fmt(_tail_mean(b, 'rollout/len_mean'), 0)}-token, entropy {fmt(_tail_mean(b, 'policy/entropy'), 2)} "
                 f"outputs with {100 * (_tail_mean(b, 'ratio/clipfrac') or 0):.0f}% clipped tokens; the LoRA-continuation inits train smoothly "
                 f"(entropy {lora_ent:.2f}, {100 * lora_clip:.1f}% clipped, |delta log p| {lora_dlp:.3f}); the fresh LoRA on full-FT weights runs hotter "
                 f"({100 * (_tail_mean(n, 'ratio/clipfrac') or 0):.1f}% clipped, |delta log p| {fmt(_tail_mean(n, 'policy/sampler_abs_dlogp'))});\n"
                 f"full-parameter RL on that init loses entropy fastest ({fmt(_tail_mean(f, 'policy/entropy'), 2)}) with the largest sampler-trainer gap "
                 f"(|delta log p| {fmt(_tail_mean(f, 'policy/sampler_abs_dlogp'))}, {100 * (_tail_mean(f, 'ratio/clipfrac') or 0):.1f}% clipped) - last-50-step means.  "
                 "RL training dynamics vs step, six arms - CISPO GRPO, 8 x 512 rollouts/step, 25 warmup, no KL; LoRA arms lr 7e-6, full-parameter arm lr 1e-6; "
                 f"thin = per-step, thick = {W}-step moving average", fontsize=10.2, x=0.01, ha="left", y=0.995)
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8.6, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.07, 1, 0.91))
    save(fig, "sft_init_train_dynamics")


def plot_bars(D, S, M):
    fig, axes = plt.subplots(1, 2, figsize=(17.5, 6), gridspec_kw={"width_ratios": [4, 3]})
    arms = list(ARMS)
    n = len(arms)
    w = 0.125
    off = lambda i: (i - (n - 1) / 2) * (w + 0.01)  # noqa: E731
    cats = [("before RL", "before_rl")] + [(f"RL step {s}", str(s)) for s in STEPS]
    ax = axes[0]
    for i, arm in enumerate(arms):
        e = M["arms"][arm]
        for j, (_, k) in enumerate(cats):
            v = e["before_rl"] if k == "before_rl" else e["at"][k]
            x = j + off(i)
            if v is None:
                ax.text(x, 0.008, "pending", rotation=90, fontsize=7.5, color=MUTED, ha="center", va="bottom")
                continue
            hollow = (arm == "initbase" and k == "before_rl")
            ax.bar(x, v, width=w, color="white" if hollow else ARMS[arm]["color"], edgecolor=ARMS[arm]["color"] if hollow else "none", lw=1.5,
                   label=ARMS[arm]["label"] if (j == 1) else None, zorder=3)
            ax.text(x, v + 0.006, f"{v:.3f}", ha="center", va="bottom", fontsize=7.2, color=INK2, rotation=90)
    rv = M["ref"]["at"].get("300")
    if rv is not None:
        ax.axhline(rv, color=REF["color"], ls="--", lw=1.2, zorder=1)
        ax.text(len(cats) - 0.55, rv + 0.004, f"RL-C reference @ 300: {rv:.3f}", fontsize=8, color=MUTED, ha="right", va="bottom")
    ax.set_xticks(np.arange(len(cats))); ax.set_xticklabels([c for c, _ in cats])
    ax.set_ylim(0, 0.53); ax.set_ylabel("held-out fidelity (mean cosine, 10 families)")
    ax.set_title("absolute level (hollow bar = floor by construction, not measured)", fontsize=10, loc="left")
    style_ax(ax)
    ax = axes[1]
    dcats = [(f"gain by step {s}", str(s)) for s in STEPS]
    for i, arm in enumerate(arms):
        for j, (_, k) in enumerate(dcats):
            v = M["arms"][arm]["delta"][k]
            x = j + off(i)
            if v is None:
                ax.text(x, 0.004, "pending", rotation=90, fontsize=7.5, color=MUTED, ha="center", va="bottom")
                continue
            ax.bar(x, v, width=w, color=ARMS[arm]["color"], zorder=3)
            ax.text(x, v + 0.004, f"{v:+.3f}", ha="center", va="bottom", fontsize=7.2, color=INK2, rotation=90)
    ax.set_xticks(np.arange(len(dcats))); ax.set_xticklabels([c for c, _ in dcats])
    ax.set_ylim(0, 0.27); ax.set_ylabel("fidelity gained from RL (after - before)")
    ax.set_title("what RL added on top of each init (no SFT: from the ~0.03 floor)", fontsize=10, loc="left")
    style_ax(ax)
    rd = M["reading"][str(S)]
    fig.suptitle(f"At step {S}: adding the midtrain to the 23M pretrain changes fidelity by {signed(rd['midtrain_on_top_of_pretrain'])}, adding the pretrain to the "
                 f"midtrain by {signed(rd['pretrain_on_top_of_midtrain'])}, the new full-FT pretrain vs the LoRA pretrain by {signed(rd['new_vs_old_pretrain'])}, "
                 f"full-parameter RL vs LoRA RL on the new init by {signed(rd.get('fullparam_vs_lora_same_init'))}; any SFT beats none by at least {signed(rd['any_sft_vs_none'])}\n"
                 "Held-out mean fidelity before RL and after 100 / 200 / 300 RL steps, six arms (five SFT starting points under the same LoRA RL recipe + one full-parameter RL), "
                 "same CISPO GRPO recipe and bank (Qwen3.6-27B activation-to-text inverter)", fontsize=10.2, x=0.01, ha="left", y=1.0)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8.6, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 0.9))
    save(fig, "sft_init_matched_step_bars")


def plot_transcripts(TX, D):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5), sharex=True)
    panels = [("cos_mean", "mean cosine of the 16 sampled rollouts"), ("n_tok_mean", "mean response length (tokens)"), ("degenerate_share", "degenerate share of sampled rollouts")]
    for ax, (key, title) in zip(axes, panels):
        for arm in ARMS:
            st = TX["stats"].get(arm) or []
            if not st:
                continue
            x = [s["step"] for s in st]; y = [s[key] for s in st]
            ax.plot(x, y, color=ARMS[arm]["color"], lw=0.9, alpha=0.28, zorder=2)
            ax.plot(x, rolling(y, 3), color=ARMS[arm]["color"], lw=2.0, label=arm_label(D, arm), zorder=3, solid_capstyle="round")
        ax.set_title(title, fontsize=10, loc="left"); ax.set_xlim(0, 305); ax.set_xticks([0, 50, 100, 150, 200, 250, 300]); ax.set_xlabel("RL step", fontsize=9)
        ax.set_ylim(0, 1.02 if key == "degenerate_share" else None)
        style_ax(ax)
    tails = {arm: (np.mean([s["n_tok_mean"] for s in st[-10:]]) if st else None, np.mean([s["degenerate_share"] for s in st[-10:]]) if st else None)
             for arm, st in TX["stats"].items()}
    tb, ta = tails.get("initbase", (None, None)), tails.get("init23m", (None, None))
    sft_len = [tails[a][0] for a in SFT_ARMS if tails.get(a) and tails[a][0] is not None]
    fig.suptitle(f"The no-SFT arm's rollouts shrink to ~{fmt(tb[0], 0)} tokens and {100 * (tb[1] or 0):.0f}% are degenerate (repetition or at the 8-token floor); "
                 f"the SFT arms hold {min(sft_len):.0f}-{max(sft_len):.0f} tokens with {100 * (ta[1] or 0):.0f}% degenerate for the 23M pretrain (last 10 logged steps)\n"
                 f"Per-step statistics of the logged rollout transcripts (16 rollouts = 4 groups x 4 samples every 5 RL steps); degenerate = <= {DEGENERATE_MAX_TOK} tokens "
                 f"or word type/token ratio < {DEGENERATE_TTR}; thick line = 3-point moving average", fontsize=10.5, x=0.01, ha="left", y=1.0)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8.6, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(rect=(0, 0.02, 1, 0.88))
    save(fig, "sft_init_transcript_stats")


# ---------------------------------------------------------------- data files ----------------------------------------------------------------
def write_data(D, TX, S, M, MR):
    dd = OUT / "data"; dd.mkdir(parents=True, exist_ok=True)
    meta = {"generated_at": D["fetched_at"], "matched_step": S, "steps": STEPS, "wandb_project": PROJ, "ref_project": REF_PROJ, "random_floor": RANDOM_FLOOR,
            "arm_order": list(ARMS), "sft_arms": SFT_ARMS, "lora_sft_arms": LORA_SFT_ARMS, "fullparam_pair": list(FULLPARAM_PAIR),
            "arms": {a: {k: d[k] for k in ("eval_run", "eval_id", "eval_state", "train_run", "train_id", "train_state", "train_last_step", "before_rl_source")}
                     | {k: ARMS[a][k] for k in ("label", "short", "init_adapter", "policy_base", "ckpt_dir", "init_desc", "pretrain", "midtrain", "color")}
                     | {"full_param": bool(ARMS[a].get("full_param")), "rl_lr": d["config"].get("lr"),
                        "eval_steps": [r["ckpt_step"] for r in d["evals"]], "sft_eval": D["sft_evals"].get(a, {}).get("run"),
                        "sft_eval_cache": D["sft_evals"].get(a, {}).get("cache"), "sft_final_steps_used": D["sft_evals"].get(a, {}).get("final_steps_used"),
                        "mlprank_names": ARMS[a]["mlprank"]} for a, d in D["arms"].items()},
            "ref": {k: D["ref"][k] for k in ("eval_run", "project", "id", "state", "before_rl_source", "note")} | {"label": REF["label"]},
            "mlprank_source": MR["source"] if MR else None,
            "family_panels": FAMILY_PANELS, "train_panels": TRAIN_PANELS, "degenerate_rule": {"max_tok": DEGENERATE_MAX_TOK, "ttr_lt": DEGENERATE_TTR}}
    json.dump(meta, open(dd / "meta.json", "w"), indent=1)
    json.dump({"arms": {a: {"label": ARMS[a]["label"], "before_rl": d["before_rl"], "before_rl_source": d["before_rl_source"], "evals": d["evals"]}
                        for a, d in D["arms"].items()},
               "ref": {"label": REF["label"], "before_rl": D["ref"]["before_rl"], "evals": D["ref"]["evals"], "note": D["ref"]["note"]},
               "sft_trajectories": {a: {"run": v["run"], "state": v["state"], "cache": v["cache"], "final_steps_used": v["final_steps_used"], "rows": v["rows"]}
                                    for a, v in D["sft_evals"].items()}}, open(dd / "eval_curves.json", "w"), indent=1)
    train = {}
    for a, d in D["arms"].items():
        t = dict(d["train"])
        for k, _ in TRAIN_PANELS:
            t[k + "__ma9"] = [None if np.isnan(v) else float(v) for v in rolling(d["train"][k], 9)]
        train[a] = {"label": ARMS[a]["label"], "run": d["train_run"], "id": d["train_id"], "state": d["train_state"], "series": t,
                    "tail50": {k: _tail_mean(d, k) for k, _ in TRAIN_PANELS}}
    json.dump(train, open(dd / "train_dynamics.json", "w"), indent=1)
    json.dump({k: v for k, v in M.items() if k != "lora_vs_fullparam"}, open(dd / "matched_step.json", "w"), indent=1)
    json.dump(M["lora_vs_fullparam"], open(dd / "lora_vs_fullparam.json", "w"), indent=1)
    json.dump({"rule": meta["degenerate_rule"], "stats": TX["stats"], "examples": TX["examples"]}, open(dd / "transcripts.json", "w"), indent=1)
    # MLP rank re-score joined per arm (before-RL init where re-scored, and the step-300 final)
    mlprank = {"source": MR["source"] if MR else None, "chance": MR["chance"] if MR else {}, "protocol_note": MR["protocol_note"] if MR else None, "arms": {}}
    for a in ARMS:
        sft_name, rl_name = ARMS[a]["mlprank"]
        ent = {}
        for tag, name in (("before_rl", sft_name), ("final_300", rl_name)):
            c = MR["checkpoints"].get(name) if (MR and name) else None
            ent[tag] = None if c is None else {"rescore_name": name, "ckpt": c["ckpt"], **{k: c["metrics"].get(k) for k, _ in RANK_COLS}, "eval/mlp/mrr": c["metrics"].get("eval/mlp/mrr")}
        mlprank["arms"][a] = ent
    json.dump(mlprank, open(dd / "mlp_rank.json", "w"), indent=1)
    # checkpoint x family table (before-RL row first), coloured in build_html relative to the 23M init's before-RL row
    tbl = {"columns": TABLE_COLS, "rank_columns": RANK_COLS, "reference_row": D["arms"]["init23m"]["before_rl"], "arms": {}}
    for a, d in D["arms"].items():
        rows = [{"ckpt_step": 0, "kind": "before RL", **{k: d["before_rl"].get(k) for k, _ in TABLE_COLS}}]
        rows += [{"ckpt_step": r["ckpt_step"], "kind": "RL", **{k: r.get(k) for k, _ in TABLE_COLS}} for r in d["evals"]]
        for row in rows:
            src = mlprank["arms"][a]["before_rl"] if row["kind"] == "before RL" else (mlprank["arms"][a]["final_300"] if row["ckpt_step"] == 300 else None)
            for k, _ in RANK_COLS:
                row[k] = src.get(k) if src else None
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
    MR = load_mlprank()
    print("mlp-rank re-score:", "none" if MR is None else sorted(MR["checkpoints"]))
    tx_rows = fetch_transcripts(list(ARMS), fetch=not args.no_fetch_transcripts)
    TX = {"stats": {a: transcript_stats(r) for a, r in tx_rows.items()}, "examples": {}}
    for a, r in tx_rows.items():
        steps = sorted({s for s in (100, max((x["step"] for x in r), default=None)) if s is not None})
        TX["examples"][a] = transcript_examples(r, steps)
    M = matched_numbers(D, S)
    write_data(D, TX, S, M, MR)
    plot_headline(D, S, M); plot_families(D, S); plot_train(D); plot_bars(D, S, M)
    if any(TX["stats"].values()):
        plot_transcripts(TX, D)
    print(json.dumps({a: {"before": e["before_rl"], "at": e["at"], "last": (e["last_step"], e["last_value"])} for a, e in M["arms"].items()}, indent=1))
    print("reading:", json.dumps(M["reading"], indent=1))
    bh = OUT / "build_html.py"
    if not args.no_html and bh.exists():
        subprocess.run(["python3", str(bh)], check=True)


if __name__ == "__main__":
    main()
