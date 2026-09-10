#!/usr/bin/env python3
"""Re-cut midtrain bank -> full-fine-tune midtrain -> full-parameter RL (Qwen3.6-27B activation->text inverter).

The chain this script reports on (all wandb project celestedeschamphelaere-personal/maxact-fast):
  pretrain   realact104m_fullft_b4096_lr1e-5 (jp82mr9a)           104M real-activation examples, full FT, held-out mean_all .369
  bank       /data/banks/mix_5m_sft (2,747,509 rows)               re-cut after four corrections: no realact_long; every target ENDS at the
                                                                    token where its direction fires (verified standalone); BSF re-mined from
                                                                    verbatim contexts; MLP windows kept only if they fire in isolation
  midtrain   mix5msft_midtrain_fft_from_fft104m (eaozkq7o)         FSDP2 full FT from the pretrain final, eff batch 4,096, lr 1e-5, 671 steps
  RL         rl_fullparam_fft104m_mix5m_8x512 (4piq4y7x)           full-parameter CISPO GRPO, 8 x 512 rollouts/step, lr 1e-6, 300 steps
  reference  rl_abl_initnewfft_fullparam_8x512 (y4teoshq)          the SAME RL recipe on the OLD-bank init (23M-ckpt pretrain + 1.45M FFT
                                                                    midtrain; targets not end-anchored, realact_long included): .424 @300

Idempotent: every run re-pulls the eval + training histories from wandb (scan_history AND sampled history unioned, deduped by _step), reads
the LoRA init-ablation arms from ~/shared/reports/maemm-sft-init-ablation/data/eval_curves.json and the pretrain numbers from
~/shared/reports/maemm-sft-fullft-104m/data/fidelity_vs_examples.json, rewrites data/*.json + every figure as PNG + PDF, then (unless
--no-html) runs the report folder's build_html.py. Re-run after the step-250/300 evals land and the headline flips on its own.

    python scripts/plot_fft104m_mix5m_fullrl.py            # refresh everything
    python scripts/plot_fft104m_mix5m_fullrl.py --no-html  # figures + data only

Outputs -> ~/shared/reports/maemm-fft104m-mix5m-fullrl/{data/*.json, fidelity_vs_rl_step, per_family_vs_rl_step, rl_dynamics, bank_recut}.{png,pdf}
"""
import argparse
import datetime as dt
import json
import math
import os
import statistics
import subprocess
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb

OUT = Path(os.environ.get("MAEMM_FULLRL_REPORT", "~/shared/reports/maemm-fft104m-mix5m-fullrl")).expanduser()
PROJ = "celestedeschamphelaere-personal/maxact-fast"
ABL_DATA = Path("~/shared/reports/maemm-sft-init-ablation/data").expanduser()     # six-arm init ablation (LoRA arms + the reference full-param arm)
PRE_DATA = Path("~/shared/reports/maemm-sft-fullft-104m/data").expanduser()       # 104M full-FT pretrain report (init of this chain, eval-noise pairs)

RUNS = {
    "this_train": ("id", "4piq4y7x", "rl_fullparam_fft104m_mix5m_8x512"),
    "this_eval": ("name", "rl_fullparam_fft104m_mix5m_8x512_eval", None),
    "ref_train": ("name", "rl_abl_initnewfft_fullparam_8x512", None),
    "ref_eval": ("name", "rl_abl_initnewfft_fullparam_8x512_eval", None),
    "mid_train": ("id", "eaozkq7o", "mix5msft_midtrain_fft_from_fft104m"),
    "mid_eval": ("name", "mix5msft_midtrain_fft_from_fft104m_eval", None),
    "pre_train": ("id", "jp82mr9a", "realact104m_fullft_b4096_lr1e-5"),
}
SAVE_STEPS = [25, 50, 100, 150, 200, 250, 300]
REF_STEP = 300                     # the reference number the headline is judged against is the reference arm's step-300 eval (.424)
NOISE_EPS = 0.002                  # eval-noise floor: duplicate-checkpoint deltas from the pretrain report (max |delta| .0023)
TABLE_EPS = 0.004                  # this-vs-reference table colouring
RL_LR = 1e-6
LORA_WALL_LR_STEPS = 2.4e-3        # LoRA arms' drift-wall onset in cumulative lr x steps (report maemm-rl-lr-level)

# Bank re-cut numbers (build_stats + the standalone-check JSONs on the Modal volume; transcribed from the run log — not on wandb).
BANK = {
    "name": "/data/banks/mix_5m_sft", "n_rows": 2_747_509, "leak_drops": 0, "seed": 2030,
    "families": {"realact": 1_000_000, "sae": 396_506, "sae_dec": 396_506, "cluster": 204_366, "bsf": 225_573, "mlp": 460_842, "mlp_pair": 57_946, "mlp_triple": 5_770},
    "family_labels": {"realact": "real activations (verbatim text, 1M slice)", "sae": "SAE features: max-act windows", "sae_dec": "SAE features: decoder-direction targets",
                      "cluster": "linear cluster probes (trimmed to peak)", "bsf": "BSF subspace blocks (verbatim re-mine)", "mlp": "MLP neurons: single",
                      "mlp_pair": "MLP neurons: co-firing pairs", "mlp_triple": "MLP neurons: co-firing triples"},
    "sae": {"rule": "peak == last token (rule=last; the earlier rule allowed the last 2 tokens)", "kept": 396_506, "total": 468_895,
            "peak_offset_hist": {"0": 396_506, "1": 13_955, "2": 6_638, ">=3": 468_895 - 396_506 - 13_955 - 6_638},
            "per_feature": "top-16 standalone 32-token max-act windows per feature -> 4 distinct windows kept per feature (mean 3.98) before the rule, ~3.4 after",
            "applies_to": ["sae", "sae_dec"], "source": "/data/banks/everything_5m_fresh/end_anchor_rows.json (rule=last regeneration, 1104 s)"},
    "cluster": {"rule": "trim the target to peak_idx+1 (causal trick: the activation at the peak token does not depend on later tokens), drop if < 8 tokens",
                "total": 241_741, "already_last": 18_868, "trimmed": 185_498, "mean_tokens_removed": 2.1, "dropped_lt8": 37_375, "kept": 204_366,
                "sample_check_n20k": {"already_last": 0.074, "trimmed": 0.771, "too_short": 0.154, "roundtrip_fail": 0.00015, "keep": 0.846, "keep_if_floor4": 0.97}},
    "mlp": {"rule": "fire_and_dirpeak_last: every member neuron fires at the last token AND the direction's argmax over the window is the last token, window forwarded alone",
            "totals": {"mlp": 810_548, "mlp_pair": 156_966, "mlp_triple": 37_446}, "kept": {"mlp": 460_842, "mlp_pair": 57_946, "mlp_triple": 5_770},
            "sample_check_2048": {"mlp": {"fire_last": 0.70, "neuron_peak_last": 0.69, "dir_peak_last": 0.60, "both": 0.57},
                                  "mlp_pair": {"fire_last": 0.41, "dir_peak_last": 0.61, "both": 0.42}, "mlp_triple": {"fire_last": 0.18, "dir_peak_last": 0.54, "both": 0.26},
                                  "note": "2048 rows/family, app maemm-mlp42-anchor-check; cosine of the window's residual with the neuron direction is tiny (median .05)"},
            "how_found": "scan 160k x 256-token fresh windows, per-neuron top-64 firing tokens, window of 16-32 tokens ending at the firing token; pairs/triples at joint-firing tokens",
            "source": "/data/banks/mlp42_5m_fresh/mlp_anchor_rows.json (8 shards, 8.5 min) union eval_cos_rows.json (+1,209 rows)"},
    "bsf": {"rule": "each 8-64-token window forwarded ALONE (with BOS); block direction from ITS last-token layer-42 residual; keep if the direction's peak is the last token",
            "kept": 225_573, "dir_peak_last_before_rule": 0.839, "dir_peak_last_after_rule": 1.0, "independent_remeasure": 0.997, "mean_tokens": 19.7,
            "bank": "/data/banks/bsf_verbatim_250k (8 x H200, 13.6 min; n_candidates 400K, cap_per_block 10)",
            "replaces": "windows cut from 512-token contexts whose direction came from the full context (context-dependent = mismatch with the standalone target)"},
    "realact_long": "RL-only family: removed from the midtrain bank on purpose (it is the family RL is supposed to generalise to)",
    "compose": "app maemm-mix-5m-bank3, dense streaming (dense_frac 0) after v1 stalled on volume random reads; 9.3 min; trim_to_peak cluster floor 8",
}
FAM_SHORT = {"realact": "real activations", "sae": "SAE features", "sae_dec": "SAE decoder dirs", "cluster": "cluster probes", "bsf": "BSF blocks",
             "mlp": "MLP single", "mlp_pair": "MLP pair", "mlp_triple": "MLP triple"}

# ---- chart chrome. Colour follows the entity: this arm = blue, the reference full-param arm = orange (validated pair, CVD dE 9.8),
#      context (LoRA arms / thresholds) = grey de-emphasis; text never wears a series colour.
THIS_C, REF_C, LORA_C = "#2a78d6", "#eb6834", "#b3b1a8"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.titlecolor": INK, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "grid.color": GRID, "grid.linewidth": 0.8, "axes.grid": True, "axes.axisbelow": True, "legend.frameon": False,
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})

FAM_PLAIN = {"eval/sae/norm_act": "SAE features", "eval/mlp/norm_act": "MLP neurons", "eval/realact/cos": "short-context real activations",
             "eval/realact_long/cos": "long-context real activations", "eval/bsf/cos": "BSF blocks", "eval/cluster/cos": "cluster probes", "eval/jlens/cos": "J-lens", "eval/mean_all": "mean"}
FAMILY_PANELS = [("eval/sae/norm_act", "SAE features: normalised peak activation"), ("eval/mlp/norm_act", "MLP neurons: fire-back (normalised activation)"),
                 ("eval/realact/cos", "real activations, short context (cosine)"), ("eval/realact_long/cos", "real activations, long context (cosine) — RL-only family"),
                 ("eval/bsf/cos", "BSF subspace blocks (cosine)"), ("eval/cluster/cos", "linear cluster probes (cosine)")]
TABLE_KEYS = [("eval/mean_all", "mean fidelity"), ("eval/sae/norm_act", "SAE"), ("eval/mlp/norm_act", "MLP"), ("eval/realact/cos", "real acts"),
              ("eval/realact_long/cos", "long-ctx acts"), ("eval/bsf/cos", "BSF"), ("eval/cluster/cos", "probes"), ("eval/jlens/cos", "J-lens"), ("eval/random/cos", "random")]
DYN_KEYS = ["reward/mean", "policy/entropy", "grad_norm", "policy/sampler_abs_dlogp", "rollout/len_mean", "time/step_s", "reward/peak_last_frac",
            "reward/peak_dist_mean", "reward/peak_in_last5_frac", "lr", "ratio/clipfrac", "reward/max", "policy/offpolicy_lag_steps", "mem/hf_peak_gb"]
MA = 9


# ------------------------------------------------------------------ fetch ------------------------------------------------------------------
def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and not (isinstance(x, float) and math.isnan(x))


def _get_run(api, key):
    kind, val, _ = RUNS[key]
    if kind == "id":
        return api.run(f"{PROJ}/{val}")
    rs = list(api.runs(PROJ, filters={"display_name": val}, order="-created_at"))
    if not rs:
        raise SystemExit(f"wandb run not found: {PROJ}/{val}")
    return rs[0]


def eval_rows(run):
    """Latest row per ckpt_step (re-evals overwrite); eval/* + extra/* keys only."""
    rows = {}
    for r in run.scan_history():
        if _num(r.get("ckpt_step")) and _num(r.get("eval/mean_all")):
            row = {k: v for k, v in r.items() if (k.startswith("eval/") or k.startswith("extra/")) and _num(v)}
            row["ckpt_step"] = int(r["ckpt_step"])
            rows[row["ckpt_step"]] = row
    return [rows[k] for k in sorted(rows)]


def train_series(run, keys):
    """scan_history AND sampled history unioned (scan_history sometimes drops ranges), deduped by _step; later non-null wins."""
    by = {}
    srcs = [("scan", run.scan_history()), ("sampled", run.history(samples=20000, pandas=False))]
    counts = {}
    for name, rows in srcs:
        n = 0
        for r in rows:
            s = r.get("_step")
            if not _num(s):
                continue
            n += 1
            d = by.setdefault(int(s), {})
            for k in keys:
                if _num(r.get(k)):
                    d[k] = float(r[k])
            if _num(r.get("_timestamp")):
                d["_timestamp"] = float(r["_timestamp"])
        counts[name] = n
    steps = sorted(by)
    out = {"step": steps, "rows_scan": counts.get("scan"), "rows_sampled": counts.get("sampled"), "n_steps": len(steps),
           "missing_steps": [s for s in range(steps[0], steps[-1] + 1) if s not in by] if steps else []}
    for k in keys + ["_timestamp"]:
        out[k] = [by[s].get(k) for s in steps]
    return out


def fetch():
    api = wandb.Api(timeout=180)
    D = {"fetched_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "runs": {}}
    R = {k: _get_run(api, k) for k in RUNS}
    for k, r in R.items():
        D["runs"][k] = {"id": r.id, "name": r.name, "state": r.state, "last_step": r.lastHistoryStep, "created_at": str(r.created_at), "url": r.url}
    D["this"] = {"evals": eval_rows(R["this_eval"]), "train": train_series(R["this_train"], DYN_KEYS),
                 "config": {k: v for k, v in R["this_train"].config.items() if not isinstance(v, (dict, list))}}
    D["ref"] = {"evals": eval_rows(R["ref_eval"]), "train": train_series(R["ref_train"], DYN_KEYS),
                "config": {k: v for k, v in R["ref_train"].config.items() if not isinstance(v, (dict, list))}}
    D["mid"] = {"evals": eval_rows(R["mid_eval"]), "train": train_series(R["mid_train"], ["loss", "lr", "ex_per_s", "peak_mem_gb", "tok_per_s"]),
                "config": {k: v for k, v in R["mid_train"].config.items() if not isinstance(v, (dict, list))}}
    return D


def load_ablation():
    """LoRA init-ablation arms (thin context lines) + the reference arm's before-RL row (its init = old-bank FFT midtrain final)."""
    p = ABL_DATA / "eval_curves.json"
    if not p.exists():
        return {"available": False, "source": str(p), "arms": {}, "ref_before_rl": None, "ref_before_rl_source": None}
    ec = json.load(open(p))
    arms = {}
    for a, v in ec["arms"].items():
        if a in ("initbase", "initnewfft_fullparam"):     # no-SFT arm sits at .21 (off this chart's story); the full-param arm is fetched live from wandb
            continue
        arms[a] = {"label": v["label"], "before_rl": v["before_rl"].get("eval/mean_all"),
                   "evals": [{"ckpt_step": r["ckpt_step"], "eval/mean_all": r["eval/mean_all"]} for r in v["evals"] if r["ckpt_step"] <= 300]}
    ref = ec["arms"].get("initnewfft_fullparam", {})
    return {"available": True, "source": str(p), "arms": arms, "ref_before_rl": ref.get("before_rl"), "ref_before_rl_source": ref.get("before_rl_source")}


def load_pretrain():
    p = PRE_DATA / "fidelity_vs_examples.json"
    if not p.exists():
        return {"available": False, "source": str(p)}
    d = json.load(open(p))
    pts = d["series"]["fft104m"]["points"]
    final = pts[-1]
    pairs = [{"steps": [a["ckpt_step"], b["ckpt_step"]], "mean_all": [a["mean_all"], b["mean_all"]], "abs_delta": abs(b["mean_all"] - a["mean_all"])}
             for a, b in zip(pts, pts[1:]) if b["ckpt_step"] - a["ckpt_step"] <= 10]
    return {"available": True, "source": str(p), "final": final, "best": d["band_check"]["best"], "lora_band": {k: d["lora_band"][k] for k in ("min", "max", "mid", "sd")},
            "duplicate_ckpt_pairs": pairs, "max_dup_delta": max([q["abs_delta"] for q in pairs], default=None), "n_ckpts": len(pts)}


# --------------------------------------------------------------- derived numbers ---------------------------------------------------------------
def rolling(y, w=MA):
    y = np.array([np.nan if v is None else v for v in y], dtype=float)
    ok = ~np.isnan(y)
    num = np.convolve(np.where(ok, y, 0.0), np.ones(w), "same"); den = np.convolve(ok.astype(float), np.ones(w), "same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def at(rows, step, key="eval/mean_all"):
    for r in rows:
        if r["ckpt_step"] == step and _num(r.get(key)):
            return r[key]
    return None


def win_mean(tr, key, a, b):
    ys = [v for s, v in zip(tr["step"], tr[key]) if a <= s <= b and _num(v)]
    return float(np.mean(ys)) if ys else None


def onsets(tr):
    S, g, dl = tr["step"], tr["grad_norm"], tr["policy/sampler_abs_dlogp"]
    o = {"rule_gnorm": "first of 2 consecutive steps with grad_norm > 1.0, step >= 100", "rule_dlogp": "first step with sampler |delta log p| > 0.05, step >= 100",
         "gnorm_gt1_2consec": None, "gnorm_gt1_for_good": None, "dlogp_gt05_first": None, "dlogp_gt05_3consec": None, "dlogp_gt05_for_good": None}
    for i in range(len(S) - 1):
        if S[i] >= 100 and _num(g[i]) and _num(g[i + 1]) and g[i] > 1 and g[i + 1] > 1:
            o["gnorm_gt1_2consec"] = S[i]; break
    le1 = [s for s, v in zip(S, g) if _num(v) and v <= 1]
    if le1 and le1[-1] < S[-1]:
        o["gnorm_gt1_for_good"] = S[S.index(le1[-1]) + 1]
    o["gnorm_le1_steps_after_100"] = [s for s in le1 if s >= 100]
    hi = [s for s, v in zip(S, dl) if s >= 100 and _num(v) and v > 0.05]
    o["dlogp_gt05_first"] = hi[0] if hi else None
    for i in range(len(S) - 2):
        if S[i] >= 100 and all(_num(dl[j]) and dl[j] > 0.05 for j in (i, i + 1, i + 2)):
            o["dlogp_gt05_3consec"] = S[i]; break
    lo = [s for s, v in zip(S, dl) if _num(v) and v <= 0.05]
    if lo and lo[-1] < S[-1] and hi:
        o["dlogp_gt05_for_good"] = S[S.index(lo[-1]) + 1]
    o["n_steps_dlogp_gt05"] = len(hi)
    gg = [(v, s) for s, v in zip(S, g) if _num(v)]; dd = [(v, s) for s, v in zip(S, dl) if _num(v)]
    o["gnorm_max"], o["gnorm_max_step"] = max(gg) if gg else (None, None)
    o["dlogp_max"], o["dlogp_max_step"] = max(dd) if dd else (None, None)
    for k in ("gnorm_gt1_2consec", "gnorm_gt1_for_good", "dlogp_gt05_first", "dlogp_gt05_3consec"):
        o[k + "_lr_steps"] = None if o[k] is None else RL_LR * o[k]
    return o


def summarize(D, ABL, PRE):
    ev, ref = D["this"]["evals"], D["ref"]["evals"]
    tr = D["this"]["train"]
    init = at(D["mid"]["evals"], max(r["ckpt_step"] for r in D["mid"]["evals"])) if D["mid"]["evals"] else None
    ref_init = (ABL["ref_before_rl"] or {}).get("eval/mean_all")
    ref_at = at(ref, REF_STEP); ref_peak = max(ref, key=lambda r: r["eval/mean_all"]) if ref else None
    best = max(ev, key=lambda r: r["eval/mean_all"]) if ev else None
    last = ev[-1] if ev else None
    evaluated = [r["ckpt_step"] for r in ev]; pending = [s for s in SAVE_STEPS if s not in evaluated]
    done = D["runs"]["this_train"]["state"] == "finished" and not pending
    if best is None or ref_at is None:
        verdict, vword = "pending", "pending"
    elif best["eval/mean_all"] > ref_at + NOISE_EPS:
        verdict, vword = "beats", "beats"
    elif abs(best["eval/mean_all"] - ref_at) <= NOISE_EPS:
        verdict, vword = "matches", f"matches within {NOISE_EPS:.3f}"
    else:
        verdict, vword = "below", "is below"
    common = [s for s in evaluated if at(ref, s) is not None]
    deltas = {str(s): at(ev, s) - at(ref, s) for s in common}
    fam_last = {}
    if common:
        S = common[-1]
        for k, _ in FAMILY_PANELS + [("eval/jlens/cos", ""), ("eval/mean_all", "")]:
            a, b = at(ev, S, k), at(ref, S, k)
            fam_last[k] = None if a is None or b is None else a - b
    o = onsets(tr)
    peak = {"peak_last_frac": {"step0": tr["reward/peak_last_frac"][0] if tr["reward/peak_last_frac"] else None,
                               "w20_30": win_mean(tr, "reward/peak_last_frac", 20, 30), "w145_155": win_mean(tr, "reward/peak_last_frac", 145, 155),
                               "w195_205": win_mean(tr, "reward/peak_last_frac", 195, 205), "last10": win_mean(tr, "reward/peak_last_frac", tr["step"][-1] - 9, tr["step"][-1]),
                               "at25": tr["reward/peak_last_frac"][tr["step"].index(25)] if 25 in tr["step"] else None,
                               "at275": tr["reward/peak_last_frac"][tr["step"].index(275)] if 275 in tr["step"] else None,
                               "min": min((v for v in tr["reward/peak_last_frac"] if _num(v)), default=None)},
            "peak_dist_mean": {"step0": tr["reward/peak_dist_mean"][0], "w20_30": win_mean(tr, "reward/peak_dist_mean", 20, 30),
                               "last10": win_mean(tr, "reward/peak_dist_mean", tr["step"][-1] - 9, tr["step"][-1])},
            "peak_in_last5_frac": {"min": min((v for v in tr["reward/peak_in_last5_frac"] if _num(v)), default=None),
                                   "max": max((v for v in tr["reward/peak_in_last5_frac"] if _num(v)), default=None),
                                   "note": "identically 1 under --reward-window-last 5: the reward IS the max over the last 5 tokens, so its argmax is always inside the window"},
            "definitions": "peak_dist = distance (tokens) of the reward argmax from the last kept token, per rollout; peak_last_frac = P(peak_dist == 0); "
                           "peak_in_last5_frac = P(peak_dist <= 4); peak_dist_mean = mean over the batch (rl/rl_disagg.py:2891-2893)"}
    def w(k, a, b):
        return win_mean(tr, k, a, b)
    last_s = tr["step"][-1]
    dyn = {"reward_w20_30": w("reward/mean", 20, 30), "reward_w145_155": w("reward/mean", 145, 155), "reward_last10": w("reward/mean", last_s - 9, last_s),
           "reward_min_from150": min((v for s, v in zip(tr["step"], tr["reward/mean"]) if s >= 150 and _num(v)), default=None),
           "reward_max_from150": max((v for s, v in zip(tr["step"], tr["reward/mean"]) if s >= 150 and _num(v)), default=None),
           "entropy_w20_30": w("policy/entropy", 20, 30), "entropy_last10": w("policy/entropy", last_s - 9, last_s),
           "entropy_min": min((v for v in tr["policy/entropy"] if _num(v)), default=None),
           "gnorm_w20_30": w("grad_norm", 20, 30), "gnorm_last10": w("grad_norm", last_s - 9, last_s),
           "dlogp_w20_30": w("policy/sampler_abs_dlogp", 20, 30), "dlogp_last10": w("policy/sampler_abs_dlogp", last_s - 9, last_s),
           "len_w20_30": w("rollout/len_mean", 20, 30), "len_last10": w("rollout/len_mean", last_s - 9, last_s),
           "step_s_median_from30": float(np.median([v for s, v in zip(tr["step"], tr["time/step_s"]) if s >= 30 and _num(v)])) if any(_num(v) for v in tr["time/step_s"]) else None,
           "clipfrac_last10": w("ratio/clipfrac", last_s - 9, last_s), "mem_peak_gb_max": max((v for v in tr["mem/hf_peak_gb"] if _num(v)), default=None),
           "lr_after_warmup": max((v for v in tr["lr"] if _num(v)), default=None), "last_step": last_s}
    rtr = D["ref"]["train"]; rl = rtr["step"][-1] if rtr["step"] else None
    ref_dyn = {"reward_last10": win_mean(rtr, "reward/mean", rl - 9, rl) if rl else None, "entropy_last10": win_mean(rtr, "policy/entropy", rl - 9, rl) if rl else None,
               "gnorm_last10": win_mean(rtr, "grad_norm", rl - 9, rl) if rl else None, "dlogp_last10": win_mean(rtr, "policy/sampler_abs_dlogp", rl - 9, rl) if rl else None,
               "onsets": onsets(rtr) if rtr["step"] else None}
    mid = D["mid"]["train"]; ml = [v for v in mid["loss"] if _num(v)]
    ts = [t for t in mid["_timestamp"] if _num(t)]
    mid_stats = {"steps": mid["step"][-1] + 1 if mid["step"] else None, "loss_first10": float(np.mean(ml[:10])) if ml else None,
                 "loss_last20": float(np.mean(ml[-20:])) if ml else None, "loss_last20_sd": float(np.std(ml[-20:])) if len(ml) > 1 else None,
                 "median_s_per_step": float(np.median(np.diff(ts))) if len(ts) > 2 else None, "wall_h": (ts[-1] - ts[0]) / 3600 if len(ts) > 1 else None,
                 "lr_peak": max((v for v in mid["lr"] if _num(v)), default=None), "peak_mem_gb": max((v for v in mid["peak_mem_gb"] if _num(v)), default=None),
                 "evals": [{"ckpt_step": r["ckpt_step"], **{k: r.get(k) for k, _ in TABLE_KEYS}} for r in D["mid"]["evals"]],
                 "mean_all_min": min(r["eval/mean_all"] for r in D["mid"]["evals"]) if D["mid"]["evals"] else None,
                 "mean_all_max": max(r["eval/mean_all"] for r in D["mid"]["evals"]) if D["mid"]["evals"] else None,
                 "config": {k: D["mid"]["config"].get(k) for k in ("lr", "batch_size", "grad_accum", "max_seq", "full_ft", "init_adapter", "data_dir", "save_dir", "n_ckpts", "epochs")}}
    cfg_this, cfg_ref = D["this"]["config"], D["ref"]["config"]
    diffs = {k: [cfg_this.get(k), cfg_ref.get(k)] for k in sorted(set(cfg_this) | set(cfg_ref))
             if json.dumps(cfg_this.get(k), default=str) != json.dumps(cfg_ref.get(k), default=str) and k not in ("mb_search",)}
    shared = {k: cfg_this.get(k) for k in ("lr", "group_size", "groups_per_step", "total_steps", "warmup_steps", "recipe", "loss", "cispo_eps_max", "reward_window_last",
                                            "kl_coef", "entropy_coef", "max_grad_norm", "save_steps", "max_new_tokens", "min_new_tokens", "length_control",
                                            "len_penalty_start", "len_penalty_per_tok", "temperature", "adv_mode", "zero_var_filter", "npr_threshold", "npr_pass_cos",
                                            "max_lag", "lr_decay", "eval_cache", "fp32_head", "autocast_bf16")}
    return {"verdict": verdict, "verdict_word": vword, "run_finished": D["runs"]["this_train"]["state"] == "finished", "all_evals_in": done,
            "evaluated_steps": evaluated, "pending_steps": pending, "init_mean_all": init, "ref_init_mean_all": ref_init,
            "best": None if best is None else {"ckpt_step": best["ckpt_step"], "mean_all": best["eval/mean_all"]},
            "last": None if last is None else {"ckpt_step": last["ckpt_step"], "mean_all": last["eval/mean_all"]},
            "gain_over_init": None if best is None or init is None else best["eval/mean_all"] - init,
            "ref_at_300": ref_at, "ref_peak": None if ref_peak is None else {"ckpt_step": ref_peak["ckpt_step"], "mean_all": ref_peak["eval/mean_all"]},
            "ref_gain_over_init": None if ref_at is None or ref_init is None else ref_at - ref_init,
            "delta_vs_ref_same_step": deltas, "delta_vs_ref_last_common_step": {"step": common[-1] if common else None, "by_key": fam_last},
            "noise_eps": NOISE_EPS, "table_eps": TABLE_EPS, "onsets": o, "peak_metrics": peak, "dynamics": dyn, "ref_dynamics": ref_dyn,
            "midtrain": mid_stats, "pretrain": PRE, "rl_lr": RL_LR, "lora_wall_lr_steps": LORA_WALL_LR_STEPS,
            "config_shared": shared, "config_diffs_this_vs_ref": diffs, "recipe_pool": "/data/banks/mix_eq_1p45m (six equal families x 241,741; the RL bank, unchanged)"}


# -------------------------------------------------------------------- plots --------------------------------------------------------------------
def style_ax(ax):
    ax.grid(True, axis="y"); ax.grid(False, axis="x"); ax.tick_params(length=0)


def save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig); print("wrote", OUT / f"{stem}.png")


def finish(fig, bottom=0.0, pad=0.012):
    """Measure the (multi-line, left-aligned) suptitle's real bounding box, then tight_layout the axes block into the space below it.
    The suptitle is hidden during tight_layout so its height is not reserved twice."""
    st = fig._suptitle
    if st is None:
        fig.tight_layout(rect=(0, bottom, 1, 1)); return
    fig.canvas.draw()
    bb = st.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    st.set_visible(False)
    fig.tight_layout(rect=(0, bottom, 1, max(0.5, bb.y0 - pad)))
    st.set_visible(True)


def wrap(s, w=150):
    return "\n".join(textwrap.fill(p, w) for p in s.split("\n"))


def fmt(v, nd=3):
    return "n/a" if v is None else f"{v:.{nd}f}"


def sgn(v, nd=3):
    if v is None:
        return "n/a"
    r = round(v, nd)
    return f"{0.0:+.{nd}f}" if r == 0 else f"{r:+.{nd}f}"


def xy(rows, key, init=None):
    xs, ys = ([0], [init]) if _num(init) else ([], [])
    for r in rows:
        if _num(r.get(key)):
            xs.append(r["ckpt_step"]); ys.append(r[key])
    return xs, ys


def this_label(D, M):
    st = D["runs"]["this_train"]["state"]
    tail = "" if st == "finished" else f"  [RL still running, step {D['runs']['this_train']['last_step']}]"
    if M["pending_steps"] and st == "finished":
        n = len(M["pending_steps"])
        tail = f"  [eval{'s' if n > 1 else ''} pending for step{'s' if n > 1 else ''} {', '.join(map(str, M['pending_steps']))}]"
    return "re-cut bank: 104M pretrain + full-FT midtrain on the re-cut bank, full-parameter RL" + tail


REF_LABEL = "reference: same full-parameter RL on the old-bank init (23M-ckpt pretrain + FFT midtrain, targets not end-anchored)"


def plot_headline(D, M, ABL):
    fig, ax = plt.subplots(figsize=(11, 6.6))
    ev, ref = D["this"]["evals"], D["ref"]["evals"]
    # LoRA ablation arms: thin grey context (identity via one legend entry; they share a recipe, lr 7e-6)
    first = True
    for a, v in ABL["arms"].items():
        xs, ys = xy(v["evals"], "eval/mean_all", v["before_rl"])
        ax.plot(xs, ys, color=LORA_C, lw=1.1, marker="o", ms=3, zorder=1, label="context: the four LoRA-RL arms of the SFT-init ablation (lr 7e-6, same RL bank, different SFT inits)" if first else None)
        first = False
    xs, ys = xy(ref, "eval/mean_all", M["ref_init_mean_all"])
    ax.plot(xs, ys, color=REF_C, lw=2, ls="--", marker="s", ms=6.5, mec="white", mew=1.2, label=REF_LABEL, zorder=3)
    xs, ys = xy(ev, "eval/mean_all", M["init_mean_all"])
    ax.plot(xs, ys, color=THIS_C, lw=2.4, marker="o", ms=7.5, mec="white", mew=1.2, label=this_label(D, M), zorder=4)
    if M["ref_at_300"] is not None:
        ax.axhline(M["ref_at_300"], color=REF_C, lw=1, ls=(0, (4, 3)), zorder=0)
        ax.axhspan(M["ref_at_300"] - NOISE_EPS, M["ref_at_300"] + NOISE_EPS, color=REF_C, alpha=0.08, lw=0, zorder=0)
        ax.text(309, M["ref_at_300"], f"reference @300: {M['ref_at_300']:.3f}\n(band = eval noise ±{NOISE_EPS:.3f})", fontsize=8.3, color=INK2, va="center")
    # end labels
    if ev:
        ax.annotate(f"{ev[-1]['eval/mean_all']:.3f} @ {ev[-1]['ckpt_step']}", xy=(ev[-1]["ckpt_step"], ev[-1]["eval/mean_all"]), xytext=(6, -14), textcoords="offset points",
                    fontsize=8.8, color=INK2, va="top")
        if M["best"] and M["best"]["ckpt_step"] != ev[-1]["ckpt_step"]:
            b = M["best"]
            ax.annotate(f"best {b['mean_all']:.3f} @ {b['ckpt_step']}", xy=(b["ckpt_step"], b["mean_all"]), xytext=(0, 10), textcoords="offset points", fontsize=8.5, color=INK2, ha="center")
    for s in M["pending_steps"]:
        ax.text(s, 0.335, f"eval @{s}\npending", fontsize=7.8, color=MUTED, ha="center", va="bottom")
    if M["init_mean_all"] is not None:
        ax.annotate(f"init after midtrain: {M['init_mean_all']:.3f}", xy=(0, M["init_mean_all"]), xytext=(16, -12), textcoords="offset points", fontsize=8.3, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.8))
    if M["ref_init_mean_all"] is not None:
        ax.annotate(f"reference init: {M['ref_init_mean_all']:.3f}", xy=(0, M["ref_init_mean_all"]), xytext=(8, 14), textcoords="offset points", fontsize=8.3, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.8))
    ax.set_xlim(-6, 372); ax.set_ylim(0.33, 0.44); ax.set_xticks([0, 25, 50, 100, 150, 200, 250, 300])
    ax.set_xlabel("RL step (4,096 rollouts per step; step 0 = the SFT init before any RL)")
    ax.set_ylabel("held-out fidelity: mean cosine over 10 direction families")
    b, r = M["best"], M["ref_at_300"]
    if b and r is not None:
        vw = {"beats": "BEATS", "matches": "MATCHES (within eval noise)", "below": "stays BELOW"}[M["verdict"]]
        head = (f"Re-cutting the midtrain bank (end-anchored, standalone-verified targets) then full-parameter RL reaches {b['mean_all']:.3f} at step {b['ckpt_step']} "
                f"(+{b['mean_all'] - M['init_mean_all']:.3f} over its {M['init_mean_all']:.3f} init) — this {vw} the {r:.3f} the same RL reached from the old-bank init"
                + ("" if M["all_evals_in"] else f"; the eval{'s' if len(M['pending_steps']) > 1 else ''} for step{'s' if len(M['pending_steps']) > 1 else ''} "
                                                f"{', '.join(map(str, M['pending_steps']))} still pending"))
    else:
        head = "Re-cut midtrain bank + full-FT midtrain + full-parameter RL vs the same RL on the old-bank init (evals pending)"
    ax.set_title(wrap(head, 160) + "\n" + wrap("Held-out mean fidelity vs RL step — Qwen3.6-27B activation-to-text inverter; full-parameter CISPO GRPO, 8 x 512 rollouts/step, "
                 "lr 1e-6 constant after 25 warmup steps, 300 steps, same six-family RL bank; 512 held-out directions per family, best-of-4 at T=1", 160),
                 fontsize=10.2, loc="left", pad=10)
    style_ax(ax)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=1, fontsize=8.6, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    save(fig, "fidelity_vs_rl_step")


def plot_families(D, M, ABL):
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8), sharex=True)
    ev, ref = D["this"]["evals"], D["ref"]["evals"]
    mid_final = D["mid"]["evals"][-1] if D["mid"]["evals"] else {}
    ref_before = ABL["ref_before_rl"] or {}
    for ax, (key, title) in zip(axes.flat, FAMILY_PANELS):
        xs, ys = xy(ref, key, ref_before.get(key))
        ax.plot(xs, ys, color=REF_C, lw=1.8, ls="--", marker="s", ms=5.5, mec="white", mew=1.1, label=REF_LABEL, zorder=3)
        xs, ys = xy(ev, key, mid_final.get(key))
        ax.plot(xs, ys, color=THIS_C, lw=2.2, marker="o", ms=6.5, mec="white", mew=1.1, label=this_label(D, M), zorder=4)
        d = M["delta_vs_ref_last_common_step"]["by_key"].get(key)
        S = M["delta_vs_ref_last_common_step"]["step"]
        ax.set_title(title + (f"\nre-cut minus reference @ step {S}: {sgn(d)}" if d is not None else ""), fontsize=9.6, loc="left")
        ax.set_xticks([0, 50, 100, 150, 200, 250, 300]); ax.set_xlim(-6, 312)
        style_ax(ax)
    for ax in axes[-1]:
        ax.set_xlabel("RL step (0 = the SFT init before RL)", fontsize=9)
    by = M["delta_vs_ref_last_common_step"]["by_key"]; S = M["delta_vs_ref_last_common_step"]["step"]
    short = FAM_PLAIN
    lead = [short[k] for k, _ in FAMILY_PANELS if by.get(k) is not None and by[k] > 0.01]
    trail = [short[k] for k, _ in FAMILY_PANELS if by.get(k) is not None and by[k] < -0.01]
    tie = [short[k] for k, _ in FAMILY_PANELS if by.get(k) is not None and abs(by[k]) <= 0.01]
    head = f"At the last common checkpoint (step {S}) the re-cut-bank arm " + \
        (f"leads on {', '.join(lead)}" if lead else "leads on no family") + (f", trails on {', '.join(trail)}" if trail else "") + \
        (f", and is within ±0.01 on {', '.join(tie)}" if tie else "") + \
        (f"; the mean gap is {sgn(by.get('eval/mean_all'))}" if by.get("eval/mean_all") is not None else "")
    fig.suptitle(wrap(head, 175) + "\n" + wrap("Per-family held-out fidelity vs RL step, this arm vs the reference (same full-parameter CISPO GRPO recipe, 8 x 512, lr 1e-6, "
                 "same RL bank; only the SFT init differs: re-cut bank vs old bank). The midtrain never saw long-context real activations (RL-only family). "
                 "MLP fire-back was not measured for this arm's init (hence no step-0 point there).", 175), fontsize=10.2, x=0.01, ha="left", y=0.995)
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=1, fontsize=8.8, bbox_to_anchor=(0.5, -0.005))
    finish(fig, bottom=0.07)
    save(fig, "per_family_vs_rl_step")


def plot_dynamics(D, M):
    tr, rtr, o = D["this"]["train"], D["ref"]["train"], M["onsets"]
    panels = [("reward/mean", "reward: mean cosine, max over the last-5-token window", False),
              ("policy/entropy", "policy entropy (nats / token)", False),
              ("grad_norm", "gradient norm before clipping (log; dotted = clip at 1.0)", True),
              ("policy/sampler_abs_dlogp", "sampler vs trainer |delta log p| per token (log; dotted = 0.05)", True),
              ("rollout/len_mean", "response length (tokens)", False),
              ("time/step_s", "wall-clock seconds per RL step", False),
              ("reward/peak_last_frac", "reward peak exactly at the last token (share of rollouts)", False),
              ("reward/peak_dist_mean", "reward peak distance from the last token (mean tokens, 0-4)", False)]
    fig, axes = plt.subplots(2, 4, figsize=(19, 8.4), sharex=True)
    for ax, (key, title, logy) in zip(axes.flat, panels):
        for tag, t, c, lw in (("ref", rtr, REF_C, 1.4), ("this", tr, THIS_C, 2.0)):
            if not t["step"] or not any(_num(v) for v in t[key]):
                continue
            x = np.array(t["step"]); yy = np.array([np.nan if v is None else v for v in t[key]], dtype=float)
            ax.plot(x, yy, color=c, lw=0.7, alpha=0.22, zorder=2)
            ax.plot(x, rolling(t[key]), color=c, lw=lw, ls="--" if tag == "ref" else "-", zorder=3,
                    label=(REF_LABEL if tag == "ref" else this_label(D, M)) if key == "reward/mean" else None, solid_capstyle="round")
        if key == "reward/peak_last_frac" and any(_num(v) for v in tr["reward/peak_in_last5_frac"]):
            ax.plot(tr["step"], tr["reward/peak_in_last5_frac"], color=MUTED, lw=1.2, zorder=2)
            ax.text(tr["step"][-1], 1.0, "peak within the last 5 (= 1 by construction)", fontsize=7.8, color=INK2, ha="right", va="bottom")
            p = M["peak_metrics"]["peak_last_frac"]
            if p["w20_30"] is not None and p["last10"] is not None:
                ax.text(150, 0.40, f"steps 20-30: {p['w20_30']:.2f}  ->  last 10 steps: {p['last10']:.2f}", fontsize=8.5, color=INK2, ha="center")
            ax.set_ylim(0.3, 1.06)
        if key == "grad_norm":
            ax.axhline(1.0, color=INK2, lw=1, ls=":", zorder=1)
        if key == "policy/sampler_abs_dlogp":
            ax.axhline(0.05, color=INK2, lw=1, ls=":", zorder=1)
        if logy:
            ax.set_yscale("log")
            ticks = [0.5, 0.7, 1.0, 1.5, 2.0, 3.0] if key == "grad_norm" else [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1]
            ax.set_yticks(ticks); ax.set_yticklabels([f"{t:g}" for t in ticks]); ax.set_yticks([], minor=True)
        if key in ("reward/mean",):
            ax.set_ylim(0.15, 0.35)
        if key == "policy/entropy":
            ax.set_ylim(0.5, 3.0)
        if key == "time/step_s":
            ax.set_ylim(0, 80)
        if key == "rollout/len_mean":
            ax.set_ylim(15, 60)
        for s, lab, ha, dx in ((o["gnorm_gt1_2consec"], f"grad norm > 1\n2 consecutive: {o['gnorm_gt1_2consec']}\nfor good: {o['gnorm_gt1_for_good']}", "right", -3),
                              (o["dlogp_gt05_first"], f"|dlogp| > .05\nfirst: {o['dlogp_gt05_first']}", "left", 3)):
            if s is not None:
                ax.axvline(s, color=INK2, lw=1, ls="--", alpha=0.7, zorder=1)
                if key in ("reward/mean",):
                    ax.text(s + dx, 0.153, lab, fontsize=7.4, color=INK2, va="bottom", ha=ha)
        ax.set_title(title, fontsize=9.4, loc="left"); ax.set_xlim(0, 305); ax.set_xticks([0, 50, 100, 150, 200, 250, 300])
        style_ax(ax)
    for ax in axes[-1]:
        ax.set_xlabel("RL step", fontsize=9)
    dy = M["dynamics"]; p = M["peak_metrics"]["peak_last_frac"]
    lrs = o["gnorm_gt1_for_good_lr_steps"] or o["gnorm_gt1_2consec_lr_steps"]
    head = (f"Full-parameter RL at lr 1e-6 hits the drift wall early: grad norm is above the clip for good from step {o['gnorm_gt1_for_good']} "
            f"(2-consecutive rule: {o['gnorm_gt1_2consec']}), sampler-trainer |delta log p| first exceeds .05 at step {o['dlogp_gt05_first']} (max {fmt(o['dlogp_max'])}), "
            f"and reward stalls at {fmt(dy['reward_min_from150'], 2)}-{fmt(dy['reward_max_from150'], 2)} from step 150 while entropy keeps falling "
            f"({fmt(dy['entropy_w20_30'], 2)} -> {fmt(dy['entropy_last10'], 2)}) — onset at cumulative lr x steps ≈ {lrs * 1e4:.1f}e-4, "
            f"about {LORA_WALL_LR_STEPS / lrs:.0f}x lower than the LoRA arms' 2.4e-3. New finding: the reward peak drifts EARLIER in the window over training "
            f"(share of rollouts peaking exactly at the last token {fmt(p['w20_30'], 2)} -> {fmt(p['last10'], 2)})")
    fig.suptitle(wrap(head, 205) + "\n" + wrap("RL training dynamics vs step — this arm (solid) vs the reference full-parameter arm on the old-bank init (dashed), same recipe; "
                 f"thin = per step, thick = {MA}-step moving average; dashed verticals = this arm's two onset markers. The reference run did not log peak_last_frac.", 205),
                 fontsize=10.2, x=0.01, ha="left", y=0.995)
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=1, fontsize=8.8, bbox_to_anchor=(0.5, -0.005))
    finish(fig, bottom=0.06)
    save(fig, "rl_dynamics")


def plot_bank():
    b = BANK
    surv = [("SAE feature windows\n(peak must be the last token)", b["sae"]["kept"], b["sae"]["total"]),
            ("cluster-probe targets\n(trim to peak; drop if < 8 tokens)", b["cluster"]["kept"], b["cluster"]["total"]),
            ("BSF verbatim windows\n(direction peak at the last token)", None, None),
            ("MLP single neurons\n(fire + peak at the last token, alone)", b["mlp"]["kept"]["mlp"], b["mlp"]["totals"]["mlp"]),
            ("MLP co-firing pairs\n(all members, alone)", b["mlp"]["kept"]["mlp_pair"], b["mlp"]["totals"]["mlp_pair"]),
            ("MLP co-firing triples\n(all members, alone)", b["mlp"]["kept"]["mlp_triple"], b["mlp"]["totals"]["mlp_triple"])]
    fracs, notes = [], []
    for lab, k, t in surv:
        if k is None:
            fracs.append(b["bsf"]["dir_peak_last_before_rule"]); notes.append(f"{b['bsf']['dir_peak_last_before_rule']:.1%} of candidates; {b['bsf']['kept']:,} kept")
        else:
            fracs.append(k / t); notes.append(f"{k / t:.1%}  ({k:,} of {t:,})")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.4), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    y = np.arange(len(surv))[::-1]
    ax.barh(y, [1.0] * len(surv), color="#eef2f8", height=0.62, zorder=1)
    ax.barh(y, fracs, color=THIS_C, height=0.62, zorder=2)
    for yi, f, n in zip(y, fracs, notes):
        ax.text(f + 0.012, yi, n, va="center", fontsize=8.6, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels([s[0] for s in surv], fontsize=8.8)
    ax.set_xlim(0, 1.0); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("share of candidate windows passing the standalone last-token check")
    ax.set_title("survival of the standalone last-token check, per family", fontsize=10, loc="left")
    ax.grid(True, axis="x"); ax.grid(False, axis="y"); ax.tick_params(length=0)
    ax = axes[1]
    fams = sorted(b["families"], key=lambda k: -b["families"][k])
    y = np.arange(len(fams))[::-1]; vals = [b["families"][f] for f in fams]
    ax.barh(y, vals, color=THIS_C, height=0.62, zorder=2)
    for yi, v in zip(y, vals):
        ax.text(v + 12_000, yi, f"{v:,}  ({v / b['n_rows']:.1%})", va="center", fontsize=8.6, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels([FAM_SHORT[f] for f in fams], fontsize=9)
    ax.set_xlim(0, 1_320_000); ax.set_xticks([0, 250_000, 500_000, 750_000, 1_000_000]); ax.set_xticklabels(["0", "250k", "500k", "750k", "1M"])
    ax.set_xlabel(f"rows in the re-cut midtrain bank ({b['n_rows']:,} total)")
    ax.set_title("final family composition of the re-cut bank", fontsize=10, loc="left")
    ax.grid(True, axis="x"); ax.grid(False, axis="y"); ax.tick_params(length=0)
    s = b["sae"]["kept"] / b["sae"]["total"]; c = b["cluster"]["kept"] / b["cluster"]["total"]
    m1, m2, m3 = (b["mlp"]["kept"][k] / b["mlp"]["totals"][k] for k in ("mlp", "mlp_pair", "mlp_triple"))
    head = (f"Forced to fire standalone at their LAST token, SAE ({s:.0%}) and probe ({c:.0%}) windows mostly survive but MLP-neuron windows mostly do not "
            f"({m1:.0%} singles, {m2:.0%} pairs, {m3:.0%} triples) — the re-cut 2.75M-row midtrain bank for the Qwen3.6-27B activation-to-text inverter")
    fig.suptitle(wrap(head, 185) + "\n" + wrap("Left: fraction of each family's candidate SFT windows that pass the check (each window forwarded alone; the target direction's "
                 "activation must peak at the window's last token). Right: rows per family after the checks (SAE decoder-direction targets share the SAE windows; "
                 "long-context real activations are deliberately absent = RL-only family).", 185), fontsize=10.2, x=0.01, ha="left", y=0.995)
    finish(fig, bottom=0.0)
    save(fig, "bank_recut")
    return {"survival": [{"label": lab.replace("\n", " "), "kept": k, "total": t, "frac": f, "note": n} for (lab, k, t), f, n in zip(surv, fracs, notes)],
            "composition": [{"family": f, "short": FAM_SHORT[f], "label": b["family_labels"][f], "rows": b["families"][f], "share": b["families"][f] / b["n_rows"]} for f in fams]}


# ---------------------------------------------------------------- data files ----------------------------------------------------------------
def write_data(D, M, ABL, bank_plot):
    dd = OUT / "data"; dd.mkdir(parents=True, exist_ok=True)
    ev, ref = D["this"]["evals"], D["ref"]["evals"]
    table = []
    for s in sorted(set(SAVE_STEPS) | {r["ckpt_step"] for r in ev} | {r["ckpt_step"] for r in ref}):
        table.append({"ckpt_step": s, "this": {k: at(ev, s, k) for k, _ in TABLE_KEYS}, "ref": {k: at(ref, s, k) for k, _ in TABLE_KEYS},
                      "delta": {k: (at(ev, s, k) - at(ref, s, k)) if (at(ev, s, k) is not None and at(ref, s, k) is not None) else None for k, _ in TABLE_KEYS},
                      "this_pending": at(ev, s) is None})
    mid_final = D["mid"]["evals"][-1] if D["mid"]["evals"] else {}
    init_row = {"ckpt_step": 0, "this": {k: mid_final.get(k) for k, _ in TABLE_KEYS}, "ref": {k: (ABL["ref_before_rl"] or {}).get(k) for k, _ in TABLE_KEYS}}
    init_row["delta"] = {k: (init_row["this"][k] - init_row["ref"][k]) if (init_row["this"][k] is not None and init_row["ref"][k] is not None) else None for k, _ in TABLE_KEYS}
    init_row["this_pending"] = False
    json.dump({"generated_at": D["fetched_at"], "table_keys": TABLE_KEYS, "family_panels": FAMILY_PANELS,
               "this": {"label": this_label(D, M), "evals": ev, "init": {"mean_all": M["init_mean_all"], "source": "midtrain final checkpoint eval (mix5msft_midtrain_fft_from_fft104m_eval, last ckpt)", "row": mid_final}},
               "ref": {"label": REF_LABEL, "evals": ref, "init": {"mean_all": M["ref_init_mean_all"], "source": ABL["ref_before_rl_source"], "row": ABL["ref_before_rl"]}},
               "lora_arms": ABL["arms"], "lora_arms_source": ABL["source"], "midtrain_evals": D["mid"]["evals"],
               "table_this_vs_ref": [init_row] + table}, open(dd / "eval_curves.json", "w"), indent=1)
    dyn = {}
    for tag in ("this", "ref"):
        t = dict(D[tag]["train"])
        for k in ("reward/mean", "policy/entropy", "grad_norm", "policy/sampler_abs_dlogp", "rollout/len_mean", "time/step_s", "reward/peak_last_frac", "reward/peak_dist_mean"):
            if any(_num(v) for v in t.get(k, [])):
                t[k + f"__ma{MA}"] = [None if np.isnan(v) else float(v) for v in rolling(t[k])]
        dyn[tag] = {"run": D["runs"][tag + "_train"], "series": t}
    json.dump({"generated_at": D["fetched_at"], "ma_window": MA, "onsets": M["onsets"], "peak_metrics": M["peak_metrics"], "dynamics_summary": M["dynamics"],
               "ref_dynamics_summary": M["ref_dynamics"], "runs": dyn}, open(dd / "rl_dynamics.json", "w"), indent=1)
    json.dump({"generated_at": D["fetched_at"], "spec": BANK, "plotted": bank_plot}, open(dd / "bank_recut.json", "w"), indent=1)
    mt = D["mid"]["train"]
    json.dump({"generated_at": D["fetched_at"], "run": D["runs"]["mid_train"], "eval_run": D["runs"]["mid_eval"], "stats": M["midtrain"],
               "series": {k: mt[k] for k in ("step", "loss", "lr", "ex_per_s", "peak_mem_gb", "_timestamp")}}, open(dd / "midtrain.json", "w"), indent=1)
    summ = {k: v for k, v in M.items() if k not in ("midtrain",)} | {"generated_at": D["fetched_at"], "runs": D["runs"], "midtrain_stats": {k: v for k, v in M["midtrain"].items() if k != "evals"},
                                                                    "bank_n_rows": BANK["n_rows"], "bank_families": BANK["families"], "save_steps": SAVE_STEPS, "ref_step": REF_STEP}
    json.dump(summ, open(dd / "summary.json", "w"), indent=1, default=str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-html", action="store_true", help="skip running the report folder's build_html.py")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    D = fetch(); ABL = load_ablation(); PRE = load_pretrain()
    M = summarize(D, ABL, PRE)
    print(f"this arm: train {D['runs']['this_train']['state']} @ step {D['runs']['this_train']['last_step']}; evals {M['evaluated_steps']} pending {M['pending_steps']}; "
          f"init {fmt(M['init_mean_all'])}; best {M['best']}; ref@300 {fmt(M['ref_at_300'])} (init {fmt(M['ref_init_mean_all'])}) -> verdict: {M['verdict_word']}")
    print("onsets:", json.dumps({k: v for k, v in M["onsets"].items() if not k.startswith("rule") and k != "gnorm_le1_steps_after_100"}))
    print("peak_last_frac:", json.dumps(M["peak_metrics"]["peak_last_frac"]))
    print("dynamics:", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in M["dynamics"].items()}))
    print("midtrain:", json.dumps({k: v for k, v in M["midtrain"].items() if k not in ("evals", "config")}, default=str))
    print("config diffs this vs ref:", json.dumps(M["config_diffs_this_vs_ref"], default=str))
    bank_plot = plot_bank()
    plot_headline(D, M, ABL); plot_families(D, M, ABL); plot_dynamics(D, M)
    write_data(D, M, ABL, bank_plot)
    bh = OUT / "build_html.py"
    if not args.no_html and bh.exists():
        subprocess.run(["python3", str(bh)], check=True)


if __name__ == "__main__":
    main()
