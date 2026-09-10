#!/usr/bin/env python3
"""Re-cut midtrain bank -> full-fine-tune midtrain -> full-parameter RL (Qwen3.6-27B activation->text inverter).

The chain this script reports on (all wandb project celestedeschamphelaere-personal/maxact-fast):
  pretrain   realact104m_fullft_b4096_lr1e-5 (jp82mr9a)           104M real-activation examples, full FT, held-out mean_all .369
  bank       /data/banks/mix_5m_sft (2,747,509 rows)               re-cut after four corrections: no realact_long; every target ENDS at the
                                                                    token where its direction fires (verified standalone); BSF re-mined from
                                                                    verbatim contexts; MLP windows kept only if they fire in isolation
  midtrain   mix5msft_midtrain_fft_from_fft104m (eaozkq7o)         FSDP2 full FT from the pretrain final, eff batch 4,096, lr 1e-5, 671 steps
  RL         rl_fullparam_fft104m_mix5m_8x512 (4piq4y7x)           full-parameter CISPO GRPO, 8 x 512 rollouts/step, lr 1e-6, 300 steps  (.425 @300)
  reference  rl_abl_initnewfft_fullparam_8x512 (y4teoshq)          the SAME RL recipe on the OLD-bank init (23M-ckpt pretrain + 1.45M FFT
                                                                    midtrain; targets not end-anchored, realact_long included): .424 @300
  + three follow-up arms from the SAME re-cut init (same pool mix_eq_1p45m, same ScaleRL/CISPO recipe, 2 rollout + 6 trainer B200, full-parameter):
  arm A      rl_fullparam_fft104m_mix5m_8x2048 (e9zit9p7)          --groups-per-step 2048 = 16,384 rollouts/step (4x the batch), lr 1e-6, 300 steps
  arm B      rl_fullparam_fft104m_mix5m_8x512_anywin (0unnognn)    --reward-window-last 0: reward = max cosine over ALL rollout tokens (evaluator unchanged = last 5)
  arm C      rl_fullparam_fft104m_mix5m_8x2048_paperlr (pvwi9pv9)  ScaleRL-paper optimizer: lr 5e-7 constant after a 100-step warmup, AdamW eps 1e-15,
                                                                    weight decay .01, 8x2048, 7,400 steps; respawned every 24 h from the latest full-model
                                                                    checkpoint with --step-offset (same wandb id) -> keeps adding checkpoints for ~10 days

Idempotent: every run re-pulls the eval + training histories from wandb (scan_history AND sampled history unioned, deduped by _step; eval runs =
every run named <run>_eval unioned, latest row per ckpt_step), reads the LoRA init-ablation arms from ~/shared/reports/maemm-sft-init-ablation/data/
eval_curves.json and the pretrain numbers from ~/shared/reports/maemm-sft-fullft-104m/data/fidelity_vs_examples.json, rewrites data/*.json + every
figure as PNG + PDF, then (unless --no-html) runs the report folder's build_html.py. Arms that are still training are labelled "(running, step N)";
figures with a step axis switch to a log axis once arm C passes 900 steps, so re-running any time during its 10-day run just works.

Cumulative lr x steps is computed from the configured schedule lr_t = lr_peak * min(1, (t+1)/warmup) for update t (0-indexed), constant after
warmup (lr_decay none); cum(s) = sum_{t<=s} lr_t. The logged `lr` series follows exactly this schedule for every arm (checked on each run;
the max relative deviation of the logged cumulative sum is written to data/new_arms.json).

    python scripts/plot_fft104m_mix5m_fullrl.py            # refresh everything
    python scripts/plot_fft104m_mix5m_fullrl.py --no-html  # figures + data only

Outputs -> ~/shared/reports/maemm-fft104m-mix5m-fullrl/{data/*.json, fidelity_vs_rl_step, fidelity_vs_rollouts, fidelity_vs_lr_steps,
           per_family_vs_rl_step, rl_dynamics, bank_recut}.{png,pdf}
"""
import argparse
import datetime as dt
import json
import math
import os
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

# ---- chart chrome. Colour follows the entity: the 8x512 last-5 arm = blue, the reference full-param arm = orange, arm A (4x batch) = purple,
#      arm B (all-token reward) = aqua, arm C (paper optimizer) = ink; the four hues validate all-pairs (CVD dE >= 9.2, normal dE >= 16.3);
#      context (LoRA arms / thresholds) = grey de-emphasis; text never wears a series colour. Aqua (2.7:1) gets direct end labels as relief.
THIS_C, REF_C, LORA_C = "#2a78d6", "#eb6834", "#b3b1a8"
A_C, B_C, C_C = "#7a3aa7", "#1baf7a", "#191919"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.titlecolor": INK, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "grid.color": GRID, "grid.linewidth": 0.8, "axes.grid": True, "axes.axisbelow": True, "legend.frameon": False,
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})

# The four RL arms that share the re-cut init. Static facts here; lr / warmup / groups / total steps / save steps are re-read from the wandb config.
ARM_ORDER = ["this", "a", "b", "c"]
ARMS = {
    "this": {"id": "4piq4y7x", "name": "rl_fullparam_fft104m_mix5m_8x512", "nick": "8 x 512, last-5 reward", "tiny": "8x512 last-5",
             "long": "8 x 512 rollouts/step, reward = max cosine over the last 5 tokens, lr 1e-6 (the .425 arm; baseline for the new arms)",
             "color": THIS_C, "marker": "o", "ls": "-", "lw": 2.4, "ms": 7.5,
             "lr": 1e-6, "warmup": 25, "groups_per_step": 512, "group_size": 8, "total_steps": 300, "reward_window_last": 5, "weight_decay": 0.0, "adam_eps": 1e-8,
             "save_steps": [25, 50, 100, 150, 200, 250, 300]},
    "a": {"id": "e9zit9p7", "name": "rl_fullparam_fft104m_mix5m_8x2048", "nick": "8 x 2048 (4x batch)", "tiny": "4x batch",
          "long": "arm A: 8 x 2048 rollouts/step = 4x the batch (16,384 rollouts/step), last-5 reward, lr 1e-6",
          "color": A_C, "marker": "D", "ls": "-", "lw": 2.0, "ms": 6.2,
          "lr": 1e-6, "warmup": 25, "groups_per_step": 2048, "group_size": 8, "total_steps": 300, "reward_window_last": 5, "weight_decay": 0.0, "adam_eps": 1e-8,
          "save_steps": [25, 50, 100, 150, 200, 250, 300]},
    "b": {"id": "0unnognn", "name": "rl_fullparam_fft104m_mix5m_8x512_anywin", "nick": "8 x 512, all-token reward", "tiny": "all-token",
          "long": "arm B: 8 x 512, reward = max cosine over ALL rollout tokens (--reward-window-last 0; the evaluator is unchanged = last 5), lr 1e-6",
          "color": B_C, "marker": "^", "ls": "-", "lw": 2.0, "ms": 7.0,
          "lr": 1e-6, "warmup": 25, "groups_per_step": 512, "group_size": 8, "total_steps": 300, "reward_window_last": 0, "weight_decay": 0.0, "adam_eps": 1e-8,
          "save_steps": [25, 50, 100, 150, 200, 250, 300]},
    "c": {"id": "pvwi9pv9", "name": "rl_fullparam_fft104m_mix5m_8x2048_paperlr", "nick": "8 x 2048, paper optimizer", "tiny": "paper lr",
          "long": "arm C: 8 x 2048, last-5 reward, ScaleRL-paper optimizer: lr 5e-7 constant after a 100-step warmup, AdamW eps 1e-15, weight decay .01, 7,400 steps",
          "color": C_C, "marker": "v", "ls": "-.", "lw": 1.9, "ms": 7.0,
          "lr": 5e-7, "warmup": 100, "groups_per_step": 2048, "group_size": 8, "total_steps": 7400, "reward_window_last": 5, "weight_decay": 0.01, "adam_eps": 1e-15,
          "save_steps": [25, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000] + list(range(1250, 7400, 250)) + [7400]},
}
NEW_ARMS = ["a", "b", "c"]
RUNS = {
    "this_train": ("id", ARMS["this"]["id"], ARMS["this"]["name"]),
    "ref_train": ("name", "rl_abl_initnewfft_fullparam_8x512", None),
    "ref_eval": ("name", "rl_abl_initnewfft_fullparam_8x512_eval", None),
    "mid_train": ("id", "eaozkq7o", "mix5msft_midtrain_fft_from_fft104m"),
    "mid_eval": ("name", "mix5msft_midtrain_fft_from_fft104m_eval", None),
    "pre_train": ("id", "jp82mr9a", "realact104m_fullft_b4096_lr1e-5"),
}
for _k in NEW_ARMS:
    RUNS[_k + "_train"] = ("id", ARMS[_k]["id"], ARMS[_k]["name"])
SAVE_STEPS = [25, 50, 100, 150, 200, 250, 300]
REF_STEP = 300                     # the reference number the headline is judged against is the reference arm's step-300 eval (.424)
NOISE_EPS = 0.002                  # eval-noise floor: duplicate-checkpoint deltas from the pretrain report (max |delta| .0023)
TABLE_EPS = 0.004                  # this-vs-reference table colouring
RL_LR = 1e-6
LORA_WALL_LR_STEPS = 2.4e-3        # LoRA arms' drift-wall onset in cumulative lr x steps (report maemm-rl-lr-level)
LOG_STEP_AXIS_FROM = 900           # once any arm passes this many steps, step axes go log (arm C runs to 7,400)
BUDGET_TOL = 0.40                  # budget hypothesis: onsets "match" if the cumulative-lr ratio C/A is within +-40%

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

FAM_PLAIN = {"eval/sae/norm_act": "SAE features", "eval/mlp/norm_act": "MLP neurons", "eval/realact/cos": "short-context real activations",
             "eval/realact_long/cos": "long-context real activations", "eval/bsf/cos": "BSF blocks", "eval/cluster/cos": "cluster probes", "eval/jlens/cos": "J-lens", "eval/mean_all": "mean"}
FAMILY_PANELS = [("eval/sae/norm_act", "SAE features: normalised peak activation"), ("eval/mlp/norm_act", "MLP neurons: fire-back (normalised activation)"),
                 ("eval/realact/cos", "real activations, short context (cosine)"), ("eval/realact_long/cos", "real activations, long context (cosine) — RL-only family"),
                 ("eval/bsf/cos", "BSF subspace blocks (cosine)"), ("eval/cluster/cos", "linear cluster probes (cosine)")]
TABLE_KEYS = [("eval/mean_all", "mean fidelity"), ("eval/sae/norm_act", "SAE"), ("eval/mlp/norm_act", "MLP"), ("eval/realact/cos", "real acts"),
              ("eval/realact_long/cos", "long-ctx acts"), ("eval/bsf/cos", "BSF"), ("eval/cluster/cos", "probes"), ("eval/jlens/cos", "J-lens"), ("eval/random/cos", "random")]
MATCH_KEYS = [("eval/mean_all", "mean"), ("eval/sae/norm_act", "SAE"), ("eval/mlp/norm_act", "MLP"), ("eval/realact/cos", "real acts"), ("eval/realact_long/cos", "long-ctx acts"),
              ("eval/bsf/cos", "BSF"), ("eval/cluster/cos", "probes")]
DYN_KEYS = ["reward/mean", "policy/entropy", "grad_norm", "policy/sampler_abs_dlogp", "rollout/len_mean", "time/step_s", "reward/peak_last_frac",
            "reward/peak_dist_mean", "reward/peak_in_last5_frac", "lr", "ratio/clipfrac", "reward/max", "policy/offpolicy_lag_steps", "mem/hf_peak_gb"]
DYN_MA_KEYS = ["reward/mean", "policy/entropy", "grad_norm", "policy/sampler_abs_dlogp", "rollout/len_mean", "time/step_s", "reward/peak_last_frac",
               "reward/peak_dist_mean", "reward/peak_in_last5_frac"]
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


def eval_rows(runs):
    """Latest row per ckpt_step across every eval run with that name (the evaluator daemon can be respawned -> several runs; re-evals overwrite);
    eval/* + extra/* keys only."""
    if not isinstance(runs, (list, tuple)):
        runs = [runs]
    rows = {}
    for run in sorted(runs, key=lambda r: str(r.created_at)):
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


def _run_meta(r):
    return {"id": r.id, "name": r.name, "state": r.state, "last_step": r.lastHistoryStep, "created_at": str(r.created_at), "url": r.url}


def _parse_save_steps(v, fallback):
    if isinstance(v, str):
        try:
            return sorted({int(x) for x in v.split(",") if x.strip()})
        except ValueError:
            return fallback
    if isinstance(v, (list, tuple)):
        return sorted({int(x) for x in v})
    return fallback


def fetch():
    api = wandb.Api(timeout=180)
    D = {"fetched_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "runs": {}}
    R = {k: _get_run(api, k) for k in RUNS}
    for k, r in R.items():
        D["runs"][k] = _run_meta(r)
    # the four arms from the re-cut init: train run by id, eval runs = every run named <run>_eval
    D["arms"] = {}
    for k in ARM_ORDER:
        r = R[k + "_train"]
        ev_runs = list(api.runs(PROJ, filters={"display_name": ARMS[k]["name"] + "_eval"}, order="-created_at"))
        D["runs"][k + "_eval"] = _run_meta(ev_runs[0]) if ev_runs else {"id": None, "name": ARMS[k]["name"] + "_eval", "state": "missing", "last_step": None, "created_at": None, "url": None}
        D["runs"][k + "_eval"]["n_runs"] = len(ev_runs)
        cfg = {kk: v for kk, v in r.config.items() if not isinstance(v, (dict, list))}
        arm = dict(ARMS[k])
        for ck, ak in (("lr", "lr"), ("warmup_steps", "warmup"), ("groups_per_step", "groups_per_step"), ("group_size", "group_size"), ("total_steps", "total_steps"),
                       ("reward_window_last", "reward_window_last"), ("weight_decay", "weight_decay"), ("adam_eps", "adam_eps")):
            if _num(cfg.get(ck)):
                arm[ak] = cfg[ck]
        arm["save_steps"] = _parse_save_steps(cfg.get("save_steps"), ARMS[k]["save_steps"])
        arm["rollouts_per_step"] = int(arm["group_size"] * arm["groups_per_step"])
        D["arms"][k] = {"meta": arm, "evals": eval_rows(ev_runs), "train": train_series(r, DYN_KEYS), "config": cfg}
    D["this"] = D["arms"]["this"]
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


def interp_at(rows, step, key="eval/mean_all", init=None):
    """Linear interpolation of an eval curve at an arbitrary step (init at step 0 if given); None outside the evaluated range."""
    xs, ys = ([0.0], [init]) if _num(init) else ([], [])
    for r in rows:
        if _num(r.get(key)):
            xs.append(float(r["ckpt_step"])); ys.append(float(r[key]))
    if not xs or step < xs[0] or step > xs[-1]:
        return None
    return float(np.interp(step, xs, ys))


def win_mean(tr, key, a, b):
    ys = [v for s, v in zip(tr["step"], tr.get(key, [])) if a <= s <= b and _num(v)]
    return float(np.mean(ys)) if ys else None


def lr_at(arm, t):
    """lr applied at update t (0-indexed): linear warmup to the peak over `warmup` updates, then constant (lr_decay none)."""
    return arm["lr"] * min(1.0, (t + 1) / max(1, arm["warmup"]))


def cum_lr(arm, s):
    """Cumulative lr x steps through update s inclusive (sum of lr_t for t = 0..s). For a checkpoint saved after s updates use cum_lr(arm, s-1)."""
    if s is None or s < 0:
        return 0.0
    w, n = max(1, arm["warmup"]), s + 1
    if n <= w:
        return arm["lr"] * n * (n + 1) / (2 * w)
    return arm["lr"] * ((w + 1) / 2 + (n - w))


def cum_lr_ckpt(arm, ckpt_step):
    return cum_lr(arm, ckpt_step - 1)


def step_for_cum_lr(arm, target):
    """Inverse of cum_lr: the update index s at which the cumulative lr first reaches `target` (float)."""
    w = max(1, arm["warmup"]); full_w = arm["lr"] * (w + 1) / 2
    if target <= full_w:
        # n(n+1)/2 * lr/w = target
        n = (-1 + math.sqrt(1 + 8 * target * w / arm["lr"])) / 2
        return max(0.0, n - 1)
    return w + (target - full_w) / arm["lr"] - 1


def onsets(tr, arm=None, running=False):
    """Drift-wall onsets by the two rules used since the first version of this report (unchanged): grad norm > 1 on 2 consecutive steps (step >= 100),
    grad norm > 1 'for good' (= the step after the last step at or below 1), sampler |dlogp| > .05 first / 3-consecutive / for good (step >= 100).
    lr_steps = lr_peak x step (the simple product used in the earlier text); cum_lr = the lr summed over the updates incl. the warmup ramp."""
    S, g, dl = tr["step"], tr["grad_norm"], tr["policy/sampler_abs_dlogp"]
    o = {"rule_gnorm": "first of 2 consecutive steps with grad_norm > 1.0, step >= 100", "rule_dlogp": "first step with sampler |delta log p| > 0.05, step >= 100",
         "gnorm_gt1_2consec": None, "gnorm_gt1_for_good": None, "dlogp_gt05_first": None, "dlogp_gt05_3consec": None, "dlogp_gt05_for_good": None,
         "provisional": bool(running), "last_step": S[-1] if S else None}
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
    lr_peak = arm["lr"] if arm else RL_LR
    for k in ("gnorm_gt1_2consec", "gnorm_gt1_for_good", "dlogp_gt05_first", "dlogp_gt05_3consec"):
        o[k + "_lr_steps"] = None if o[k] is None else lr_peak * o[k]
        o[k + "_cum_lr"] = None if (o[k] is None or arm is None) else cum_lr(arm, o[k])
    if arm is not None and S:
        o["cum_lr_at_last_step"] = cum_lr(arm, S[-1])
        o["dlogp_last10"] = win_mean(tr, "policy/sampler_abs_dlogp", S[-1] - 9, S[-1])
        o["gnorm_last10"] = win_mean(tr, "grad_norm", S[-1] - 9, S[-1])
    return o


def dyn_summary(tr):
    if not tr["step"]:
        return {}
    last_s = tr["step"][-1]
    def w(k, a, b):
        return win_mean(tr, k, a, b)
    return {"reward_w20_30": w("reward/mean", 20, 30), "reward_w145_155": w("reward/mean", 145, 155), "reward_last10": w("reward/mean", last_s - 9, last_s),
            "reward_min_from150": min((v for s, v in zip(tr["step"], tr["reward/mean"]) if s >= 150 and _num(v)), default=None),
            "reward_max_from150": max((v for s, v in zip(tr["step"], tr["reward/mean"]) if s >= 150 and _num(v)), default=None),
            "entropy_w20_30": w("policy/entropy", 20, 30), "entropy_last10": w("policy/entropy", last_s - 9, last_s),
            "entropy_min": min((v for v in tr["policy/entropy"] if _num(v)), default=None),
            "gnorm_w20_30": w("grad_norm", 20, 30), "gnorm_last10": w("grad_norm", last_s - 9, last_s),
            "dlogp_w20_30": w("policy/sampler_abs_dlogp", 20, 30), "dlogp_last10": w("policy/sampler_abs_dlogp", last_s - 9, last_s),
            "len_w20_30": w("rollout/len_mean", 20, 30), "len_last10": w("rollout/len_mean", last_s - 9, last_s),
            "len_max": max((v for v in tr["rollout/len_mean"] if _num(v)), default=None), "len_mean_from50": win_mean(tr, "rollout/len_mean", 50, last_s),
            "step_s_median_from30": float(np.median([v for s, v in zip(tr["step"], tr["time/step_s"]) if s >= 30 and _num(v)])) if any(_num(v) for v in tr["time/step_s"]) else None,
            "clipfrac_last10": w("ratio/clipfrac", last_s - 9, last_s), "mem_peak_gb_max": max((v for v in tr["mem/hf_peak_gb"] if _num(v)), default=None),
            "lr_logged_max": max((v for v in tr["lr"] if _num(v)), default=None), "last_step": last_s}


def peak_summary(tr):
    if not tr["step"]:
        return {}
    last_s = tr["step"][-1]
    def w(k, a, b):
        return win_mean(tr, k, a, b)
    def first(k):
        return next((v for v in tr[k] if _num(v)), None)
    return {"peak_last_frac": {"step0": first("reward/peak_last_frac"), "w20_30": w("reward/peak_last_frac", 20, 30), "w145_155": w("reward/peak_last_frac", 145, 155),
                               "w195_205": w("reward/peak_last_frac", 195, 205), "last10": w("reward/peak_last_frac", last_s - 9, last_s),
                               "at25": tr["reward/peak_last_frac"][tr["step"].index(25)] if 25 in tr["step"] else None,
                               "at275": tr["reward/peak_last_frac"][tr["step"].index(275)] if 275 in tr["step"] else None,
                               "min": min((v for v in tr["reward/peak_last_frac"] if _num(v)), default=None),
                               "mean_from50": win_mean(tr, "reward/peak_last_frac", 50, last_s)},
            "peak_dist_mean": {"step0": first("reward/peak_dist_mean"), "w20_30": w("reward/peak_dist_mean", 20, 30), "w145_155": w("reward/peak_dist_mean", 145, 155),
                               "last10": w("reward/peak_dist_mean", last_s - 9, last_s), "mean_from50": win_mean(tr, "reward/peak_dist_mean", 50, last_s),
                               "min_from50": min((v for s, v in zip(tr["step"], tr["reward/peak_dist_mean"]) if s >= 50 and _num(v)), default=None),
                               "max_from50": max((v for s, v in zip(tr["step"], tr["reward/peak_dist_mean"]) if s >= 50 and _num(v)), default=None)},
            "peak_in_last5_frac": {"step0": first("reward/peak_in_last5_frac"), "w20_30": w("reward/peak_in_last5_frac", 20, 30), "last10": w("reward/peak_in_last5_frac", last_s - 9, last_s),
                                   "min": min((v for v in tr["reward/peak_in_last5_frac"] if _num(v)), default=None),
                                   "max": max((v for v in tr["reward/peak_in_last5_frac"] if _num(v)), default=None),
                                   "note": "identically 1 under --reward-window-last 5 (the reward IS the max over the last 5 tokens); informative only under the all-token reward"},
            "definitions": "peak_dist = distance (tokens) of the reward argmax from the last kept token, per rollout; peak_last_frac = P(peak_dist == 0); "
                           "peak_in_last5_frac = P(peak_dist <= 4); peak_dist_mean = mean over the batch (rl/rl_disagg.py:2891-2893)"}


def arm_status(D, k):
    """(state, last_step, pending_steps): pending = checkpoints already saved (<= last train step) whose eval has not landed."""
    run, arm, ev = D["runs"][k + "_train"], D["arms"][k]["meta"], D["arms"][k]["evals"]
    last = run["last_step"] or 0
    evaluated = {r["ckpt_step"] for r in ev}
    pending = [s for s in arm["save_steps"] if s <= last + 1 and s not in evaluated]
    return run["state"], last, pending


def arm_label(D, k, long=True):
    arm = D["arms"][k]["meta"]
    st, last, pending = arm_status(D, k)
    base = arm["long"] if long else arm["nick"]
    if st != "finished":
        tot = f" of {arm['total_steps']:,}" if arm.get("total_steps") else ""
        return base + f" (running, step {last:,}{tot})"
    if pending:
        return base + f" (eval pending for step{'s' if len(pending) > 1 else ''} {', '.join(map(str, pending))})"
    return base


def this_label(D, M):
    return arm_label(D, "this")


REF_LABEL = "reference: same full-parameter RL on the old-bank init (23M-ckpt pretrain + FFT midtrain, targets not end-anchored)"


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
    o = onsets(tr, D["this"]["meta"], running=D["runs"]["this_train"]["state"] != "finished")
    peak = peak_summary(tr)
    dyn = dyn_summary(tr)
    rtr = D["ref"]["train"]; rl = rtr["step"][-1] if rtr["step"] else None
    ref_dyn = {"reward_last10": win_mean(rtr, "reward/mean", rl - 9, rl) if rl else None, "entropy_last10": win_mean(rtr, "policy/entropy", rl - 9, rl) if rl else None,
               "gnorm_last10": win_mean(rtr, "grad_norm", rl - 9, rl) if rl else None, "dlogp_last10": win_mean(rtr, "policy/sampler_abs_dlogp", rl - 9, rl) if rl else None,
               "onsets": onsets(rtr, D["this"]["meta"]) if rtr["step"] else None}
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


def summarize_arms(D, M):
    """Everything about the three follow-up arms vs the 8x512 last-5 arm: eval tables with cumulative rollouts + cumulative lr, onsets by both rules,
    matched-step and matched-rollout deltas, smearing numbers, the budget-hypothesis check (arm C vs arm A vs the .425 arm)."""
    init = M["init_mean_all"]
    base = D["arms"]["this"]
    N = {"arms": {}, "base_arm": "this", "init_mean_all": init, "noise_eps": NOISE_EPS, "table_eps": TABLE_EPS,
         "lr_schedule": "lr_t = lr_peak * min(1, (t+1)/warmup) for update t (0-indexed), constant afterwards (lr_decay none); cum_lr(s) = sum_{t<=s} lr_t; "
                        "checkpoints saved after s updates use cum_lr(s-1). Verified against the logged lr series per arm: see lr_logged_check.",
         "lr_logged_check": {},
         "rollouts_rule": "cumulative rollouts at checkpoint s = s x group_size x groups_per_step (4,096/step for the 8x512 arms, 16,384/step for the 8x2048 arms)"}
    for k in ARM_ORDER:
        A = D["arms"][k]; arm = A["meta"]; st, last, pending = arm_status(D, k)
        evs = []
        for r in A["evals"]:
            s = r["ckpt_step"]
            evs.append({"ckpt_step": s, "cum_rollouts": s * arm["rollouts_per_step"], "cum_lr": cum_lr_ckpt(arm, s), "lr_x_step": arm["lr"] * s,
                        **{kk: r.get(kk) for kk, _ in TABLE_KEYS}})
        best = max(A["evals"], key=lambda r: r["eval/mean_all"]) if A["evals"] else None
        o = onsets(A["train"], arm, running=st != "finished") if A["train"]["step"] else None
        tr = A["train"]
        if tr["step"] and any(_num(v) for v in tr["lr"]):
            logged = np.cumsum([v if _num(v) else 0.0 for v in tr["lr"]]); analytic = np.array([cum_lr(arm, s_) for s_ in tr["step"]])
            with np.errstate(invalid="ignore", divide="ignore"):
                rel = np.abs(logged - analytic) / np.where(analytic > 0, analytic, np.nan)
            N["lr_logged_check"][k] = {"n_steps": len(tr["step"]), "n_lr_missing": sum(not _num(v) for v in tr["lr"]), "lr_logged_max": float(np.nanmax([v for v in tr["lr"] if _num(v)])),
                                       "max_rel_dev_cumsum": float(np.nanmax(rel)) if np.isfinite(rel).any() else None,
                                       "logged_lr_first3": [v for v in tr["lr"][:3]], "logged_cumsum_last": float(logged[-1]), "analytic_last": float(analytic[-1])}
        N["arms"][k] = {"key": k, "nick": arm["nick"], "long": arm["long"], "label": arm_label(D, k), "wandb_id": arm["id"], "run_name": arm["name"],
                        "state": st, "last_step": last, "total_steps": arm["total_steps"], "pending_steps": pending, "finished": st == "finished",
                        "all_evals_in": st == "finished" and not pending,
                        "config": {kk: arm.get(kk) for kk in ("lr", "warmup", "groups_per_step", "group_size", "rollouts_per_step", "total_steps", "reward_window_last", "weight_decay", "adam_eps", "save_steps")},
                        "evals": evs, "best": None if best is None else {"ckpt_step": best["ckpt_step"], "mean_all": best["eval/mean_all"]},
                        "last_eval": None if not evs else {"ckpt_step": evs[-1]["ckpt_step"], "mean_all": evs[-1]["eval/mean_all"]},
                        "gain_over_init": None if best is None or init is None else best["eval/mean_all"] - init,
                        "onsets": o, "dynamics": dyn_summary(A["train"]), "peak_metrics": peak_summary(A["train"]),
                        "eval_runs": D["runs"][k + "_eval"], "train_run": D["runs"][k + "_train"]}
    # matched steps vs the 8x512 last-5 arm
    steps_all = sorted({r["ckpt_step"] for k in ARM_ORDER for r in D["arms"][k]["evals"]})
    N["matched_steps_table"] = []
    for s in steps_all:
        row = {"ckpt_step": s, "values": {}, "delta_vs_base": {}}
        for k in ARM_ORDER:
            row["values"][k] = {kk: at(D["arms"][k]["evals"], s, kk) for kk, _ in MATCH_KEYS}
            if k != "this":
                row["delta_vs_base"][k] = {kk: (row["values"][k][kk] - row["values"]["this"][kk]) if (row["values"][k][kk] is not None and row["values"]["this"][kk] is not None) else None
                                           for kk, _ in MATCH_KEYS}
        N["matched_steps_table"].append(row)
    N["matched_steps_delta_mean"] = {k: {str(r["ckpt_step"]): r["delta_vs_base"][k]["eval/mean_all"] for r in N["matched_steps_table"] if r["delta_vs_base"].get(k, {}).get("eval/mean_all") is not None}
                                     for k in NEW_ARMS}
    # matched rollouts: for every checkpoint of every arm, the 8x512 arm's value at the same cumulative rollouts (exact when 4s is an evaluated step, else interpolated)
    base_rps = base["meta"]["rollouts_per_step"]
    mr = []
    for k in NEW_ARMS:
        arm = D["arms"][k]["meta"]
        for r in D["arms"][k]["evals"]:
            s = r["ckpt_step"]; roll = s * arm["rollouts_per_step"]; bs = roll / base_rps
            exact = at(base["evals"], int(bs)) if float(bs).is_integer() else None
            bval = exact if exact is not None else interp_at(base["evals"], bs, init=init)
            mr.append({"arm": k, "nick": arm["nick"], "ckpt_step": s, "cum_rollouts": roll, "mean_all": r["eval/mean_all"], "base_equiv_step": bs,
                       "base_mean_all": bval, "base_interpolated": exact is None and bval is not None,
                       "delta_vs_base": None if bval is None else r["eval/mean_all"] - bval})
        # and the reverse: where the base arm's LAST checkpoint sits on this arm's rollout axis
        if base["evals"] and arm["rollouts_per_step"] != base_rps:
            bl = base["evals"][-1]; roll = bl["ckpt_step"] * base_rps; ks = roll / arm["rollouts_per_step"]
            kval = at(D["arms"][k]["evals"], int(ks)) if float(ks).is_integer() else None
            kv = kval if kval is not None else interp_at(D["arms"][k]["evals"], ks, init=init)
            mr.append({"arm": k, "nick": arm["nick"], "ckpt_step": ks, "cum_rollouts": roll, "mean_all": kv, "arm_interpolated": kval is None and kv is not None,
                       "base_equiv_step": bl["ckpt_step"], "base_mean_all": bl["eval/mean_all"], "base_interpolated": False,
                       "delta_vs_base": None if kv is None else kv - bl["eval/mean_all"], "reverse": True})
    N["matched_rollouts"] = mr
    # best overall
    cands = [(v["best"]["mean_all"], k, v["best"]["ckpt_step"]) for k, v in N["arms"].items() if v["best"]]
    if cands:
        top = max(cands)
        ties = [(k, s, m) for m, k, s in cands if abs(m - top[0]) <= NOISE_EPS and (k, s) != (top[1], top[2])]
        N["best_overall"] = {"arm": top[1], "nick": N["arms"][top[1]]["nick"], "ckpt_step": top[2], "mean_all": top[0],
                             "ties_within_noise": [{"arm": k, "nick": N["arms"][k]["nick"], "ckpt_step": s, "mean_all": m} for k, s, m in ties],
                             "delta_vs_base_final": None if not base["evals"] else top[0] - base["evals"][-1]["eval/mean_all"],
                             "base_final": None if not base["evals"] else {"ckpt_step": base["evals"][-1]["ckpt_step"], "mean_all": base["evals"][-1]["eval/mean_all"]}}
    else:
        N["best_overall"] = None
    # smearing: arm B vs the last-5 arms
    N["smearing"] = {k: {"peak_metrics": N["arms"][k]["peak_metrics"], "len": {kk: N["arms"][k]["dynamics"].get(kk) for kk in ("len_w20_30", "len_last10", "len_max", "len_mean_from50")},
                         "reward_window_last": N["arms"][k]["config"]["reward_window_last"]} for k in ARM_ORDER}
    # budget hypothesis: the drift wall sits at a fixed cumulative lr x steps -> arm C (half the lr) should cross at ~2x arm A's step count
    N["budget_hypothesis"] = budget_check(N)
    return N


def budget_check(N):
    A, C, T = N["arms"]["a"], N["arms"]["c"], N["arms"]["this"]
    armA, armC = {"lr": A["config"]["lr"], "warmup": A["config"]["warmup"]}, {"lr": C["config"]["lr"], "warmup": C["config"]["warmup"]}
    out = {"tolerance": BUDGET_TOL, "rules": {}, "c_cum_lr_now": (C["onsets"] or {}).get("cum_lr_at_last_step"), "c_last_step": C["last_step"],
           "c_dlogp_last10": (C["onsets"] or {}).get("dlogp_last10"), "c_gnorm_last10": (C["onsets"] or {}).get("gnorm_last10"),
           "lr_ratio_a_over_c": armA["lr"] / armC["lr"] if armC["lr"] else None}
    rules = [("dlogp_gt05_first", "sampler |dlogp| > .05, first step >= 100"), ("dlogp_gt05_3consec", "sampler |dlogp| > .05, first of 3 consecutive"),
             ("gnorm_gt1_2consec", "grad norm > 1, first of 2 consecutive steps >= 100"), ("gnorm_gt1_for_good", "grad norm > 1 for good (step after the last <= 1)")]
    verdicts = []
    for key, desc in rules:
        r = {"rule": desc}
        for tag, arm in (("this", T), ("a", A), ("c", C)):
            o = arm["onsets"] or {}
            r[tag] = {"step": o.get(key), "cum_lr": o.get(key + "_cum_lr"), "lr_x_step": o.get(key + "_lr_steps"), "provisional": o.get("provisional", False)}
        a_cum, c_cum = r["a"]["cum_lr"], r["c"]["cum_lr"]
        if a_cum:
            r["c_predicted_step"] = round(step_for_cum_lr(armC, a_cum))
            r["c_predicted_step_ratio_vs_a"] = r["c_predicted_step"] / r["a"]["step"] if r["a"]["step"] else None
        if a_cum and c_cum:
            ratio = c_cum / a_cum; r["c_over_a_cum_lr"] = ratio; r["c_over_a_steps"] = r["c"]["step"] / r["a"]["step"]
            r["verdict"] = "match" if abs(ratio - 1) <= BUDGET_TOL else ("C later than the budget predicts" if ratio > 1 else "C earlier than the budget predicts")
        elif a_cum and out["c_cum_lr_now"] is not None:
            now = out["c_cum_lr_now"]; r["c_over_a_cum_lr_so_far"] = now / a_cum
            if now < a_cum * (1 - BUDGET_TOL):
                r["verdict"] = "pending: C has not yet reached A's budget"
            elif now <= a_cum * (1 + BUDGET_TOL):
                r["verdict"] = "pending, inside the window: C is at A's budget without crossing yet"
            else:
                r["verdict"] = "strained: C has passed A's budget by more than the tolerance without crossing"
        else:
            r["verdict"] = "pending: no onset for A yet"
        out["rules"][key] = r
        verdicts.append(r["verdict"])
    out["summary"] = "; ".join(f"{k}: {v['verdict']}" for k, v in out["rules"].items())
    out["any_match"] = any(v == "match" for v in verdicts); out["any_strained"] = any(v.startswith("strained") or "than the budget" in v for v in verdicts)
    return out


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


def sci(v):
    """1.23e-4 -> '1.2e-4' (short scientific for lr x steps)."""
    if v is None:
        return "n/a"
    e = int(math.floor(math.log10(abs(v)))) if v else 0
    return f"{v / 10 ** e:.1f}e{e}"


def xy(rows, key, init=None, xmin=None):
    xs, ys = ([0], [init]) if _num(init) else ([], [])
    for r in rows:
        if _num(r.get(key)):
            xs.append(r["ckpt_step"]); ys.append(r[key])
    if xmin is not None:
        keep = [i for i, x in enumerate(xs) if x >= xmin]
        xs, ys = [xs[i] for i in keep], [ys[i] for i in keep]
    return xs, ys


def step_axis(ax, xmax, pad=5, allow_zero=True):
    """Linear step axis while every arm is short; log once arm C runs long (LOG_STEP_AXIS_FROM). Returns True if log (step-0 points cannot be drawn)."""
    if xmax <= LOG_STEP_AXIS_FROM:
        hi = max(300, xmax)
        ax.set_xlim(-6 if allow_zero else 0, hi + pad + (hi - 300) * 0.02)
        ax.set_xticks([0, 50, 100, 150, 200, 250, 300] if hi <= 330 else list(range(0, int(hi) + 1, 100)))
        return False
    ax.set_xscale("log"); ax.set_xlim(9, xmax * 1.12)
    ticks = [t for t in (10, 25, 50, 100, 200, 300, 500, 1000, 2000, 3000, 5000, 7400) if t <= xmax * 1.05]
    ax.set_xticks(ticks); ax.set_xticklabels([f"{t:,}" for t in ticks]); ax.set_xticks([], minor=True)
    return True


def spread(ys, gap, lo=None, hi=None):
    """Nudge label y-positions apart (keeps order, moves both ways, stays inside [lo, hi]) so end labels do not overprint."""
    idx = sorted(range(len(ys)), key=lambda i: ys[i]); out = list(ys)
    for _ in range(60):
        moved = False
        for a, b in zip(idx, idx[1:]):
            d = out[b] - out[a]
            if d < gap - 1e-12:
                sh = (gap - d) / 2; out[a] -= sh; out[b] += sh; moved = True
        if lo is not None and out[idx[0]] < lo:
            k = lo - out[idx[0]]; out = [v + k for v in out]; moved = True
        if hi is not None and out[idx[-1]] > hi:
            k = out[idx[-1]] - hi; out = [v - k for v in out]; moved = True
        if not moved:
            break
    return out


def plot_arm(ax, D, k, key="eval/mean_all", init=None, label=None, xmin=None, lw_scale=1.0, ms_scale=1.0, zorder=4, ls=None, x_transform=None, max_step=None):
    arm = D["arms"][k]["meta"]
    rows = D["arms"][k]["evals"] if max_step is None else [r for r in D["arms"][k]["evals"] if r["ckpt_step"] <= max_step]
    xs, ys = xy(rows, key, init, xmin)
    if x_transform is not None:
        xs = [x_transform(x) for x in xs]
    if not xs:
        return xs, ys
    ax.plot(xs, ys, color=arm["color"], lw=arm["lw"] * lw_scale, ls=ls or arm["ls"], marker=arm["marker"], ms=arm["ms"] * ms_scale, mec="white", mew=1.1,
            label=label, zorder=zorder, solid_capstyle="round")
    return xs, ys


def headline_claim(M, N):
    T, A, B, C = (N["arms"][k] for k in ARM_ORDER)
    bo = N["best_overall"]
    tf = M["last"]["mean_all"] if M["last"] else None
    bits = []
    if bo and tf is not None:
        d = bo["mean_all"] - tf
        tie = ", ".join(f"{t['nick']} @{t['ckpt_step']}" for t in bo["ties_within_noise"])
        bits.append(f"From one re-cut-bank SFT init, the best full-parameter RL checkpoint is {bo['mean_all']:.3f} ({bo['nick']} @{bo['ckpt_step']}"
                    + (f"; {tie} ties within the ±{NOISE_EPS:.3f} eval noise" if tie else "") + f") — {sgn(d)} over the 8 x 512 last-5 arm's {tf:.3f} @{M['last']['ckpt_step']}"
                    + (", past the eval noise" if abs(d) > NOISE_EPS else ", within the eval noise"))
    if A["last_eval"] and T["evals"]:
        ms = N["matched_steps_delta_mean"]["a"]
        mr = [r for r in N["matched_rollouts"] if r["arm"] == "a" and not r.get("reverse") and not r["base_interpolated"] and r["delta_vs_base"] is not None]
        bits.append(f"4x the rollouts per step is ahead at every matched step ({', '.join(sgn(v) for v in ms.values())})"
                    + (f" but behind at matched rollouts ({', '.join(sgn(r['delta_vs_base']) for r in mr)})" if mr else ""))
    if B["last_eval"]:
        bits.append(f"the all-token reward finishes at {B['last_eval']['mean_all']:.3f} on the unchanged last-5 evaluator despite smearing the reward peak "
                    f"{fmt(B['peak_metrics']['peak_dist_mean']['mean_from50'], 0)} tokens into the rollout")
    if C["last_eval"]:
        bits.append(f"the paper optimizer (lr 5e-7) trails at matched steps ({C['last_eval']['mean_all']:.3f} @{C['last_eval']['ckpt_step']}"
                    + (f", running, step {C['last_step']:,} of {C['total_steps']:,}" if not C["finished"] else "") + ")")
    return "; ".join(bits) if bits else "Held-out fidelity vs RL step for the re-cut-init arms (evals pending)"


def plot_headline(D, M, ABL, N):
    c_evals = D["arms"]["c"]["evals"]
    c_beyond = [r for r in c_evals if r["ckpt_step"] > 300]
    if c_beyond:
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.8), gridspec_kw={"width_ratios": [3, 2]}, sharey=True)
    else:
        fig, ax = plt.subplots(figsize=(11.5, 6.8)); ax2 = None
    ref = D["ref"]["evals"]
    first = True
    for a, v in ABL["arms"].items():
        xs, ys = xy(v["evals"], "eval/mean_all", v["before_rl"])
        ax.plot(xs, ys, color=LORA_C, lw=1.0, marker="o", ms=2.8, zorder=1, label="context: the four LoRA-RL arms of the SFT-init ablation (lr 7e-6, same RL bank, different SFT inits)" if first else None)
        first = False
    xs, ys = xy(ref, "eval/mean_all", M["ref_init_mean_all"])
    ax.plot(xs, ys, color=REF_C, lw=1.8, ls="--", marker="s", ms=6, mec="white", mew=1.1, label=REF_LABEL, zorder=3)
    ends = []
    for k in ARM_ORDER:
        xs, ys = plot_arm(ax, D, k, init=M["init_mean_all"], label=arm_label(D, k), max_step=300, zorder=5 if k == "this" else 4)
        if len(xs) > 1:
            ends.append((k, xs[-1], ys[-1]))
    # reference lines: the 8x512 last-5 arm's final (with the eval-noise band) and the best checkpoint of any arm
    tf = M["last"]["mean_all"] if M["last"] else None
    if tf is not None:
        ax.hlines(tf, -6, 300, color=THIS_C, lw=1, ls=(0, (4, 3)), zorder=0)
        ax.fill_between([-6, 300], tf - NOISE_EPS, tf + NOISE_EPS, color=THIS_C, alpha=0.07, lw=0, zorder=0)
    bo = N["best_overall"]
    if bo and tf is not None and bo["mean_all"] > tf + 1e-9:
        ax.hlines(bo["mean_all"], -6, 300, color=INK2, lw=1, ls=(0, (2, 3)), zorder=0)
    # right-margin end labels (the two dotted reference lines are labelled through the arms that define them), spread so they never overprint
    labs = []
    for k, x, v in ends:
        arm = D["arms"][k]["meta"]; st, _, _ = arm_status(D, k)
        extra = f" (band = eval noise ±{NOISE_EPS:.3f})" if k == "this" else (" — best of any arm" if bo and bo["arm"] == k and bo["ckpt_step"] == x else "") + (" (running)" if st != "finished" else "")
        labs.append((f"{v:.3f} @{x} · {arm['tiny']}{extra}", x, v, arm["color"]))
    if M["ref_at_300"] is not None:
        labs.append((f"{M['ref_at_300']:.3f} @300 · old-bank reference", 300, M["ref_at_300"], REF_C))
    ys_lab = spread([v for _, _, v, _ in labs], 0.0048, lo=0.336, hi=0.4425)
    for (txt, x, v, col), yl in zip(labs, ys_lab):
        ax.annotate(txt, xy=(x, v), xytext=(308, yl), textcoords="data", fontsize=8.1, color=INK2, va="center",
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.7, alpha=0.7, shrinkA=0, shrinkB=2))
    for k in ARM_ORDER:
        st, last, pending = arm_status(D, k)
        for s in pending:
            if s <= 300:
                ax.text(s, 0.334 + 0.004 * ARM_ORDER.index(k), f"eval @{s} pending ({D['arms'][k]['meta']['nick']})", fontsize=7.2, color=MUTED, ha="center", va="bottom")
    if M["init_mean_all"] is not None:
        ax.annotate(f"shared SFT init after midtrain: {M['init_mean_all']:.3f}", xy=(0, M["init_mean_all"]), xytext=(16, -14), textcoords="offset points", fontsize=8.3, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.8))
    if M["ref_init_mean_all"] is not None:
        ax.annotate(f"reference init: {M['ref_init_mean_all']:.3f}", xy=(0, M["ref_init_mean_all"]), xytext=(8, 14), textcoords="offset points", fontsize=8.3, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.8))
    ax.set_xlim(-6, 392); ax.set_ylim(0.33, 0.445); ax.set_xticks([0, 25, 50, 100, 150, 200, 250, 300])
    ax.set_xlabel("RL step (4,096 rollouts per step for the 8 x 512 arms, 16,384 for the 8 x 2048 arms; step 0 = the SFT init before any RL)")
    ax.set_ylabel("held-out fidelity: mean cosine over 10 direction families")
    style_ax(ax)
    if ax2 is not None:
        plot_arm(ax2, D, "c", init=None, label=None)
        for k in ("this", "a", "b"):
            e = D["arms"][k]["evals"]
            if e:
                ax2.axhline(e[-1]["eval/mean_all"], color=D["arms"][k]["meta"]["color"], lw=1, ls=(0, (4, 3)), zorder=0)
        c_last = c_evals[-1]
        ax2.annotate(f"{c_last['eval/mean_all']:.3f} @{c_last['ckpt_step']:,}", xy=(c_last["ckpt_step"], c_last["eval/mean_all"]), xytext=(6, -12), textcoords="offset points", fontsize=8.3, color=INK2)
        ax2.set_xlim(0, c_last["ckpt_step"] * 1.12); ax2.set_title(f"arm C continues: {arm_label(D, 'c', long=False)}; dotted = the other arms' final values", fontsize=9.4, loc="left")
        ax2.set_xlabel("RL step (arm C, 16,384 rollouts per step)"); style_ax(ax2)
    head = headline_claim(M, N)
    fig.suptitle(wrap(head, 165) + "\n" + wrap("Held-out mean fidelity vs RL step — Qwen3.6-27B activation-to-text inverter; full-parameter CISPO GRPO from the same SFT init "
                 "(104M pretrain + full-FT midtrain on the re-cut bank), same six-family RL bank; 512 held-out directions per family, best-of-4 at T=1. "
                 "Orange dashed = the same 8 x 512 recipe from the old-bank init.", 165), fontsize=10.2, x=0.01, ha="left", y=0.995)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=1, fontsize=8.4, bbox_to_anchor=(0.5, -0.005))
    finish(fig, bottom=0.2 if ax2 is None else 0.17)
    save(fig, "fidelity_vs_rl_step")


def _roll_ticks(ax, xmax):
    ticks = [t for t in (1e5, 2e5, 5e5, 1e6, 2e6, 5e6, 1e7, 2e7, 5e7, 1e8, 2e8) if 8e4 <= t <= xmax * 1.6]
    ax.set_xticks(ticks); ax.set_xticklabels([f"{t / 1e6:g}M" if t >= 1e6 else f"{t / 1e3:g}k" for t in ticks]); ax.set_xticks([], minor=True)


def plot_rollouts(D, M, N):
    fig, ax = plt.subplots(figsize=(11.5, 6.6))
    init = M["init_mean_all"]
    xmax = 0
    for k in ARM_ORDER:
        arm = D["arms"][k]["meta"]
        xs, ys = plot_arm(ax, D, k, init=None, label=arm_label(D, k), x_transform=lambda s, r=arm["rollouts_per_step"]: s * r, zorder=5 if k == "this" else 4)
        if xs:
            xmax = max(xmax, xs[-1])
            ax.annotate(f"{ys[-1]:.3f}", xy=(xs[-1], ys[-1]), xytext=(7, 0), textcoords="offset points", fontsize=8.2, color=INK2, va="center")
    if xmax == 0:
        plt.close(fig); return
    ax.set_xscale("log"); ax.set_xlim(8e4, xmax * 2.2); _roll_ticks(ax, xmax)
    if init is not None:
        ax.axhline(init, color=AXIS, lw=1, ls=":", zorder=0); ax.text(8.6e4, init + 0.001, f"shared SFT init {init:.3f}", fontsize=8.2, color=INK2, va="bottom")
    # exact matched-rollout pairs (arm at step s vs the 8x512 arm at step 4s)
    exact = [r for r in N["matched_rollouts"] if r["arm"] == "a" and not r.get("reverse") and r["delta_vs_base"] is not None and not r["base_interpolated"]]
    block = ["same rollouts, 8 x 2048 (4x batch) vs 8 x 512 last-5:"]
    for r in exact:
        ax.axvline(r["cum_rollouts"], color=GRID, lw=1, zorder=0)
        ax.annotate("", xy=(r["cum_rollouts"], r["mean_all"]), xytext=(r["cum_rollouts"], r["base_mean_all"]), arrowprops=dict(arrowstyle="-", color=INK2, lw=1.2, shrinkA=0, shrinkB=0))
        ax.text(r["cum_rollouts"] * 1.03, min(r["mean_all"], r["base_mean_all"]) - 0.002, f"{r['cum_rollouts'] / 1e3:,.0f}k: {sgn(r['delta_vs_base'])}", fontsize=8, color=INK2, va="top")
        block.append(f"{r['cum_rollouts'] / 1e3:,.0f}k rollouts: @{r['ckpt_step']} {r['mean_all']:.3f} vs @{int(r['base_equiv_step'])} {r['base_mean_all']:.3f} -> {sgn(r['delta_vs_base'])}")
    rev0 = next((r for r in N["matched_rollouts"] if r["arm"] == "a" and r.get("reverse") and r["delta_vs_base"] is not None), None)
    if rev0:
        block.append(f"{rev0['cum_rollouts'] / 1e6:.2f}M rollouts: @{rev0['ckpt_step']:g} ≈{rev0['mean_all']:.3f} (interpolated) vs @{rev0['base_equiv_step']} {rev0['base_mean_all']:.3f} -> {sgn(rev0['delta_vs_base'])}")
    ax.text(0.99, 0.03, "\n".join(block), transform=ax.transAxes, fontsize=8, color=INK2, ha="right", va="bottom", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRID, alpha=0.92))
    ax.set_ylim(0.33, 0.44)
    ax.set_xlabel("cumulative rollouts sampled (log; RL step x 8 gens x groups per step = 4,096/step for 8 x 512, 16,384/step for 8 x 2048)")
    ax.set_ylabel("held-out fidelity: mean cosine over 10 direction families")
    ex = exact
    rev = next((r for r in N["matched_rollouts"] if r["arm"] == "a" and r.get("reverse") and r["delta_vs_base"] is not None), None)
    head = ("Rollout for rollout, 4x the batch is LESS sample-efficient: at the same number of rollouts the 8 x 2048 arm sits "
            + (", ".join(f"{sgn(r['delta_vs_base'])} ({r['mean_all']:.3f} vs {r['base_mean_all']:.3f} at {r['cum_rollouts'] / 1e3:,.0f}k)" for r in ex) if ex else "n/a")
            + " below the 8 x 512 arm" + (f", and where the 8 x 512 arm ends ({rev['cum_rollouts'] / 1e6:.2f}M rollouts) the 4x arm is at ≈{rev['mean_all']:.3f} (interpolated)" if rev else "")
            + " — its per-step lead is bought with 4x the samples; the all-token-reward arm tracks the 8 x 512 arm rollout-for-rollout and ends higher")
    ax.set_title(wrap(head, 158) + "\n" + wrap("Held-out mean fidelity vs cumulative rollouts — the four full-parameter RL arms from the same re-cut-bank SFT init; "
                 "vertical connectors join checkpoints with identical rollout counts. Arm C's axis extends as its 7,400-step run proceeds.", 158), fontsize=10.2, loc="left", pad=10)
    style_ax(ax)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=1, fontsize=8.4, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    save(fig, "fidelity_vs_rollouts")


def plot_lr_steps(D, M, N):
    arms = ["this", "a", "c"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.5, 9.2), sharex=True, gridspec_kw={"height_ratios": [1.1, 1]})
    xmax = 1e-5
    for k in arms:
        arm = D["arms"][k]["meta"]
        xs, ys = plot_arm(ax1, D, k, init=None, label=arm_label(D, k), x_transform=lambda s, a=arm: cum_lr_ckpt(a, s), zorder=5 if k == "this" else 4)
        if xs:
            xmax = max(xmax, xs[-1])
            if k == "c":
                ax1.annotate(f"{ys[-1]:.3f} @{D['arms'][k]['evals'][-1]['ckpt_step']:,} (running)", xy=(xs[-1], ys[-1]), xytext=(-7, 9), textcoords="offset points", fontsize=8.2, color=INK2, ha="right", va="bottom")
            else:
                ax1.annotate(f"{ys[-1]:.3f}" + (" (running)" if arm_status(D, k)[0] != "finished" else ""), xy=(xs[-1], ys[-1]), xytext=(7, 0), textcoords="offset points", fontsize=8.2, color=INK2, va="center")
        tr = D["arms"][k]["train"]
        if tr["step"]:
            cx = np.array([cum_lr(arm, s) for s in tr["step"]]); yy = np.array([np.nan if v is None else v for v in tr["policy/sampler_abs_dlogp"]], dtype=float)
            ax2.plot(cx, yy, color=arm["color"], lw=0.7, alpha=0.22, zorder=2)
            ax2.plot(cx, rolling(tr["policy/sampler_abs_dlogp"]), color=arm["color"], lw=arm["lw"], ls=arm["ls"], zorder=3, solid_capstyle="round")
            xmax = max(xmax, cx[-1])
    if M["init_mean_all"] is not None:
        ax1.axhline(M["init_mean_all"], color=AXIS, lw=1, ls=":", zorder=0); ax1.text(1.05e-5, M["init_mean_all"] + 0.001, f"shared SFT init {M['init_mean_all']:.3f}", fontsize=8.2, color=INK2, va="bottom")
    ax2.axhline(0.05, color=INK2, lw=1, ls=":", zorder=1); ax2.text(1.05e-5, 0.0505, "|delta log p| = .05 (the drift-wall threshold)", fontsize=8, color=INK2, va="bottom")
    # the 8x512 arm's wall as a band (from its earliest to its latest onset), then per-arm onset markers by both rules
    bh = N["budget_hypothesis"]["rules"]
    t_on = [v["this"]["cum_lr"] for v in bh.values() if v["this"]["cum_lr"]]
    if t_on:
        for a in (ax1, ax2):
            a.axvspan(min(t_on), max(t_on), color=THIS_C, alpha=0.06, lw=0, zorder=0)
        ax1.text(math.sqrt(min(t_on) * max(t_on)), 0.3335, f"8 x 512 arm's wall\n{sci(min(t_on))} – {sci(max(t_on))}", fontsize=7.8, color=INK2, ha="center", va="bottom")
    lines = []
    for k in arms:
        arm = D["arms"][k]["meta"]; o = N["arms"][k]["onsets"] or {}
        d1, g1 = o.get("dlogp_gt05_first"), o.get("gnorm_gt1_for_good")
        if d1 is not None:
            ax2.axvline(o["dlogp_gt05_first_cum_lr"], color=arm["color"], lw=1.1, ls="--", alpha=0.85, zorder=1)
        if g1 is not None:
            ax2.axvline(o["gnorm_gt1_for_good_cum_lr"], color=arm["color"], lw=1.1, ls=":", alpha=0.85, zorder=1)
        tag = arm["nick"] + (" (running)" if o.get("provisional") else "")
        lines.append(f"{tag}: |dlogp| > .05 first " + (f"step {d1} = {sci(o['dlogp_gt05_first_cum_lr'])}" if d1 is not None else f"not yet (at {sci(o.get('cum_lr_at_last_step'))}, last-10 mean {fmt(o.get('dlogp_last10'))})")
                     + " · grad norm > 1 for good " + (f"step {g1} = {sci(o['gnorm_gt1_for_good_cum_lr'])}" if g1 is not None else f"not yet (last-10 mean {fmt(o.get('gnorm_last10'), 2)})"))
    ax2.text(0.01, 0.98, "\n".join(lines), transform=ax2.transAxes, fontsize=8, color=INK2, va="top", ha="left",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRID, alpha=0.92))
    for a in (ax1, ax2):
        a.set_xscale("log"); style_ax(a)
    ax1.set_xlim(1e-5, xmax * 1.6); ax1.set_ylim(0.33, 0.44)
    ax2.set_yscale("log"); ticks = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1]; ax2.set_yticks(ticks); ax2.set_yticklabels([f"{t:g}" for t in ticks]); ax2.set_yticks([], minor=True); ax2.set_ylim(0.018, 0.175)
    ax1.set_ylabel("held-out fidelity: mean cosine over 10 families"); ax2.set_ylabel("sampler vs trainer |delta log p| (log)")
    ax2.set_xlabel("cumulative learning rate x steps (log; lr summed over the updates, warmup ramp included)")
    ax1.set_title("held-out fidelity vs cumulative lr x steps", fontsize=9.6, loc="left")
    ax2.set_title(f"policy drift vs cumulative lr x steps (thin = per step, thick = {MA}-step mean) — dashed verticals = first |delta log p| > .05, dotted = grad norm above the clip for good (arm colours)", fontsize=9.6, loc="left")
    # claim
    r = bh["dlogp_gt05_first"]; g = bh["gnorm_gt1_for_good"]
    a_d, c_d = r["a"], r["c"]
    if a_d["cum_lr"] and c_d["cum_lr"]:
        verdict = (f"arm C crossed at step {c_d['step']} = {sci(c_d['cum_lr'])} vs arm A's step {a_d['step']} = {sci(a_d['cum_lr'])} (ratio {r['c_over_a_cum_lr']:.2f} in lr x steps, "
                   f"{r['c_over_a_steps']:.2f} in steps; a fixed budget predicts step {r['c_predicted_step']}) — {r['verdict']}")
    elif a_d["cum_lr"]:
        verdict = (f"arm A crossed at step {a_d['step']} = {sci(a_d['cum_lr'])}; a fixed budget predicts arm C crosses at step ≈{r['c_predicted_step']} — "
                   f"arm C is at step {N['budget_hypothesis']['c_last_step']:,} = {sci(N['budget_hypothesis']['c_cum_lr_now'])} with |delta log p| {fmt(N['budget_hypothesis']['c_dlogp_last10'])} "
                   f"(last-10 mean), so far {r['verdict']}")
    else:
        verdict = "onsets pending"
    head = (f"Does the drift wall sit at a fixed cumulative-lr budget rather than a fixed step count? Halving the lr (5e-7 vs 1e-6, same 8 x 2048 batch) should double the onset step: "
            f"{verdict}. Grad-norm rule: " + (f"A step {g['a']['step']} = {sci(g['a']['cum_lr'])}, C " + (f"step {g['c']['step']} = {sci(g['c']['cum_lr'])} ({g['verdict']})" if g["c"]["cum_lr"]
            else f"not yet (predicted step ≈{g.get('c_predicted_step')})") if g["a"]["cum_lr"] else "pending"))
    fig.suptitle(wrap(head, 175) + "\n" + wrap("Held-out fidelity (top) and sampler-trainer policy drift (bottom) vs cumulative lr x steps for the 8 x 512 arm (lr 1e-6, 25 warmup), "
                 "arm A (8 x 2048, lr 1e-6, 25 warmup) and arm C (8 x 2048, lr 5e-7 after a 100-step warmup, AdamW eps 1e-15, wd .01). Shaded band = the 8 x 512 arm's onset range by the two rules. "
                 "Arm B shares the 8 x 512 arm's schedule and is omitted.", 175), fontsize=10.2, x=0.01, ha="left", y=0.995)
    h, l = ax1.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=1, fontsize=8.4, bbox_to_anchor=(0.5, -0.005))
    finish(fig, bottom=0.1)
    save(fig, "fidelity_vs_lr_steps")


def plot_families(D, M, ABL, N):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.6), sharex=True)
    ref = D["ref"]["evals"]
    mid_final = D["mid"]["evals"][-1] if D["mid"]["evals"] else {}
    ref_before = ABL["ref_before_rl"] or {}
    xmax = max([D["runs"][k + "_train"]["last_step"] or 0 for k in ARM_ORDER] + [r["ckpt_step"] for k in ARM_ORDER for r in D["arms"][k]["evals"]] + [300])
    logx = xmax > LOG_STEP_AXIS_FROM
    leaders = {}
    for ax, (key, title) in zip(axes.flat, FAMILY_PANELS):
        xs, ys = xy(ref, key, ref_before.get(key), xmin=1 if logx else None)
        ax.plot(xs, ys, color=REF_C, lw=1.6, ls="--", marker="s", ms=5, mec="white", mew=1.0, label=REF_LABEL, zorder=3)
        for k in ARM_ORDER:
            plot_arm(ax, D, k, key=key, init=mid_final.get(key), label=arm_label(D, k), xmin=1 if logx else None, lw_scale=0.85, ms_scale=0.85, zorder=5 if k == "this" else 4)
        if logx and _num(mid_final.get(key)):
            ax.axhline(mid_final[key], color=AXIS, lw=1, ls=":", zorder=0)
        # who leads this family at step 300 (or the last step every finished arm has)
        vals = [(at(D["arms"][k]["evals"], 300, key), k) for k in ARM_ORDER if at(D["arms"][k]["evals"], 300, key) is not None]
        rv = at(ref, 300, key)
        if rv is not None:
            vals.append((rv, "ref"))
        sub = ""
        if vals:
            v, k = max(vals); nick = "old-bank reference" if k == "ref" else D["arms"][k]["meta"]["nick"]
            leaders[key] = {"arm": k, "nick": nick, "value": v, "step": 300, "all": {kk: vv for vv, kk in vals}}
            tiny = lambda kk: "reference" if kk == "ref" else D["arms"][kk]["meta"]["tiny"]
            rest = sorted(vals, reverse=True)[1:2]
            sub = f"\nleader @300: {tiny(k)} {v:.3f}" + (f" · next {tiny(rest[0][1])} {rest[0][0]:.3f}" if rest else "")
        ax.set_title(title + sub, fontsize=8.9, loc="left")
        step_axis(ax, xmax); style_ax(ax)
    for ax in axes[-1]:
        ax.set_xlabel("RL step (0 = the SFT init before RL)" if not logx else "RL step (log; dotted = the shared SFT init)", fontsize=9)
    fam_names = {k: FAM_PLAIN[k] for k, _ in FAMILY_PANELS}
    by_leader = {}
    for key, L in leaders.items():
        by_leader.setdefault(L["nick"], []).append(fam_names[key])
    head = "At step 300 the family leaders are: " + "; ".join(f"{n} on {', '.join(f)}" for n, f in by_leader.items()) if leaders else "Per-family held-out fidelity vs RL step"
    B = N["arms"]["b"]
    if B["last_eval"] and at(D["arms"]["b"]["evals"], 300, "eval/sae/norm_act") is not None:
        head += f" — the all-token reward lifts SAE fire-back to {at(D['arms']['b']['evals'], 300, 'eval/sae/norm_act'):.3f} (above the corpus max = 1)"
    fig.suptitle(wrap(head, 185) + "\n" + wrap("Per-family held-out fidelity vs RL step: the four full-parameter arms from the same re-cut-bank init (8 x 512 last-5 = blue; "
                 "8 x 2048 = purple; all-token reward = aqua; paper optimizer = ink, running) and the old-bank reference (orange dashed). The midtrain never saw long-context real "
                 "activations (RL-only family). MLP fire-back was not measured for the re-cut init (no step-0 point there).", 185), fontsize=10.2, x=0.01, ha="left", y=0.995)
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=1, fontsize=8.3, bbox_to_anchor=(0.5, -0.005))
    finish(fig, bottom=0.14)
    save(fig, "per_family_vs_rl_step")
    return leaders


def plot_dynamics(D, M, N):
    panels = [("reward/mean", "reward: mean cosine (max over the last 5 tokens; arm B: over all tokens)", False),
              ("policy/entropy", "policy entropy (nats / token)", False),
              ("grad_norm", "gradient norm before clipping (log; dotted = clip at 1.0)", True),
              ("policy/sampler_abs_dlogp", "sampler vs trainer |delta log p| per token (log; dotted = 0.05)", True),
              ("rollout/len_mean", "response length (tokens)", False),
              ("time/step_s", "wall-clock seconds per RL step", False),
              ("reward/peak_last_frac", "reward peak exactly at the last token (share of rollouts)", False),
              ("reward/peak_dist_mean", "reward peak distance from the last token (mean tokens, log)", True)]
    series = [("ref", D["ref"]["train"], REF_C, 1.3, "--", REF_LABEL)] + [(k, D["arms"][k]["train"], D["arms"][k]["meta"]["color"], D["arms"][k]["meta"]["lw"] * 0.85, D["arms"][k]["meta"]["ls"], arm_label(D, k)) for k in ARM_ORDER]
    xmax = max([t["step"][-1] for _, t, *_ in series if t["step"]] + [300])
    fig, axes = plt.subplots(2, 4, figsize=(19.5, 8.8), sharex=True)
    for ax, (key, title, logy) in zip(axes.flat, panels):
        for tag, t, c, lw, ls, lab in series:
            if not t["step"] or not any(_num(v) for v in t.get(key, [])):
                continue
            x = np.array(t["step"]); yy = np.array([np.nan if v is None else v for v in t[key]], dtype=float)
            if xmax > LOG_STEP_AXIS_FROM:
                keep = x >= 1; x, yy = x[keep], yy[keep]; ma = rolling(t[key])[keep]
            else:
                ma = rolling(t[key])
            ax.plot(x, yy, color=c, lw=0.6, alpha=0.2, zorder=2)
            ax.plot(x, ma, color=c, lw=lw, ls=ls, zorder=4 if tag == "this" else 3, label=lab if key == "reward/mean" else None, solid_capstyle="round")
        if key == "reward/peak_last_frac":
            tb = D["arms"]["b"]["train"]
            if tb["step"] and any(_num(v) for v in tb["reward/peak_in_last5_frac"]):
                ax.plot(tb["step"], rolling(tb["reward/peak_in_last5_frac"]), color=B_C, lw=1.1, ls=":", zorder=2)
                ax.text(0.99, 0.02, "aqua dotted = arm B's share of peaks inside the last 5 tokens\n(= 1 by construction for the last-5 arms)", transform=ax.transAxes, fontsize=7.6, color=INK2, ha="right", va="bottom")
            ax.set_ylim(0.2, 1.06)
        if key == "grad_norm":
            ax.axhline(1.0, color=INK2, lw=1, ls=":", zorder=1)
            txt = ["grad norm > 1 for good (2-consecutive rule):"]
            for k in ARM_ORDER:
                o = N["arms"][k]["onsets"] or {}
                txt.append(f"{D['arms'][k]['meta']['nick']}: {o.get('gnorm_gt1_for_good') if o.get('gnorm_gt1_for_good') is not None else 'not yet'} ({o.get('gnorm_gt1_2consec') if o.get('gnorm_gt1_2consec') is not None else '–'})"
                           + (" running" if o.get("provisional") else ""))
            ax.text(0.01, 0.98, "\n".join(txt), transform=ax.transAxes, fontsize=7.4, color=INK2, va="top", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=GRID, alpha=0.9))
        if key == "policy/sampler_abs_dlogp":
            ax.axhline(0.05, color=INK2, lw=1, ls=":", zorder=1)
            txt = ["|delta log p| > .05 first (3-consecutive):"]
            for k in ARM_ORDER:
                o = N["arms"][k]["onsets"] or {}
                txt.append(f"{D['arms'][k]['meta']['nick']}: {o.get('dlogp_gt05_first') if o.get('dlogp_gt05_first') is not None else 'not yet'} ({o.get('dlogp_gt05_3consec') if o.get('dlogp_gt05_3consec') is not None else '–'})"
                           + (" running" if o.get("provisional") else ""))
            ax.text(0.01, 0.98, "\n".join(txt), transform=ax.transAxes, fontsize=7.4, color=INK2, va="top", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=GRID, alpha=0.9))
        if logy:
            ax.set_yscale("log")
            ticks = {"grad_norm": [0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0], "policy/sampler_abs_dlogp": [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1],
                     "reward/peak_dist_mean": [0.5, 1, 2, 5, 10, 20]}[key]
            ax.set_yticks(ticks); ax.set_yticklabels([f"{t:g}" for t in ticks]); ax.set_yticks([], minor=True)
            ax.set_ylim(ticks[0] * 0.9, ticks[-1] * (1.9 if key in ("grad_norm", "policy/sampler_abs_dlogp") else 1.15))
        if key == "reward/mean":
            ax.set_ylim(0.15, 0.35)
        if key == "policy/entropy":
            ax.set_ylim(0.5, 3.0)
        if key == "time/step_s":
            ax.set_ylim(0, 180)
        if key == "rollout/len_mean":
            ax.set_ylim(15, 80)
        ax.set_title(title, fontsize=9.2, loc="left")
        step_axis(ax, xmax, allow_zero=False); style_ax(ax)
    for ax in axes[-1]:
        ax.set_xlabel("RL step" + (" (log)" if xmax > LOG_STEP_AXIS_FROM else ""), fontsize=9)
    T, A, B, C = (N["arms"][k] for k in ARM_ORDER)
    oT, oA, oB, oC = (N["arms"][k]["onsets"] or {} for k in ARM_ORDER)
    pB, pT = B["peak_metrics"], T["peak_metrics"]
    head = (f"4x the batch delays the grad-norm wall in steps (above the clip for good from {oA.get('gnorm_gt1_for_good', 'n/a')} vs {oT.get('gnorm_gt1_for_good', 'n/a')} for 8 x 512) but brings the "
            f"|delta log p| crossing EARLIER ({oA.get('dlogp_gt05_first', 'not yet')} vs {oT.get('dlogp_gt05_first', 'n/a')}); the all-token reward smears the peak "
            f"{fmt(pB['peak_dist_mean']['mean_from50'], 0) if pB else 'n/a'} tokens into the rollout (share peaking at the last token {fmt(pB['peak_last_frac']['mean_from50'], 2) if pB else 'n/a'} vs "
            f"{fmt(pT['peak_last_frac']['mean_from50'], 2) if pT else 'n/a'}; responses {fmt(B['dynamics'].get('len_mean_from50'), 0)} vs {fmt(T['dynamics'].get('len_mean_from50'), 0)} tokens); "
            f"the paper optimizer (lr 5e-7) keeps grad norm at {fmt(C['dynamics'].get('gnorm_last10'), 2)} and |delta log p| at {fmt(C['dynamics'].get('dlogp_last10'))} at step {C['last_step']:,}"
            + (" (running)" if not C["finished"] else ""))
    fig.suptitle(wrap(head, 215) + "\n" + wrap(f"RL training dynamics vs step — the four full-parameter arms from the same re-cut init plus the old-bank reference (orange dashed); thin = per step, "
                 f"thick = {MA}-step moving average. Onset steps in the boxes use the rules of the earlier sections (grad norm > 1 on 2 consecutive steps / for good; |delta log p| > .05 first / 3 consecutive; step >= 100). "
                 "The reference run did not log the peak-position metrics.", 215), fontsize=10.2, x=0.01, ha="left", y=0.995)
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=1, fontsize=8.3, bbox_to_anchor=(0.5, -0.005))
    finish(fig, bottom=0.13)
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
def write_data(D, M, N, ABL, bank_plot, leaders):
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
    arms_ev = {k: {"label": arm_label(D, k), "nick": D["arms"][k]["meta"]["nick"], "color": D["arms"][k]["meta"]["color"], "wandb_id": D["arms"][k]["meta"]["id"],
                   "rollouts_per_step": D["arms"][k]["meta"]["rollouts_per_step"], "evals": D["arms"][k]["evals"], "init": {"mean_all": M["init_mean_all"], "row": mid_final}}
               for k in ARM_ORDER}
    json.dump({"generated_at": D["fetched_at"], "table_keys": TABLE_KEYS, "family_panels": FAMILY_PANELS,
               "this": {"label": this_label(D, M), "evals": ev, "init": {"mean_all": M["init_mean_all"], "source": "midtrain final checkpoint eval (mix5msft_midtrain_fft_from_fft104m_eval, last ckpt)", "row": mid_final}},
               "ref": {"label": REF_LABEL, "evals": ref, "init": {"mean_all": M["ref_init_mean_all"], "source": ABL["ref_before_rl_source"], "row": ABL["ref_before_rl"]}},
               "arms": arms_ev, "arm_order": ARM_ORDER, "family_leaders_at_300": leaders,
               "lora_arms": ABL["arms"], "lora_arms_source": ABL["source"], "midtrain_evals": D["mid"]["evals"],
               "table_this_vs_ref": [init_row] + table}, open(dd / "eval_curves.json", "w"), indent=1)
    dyn = {}
    for tag in ["ref"] + ARM_ORDER:
        t = dict(D[tag]["train"] if tag == "ref" else D["arms"][tag]["train"])
        for k in DYN_MA_KEYS:
            if any(_num(v) for v in t.get(k, [])):
                t[k + f"__ma{MA}"] = [None if np.isnan(v) else float(v) for v in rolling(t[k])]
        if tag != "ref":
            arm = D["arms"][tag]["meta"]
            t["cum_lr"] = [cum_lr(arm, s) for s in t["step"]]
            t["cum_rollouts"] = [(s + 1) * arm["rollouts_per_step"] for s in t["step"]]
        dyn[tag] = {"run": D["runs"][tag + "_train"], "series": t, "label": REF_LABEL if tag == "ref" else arm_label(D, tag)}
    dyn["this"]["series"] = dyn["this"]["series"]
    json.dump({"generated_at": D["fetched_at"], "ma_window": MA, "onsets": M["onsets"], "peak_metrics": M["peak_metrics"], "dynamics_summary": M["dynamics"],
               "ref_dynamics_summary": M["ref_dynamics"], "arm_onsets": {k: N["arms"][k]["onsets"] for k in ARM_ORDER},
               "arm_dynamics_summary": {k: N["arms"][k]["dynamics"] for k in ARM_ORDER}, "arm_peak_metrics": {k: N["arms"][k]["peak_metrics"] for k in ARM_ORDER},
               "lr_schedule": N["lr_schedule"], "runs": dyn}, open(dd / "rl_dynamics.json", "w"), indent=1)
    json.dump({"generated_at": D["fetched_at"], "spec": BANK, "plotted": bank_plot}, open(dd / "bank_recut.json", "w"), indent=1)
    mt = D["mid"]["train"]
    json.dump({"generated_at": D["fetched_at"], "run": D["runs"]["mid_train"], "eval_run": D["runs"]["mid_eval"], "stats": M["midtrain"],
               "series": {k: mt[k] for k in ("step", "loss", "lr", "ex_per_s", "peak_mem_gb", "_timestamp")}}, open(dd / "midtrain.json", "w"), indent=1)
    json.dump({"generated_at": D["fetched_at"], **N, "family_leaders_at_300": leaders}, open(dd / "new_arms.json", "w"), indent=1, default=str)
    summ = {k: v for k, v in M.items() if k not in ("midtrain",)} | {"generated_at": D["fetched_at"], "runs": D["runs"], "midtrain_stats": {k: v for k, v in M["midtrain"].items() if k != "evals"},
                                                                    "bank_n_rows": BANK["n_rows"], "bank_families": BANK["families"], "save_steps": SAVE_STEPS, "ref_step": REF_STEP,
                                                                    "best_overall": N["best_overall"], "arm_status": {k: {"state": N["arms"][k]["state"], "last_step": N["arms"][k]["last_step"],
                                                                                                                         "pending_steps": N["arms"][k]["pending_steps"], "label": N["arms"][k]["label"],
                                                                                                                         "best": N["arms"][k]["best"], "last_eval": N["arms"][k]["last_eval"]} for k in ARM_ORDER}}
    json.dump(summ, open(dd / "summary.json", "w"), indent=1, default=str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-html", action="store_true", help="skip running the report folder's build_html.py")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    D = fetch(); ABL = load_ablation(); PRE = load_pretrain()
    M = summarize(D, ABL, PRE)
    N = summarize_arms(D, M)
    print(f"this arm: train {D['runs']['this_train']['state']} @ step {D['runs']['this_train']['last_step']}; evals {M['evaluated_steps']} pending {M['pending_steps']}; "
          f"init {fmt(M['init_mean_all'])}; best {M['best']}; ref@300 {fmt(M['ref_at_300'])} (init {fmt(M['ref_init_mean_all'])}) -> verdict: {M['verdict_word']}")
    for k in ARM_ORDER:
        A = N["arms"][k]
        print(f"arm {k}: {A['label']}\n   evals {[(e['ckpt_step'], round(e['eval/mean_all'], 4)) for e in A['evals']]} pending {A['pending_steps']}")
        o = A["onsets"] or {}
        print("   onsets:", json.dumps({kk: (round(v, 7) if isinstance(v, float) else v) for kk, v in o.items() if not kk.startswith("rule") and kk != "gnorm_le1_steps_after_100"}))
    print("matched-step deltas vs 8x512:", json.dumps({k: {s: round(v, 4) for s, v in d.items()} for k, d in N["matched_steps_delta_mean"].items()}))
    print("matched rollouts:", json.dumps([{kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in r.items() if kk in ("arm", "ckpt_step", "cum_rollouts", "mean_all", "base_equiv_step", "base_mean_all", "base_interpolated", "delta_vs_base", "reverse")} for r in N["matched_rollouts"] if r["arm"] != "b"]))
    print("best overall:", N["best_overall"])
    print("budget hypothesis:", N["budget_hypothesis"]["summary"])
    bank_plot = plot_bank()
    plot_headline(D, M, ABL, N); plot_rollouts(D, M, N); plot_lr_steps(D, M, N); leaders = plot_families(D, M, ABL, N); plot_dynamics(D, M, N)
    write_data(D, M, N, ABL, bank_plot, leaders)
    bh = OUT / "build_html.py"
    if not args.no_html and bh.exists():
        subprocess.run(["python3", str(bh)], check=True)


if __name__ == "__main__":
    main()
