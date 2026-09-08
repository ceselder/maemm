#!/usr/bin/env python3
"""Re-score the key inverter checkpoints with the MLP RANK evaluator (branch eval-mlp-rank, Modal app maemm-eval-ckpt-mlprank).

One --once daemon per checkpoint (1 x B200 each, ~15-20 min: the full held-out protocol, 512/family Bo4, extra evals OFF, no wandb,
per-direction dump ON, v2 eval cache with the mlp / mlp_pair families, --mlp-stats default = /data/mlp42/neuron_stats.npz).
Outputs: /data/eval_ckpt/mlprank_<name>/{ckpt_<k>.json, perdir_ckpt_<k>.json}.

    EVAL_APP=maemm-eval-ckpt-mlprank MODAL_PROFILE=safety-sahan modal deploy eval/modal_eval_ckpt.py      # from the eval-mlp-rank worktree
    MODAL_PROFILE=safety-sahan python scripts/launchers/spawn_mlprank_rescore.py [name ...]              # default: every checkpoint below
    MODAL_PROFILE=safety-sahan python scripts/launchers/spawn_mlprank_rescore.py --list                  # show the table, spawn nothing

RL ablation finals: the runs saved `final` (300 steps) rather than step_300 -> only_step = final_step = 300 resolves to <dir>/final.
initnewfft's adapters were trained on the full-fine-tune midtrain (run_meta.json policy_base) -> the daemon's --policy-base auto picks it up.
"""
import argparse
import json
import os
import time

import modal

APP = os.environ.get("EVAL_APP", "maemm-eval-ckpt-mlprank")
CACHE = "/data/eval_universal_ho/eval_sets_heldout_v2.pt"
EXTRA = f"--no-extra-evals --no-wandb --dump-per-dir --eval-cache {CACHE}"
IDS = os.path.expanduser("~/shared/overnight/mlprank_rescore_ids.json")

# name -> (ckpt_dir, ckpt_step to evaluate, final_step, note)
CKPTS = {
    "sft_realact23m":   ("/data/sft_mix/realact20m_prefix_lr1e-4/final",       0,   0,   "SFT baseline: 23M real-activation LoRA pretrain (final)"),
    "sft_midtrain":     ("/data/sft_mix/mixeq_midtrain_only_from_base/final",  0,   0,   "SFT baseline: midtrain-only LoRA on the equal six-family bank (final)"),
    "rl_abl_init23m":   ("/data/ckpts_rl_abl_init23m",                          300, 300, "RL ablation arm: 23M pretrain init, 300 steps (final)"),
    "rl_abl_initmid":   ("/data/ckpts_rl_abl_initmid",                          300, 300, "RL ablation arm: midtrain-only init, 300 steps (final)"),
    "rl_abl_initbase":  ("/data/ckpts_rl_abl_initbase",                         300, 300, "RL ablation arm: no SFT (fresh LoRA), 300 steps (final)"),
    "rl_abl_initboth":  ("/data/ckpts_rl_abl_initboth",                         300, 300, "RL ablation arm: 23M pretrain + midtrain init, 300 steps (final)"),
    "rl_abl_initnewfft": ("/data/ckpts_rl_abl_initnewfft",                      300, 300, "RL ablation arm: LoRA on the full-FT midtrain policy base, 300 steps (final)"),
    "rl_I_8x4096_nowarm": ("/data/ckpts_rl_I_8x4096_nowarm",                    150, 1000, "best production RL checkpoint (step 150)"),
}


def spawn(name, ckpt_dir, step, final_step, note):
    """A checkpoint saved as a bare adapter dir (the SFT finals) is served through a wrapper layout: the daemon scans <ckpt_dir>/step_* +
    final, so point it at the PARENT run dir and evaluate `final` (final_step = 0 -> ckpt_step 0, like the SFT-eval daemons did)."""
    if ckpt_dir.endswith("/final"):
        ckpt_dir = ckpt_dir[: -len("/final")]
    f = modal.Function.from_name(APP, "daemon")
    c = f.spawn(ckpt_dir=ckpt_dir, tag=f"mlprank_{name}", once=True, only_step=step, final_step=final_step, extra_args=EXTRA)
    rec = {"call_id": c.object_id, "ckpt_dir": ckpt_dir, "ckpt_step": step, "final_step": final_step, "note": note,
           "spawned_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "app": APP, "extra_args": EXTRA}
    print(f"[spawn] {name}: {c.object_id}  ({ckpt_dir} step {step}; {note})", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", help=f"subset of {list(CKPTS)} (default all)")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    names = a.names or list(CKPTS)
    for n in names:
        assert n in CKPTS, f"unknown checkpoint {n}; known: {list(CKPTS)}"
    if a.list:
        for n in names:
            print(n, CKPTS[n])
        return
    ids = json.load(open(IDS)) if os.path.exists(IDS) else {}
    for n in names:
        ids[n] = spawn(n, *CKPTS[n])
    os.makedirs(os.path.dirname(IDS), exist_ok=True)
    json.dump(ids, open(IDS, "w"), indent=1)
    print(f"[spawn] call ids -> {IDS}")


if __name__ == "__main__":
    main()
