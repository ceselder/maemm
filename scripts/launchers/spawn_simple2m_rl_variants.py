"""YOLO RL variants of the simple2m chain (user, 2026-09-17 18:3xZ): same init (/data/sft_mix/simple2m_sft/final), same even-split pool
(/data/banks/mix_simple2m_rl), same 300-step ScaleRL/CISPO recipe and cache-v3 evaluator as rl_simple2m_8x2048_anywin, varying:
    v16x1024   --group-size 16 --groups-per-step 1024   lr 1e-6, whole-span reward   (same 16,384 rollouts/step, 2x groups... half as many prompts)
    v8x2048_last16_lr5e-7   --groups-per-step 2048 --reward-window-last 16, lr 5e-7 (AdamW eps 1e-8, wd 0 -- NOT the paper optimizer)
    v8x2048_last16_lr1e-6   --groups-per-step 2048 --reward-window-last 16, lr 1e-6
Idempotent by key; ids -> ~/shared/overnight/simple2m/ids.json["rl_variants"].
    source ~/modal_venv/bin/activate && MODAL_PROFILE=safety-sahan python3 scripts/launchers/spawn_simple2m_rl_variants.py [key ...]
"""
import json
import os
import sys
import time

import modal

IDS = os.path.expanduser("~/shared/overnight/simple2m/ids.json")
APP, EVAL_APP = "maemm-rl-disagg-fullparam-s2m", "maemm-eval-ckpt-s2m"
POLICY, POOL, V3 = "/data/sft_mix/simple2m_sft/final", "/data/banks/mix_simple2m_rl", "/data/eval_universal_ho/eval_sets_heldout_v3.pt"
BASE = ("--recipe scalerl --loss cispo --cispo-eps-max 5 --loss-agg prompt --adv-mode batch --zero-var-filter --npr-threshold 0.9 --npr-pass-cos 0.7 "
        "--max-lag 2 --fp32-head --autocast-bf16 --length-control penalty --kl-coef 0 --entropy-coef 0 --entropy-target 0 "
        "--warmup-steps 25 --len-penalty-start 8 --len-penalty-per-tok 0.00025 --max-new-tokens 192 "
        "--prefix-cache --score-length-bucket --cuda-graphs --max-num-seqs 512 --rollout-block-groups 32 --save-every 0 --transcript-every 5 "
        f"--save-steps 25,50,100,150,200,250,300 --eval-cache {V3}")
VARIANTS = {
    "v16x1024": {"run": "rl_simple2m_16x1024_anywin", "flags": "--group-size 16 --groups-per-step 1024 --reward-window-last 0 --lr 1e-6",
                 "desc": "16 samples x 1,024 directions per step (16,384 rollouts/step), lr 1e-6, whole-span reward"},
    "v8x2048_last16_lr5e-7": {"run": "rl_simple2m_8x2048_last16_lr5e-7", "flags": "--group-size 8 --groups-per-step 2048 --reward-window-last 16 --lr 5e-7",
                              "desc": "8 x 2,048, lr 5e-7 (AdamW eps 1e-8, wd 0), reward = max cosine over the LAST 16 rollout tokens"},
    "v8x2048_last16_lr1e-6": {"run": "rl_simple2m_8x2048_last16_lr1e-6", "flags": "--group-size 8 --groups-per-step 2048 --reward-window-last 16 --lr 1e-6",
                              "desc": "8 x 2,048, lr 1e-6, reward = max cosine over the LAST 16 rollout tokens"},
}
keys = sys.argv[1:] or list(VARIANTS)
d = json.load(open(IDS)); d.setdefault("rl_variants", {})
for k in keys:
    if k in d["rl_variants"]:
        print(k, "already spawned:", d["rl_variants"][k]["train"]); continue
    v = VARIANTS[k]; save = f"/data/ckpts_{v['run']}"
    extra = f"{BASE} {v['flags']} --run-name {v['run']} --save-dir {save}"
    t = modal.Function.from_name(APP, "train").spawn(n_rollout=2, n_trainer=6, total_steps=300, extra_args=extra, pool_dir=POOL, policy_base=POLICY, full_param=True)
    e = modal.Function.from_name(EVAL_APP, "fullmodel_daemon").spawn(ckpt_dir=save, tag=v["run"], wandb_name=f"{v['run']}_eval", final_step=300,
                                                                     extra_args=f"--eval-cache {V3} --no-extra-evals")
    d["rl_variants"][k] = {"train": t.object_id, "eval": e.object_id, "run": v["run"], "save": save, "policy_base": POLICY, "pool": POOL, "steps": 300,
                           "flags": v["flags"], "desc": v["desc"], "app": APP, "eval_app": EVAL_APP, "eval_cache": V3, "spawned": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    print(k, "spawned: train", t.object_id, "eval", e.object_id, "|", v["desc"])
json.dump(d, open(IDS, "w"), indent=1)
