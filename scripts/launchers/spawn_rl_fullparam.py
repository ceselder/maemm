"""FULL-PARAMETER RL (rl_disagg --full-param, rl/rl_fullparam.py) on the deployed app maemm-rl-disagg-fullparam (8x B200: 3 vLLM
rollout + 5 FSDP2 trainer GPUs). Same ablation recipe as scripts/launchers/spawn_rl_ablation.py (8 samples x 512 directions per step,
ScaleRL/CISPO, pool mix_eq_1p45m), policy = the new-pretrain + full-fine-tune midtrain weights; the whole model is trained.

    python3 spawn_rl_fullparam.py smoke   [lr]     6 steps, no mid-run save (dev budget)
    python3 spawn_rl_fullparam.py val50   [lr]     50 steps, save at 25 + final, fullmodel eval daemon on the checkpoints
    python3 spawn_rl_fullparam.py prod    [lr]     the production arm: 300 steps, saves 25..300, run rl_abl_initnewfft_fullparam_8x512,
                                                   evaluator fullmodel_daemon (eval cache v2, no judge extras); ids -> rl_ablation_ids.json["initnewfft_fullparam"]
    python3 spawn_rl_fullparam.py bench5 <tag> [lr] 5 steps, no saves, run rl_fullparam_bench5_<tag> (step-time / memory benchmark of one config)
Extra rl_disagg flags after `--`, e.g. `python3 spawn_rl_fullparam.py smoke 1e-6 -- --publish-mode fs`; the split with
`--split 2+6` (n_rollout+n_trainer, default 3+5). The FAST step-time knobs (--suffix-ckpt --chunked-head --fsdp-prefetch 2; 68 -> 44 s/step)
are rl_disagg's DEFAULT for --full-param since 2026-09-09; the production arm rl_abl_initnewfft_fullparam_8x512 (launched 00:30Z that day)
runs the first validated configuration (= `-- --no-suffix-ckpt --no-chunked-head --fsdp-prefetch 0`). Recommended for FUTURE runs:
`--split 2+6` (fast default reaches micro-batch 32 there: 35 s/step vs 44 s on 3+5 vs 68 s for the production configuration; the two
engines still keep the 32-block rollout queue full)."""
import json
import os
import sys
import time

import modal

P = "/home/celeste/shared/overnight/rl_ablation_ids.json"
IDS_FULLRL = "/home/celeste/shared/overnight/rl_fullparam_ids.json"
POOL = "/data/banks/mix_eq_1p45m"
POLICY_BASE = "/data/sft_mix/mixeq_midtrain_fft_from_fft23m_v2/final"
APP, EVAL_APP = "maemm-rl-disagg-fullparam", "maemm-eval-ckpt-fullrl"
RECIPE = ("--recipe scalerl --loss cispo --cispo-eps-max 5 --loss-agg prompt --adv-mode batch --zero-var-filter --npr-threshold 0.9 --npr-pass-cos 0.7 "
          "--max-lag 2 --fp32-head --autocast-bf16 --length-control penalty --kl-coef 0 --entropy-coef 0 --entropy-target 0 --groups-per-step 512 "
          "--group-size 8 --warmup-steps 25 --len-penalty-start 8 --len-penalty-per-tok 0.00025 --max-new-tokens 192 --reward-window-last 5 "
          "--prefix-cache --score-length-bucket --cuda-graphs --max-num-seqs 512 --rollout-block-groups 32 --save-every 0 --transcript-every 5")

FAST = "--suffix-ckpt --chunked-head --fsdp-prefetch 2"
what = sys.argv[1]
rest = sys.argv[2:]
extra_flags = ""
if "--" in rest:
    i = rest.index("--"); extra_flags = " ".join(rest[i + 1:]); rest = rest[:i]
split = "3+5"
if "--split" in rest:
    i = rest.index("--split"); split = rest[i + 1]; rest = rest[:i] + rest[i + 2:]
n_rollout, n_trainer = (int(x) for x in split.split("+"))
tag = ""
if what == "bench5":
    tag, rest = rest[0], rest[1:]
lr = rest[0] if rest else "1e-6"
if what == "smoke":
    run, save, steps, saves = "rl_fullparam_smoke6", "/data/ckpts_fullrl_smoke6", 6, "999"
elif what == "val50":
    run, save, steps, saves = f"rl_fullparam_val50_lr{lr}", f"/data/ckpts_fullrl_val50_lr{lr}", 50, "25"
elif what == "prod":
    run, save, steps, saves = "rl_abl_initnewfft_fullparam_8x512", "/data/ckpts_rl_abl_initnewfft_fullparam", 300, "25,50,100,150,200,250,300"
elif what == "bench5":
    run, save, steps, saves = f"rl_fullparam_bench5_{tag}", f"/data/ckpts_fullrl_bench5_{tag}", 5, "999"
else:
    raise SystemExit(__doc__)
extra = f"{RECIPE} --lr {lr} --save-steps {saves} --run-name {run} --save-dir {save}" + (f" {extra_flags}" if extra_flags else "")
t = modal.Function.from_name(APP, "train").spawn(n_rollout=n_rollout, n_trainer=n_trainer, total_steps=steps, extra_args=extra, pool_dir=POOL,
                                                 policy_base=POLICY_BASE, full_param=True)
rec = {"train": t.object_id, "run": run, "save": save, "policy_base": POLICY_BASE, "pool": POOL, "lr": lr, "group_size": 8, "groups_per_step": 512,
       "steps": steps, "save_steps": saves, "app": APP, "full_param": True, "extra_flags": extra_flags, "split": split,
       "spawned": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "recipe": "FULL-PARAMETER RL (every weight, FSDP2 over 5 trainer GPUs, NCCL bf16 publish to 3 vLLM engines), ScaleRL/CISPO 8x512, "
                 f"lr {lr} constant after 25-step warmup, policy init = new-pretrain + FFT midtrain (mixeq_midtrain_fft_from_fft23m_v2/final)"}
if what in ("val50", "prod"):
    e = modal.Function.from_name(EVAL_APP, "fullmodel_daemon").spawn(ckpt_dir=save, tag=run, wandb_name=f"{run}_eval", final_step=steps,
                                                                    extra_args="--eval-cache /data/eval_universal_ho/eval_sets_heldout_v2.pt --no-extra-evals")
    rec["eval"] = e.object_id
    rec["eval_app"] = EVAL_APP
    rec["eval_note"] = "fullmodel_daemon: one eval_ckpt_daemon --full-model process per full-model checkpoint dir (SAVE_DONE), eval cache v2, no judge extras"
d = json.load(open(IDS_FULLRL)) if os.path.exists(IDS_FULLRL) else {}
d[what if what != "bench5" else f"bench5_{tag}"] = rec
json.dump(d, open(IDS_FULLRL, "w"), indent=1)
if what == "prod":
    a = json.load(open(P)) if os.path.exists(P) else {}
    a["initnewfft_fullparam"] = {**rec, "init": "none (full-parameter: the policy IS the FFT midtrain weights)"}
    json.dump(a, open(P, "w"), indent=1)
print(f"{what}: train {t.object_id}" + (f" eval {rec['eval']}" if "eval" in rec else "") + f" | run {run} | save {save} | lr {lr} | split {split} | extra {extra_flags!r}")
