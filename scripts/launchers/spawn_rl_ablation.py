"""SFT-init ablation (user 2026-09-08 07:20Z): RL 8x512 (group-size 8 x 512 groups = 4096 rollouts/step), constant lr 7e-6 (warmup 25), same recipe as RL-E..I,
ONE bank for every arm (/data/banks/mix_eq_1p45m, equal 6 families), 300 steps, saves 25..300, 3 rollout + 5 trainer B200. Arms:
  init23m  : /data/sft_mix/realact20m_prefix_lr1e-4/final       (23M real-activation LoRA pretrain, no midtrain)
  initbase : --init-adapter none                                  (fresh rsLoRA on the base model, no SFT at all)
  initmid  : /data/sft_mix/mixeq_midtrain_only_from_base/final   (midtrain-only LoRA on the equal bank, no pretrain; spawned by this script as SFT first,
                                                                  its RL arm is launched by /tmp/abl_armA_driver.py once the SFT finishes)
Usage: python3 spawn_rl_ablation.py sft | init23m | initbase | initmid"""
import json, os, sys, modal
P = "/home/celeste/shared/overnight/rl_ablation_ids.json"
d = json.load(open(P)) if os.path.exists(P) else {}
POOL = "/data/banks/mix_eq_1p45m"
SFT_RUN = "mixeq_midtrain_only_from_base"
INITS = {"init23m": "/data/sft_mix/realact20m_prefix_lr1e-4/final", "initbase": "none", "initmid": f"/data/sft_mix/{SFT_RUN}/final"}
what = sys.argv[1]
if what == "sft":
    t = modal.Function.from_name("maemm-sft-fullft", "train").spawn(run_name=SFT_RUN, data_dir=POOL, n_ckpts=8, epochs=1, batch_size=32, lr=1e-4, max_seq=160, backend="nccl",
                                                                   extra_args="--prefix-cache --grad-ckpt 0 --autocast-bf16 --log-steps 20")
    e = modal.Function.from_name("maemm-eval-ckpt-mlp", "daemon").spawn(ckpt_dir=f"/data/sft_mix/{SFT_RUN}", tag=f"sft_{SFT_RUN}", rl_run_id="", wandb_name=f"{SFT_RUN}_eval",
                                                                       final_step=1_450_446 // 256, extra_args="--eval-cache /data/eval_universal_ho/eval_sets_heldout_v2.pt")
    d["sft_initmid"] = {"train": t.object_id, "eval": e.object_id, "save": f"/data/sft_mix/{SFT_RUN}", "recipe": "LoRA r64 on base, mix_eq_1p45m, batch 32x8=256, lr 1e-4 OneCycle, 1 epoch (mirrors mix1m_from_realact23m minus --init-adapter)", "app": "maemm-sft-fullft"}
    print("SFT midtrain-only spawned:", t.object_id, "eval", e.object_id)
else:
    run = f"rl_abl_{what}_8x512_lr7e-6"; save = f"/data/ckpts_rl_abl_{what}"
    extra = ("--recipe scalerl --loss cispo --cispo-eps-max 5 --loss-agg prompt --adv-mode batch --zero-var-filter --npr-threshold 0.9 --npr-pass-cos 0.7 --max-lag 2 --fp32-head --autocast-bf16 --length-control penalty "
             "--kl-coef 0 --entropy-coef 0 --entropy-target 0 --groups-per-step 512 --group-size 8 --lr 7e-6 --warmup-steps 25 --len-penalty-start 8 --len-penalty-per-tok 0.00025 --max-new-tokens 192 --reward-window-last 5 "
             f"--init-adapter {INITS[what]} --prefix-cache --score-length-bucket --cuda-graphs --max-num-seqs 512 --rollout-block-groups 32 --save-every 0 --save-steps 25,50,100,150,200,250,300 --transcript-every 5 "
             f"--run-name {run} --save-dir {save}")
    t = modal.Function.from_name("maemm-rl-disagg-fast", "train").spawn(n_rollout=3, n_trainer=5, total_steps=300, extra_args=extra, pool_dir=POOL)
    e = modal.Function.from_name("maemm-eval-ckpt-mlp", "daemon").spawn(ckpt_dir=save, tag=f"rl_abl_{what}", rl_run_id="", wandb_name=f"{run}_eval", final_step=300,
                                                                       extra_args="--eval-cache /data/eval_universal_ho/eval_sets_heldout_v2.pt")
    d[what] = {"train": t.object_id, "eval": e.object_id, "run": run, "save": save, "init": INITS[what], "pool": POOL, "lr": "7e-6", "group_size": 8, "groups_per_step": 512, "steps": 300, "app": "maemm-rl-disagg-fast"}
    print(f"RL arm {what} spawned: train {t.object_id} eval {e.object_id}")
json.dump(d, open(P, "w"), indent=1)
