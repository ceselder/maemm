"""Switch RL-I (8x4096, no warmup) onto the prefix-cache trainer: resume from its step_25 checkpoint (adapter + optimizer state) on app
maemm-rl-disagg-fast (same 3+5 split, same flags) with --prefix-cache --score-length-bucket, same wandb run (t8brhw8t), same save dir
(the running evaluator keeps watching it). Run AFTER cancelling the old trainer call + stopping its container. Usage: python3 /tmp/spawn_rl_I_fast.py <resume_step>"""
import json, sys, modal
step = int(sys.argv[1]) if len(sys.argv) > 1 else 25
max_seqs = int(sys.argv[2]) if len(sys.argv) > 2 else 512      # vLLM max_num_seqs per rollout engine (<=1024 with cuda-graphs)
block_groups = int(sys.argv[3]) if len(sys.argv) > 3 else 32
n_roll = int(sys.argv[4]) if len(sys.argv) > 4 else 3
n_train = int(sys.argv[5]) if len(sys.argv) > 5 else 5   # groups per rollout block (x8 samples = seqs per block)
saves = "25,50,100,150,200,250,300"
extra = (f"--recipe scalerl --loss cispo --cispo-eps-max 5 --loss-agg prompt --adv-mode batch --zero-var-filter --npr-threshold 0.9 --npr-pass-cos 0.7 --max-lag 2 --fp32-head --autocast-bf16 --length-control penalty "
         f"--kl-coef 0 --entropy-coef 0 --entropy-target 0 --groups-per-step 4096 --group-size 8 --lr 1e-5 --warmup-steps 0 --len-penalty-start 8 --len-penalty-per-tok 0.00025 --max-new-tokens 192 --reward-window-last 5 "
         f"--init-adapter /data/ckpts_rl_I_8x4096_nowarm/step_{step} --ref-adapter /data/sft_mix/mix1m_from_realact23m/final --step-offset {step + 1} --wandb-id t8brhw8t "
         f"--prefix-cache --score-length-bucket --cuda-graphs --max-num-seqs {max_seqs} --rollout-block-groups {block_groups} --save-every 0 --save-steps {saves} --transcript-every 5 "
         f"--run-name rl_I_mixmlp_8x4096_lr1e-5_nowarm --save-dir /data/ckpts_rl_I_8x4096_nowarm")
t = modal.Function.from_name("maemm-rl-disagg-fast", "train").spawn(n_rollout=n_roll, n_trainer=n_train, total_steps=300, extra_args=extra, pool_dir="/data/banks/mix_1m_mlp")
p = "/home/celeste/shared/overnight/rl_I_ids.json"; d = json.load(open(p)); k = "I_8x4096_lr1e-5_nowarm"
d[k]["train_slow_first_leg"] = d[k]["train"]; d[k]["train"] = t.object_id; d[k]["resumed_from_step"] = step; d[k]["app"] = "maemm-rl-disagg-fast"; d[k]["fast_flags"] = f"--prefix-cache --score-length-bucket --max-num-seqs {max_seqs} --rollout-block-groups {block_groups} split {n_roll}+{n_train}"
json.dump(d, open(p, "w"), indent=1); print("RL-I fast leg spawned", t.object_id, "from step", step)
