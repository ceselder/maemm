"""RL-J relaunch as 3 rollout + 5 trainer (the 2+6 split was rollout-bound at ~420 s/step for 32,768 rollouts). Same run name / save dir;
the evaluator spawned by /tmp/spawn_rl_J.py (fc-01M1X0SPCTN22E770D5HY2NJYM) stays attached to /data/ckpts_rl_J_8x4096_lr1e-5."""
import json, modal
saves = "25,50,100,160,200,250,300,320"
extra = (f"--recipe scalerl --loss cispo --cispo-eps-max 5 --loss-agg prompt --adv-mode batch --zero-var-filter --npr-threshold 0.9 --npr-pass-cos 0.7 --max-lag 2 --fp32-head --autocast-bf16 --length-control penalty "
         f"--kl-coef 0 --entropy-coef 0 --entropy-target 0 --groups-per-step 4096 --group-size 8 --lr 1e-5 --warmup-steps 25 --len-penalty-start 8 --len-penalty-per-tok 0.00025 --max-new-tokens 192 --reward-window-last 5 "
         f"--init-adapter /data/sft_mix/mix1m_from_realact23m/final --cuda-graphs --max-num-seqs 512 --rollout-block-groups 32 --save-every 0 --save-steps {saves} --transcript-every 5 "
         f"--run-name rl_J_mixmlp_8x4096_lr1e-5 --save-dir /data/ckpts_rl_J_8x4096_lr1e-5")
t = modal.Function.from_name("maemm-rl-disagg", "train").spawn(n_rollout=3, n_trainer=5, total_steps=320, extra_args=extra, pool_dir="/data/banks/mix_1m_mlp")
p = "/home/celeste/shared/overnight/rl_J_ids.json"; d = json.load(open(p))
d["J_8x4096_lr1e-5"]["train_first_2p6_cancelled"] = "fc-01M1X0SP5NAQHGD3ZB5DWRJEZW"; d["J_8x4096_lr1e-5"]["train"] = t.object_id; d["J_8x4096_lr1e-5"]["split"] = "3+5"
json.dump(d, open(p, "w"), indent=1); print("J relaunched (3+5):", t.object_id)
