"""RL-I: 8 samples x 4096 directions per step (32,768 rollouts/step, 4x RL-F, 8x RL-E), otherwise identical to RL-E/F (init mix1m_from_realact23m/final,
bank mix_1m_mlp, ScaleRL/CISPO, constant lr 1e-5, 3 rollout + 5 trainer B200, --max-lag 2). User hypothesis: a ScaleRL-scale batch stabilises the constant-lr run.
Drift-budget prediction: onset ~ step 240-260 regardless. 300 steps; ~300+ s/step expected."""
import json, modal
saves = "25,50,100,150,200,250,300"
extra = (f"--recipe scalerl --loss cispo --cispo-eps-max 5 --loss-agg prompt --adv-mode batch --zero-var-filter --npr-threshold 0.9 --npr-pass-cos 0.7 --max-lag 2 --fp32-head --autocast-bf16 --length-control penalty "
         f"--kl-coef 0 --entropy-coef 0 --entropy-target 0 --groups-per-step 4096 --group-size 8 --lr 1e-5 --warmup-steps 0 --len-penalty-start 8 --len-penalty-per-tok 0.00025 --max-new-tokens 192 --reward-window-last 5 "
         f"--init-adapter /data/sft_mix/mix1m_from_realact23m/final --cuda-graphs --max-num-seqs 512 --rollout-block-groups 32 --save-every 0 --save-steps {saves} --transcript-every 5 "
         f"--run-name rl_I_mixmlp_8x4096_lr1e-5_nowarm --save-dir /data/ckpts_rl_I_8x4096_nowarm")
t = modal.Function.from_name("maemm-rl-disagg", "train").spawn(n_rollout=3, n_trainer=5, total_steps=300, extra_args=extra, pool_dir="/data/banks/mix_1m_mlp")
e = modal.Function.from_name("maemm-eval-ckpt-mlp", "daemon").spawn(ckpt_dir="/data/ckpts_rl_I_8x4096_nowarm", tag="rl_I_8x4096_nowarm", rl_run_id="", wandb_name="rl_I_mixmlp_8x4096_lr1e-5_nowarm_eval", final_step=300, extra_args="--eval-cache /data/eval_universal_ho/eval_sets_heldout_v2.pt")
json.dump({"I_8x4096_lr1e-5_nowarm": {"train": t.object_id, "eval": e.object_id, "run": "rl_I_mixmlp_8x4096_lr1e-5_nowarm", "save": "/data/ckpts_rl_I_8x4096_nowarm", "lr": "1e-5", "steps": 300, "group_size": 8, "groups_per_step": 4096}}, open("/home/celeste/shared/overnight/rl_I_ids.json", "w"), indent=1)
print("I train", t.object_id, "eval", e.object_id)
