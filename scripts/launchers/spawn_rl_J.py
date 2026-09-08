"""RL-J: "8x4096" (group size 8 x 4096 prompts = 32,768 rollouts/step; 2x RL-I) at constant lr 1e-5 -- user: "if it is not too much time try 8x4096 even for RL".
Identical to RL-I otherwise (init mix1m_from_realact23m/final, bank mix_1m_mlp, ScaleRL/CISPO, --max-lag 2, 2 rollout + 6 trainer on 8xB200) but 320 steps
(enough to pass the ~250-step wall of E/F/I-at-1e-5 with margin; ~280 s/step x 320 = ~25 h -> may need one resume leg at the 24 h call timeout)."""
import json, modal
saves = "25,50,100,160,200,250,300,320"
extra = (f"--recipe scalerl --loss cispo --cispo-eps-max 5 --loss-agg prompt --adv-mode batch --zero-var-filter --npr-threshold 0.9 --npr-pass-cos 0.7 --max-lag 2 --fp32-head --autocast-bf16 --length-control penalty "
         f"--kl-coef 0 --entropy-coef 0 --entropy-target 0 --groups-per-step 4096 --group-size 8 --lr 1e-5 --warmup-steps 25 --len-penalty-start 8 --len-penalty-per-tok 0.00025 --max-new-tokens 192 --reward-window-last 5 "
         f"--init-adapter /data/sft_mix/mix1m_from_realact23m/final --cuda-graphs --max-num-seqs 512 --rollout-block-groups 32 --save-every 0 --save-steps {saves} --transcript-every 5 "
         f"--run-name rl_J_mixmlp_8x4096_lr1e-5 --save-dir /data/ckpts_rl_J_8x4096_lr1e-5")
t = modal.Function.from_name("maemm-rl-disagg", "train").spawn(n_rollout=3, n_trainer=5, total_steps=320, extra_args=extra, pool_dir="/data/banks/mix_1m_mlp")
e = modal.Function.from_name("maemm-eval-ckpt-mlp", "daemon").spawn(ckpt_dir="/data/ckpts_rl_J_8x4096_lr1e-5", tag="rl_J_8x4096_lr1e-5", rl_run_id="", wandb_name="rl_J_mixmlp_8x4096_lr1e-5_eval", final_step=320, extra_args="--eval-cache /data/eval_universal_ho/eval_sets_heldout_v2.pt")
json.dump({"J_8x4096_lr1e-5": {"train": t.object_id, "eval": e.object_id, "run": "rl_J_mixmlp_8x4096_lr1e-5", "save": "/data/ckpts_rl_J_8x4096_lr1e-5", "lr": "1e-5", "steps": 320, "geometry": "8x4096", "split": "2+6"}}, open("/home/celeste/shared/overnight/rl_J_ids.json", "w"), indent=1)
print("J train", t.object_id, "eval", e.object_id)
