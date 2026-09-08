"""RL run A: init = realact-only 23M SFT FINAL ckpt; bank = realact at uniformly random ctx lens. 8xB200 (2 rollout + 6 trainer), 16x256."""
import modal
saves = "25,40,63,100,160,200,250,300,400"
extra = (f"--recipe scalerl --loss cispo --cispo-eps-max 5 --loss-agg prompt --adv-mode batch --zero-var-filter --npr-threshold 0.9 --npr-pass-cos 0.7 --max-lag 8 --fp32-head --autocast-bf16 --length-control penalty "
         f"--kl-coef 0 --entropy-coef 0 --entropy-target 0 --groups-per-step 256 --group-size 16 --lr 7e-6 --warmup-steps 25 --len-penalty-start 8 --len-penalty-per-tok 0.00025 --max-new-tokens 192 --reward-window-last 5 "
         f"--init-adapter /data/sft_mix/realact20m_prefix_lr1e-4/final --cuda-graphs --max-num-seqs 512 --rollout-block-groups 32 --save-every 0 --save-steps {saves} --transcript-every 5 "
         f"--run-name rl_A_randctx_from_realact23m --save-dir /data/ckpts_rl_A_randctx")
c = modal.Function.from_name("maemm-rl-disagg", "train").spawn(n_rollout=2, n_trainer=6, total_steps=400, extra_args=extra, pool_dir="/data/banks/rl_randctx"); print(c.object_id)
