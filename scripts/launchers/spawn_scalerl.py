import modal
saves="25,40,63,100,160,200,250,300,400,500,600,630,700,800,900,1000"
extra=(f"--recipe scalerl --loss cispo --cispo-eps-max 5 --loss-agg prompt --adv-mode batch --zero-var-filter --npr-threshold 0.9 --npr-pass-cos 0.7 --max-lag 8 --fp32-head --length-control penalty "
       f"--kl-coef 0 --entropy-coef 0 --entropy-target 0 --groups-per-step 256 --group-size 8 --lr 7e-6 --warmup-steps 25 --len-penalty-start 8 --len-penalty-per-tok 0.00025 --reward-window-last 5 "
       f"--init-adapter /data/sft_mix/last5_rp/final --cuda-graphs --max-num-seqs 512 --rollout-block-groups 64 --save-every 0 --save-steps {saves} --transcript-every 5 "
       f"--run-name rl_everything_8x256_disagg_scalerl_lr7e-6 --save-dir /data/ckpts_disagg_scalerl")
c=modal.Function.from_name("maemm-rl-disagg","train").spawn(n_rollout=2, n_trainer=6, total_steps=1000, extra_args=extra); print(c.object_id)
e=modal.Function.from_name("maemm-eval-ckpt","daemon").spawn(ckpt_dir="/data/ckpts_disagg_scalerl", tag="disagg_scalerl", rl_run_id="", wandb_name="rl_everything_8x256_disagg_scalerl_lr7e-6_eval", final_step=1000); print(e.object_id)
