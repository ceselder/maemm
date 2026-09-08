import modal
saves="25,40,63,100,160,200,250,300,400,500,600,630,700,800,900,1000"
extra=(f"--recipe scalerl --loss cispo --cispo-eps-max 5 --loss-agg prompt --adv-mode batch --zero-var-filter --npr-threshold 0.9 --npr-pass-cos 0.7 --max-lag 8 --fp32-head --autocast-bf16 --length-control penalty "
       f"--kl-coef 0 --entropy-coef 0 --entropy-target 0 --groups-per-step 256 --group-size 16 --lr 7e-6 --warmup-steps 25 --len-penalty-start 8 --len-penalty-per-tok 0.00025 --max-new-tokens 192 --reward-window-last 5 "
       f"--init-adapter /data/sft_mix/last5_rp/final --cuda-graphs --max-num-seqs 512 --rollout-block-groups 32 --save-every 0 --save-steps {saves} --transcript-every 5 "
       f"--run-name rl_everything_16x256_disagg_scalerl_lr7e-6 --save-dir /data/ckpts_disagg_scalerl_16x256")
c=modal.Function.from_name("maemm-rl-disagg","train").spawn(n_rollout=2, n_trainer=6, total_steps=1000, extra_args=extra); print(c.object_id)
