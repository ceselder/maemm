import modal
f=modal.Function.from_name("maemm-rl-disagg","train")
saves="25,40,63,100,160,200,250,300,400,500,600,630,700,800,900,1000"   # log-spaced (x10^0.2) UNION every-100
c=f.spawn(n_rollout=2, n_trainer=6, total_steps=1000, extra_args=f"--cuda-graphs --max-num-seqs 512 --rollout-block-groups 64 --groups-per-step 256 --warmup-steps 25 --kl-coef 0 --entropy-coef 0.005 --entropy-target 2.0 --save-every 0 --save-steps {saves} --save-dir /data/ckpts_last5_disagg_2x6 --run-name rl_everything_8x256_disagg_entropy2.0_last5win --transcript-every 5")
print(c.object_id)
